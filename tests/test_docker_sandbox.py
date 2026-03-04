"""Tests for Docker sandbox module.

All tests use mocked Docker client — no real Docker daemon required.
The ``_DOCKER_AVAILABLE`` flag is patched to ``True`` so that
``DockerSandbox`` and ``DockerSandboxPool`` can be instantiated with
a mocked client.
"""

from __future__ import annotations

import io
import tarfile
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

import swarm.core.docker_sandbox as ds_module

# ---------------------------------------------------------------------------
# Auto-use fixture: pretend Docker SDK is available
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_docker_available():
    """Patch _DOCKER_AVAILABLE to True for all tests in this module."""
    original = ds_module._DOCKER_AVAILABLE
    ds_module._DOCKER_AVAILABLE = True
    yield
    ds_module._DOCKER_AVAILABLE = original


# ---------------------------------------------------------------------------
# Fixtures for a mock Docker client
# ---------------------------------------------------------------------------


class MockContainer:
    """Minimal Docker container mock."""

    def __init__(self, name: str = "test-container", image_name: str = "python:3.12-slim"):
        self.id = "abc123def456"
        self.short_id = "abc123d"
        self.name = name
        self.status = "running"
        self.image = MagicMock()
        self.image.tags = [image_name]
        self.image.id = "sha256:abc123"
        self.labels = {"managed-by": "swarm"}
        self._paused = False

    def stop(self, timeout: int = 10) -> None:
        self.status = "exited"

    def kill(self) -> None:
        self.status = "exited"

    def remove(self, force: bool = False) -> None:
        self.status = "removed"

    def pause(self) -> None:
        self._paused = True
        self.status = "paused"

    def unpause(self) -> None:
        self._paused = False
        self.status = "running"

    def commit(self, repository: str = "", tag: str = "") -> MagicMock:
        img = MagicMock()
        img.id = f"sha256:snapshot-{tag}"
        return img

    def logs(self, tail: int = 100) -> bytes:
        return b"container log output\n"

    def stats(self, stream: bool = False) -> Dict[str, Any]:
        return {
            "cpu_stats": {"cpu_usage": {"total_usage": 100000}},
            "memory_stats": {"usage": 50_000_000, "limit": 500_000_000},
        }

    def put_archive(self, path: str, data: Any) -> bool:
        return True

    def get_archive(self, path: str) -> tuple:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            content = b"file content"
            info = tarfile.TarInfo(name="test.txt")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        buf.seek(0)
        return ([buf.read()], {"size": 12})


class MockDockerClient:
    """Minimal Docker client mock."""

    def __init__(self) -> None:
        self._containers: List[MockContainer] = []
        self.containers = self
        self.images = self
        self.api = MockDockerAPI(self)

    def run(self, **kwargs: Any) -> MockContainer:
        container = MockContainer(
            name=kwargs.get("name", "mock-container"),
            image_name=kwargs.get("image", "python:3.12-slim"),
        )
        self._containers.append(container)
        return container

    def list(self, all: bool = False, filters: Optional[Dict] = None) -> List[MockContainer]:
        return list(self._containers)

    def pull(self, image: str, **kwargs: Any) -> MagicMock:
        return MagicMock()

    def ping(self) -> bool:
        return True


class MockDockerAPI:
    """Minimal low-level Docker API mock for exec operations."""

    def __init__(self, client: MockDockerClient):
        self._client = client

    def exec_create(
        self,
        container_id: str,
        cmd: List[str],
        user: str = "",
        workdir: str = "",
        stdout: bool = True,
        stderr: bool = True,
    ) -> Dict[str, str]:
        return {"Id": "exec-001"}

    def exec_start(self, exec_id: str, demux: bool = False) -> tuple:
        return (b"hello world\n", b"")

    def exec_inspect(self, exec_id: str) -> Dict[str, Any]:
        return {"ExitCode": 0, "Running": False}


# ---------------------------------------------------------------------------
# Tests for ContainerSpec and contract_to_spec
# ---------------------------------------------------------------------------


class TestContainerSpec:
    def test_default_spec(self):
        spec = ds_module.ContainerSpec()
        assert spec.image == "python:3.12-slim"
        assert spec.mem_limit == "512m"
        assert spec.network_mode == "none"
        assert spec.read_only_root is True
        assert "no-new-privileges:true" in spec.security_opt
        assert "ALL" in spec.cap_drop
        assert spec.pids_limit == 256

    def test_custom_spec(self):
        spec = ds_module.ContainerSpec(
            image="ubuntu:22.04",
            mem_limit="1g",
            cpu_shares=2048,
            network_mode="bridge",
            pids_limit=512,
        )
        assert spec.image == "ubuntu:22.04"
        assert spec.mem_limit == "1g"
        assert spec.cpu_shares == 2048
        assert spec.network_mode == "bridge"
        assert spec.pids_limit == 512


class TestContractToSpec:
    def test_deny_all_network(self):
        from swarm.bridges.opensandbox.config import GovernanceContract, NetworkPolicy

        contract = GovernanceContract(
            contract_id="test",
            tier="restricted",
            network=NetworkPolicy.DENY_ALL,
            max_memory_mb=256,
            max_cpu_shares=512,
        )
        spec = ds_module.contract_to_spec(contract, agent_id="agent-1")
        assert spec.network_mode == "none"
        assert spec.mem_limit == "256m"
        assert spec.cpu_shares == 512
        assert "swarm.agent_id" in spec.labels
        assert spec.labels["swarm.agent_id"] == "agent-1"
        assert spec.labels["swarm.tier"] == "restricted"
        assert spec.env["SWARM_AGENT_ID"] == "agent-1"

    def test_allowlist_network_fails_closed(self):
        """H4 fix: ALLOWLIST falls back to 'none' until enforced."""
        from swarm.bridges.opensandbox.config import GovernanceContract, NetworkPolicy

        contract = GovernanceContract(
            contract_id="standard",
            tier="standard",
            network=NetworkPolicy.ALLOWLIST,
        )
        spec = ds_module.contract_to_spec(contract)
        assert spec.network_mode == "none"  # fail-closed

    def test_full_network(self):
        from swarm.bridges.opensandbox.config import GovernanceContract, NetworkPolicy

        contract = GovernanceContract(
            contract_id="priv",
            tier="privileged",
            network=NetworkPolicy.FULL,
        )
        spec = ds_module.contract_to_spec(contract)
        assert spec.network_mode == "bridge"

    def test_mounts_from_contract(self):
        from swarm.bridges.opensandbox.config import GovernanceContract

        contract = GovernanceContract(
            contract_id="mount-test",
            allowed_mounts=["/data/shared", "/models"],
        )
        spec = ds_module.contract_to_spec(contract)
        assert len(spec.mounts) == 2
        assert spec.mounts[0]["source"] == "/data/shared"
        assert spec.mounts[0]["read_only"] is True

    def test_extra_env(self):
        from swarm.bridges.opensandbox.config import GovernanceContract

        contract = GovernanceContract(contract_id="env-test")
        spec = ds_module.contract_to_spec(
            contract,
            extra_env={"CUSTOM_VAR": "value"},
        )
        assert spec.env["CUSTOM_VAR"] == "value"

    def test_image_override(self):
        from swarm.bridges.opensandbox.config import GovernanceContract

        contract = GovernanceContract(contract_id="img-test")
        spec = ds_module.contract_to_spec(contract, image="custom:latest")
        assert spec.image == "custom:latest"


# ---------------------------------------------------------------------------
# Tests for DockerSandbox (with mocked client)
# ---------------------------------------------------------------------------


class TestDockerSandbox:
    def _make_sandbox(self) -> ds_module.DockerSandbox:
        """Create a DockerSandbox with a mocked client."""
        spec = ds_module.ContainerSpec(name="test-sandbox")
        client = MockDockerClient()
        return ds_module.DockerSandbox(spec, client=client)

    def test_start_creates_container(self):
        sandbox = self._make_sandbox()
        cid = sandbox.start()
        assert cid is not None
        assert sandbox.state.value == "running"
        assert sandbox.container_id is not None

    def test_exec_returns_result(self):
        sandbox = self._make_sandbox()
        sandbox.start()
        result = sandbox.exec("echo hello")
        assert result.exit_code == 0
        assert "hello world" in result.stdout
        assert result.duration_ms >= 0
        assert result.command == "echo hello"
        assert not result.timed_out

    def test_exec_log_accumulates(self):
        sandbox = self._make_sandbox()
        sandbox.start()
        sandbox.exec("echo 1")
        sandbox.exec("echo 2")
        sandbox.exec("echo 3")
        assert len(sandbox.exec_log) == 3

    def test_exec_on_stopped_container_raises(self):
        sandbox = self._make_sandbox()
        sandbox.start()
        sandbox.stop()
        with pytest.raises(RuntimeError, match="not running"):
            sandbox.exec("echo fail")

    def test_snapshot(self):
        sandbox = self._make_sandbox()
        sandbox.start()
        image_id = sandbox.snapshot(tag="v1")
        assert image_id is not None
        assert len(sandbox.snapshots) == 1

    def test_pause_unpause(self):
        sandbox = self._make_sandbox()
        sandbox.start()
        sandbox.pause()
        assert sandbox.state.value == "paused"
        sandbox.unpause()
        assert sandbox.state.value == "running"

    def test_stop_and_remove(self):
        sandbox = self._make_sandbox()
        sandbox.start()
        sandbox.stop()
        assert sandbox.state.value == "stopped"
        sandbox.remove()
        assert sandbox.state.value == "removed"
        assert sandbox.container_id is None

    def test_context_manager(self):
        spec = ds_module.ContainerSpec(name="ctx-test")
        client = MockDockerClient()
        with ds_module.DockerSandbox(spec, client=client) as sandbox:
            result = sandbox.exec("echo hi")
            assert result.exit_code == 0
        assert sandbox.state.value == "removed"

    def test_get_stats(self):
        sandbox = self._make_sandbox()
        sandbox.start()
        stats = sandbox.get_stats()
        assert "cpu_stats" in stats

    def test_get_logs(self):
        sandbox = self._make_sandbox()
        sandbox.start()
        logs = sandbox.get_logs()
        assert "container log" in logs

    def test_copy_from(self):
        sandbox = self._make_sandbox()
        sandbox.start()
        data = sandbox.copy_from("/workspace/test.txt")
        assert data == b"file content"

    def test_lifetime_exceeded_returns_timeout(self):
        """Container lifetime check in exec."""
        spec = ds_module.ContainerSpec(name="timeout-test", timeout_seconds=0)
        client = MockDockerClient()
        sandbox = ds_module.DockerSandbox(spec, client=client)
        sandbox.start()
        # Force created_at to past
        sandbox._created_at = 0.0
        result = sandbox.exec("echo hi")
        assert result.timed_out
        assert result.exit_code == -1


# ---------------------------------------------------------------------------
# Tests for DockerSandboxPool
# ---------------------------------------------------------------------------


class TestDockerSandboxPool:
    def _make_pool(self, max_containers: int = 5) -> ds_module.DockerSandboxPool:
        client = MockDockerClient()
        return ds_module.DockerSandboxPool(max_containers=max_containers, client=client)

    def test_create_and_start(self):
        pool = self._make_pool()
        spec = ds_module.ContainerSpec(name="pool-test")
        sandbox = pool.create(spec)
        sandbox.start()
        assert pool.active_count == 1

    def test_capacity_limit(self):
        pool = self._make_pool(max_containers=2)
        pool.create(ds_module.ContainerSpec(name="s1"))
        pool.create(ds_module.ContainerSpec(name="s2"))
        with pytest.raises(RuntimeError, match="capacity"):
            pool.create(ds_module.ContainerSpec(name="s3"))

    def test_destroy(self):
        pool = self._make_pool()
        sandbox = pool.create(ds_module.ContainerSpec(name="d1"))
        sandbox.start()
        pool.destroy(sandbox)
        assert pool.active_count == 0

    def test_destroy_all(self):
        pool = self._make_pool()
        for i in range(3):
            s = pool.create(ds_module.ContainerSpec(name=f"batch-{i}"))
            s.start()
        count = pool.destroy_all()
        assert count == 3
        assert pool.active_count == 0

    def test_list_active(self):
        pool = self._make_pool()
        s = pool.create(ds_module.ContainerSpec(name="list-test"))
        s.start()
        active = pool.list_active()
        assert len(active) == 1
        assert active[0]["name"] == "list-test"
        assert active[0]["state"] == "running"

    def test_cleanup_stale(self):
        pool = self._make_pool()
        spec = ds_module.ContainerSpec(name="stale", timeout_seconds=0)
        s = pool.create(spec)
        s.start()
        s._created_at = 0.0  # Force expired
        cleaned = pool.cleanup_stale()
        assert cleaned == 1
        assert pool.active_count == 0


# ---------------------------------------------------------------------------
# Tests for DockerSandboxBackend (FailoverChain integration)
# ---------------------------------------------------------------------------


class TestDockerSandboxBackend:
    def test_backend_properties(self):
        """Verify the backend implements ExecutionBackend correctly."""
        spec = ds_module.ContainerSpec(name="backend-test")
        backend = ds_module.DockerSandboxBackend(spec)
        assert backend.name.startswith("docker:")
        assert backend.retry_policy.max_retries == 2


# ---------------------------------------------------------------------------
# Tests for utility functions
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_docker_unavailable_error(self):
        with pytest.raises(ds_module.DockerUnavailableError):
            raise ds_module.DockerUnavailableError("test")

    def test_container_state_enum(self):
        assert ds_module.ContainerState.CREATED.value == "created"
        assert ds_module.ContainerState.RUNNING.value == "running"
        assert ds_module.ContainerState.STOPPED.value == "stopped"
        assert ds_module.ContainerState.REMOVED.value == "removed"

    def test_exec_result_fields(self):
        r = ds_module.ExecResult(
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_ms=42.5,
            command="echo ok",
        )
        assert r.exit_code == 0
        assert r.stdout == "ok"
        assert r.duration_ms == 42.5
        assert not r.timed_out

    def test_ensure_docker_raises_when_unavailable(self):
        """When _DOCKER_AVAILABLE is False, _ensure_docker raises."""
        original = ds_module._DOCKER_AVAILABLE
        try:
            ds_module._DOCKER_AVAILABLE = False
            with pytest.raises(ds_module.DockerUnavailableError):
                ds_module._ensure_docker()
        finally:
            ds_module._DOCKER_AVAILABLE = original


# ---------------------------------------------------------------------------
# Tests for security fixes
# ---------------------------------------------------------------------------


class TestSecurityValidation:
    """Tests for security hardening fixes (C1, C2, H1, H4, M5, L2)."""

    def test_validate_path_blocks_traversal(self):
        """C2 fix: path traversal is rejected."""
        with pytest.raises(ValueError, match="traversal"):
            ds_module._validate_container_path("../../../etc/passwd")

    def test_validate_path_blocks_shell_metachar(self):
        """C1 fix: shell metacharacters in paths are rejected."""
        for bad_char in (";", "&", "|", "`", "$", "(", ")"):
            with pytest.raises(ValueError, match="Invalid character"):
                ds_module._validate_container_path(f"/workspace/file{bad_char}name")

    def test_validate_path_accepts_normal_paths(self):
        """Normal paths pass validation."""
        # Should not raise
        ds_module._validate_container_path("/workspace/main.py")
        ds_module._validate_container_path("/tmp/data/output.json")
        ds_module._validate_container_path("relative/path.txt")

    def test_sanitize_agent_id(self):
        """L2 fix: agent_id is sanitized for Docker names."""
        assert ds_module._sanitize_agent_id("agent-1") == "agent-1"
        assert ds_module._sanitize_agent_id("agent with spaces") == "agent-with-spaces"
        assert ds_module._sanitize_agent_id("agent;rm -rf /") == "agent-rm--rf--"

    def test_shell_quote(self):
        """Verify shell quoting handles single quotes."""
        assert ds_module._shell_quote("hello") == "'hello'"
        assert ds_module._shell_quote("it's") == "'it'\\''s'"

    def test_contract_to_spec_explicit_security_fields(self):
        """H3 fix: security fields are explicitly set."""
        from swarm.bridges.opensandbox.config import GovernanceContract

        contract = GovernanceContract(contract_id="h3-test")
        spec = ds_module.contract_to_spec(contract)
        assert spec.cap_drop == ["ALL"]
        assert "no-new-privileges:true" in spec.security_opt
        assert spec.cap_add == []
        assert spec.read_only_root is True
        assert spec.pids_limit == 256
        assert spec.cpu_quota == 100_000  # I4 fix: hard CPU limit

    def test_contract_to_spec_blocks_swarm_env_override(self):
        """M5 fix: extra_env cannot override SWARM_ namespace."""
        from swarm.bridges.opensandbox.config import GovernanceContract

        contract = GovernanceContract(contract_id="m5-test")
        spec = ds_module.contract_to_spec(
            contract,
            extra_env={"SWARM_TIER": "malicious", "CUSTOM_VAR": "ok"},
        )
        # SWARM_TIER should NOT be overridden
        assert spec.env.get("SWARM_TIER") != "malicious"
        # Custom var should be set
        assert spec.env["CUSTOM_VAR"] == "ok"

    def test_allowlist_fails_closed(self):
        """H4 fix: ALLOWLIST → none (not bridge)."""
        from swarm.bridges.opensandbox.config import GovernanceContract, NetworkPolicy

        contract = GovernanceContract(
            contract_id="h4-test",
            network=NetworkPolicy.ALLOWLIST,
        )
        spec = ds_module.contract_to_spec(contract)
        assert spec.network_mode == "none"

    def test_copy_to_rejects_traversal(self):
        """C2 fix: copy_to validates container_path."""
        sandbox = TestDockerSandbox()._make_sandbox()
        sandbox.start()
        with pytest.raises(ValueError, match="traversal"):
            sandbox.copy_to("/tmp/safe_file", "../../../etc/evil")

    def test_copy_from_validates_path(self):
        """H1 fix: copy_from validates container_path."""
        sandbox = TestDockerSandbox()._make_sandbox()
        sandbox.start()
        with pytest.raises(ValueError, match="Invalid character"):
            sandbox.copy_from("/workspace/$(malicious)")

    def test_docker_exec_user_root_overridden(self):
        """I5 fix: docker_exec_user='root' is overridden to 'nobody'."""
        from swarm.bridges.opensandbox.config import OpenSandboxConfig

        config = OpenSandboxConfig(docker_exec_user="root")
        assert config.docker_exec_user == "nobody"

    def test_docker_exec_user_zero_overridden(self):
        """I5 fix: docker_exec_user='0' (root uid) is overridden."""
        from swarm.bridges.opensandbox.config import OpenSandboxConfig

        config = OpenSandboxConfig(docker_exec_user="0")
        assert config.docker_exec_user == "nobody"


# ---------------------------------------------------------------------------
# Tests for OpenSandboxBridge Docker integration
# ---------------------------------------------------------------------------


class TestBridgeDockerIntegration:
    """Test that the OpenSandbox bridge correctly uses Docker when enabled."""

    def _make_bridge_with_docker(self) -> Any:
        """Create a bridge with Docker enabled and mocked pool."""
        from swarm.bridges.opensandbox.bridge import OpenSandboxBridge
        from swarm.bridges.opensandbox.config import OpenSandboxConfig

        config = OpenSandboxConfig(docker_enabled=True)
        bridge = OpenSandboxBridge(config)

        # Inject a mock Docker pool and container map
        mock_pool = MagicMock()
        mock_sandbox = MagicMock()
        mock_sandbox.container_id = "mock-cid"
        mock_sandbox.exec.return_value = MagicMock(
            exit_code=0,
            stdout="executed",
            stderr="",
            duration_ms=100.0,
        )
        mock_pool.create.return_value = mock_sandbox
        mock_pool.active_count = 0
        mock_pool.max_containers = 20
        mock_pool.list_active.return_value = []
        bridge._docker_pool = mock_pool

        return bridge, mock_pool, mock_sandbox

    def test_docker_status_when_disabled(self):
        from swarm.bridges.opensandbox.bridge import OpenSandboxBridge
        from swarm.bridges.opensandbox.config import OpenSandboxConfig

        bridge = OpenSandboxBridge(OpenSandboxConfig(docker_enabled=False))
        status = bridge.get_docker_status()
        assert status["enabled"] is False

    def test_docker_status_when_enabled(self):
        bridge, _, _ = self._make_bridge_with_docker()
        status = bridge.get_docker_status()
        assert status["enabled"] is True

    def test_execute_command_uses_docker_container(self):
        """When Docker container exists for sandbox, exec runs there."""
        from swarm.bridges.opensandbox.config import (
            CapabilityManifest,
            GovernanceContract,
        )

        bridge, mock_pool, mock_sandbox = self._make_bridge_with_docker()

        # Publish contract, screen agent, create sandbox
        contract = GovernanceContract(
            contract_id="std",
            tier="standard",
            capabilities=["python", "echo"],
        )
        bridge.publish_contract(contract)

        manifest = CapabilityManifest(agent_id="agent-1")
        assignment = bridge.screen_agent(manifest)

        sandbox_id = bridge.create_sandbox(assignment)

        # Manually inject the mock Docker sandbox for this sandbox_id
        bridge._docker_containers[sandbox_id] = mock_sandbox

        # Execute command
        interaction = bridge.execute_command("agent-1", "echo hello")
        mock_sandbox.exec.assert_called_once()
        assert interaction.p >= 0.0
        assert interaction.p <= 1.0

    def test_shutdown_cleans_docker_pool(self):
        bridge, mock_pool, _ = self._make_bridge_with_docker()
        bridge.shutdown()
        mock_pool.shutdown.assert_called_once()
