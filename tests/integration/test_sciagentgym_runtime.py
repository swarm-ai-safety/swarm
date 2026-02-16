"""Integration tests: SciAgentGym Runtime - Environment lifecycle + tool invocation.

Tests environment lifecycle (create → use → cleanup) and verifies no resource leaks.
Follows patterns from test_worktree_bridge.py for comprehensive lifecycle testing.
"""

import gc
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Mock SciAgentGym Runtime Infrastructure
# ---------------------------------------------------------------------------


class ToolInvocation:
    """Represents a single tool invocation with inputs and outputs."""

    def __init__(
        self,
        tool_name: str,
        args: Dict[str, Any],
        env_id: str,
        timestamp: float,
    ):
        self.tool_name = tool_name
        self.args = args
        self.env_id = env_id
        self.timestamp = timestamp
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.completed: bool = False

    def execute(self) -> Dict[str, Any]:
        """Execute the tool and return results."""
        if self.tool_name == "read_file":
            self.result = {"content": f"Mock content from {self.args.get('path')}"}
        elif self.tool_name == "write_file":
            self.result = {"success": True, "bytes_written": len(self.args.get("content", ""))}
        elif self.tool_name == "list_dir":
            self.result = {"files": ["file1.txt", "file2.py", "file3.md"]}
        elif self.tool_name == "run_command":
            self.result = {"stdout": "Command output", "stderr": "", "exit_code": 0}
        else:
            self.error = f"Unknown tool: {self.tool_name}"
            self.result = None

        self.completed = True
        return {"result": self.result, "error": self.error}


class SciAgentGymEnvironment:
    """Mock SciAgentGym environment for lifecycle testing.

    Tracks resources (files, processes, memory) to detect leaks.
    """

    def __init__(self, env_id: str, workspace_dir: Path):
        self.env_id = env_id
        self.workspace_dir = workspace_dir
        self.created_at = time.time()
        self.last_active = self.created_at
        self.is_active = True
        self.invocations: List[ToolInvocation] = []
        self.resource_usage: Dict[str, int] = {
            "files_created": 0,
            "commands_run": 0,
            "memory_mb": 0,
        }

        # Create workspace directory
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def invoke_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a tool in this environment."""
        if not self.is_active:
            raise RuntimeError(f"Environment {self.env_id} is not active")

        invocation = ToolInvocation(
            tool_name=tool_name,
            args=args,
            env_id=self.env_id,
            timestamp=time.time(),
        )

        result = invocation.execute()
        self.invocations.append(invocation)
        self.last_active = time.time()

        # Track resource usage
        if tool_name == "write_file":
            self.resource_usage["files_created"] += 1
        elif tool_name == "run_command":
            self.resource_usage["commands_run"] += 1

        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Get environment metrics."""
        return {
            "env_id": self.env_id,
            "uptime_seconds": time.time() - self.created_at,
            "invocations_count": len(self.invocations),
            "last_active_seconds_ago": time.time() - self.last_active,
            "resource_usage": self.resource_usage.copy(),
            "is_active": self.is_active,
        }

    def cleanup(self):
        """Clean up environment resources."""
        if not self.is_active:
            return

        # Clean up workspace directory
        if self.workspace_dir.exists():
            import shutil
            shutil.rmtree(self.workspace_dir)

        self.is_active = False
        self.invocations.clear()


class SciAgentGymRuntime:
    """Mock SciAgentGym runtime managing multiple environments.

    Implements lifecycle management with resource tracking and leak detection.
    """

    def __init__(self, workspace_root: Path, max_environments: int = 10):
        self.workspace_root = workspace_root
        self.max_environments = max_environments
        self.environments: Dict[str, SciAgentGymEnvironment] = {}
        self.created_at = time.time()

        # Ensure workspace root exists
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def create_environment(self, env_id: str) -> SciAgentGymEnvironment:
        """Create a new environment."""
        if env_id in self.environments:
            raise ValueError(f"Environment {env_id} already exists")

        if len(self.environments) >= self.max_environments:
            raise RuntimeError(
                f"Max environments ({self.max_environments}) reached"
            )

        workspace_dir = self.workspace_root / env_id
        env = SciAgentGymEnvironment(env_id, workspace_dir)
        self.environments[env_id] = env
        return env

    def get_environment(self, env_id: str) -> Optional[SciAgentGymEnvironment]:
        """Get an existing environment."""
        return self.environments.get(env_id)

    def destroy_environment(self, env_id: str) -> bool:
        """Destroy an environment and clean up resources."""
        env = self.environments.get(env_id)
        if env is None:
            return False

        env.cleanup()
        del self.environments[env_id]
        return True

    def gc_stale_environments(self, stale_seconds: float = 60.0) -> List[str]:
        """Garbage collect stale environments."""
        now = time.time()
        stale_ids = []

        for env_id, env in list(self.environments.items()):
            if now - env.last_active > stale_seconds:
                env.cleanup()
                del self.environments[env_id]
                stale_ids.append(env_id)

        return stale_ids

    def shutdown(self):
        """Shutdown runtime and clean up all environments."""
        for env in list(self.environments.values()):
            env.cleanup()
        self.environments.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """Get runtime metrics."""
        return {
            "uptime_seconds": time.time() - self.created_at,
            "active_environments": len(self.environments),
            "max_environments": self.max_environments,
            "environment_metrics": {
                env_id: env.get_metrics()
                for env_id, env in self.environments.items()
            },
        }


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_workspace():
    """Provide a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def runtime(temp_workspace):
    """Provide a SciAgentGym runtime."""
    rt = SciAgentGymRuntime(workspace_root=temp_workspace / "workspaces")
    yield rt
    rt.shutdown()


# ---------------------------------------------------------------------------
# 1. Environment Lifecycle Tests
# ---------------------------------------------------------------------------


class TestEnvironmentLifecycle:
    """Test environment create → use → cleanup lifecycle."""

    def test_create_environment(self, runtime):
        """Test environment creation."""
        env = runtime.create_environment("test-env-1")

        assert env.env_id == "test-env-1"
        assert env.is_active
        assert env.workspace_dir.exists()
        assert len(env.invocations) == 0

    def test_create_duplicate_environment_raises(self, runtime):
        """Test that creating duplicate environment raises."""
        runtime.create_environment("test-env-1")

        with pytest.raises(ValueError, match="already exists"):
            runtime.create_environment("test-env-1")

    def test_max_environments_enforced(self, temp_workspace):
        """Test max environments limit."""
        runtime = SciAgentGymRuntime(
            workspace_root=temp_workspace / "workspaces",
            max_environments=2,
        )

        runtime.create_environment("env-1")
        runtime.create_environment("env-2")

        with pytest.raises(RuntimeError, match="Max environments"):
            runtime.create_environment("env-3")

        runtime.shutdown()

    def test_destroy_environment_cleanup(self, runtime):
        """Test environment destruction cleans up resources."""
        env = runtime.create_environment("test-env-1")
        workspace_path = env.workspace_dir

        assert workspace_path.exists()
        assert runtime.get_environment("test-env-1") is not None

        runtime.destroy_environment("test-env-1")

        assert not env.is_active
        assert not workspace_path.exists()
        assert runtime.get_environment("test-env-1") is None

    def test_full_lifecycle(self, runtime):
        """Test full lifecycle: create → use → cleanup."""
        # Create
        env = runtime.create_environment("test-env-1")
        assert env.is_active

        # Use
        result = env.invoke_tool("read_file", {"path": "/test/file.txt"})
        assert result["result"]["content"] == "Mock content from /test/file.txt"

        # Get metrics
        metrics = env.get_metrics()
        assert metrics["invocations_count"] == 1
        assert metrics["is_active"]

        # Cleanup
        runtime.destroy_environment("test-env-1")
        assert not env.is_active
        assert runtime.get_environment("test-env-1") is None


# ---------------------------------------------------------------------------
# 2. Tool Invocation Tests
# ---------------------------------------------------------------------------


class TestToolInvocation:
    """Test tool invocation infrastructure."""

    def test_read_file_tool(self, runtime):
        """Test read_file tool invocation."""
        env = runtime.create_environment("test-env-1")

        result = env.invoke_tool("read_file", {"path": "/data/file.txt"})

        assert result["result"]["content"] == "Mock content from /data/file.txt"
        assert result["error"] is None
        assert len(env.invocations) == 1

    def test_write_file_tool(self, runtime):
        """Test write_file tool invocation."""
        env = runtime.create_environment("test-env-1")

        result = env.invoke_tool("write_file", {
            "path": "/output/result.txt",
            "content": "Test content",
        })

        assert result["result"]["success"]
        assert result["result"]["bytes_written"] == 12
        assert env.resource_usage["files_created"] == 1

    def test_list_dir_tool(self, runtime):
        """Test list_dir tool invocation."""
        env = runtime.create_environment("test-env-1")

        result = env.invoke_tool("list_dir", {"path": "/workspace"})

        assert "files" in result["result"]
        assert len(result["result"]["files"]) == 3

    def test_run_command_tool(self, runtime):
        """Test run_command tool invocation."""
        env = runtime.create_environment("test-env-1")

        result = env.invoke_tool("run_command", {"command": "ls -la"})

        assert result["result"]["exit_code"] == 0
        assert result["result"]["stdout"] == "Command output"
        assert env.resource_usage["commands_run"] == 1

    def test_unknown_tool_error(self, runtime):
        """Test unknown tool returns error."""
        env = runtime.create_environment("test-env-1")

        result = env.invoke_tool("unknown_tool", {})

        assert result["error"] is not None
        assert "Unknown tool" in result["error"]

    def test_invoke_on_inactive_environment_raises(self, runtime):
        """Test invoking tool on inactive environment raises."""
        env = runtime.create_environment("test-env-1")
        runtime.destroy_environment("test-env-1")

        with pytest.raises(RuntimeError, match="not active"):
            env.invoke_tool("read_file", {"path": "/test"})

    def test_multiple_invocations_tracked(self, runtime):
        """Test multiple invocations are tracked."""
        env = runtime.create_environment("test-env-1")

        for i in range(5):
            env.invoke_tool("read_file", {"path": f"/file{i}.txt"})

        assert len(env.invocations) == 5
        metrics = env.get_metrics()
        assert metrics["invocations_count"] == 5


# ---------------------------------------------------------------------------
# 3. Resource Leak Detection Tests
# ---------------------------------------------------------------------------


class TestResourceLeakDetection:
    """Test resource leak detection and prevention."""

    def test_workspace_cleanup_no_leak(self, runtime):
        """Test workspace directories are cleaned up."""
        workspace_paths = []

        for i in range(5):
            env = runtime.create_environment(f"env-{i}")
            workspace_paths.append(env.workspace_dir)
            # Create some files
            (env.workspace_dir / f"file{i}.txt").write_text("test")

        # Verify all workspaces exist
        for path in workspace_paths:
            assert path.exists()

        # Destroy all environments
        for i in range(5):
            runtime.destroy_environment(f"env-{i}")

        # Verify no workspace directories remain
        for path in workspace_paths:
            assert not path.exists()

    def test_shutdown_cleanup_all(self, temp_workspace):
        """Test shutdown cleans up all environments."""
        runtime = SciAgentGymRuntime(
            workspace_root=temp_workspace / "workspaces"
        )

        workspace_paths = []
        for i in range(3):
            env = runtime.create_environment(f"env-{i}")
            workspace_paths.append(env.workspace_dir)

        # Verify all exist
        for path in workspace_paths:
            assert path.exists()

        # Shutdown
        runtime.shutdown()

        # Verify all cleaned up
        for path in workspace_paths:
            assert not path.exists()
        assert len(runtime.environments) == 0

    def test_gc_stale_environments(self, runtime):
        """Test garbage collection of stale environments."""
        # Create environments
        env1 = runtime.create_environment("env-1")
        env2 = runtime.create_environment("env-2")
        env3 = runtime.create_environment("env-3")

        # Manually set last_active times
        env1.last_active = time.time() - 100  # Very stale
        env2.last_active = time.time() - 50   # Stale
        env3.last_active = time.time()        # Active

        # Run GC with 30 second threshold
        stale_ids = runtime.gc_stale_environments(stale_seconds=30.0)

        # Verify stale environments were collected
        assert len(stale_ids) == 2
        assert "env-1" in stale_ids
        assert "env-2" in stale_ids
        assert "env-3" not in stale_ids

        # Verify only active environment remains
        assert runtime.get_environment("env-3") is not None
        assert runtime.get_environment("env-1") is None
        assert runtime.get_environment("env-2") is None

    def test_no_python_object_leaks(self, runtime):
        """Test no Python object leaks after cleanup."""
        initial_objects = len(gc.get_objects())

        # Create and destroy many environments
        for i in range(10):
            env = runtime.create_environment(f"env-{i}")
            for j in range(5):
                env.invoke_tool("read_file", {"path": f"/file{j}.txt"})
            runtime.destroy_environment(f"env-{i}")

        # Force garbage collection
        gc.collect()

        final_objects = len(gc.get_objects())

        # Allow some variance but no major leak
        # (exact count may vary due to test infrastructure)
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Potential object leak: {object_growth} new objects"

    def test_invocation_history_cleared_on_cleanup(self, runtime):
        """Test invocation history is cleared on cleanup."""
        env = runtime.create_environment("test-env-1")

        # Generate invocations
        for i in range(10):
            env.invoke_tool("read_file", {"path": f"/file{i}.txt"})

        assert len(env.invocations) == 10

        # Cleanup
        runtime.destroy_environment("test-env-1")

        # Verify invocations cleared
        assert len(env.invocations) == 0


# ---------------------------------------------------------------------------
# 4. Metrics and Monitoring Tests
# ---------------------------------------------------------------------------


class TestMetricsAndMonitoring:
    """Test metrics collection and monitoring."""

    def test_environment_metrics(self, runtime):
        """Test environment metrics collection."""
        env = runtime.create_environment("test-env-1")

        # Generate some activity
        env.invoke_tool("write_file", {"path": "/test.txt", "content": "data"})
        env.invoke_tool("run_command", {"command": "test"})

        metrics = env.get_metrics()

        assert metrics["env_id"] == "test-env-1"
        assert metrics["invocations_count"] == 2
        assert metrics["is_active"]
        assert metrics["uptime_seconds"] > 0
        assert metrics["resource_usage"]["files_created"] == 1
        assert metrics["resource_usage"]["commands_run"] == 1

    def test_runtime_metrics(self, runtime):
        """Test runtime-level metrics."""
        # Create multiple environments
        for i in range(3):
            env = runtime.create_environment(f"env-{i}")
            env.invoke_tool("read_file", {"path": "/test.txt"})

        metrics = runtime.get_metrics()

        assert metrics["active_environments"] == 3
        assert metrics["max_environments"] == 10
        assert metrics["uptime_seconds"] > 0
        assert len(metrics["environment_metrics"]) == 3

        for env_id in ["env-0", "env-1", "env-2"]:
            assert env_id in metrics["environment_metrics"]
            env_metrics = metrics["environment_metrics"][env_id]
            assert env_metrics["invocations_count"] == 1

    def test_last_active_tracking(self, runtime):
        """Test last_active timestamp tracking."""
        env = runtime.create_environment("test-env-1")
        initial_active = env.last_active

        time.sleep(0.1)
        env.invoke_tool("read_file", {"path": "/test.txt"})

        assert env.last_active > initial_active

        metrics = env.get_metrics()
        assert metrics["last_active_seconds_ago"] < 1.0


# ---------------------------------------------------------------------------
# 5. Concurrent Access Tests
# ---------------------------------------------------------------------------


class TestConcurrentAccess:
    """Test concurrent environment access patterns."""

    def test_multiple_environments_independent(self, runtime):
        """Test multiple environments operate independently."""
        env1 = runtime.create_environment("env-1")
        env2 = runtime.create_environment("env-2")

        # Use both environments
        env1.invoke_tool("write_file", {"path": "/file1.txt", "content": "env1"})
        env2.invoke_tool("write_file", {"path": "/file2.txt", "content": "env2"})

        # Verify independence
        assert len(env1.invocations) == 1
        assert len(env2.invocations) == 1
        assert env1.resource_usage["files_created"] == 1
        assert env2.resource_usage["files_created"] == 1

        # Destroy one shouldn't affect the other
        runtime.destroy_environment("env-1")

        assert not env1.is_active
        assert env2.is_active
        assert runtime.get_environment("env-2") is not None

    def test_interleaved_operations(self, runtime):
        """Test interleaved operations on multiple environments."""
        envs = [runtime.create_environment(f"env-{i}") for i in range(3)]

        # Interleaved operations
        for i in range(5):
            for j, env in enumerate(envs):
                env.invoke_tool("read_file", {"path": f"/file{i}-{j}.txt"})

        # Verify each environment has correct count
        for env in envs:
            assert len(env.invocations) == 5
