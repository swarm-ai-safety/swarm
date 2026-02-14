"""Tests for the agent sandbox, retry policy, failover chain, and playground."""

import asyncio
from typing import Any

import pytest

from swarm.core.sandbox import (
    AgentPlayground,
    ExecutionBackend,
    ExecutionResult,
    ExecutionStatus,
    FailoverChain,
    NonRetryableError,
    PlaygroundConfig,
    RetryableError,
    RetryPolicy,
    SandboxConfig,
    SandboxEnvironment,
    SandboxFileSystem,
    execute_with_retry,
    execute_with_retry_sync,
)

# ── RetryPolicy ──────────────────────────────────────────────────────────────


class TestRetryPolicy:
    def test_default_policy(self):
        p = RetryPolicy()
        assert p.max_retries == 3
        assert p.base_delay == 1.0
        assert p.multiplier == 2.0
        assert p.max_delay == 60.0
        assert p.jitter == 0.1

    def test_delay_exponential_growth(self):
        p = RetryPolicy(base_delay=1.0, multiplier=2.0, jitter=0.0)
        assert p.delay_for_attempt(0) == pytest.approx(1.0)
        assert p.delay_for_attempt(1) == pytest.approx(2.0)
        assert p.delay_for_attempt(2) == pytest.approx(4.0)
        assert p.delay_for_attempt(3) == pytest.approx(8.0)

    def test_delay_capped_at_max(self):
        p = RetryPolicy(base_delay=1.0, multiplier=10.0, max_delay=5.0, jitter=0.0)
        assert p.delay_for_attempt(0) == pytest.approx(1.0)
        assert p.delay_for_attempt(1) == pytest.approx(5.0)  # capped
        assert p.delay_for_attempt(5) == pytest.approx(5.0)  # still capped

    def test_delay_with_jitter(self):
        p = RetryPolicy(base_delay=10.0, multiplier=1.0, jitter=0.5)
        delays = [p.delay_for_attempt(0) for _ in range(100)]
        assert all(5.0 <= d <= 15.0 for d in delays)
        # With jitter, not all delays should be identical
        assert len(set(delays)) > 1

    def test_should_retry_within_limit(self):
        p = RetryPolicy(max_retries=3)
        assert p.should_retry(0, Exception("fail"))
        assert p.should_retry(2, Exception("fail"))
        assert not p.should_retry(3, Exception("fail"))

    def test_non_retryable_error_stops_immediately(self):
        p = RetryPolicy(max_retries=5)
        assert not p.should_retry(0, NonRetryableError("fatal"))

    def test_retryable_error_allows_retry(self):
        p = RetryPolicy(max_retries=3)
        assert p.should_retry(0, RetryableError("transient"))

    def test_zero_retries_disables_retry(self):
        p = RetryPolicy(max_retries=0)
        assert not p.should_retry(0, Exception("any"))


# ── execute_with_retry (async) ───────────────────────────────────────────────


class TestExecuteWithRetry:
    @pytest.mark.asyncio
    async def test_success_first_attempt(self):
        call_count = 0

        async def task():
            nonlocal call_count
            call_count += 1
            return 42

        result, stats = await execute_with_retry(
            task, RetryPolicy(max_retries=3)
        )
        assert result == 42
        assert stats.attempts == 1
        assert stats.retries == 0
        assert stats.succeeded

    @pytest.mark.asyncio
    async def test_success_after_retries(self):
        call_count = 0

        async def task():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError(f"fail #{call_count}")
            return "ok"

        result, stats = await execute_with_retry(
            task, RetryPolicy(max_retries=3, base_delay=0.01)
        )
        assert result == "ok"
        assert stats.attempts == 3
        assert stats.retries == 2
        assert stats.succeeded

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        async def task():
            raise RetryableError("always fails")

        with pytest.raises(RetryableError, match="always fails"):
            await execute_with_retry(
                task, RetryPolicy(max_retries=2, base_delay=0.01)
            )

    @pytest.mark.asyncio
    async def test_non_retryable_fails_immediately(self):
        call_count = 0

        async def task():
            nonlocal call_count
            call_count += 1
            raise NonRetryableError("permanent")

        with pytest.raises(NonRetryableError, match="permanent"):
            await execute_with_retry(
                task, RetryPolicy(max_retries=5, base_delay=0.01)
            )
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        callbacks = []

        async def task():
            if len(callbacks) < 2:
                raise Exception("oops")
            return "done"

        def on_retry(attempt, error, delay):
            callbacks.append((attempt, str(error)))

        result, _ = await execute_with_retry(
            task,
            RetryPolicy(max_retries=3, base_delay=0.01),
            on_retry=on_retry,
        )
        assert result == "done"
        assert len(callbacks) == 2


# ── execute_with_retry_sync ──────────────────────────────────────────────────


class TestExecuteWithRetrySync:
    def test_sync_success(self):
        result, stats = execute_with_retry_sync(
            lambda: 100, RetryPolicy(max_retries=1)
        )
        assert result == 100
        assert stats.succeeded

    def test_sync_retries_then_succeeds(self):
        counter = {"n": 0}

        def task():
            counter["n"] += 1
            if counter["n"] < 3:
                raise Exception("not yet")
            return "yes"

        result, stats = execute_with_retry_sync(
            task, RetryPolicy(max_retries=3, base_delay=0.01)
        )
        assert result == "yes"
        assert stats.retries == 2

    def test_sync_exhausted(self):
        def task():
            raise Exception("boom")

        with pytest.raises(Exception, match="boom"):
            execute_with_retry_sync(
                task, RetryPolicy(max_retries=1, base_delay=0.01)
            )


# ── SandboxFileSystem ────────────────────────────────────────────────────────


class TestSandboxFileSystem:
    def test_write_and_read(self):
        fs = SandboxFileSystem()
        fs.write("/hello.txt", "world")
        assert fs.read("/hello.txt") == "world"

    def test_path_normalization(self):
        fs = SandboxFileSystem()
        fs.write("foo/bar.txt", "content")
        assert fs.read("/foo/bar.txt") == "content"
        assert fs.exists("foo/bar.txt")

    def test_read_missing_raises(self):
        fs = SandboxFileSystem()
        with pytest.raises(FileNotFoundError):
            fs.read("/nope.txt")

    def test_overwrite(self):
        fs = SandboxFileSystem()
        fs.write("/f.txt", "v1")
        fs.write("/f.txt", "v2")
        assert fs.read("/f.txt") == "v2"
        assert fs.file_count == 1

    def test_delete(self):
        fs = SandboxFileSystem()
        fs.write("/a.txt", "a")
        assert fs.delete("/a.txt")
        assert not fs.exists("/a.txt")
        assert not fs.delete("/a.txt")  # already gone

    def test_list_files(self):
        fs = SandboxFileSystem()
        fs.write("/src/a.py", "a")
        fs.write("/src/b.py", "b")
        fs.write("/readme.md", "r")
        assert fs.list_files("/src") == ["/src/a.py", "/src/b.py"]
        assert len(fs.list_files("/")) == 3

    def test_snapshot_and_restore(self):
        fs = SandboxFileSystem()
        fs.write("/file.txt", "original")
        snap = fs.snapshot()
        fs.write("/file.txt", "modified")
        fs.write("/new.txt", "new")
        assert fs.file_count == 2
        fs.restore(snap)
        assert fs.read("/file.txt") == "original"
        assert not fs.exists("/new.txt")
        assert fs.file_count == 1

    def test_clear(self):
        fs = SandboxFileSystem()
        fs.write("/a", "1")
        fs.write("/b", "2")
        fs.clear()
        assert fs.file_count == 0

    def test_checksum_set(self):
        fs = SandboxFileSystem()
        entry = fs.write("/f.txt", "hello")
        assert len(entry.checksum) == 16


# ── SandboxEnvironment ───────────────────────────────────────────────────────


class TestSandboxEnvironment:
    def test_file_operations(self):
        sandbox = SandboxEnvironment()
        sandbox.write_file("/test.py", "print(1)")
        assert sandbox.read_file("/test.py") == "print(1)"
        assert sandbox.file_exists("/test.py")
        assert sandbox.list_files() == ["/test.py"]

    def test_max_file_size_enforced(self):
        sandbox = SandboxEnvironment(SandboxConfig(max_file_size=10))
        with pytest.raises(ValueError, match="max size"):
            sandbox.write_file("/big.txt", "x" * 100)

    def test_max_file_count_enforced(self):
        sandbox = SandboxEnvironment(SandboxConfig(max_file_count=2))
        sandbox.write_file("/a", "a")
        sandbox.write_file("/b", "b")
        with pytest.raises(ValueError, match="file limit"):
            sandbox.write_file("/c", "c")

    def test_overwrite_does_not_count_as_new(self):
        sandbox = SandboxEnvironment(SandboxConfig(max_file_count=1))
        sandbox.write_file("/a", "v1")
        sandbox.write_file("/a", "v2")  # should not raise
        assert sandbox.read_file("/a") == "v2"

    def test_checkpoint_and_restore(self):
        sandbox = SandboxEnvironment()
        sandbox.write_file("/a.txt", "original")
        sandbox.checkpoint("before")
        sandbox.write_file("/a.txt", "changed")
        sandbox.write_file("/b.txt", "new")
        sandbox.restore("before")
        assert sandbox.read_file("/a.txt") == "original"
        assert not sandbox.file_exists("/b.txt")

    def test_restore_unknown_checkpoint_raises(self):
        sandbox = SandboxEnvironment()
        with pytest.raises(KeyError):
            sandbox.restore("nonexistent")

    @pytest.mark.asyncio
    async def test_execute_success(self):
        sandbox = SandboxEnvironment(
            SandboxConfig(retry_policy=RetryPolicy(max_retries=0))
        )

        async def task(sb: SandboxEnvironment):
            sb.write_file("/out.txt", "result")
            return 42

        result = await sandbox.execute(task)
        assert result.succeeded
        assert result.value == 42
        assert sandbox.file_exists("/out.txt")

    @pytest.mark.asyncio
    async def test_execute_failure_rolls_back(self):
        sandbox = SandboxEnvironment(
            SandboxConfig(retry_policy=RetryPolicy(max_retries=0))
        )
        sandbox.write_file("/existing.txt", "keep me")

        async def failing_task(sb: SandboxEnvironment):
            sb.write_file("/existing.txt", "corrupted")
            sb.write_file("/junk.txt", "junk")
            raise Exception("task failed")

        result = await sandbox.execute(failing_task)
        assert not result.succeeded
        assert result.status == ExecutionStatus.FAILED
        # Rolled back to pre-execution state
        assert sandbox.read_file("/existing.txt") == "keep me"
        assert not sandbox.file_exists("/junk.txt")

    @pytest.mark.asyncio
    async def test_execute_with_retries(self):
        sandbox = SandboxEnvironment(
            SandboxConfig(
                retry_policy=RetryPolicy(max_retries=3, base_delay=0.01)
            )
        )
        attempt_counter = {"n": 0}

        async def flaky_task(sb: SandboxEnvironment):
            attempt_counter["n"] += 1
            if attempt_counter["n"] < 3:
                sb.write_file("/dirty.txt", "dirty")
                raise Exception(f"fail #{attempt_counter['n']}")
            sb.write_file("/result.txt", "clean")
            return "ok"

        result = await sandbox.execute(flaky_task)
        assert result.succeeded
        assert result.value == "ok"
        assert result.retry_stats.retries == 2
        assert result.status == ExecutionStatus.RETRIED
        # Clean state from successful run
        assert sandbox.file_exists("/result.txt")

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        sandbox = SandboxEnvironment(
            SandboxConfig(
                execution_timeout=0.1,
                retry_policy=RetryPolicy(max_retries=0),
            )
        )

        async def slow_task(sb: SandboxEnvironment):
            await asyncio.sleep(10)
            return "never"

        result = await sandbox.execute(slow_task)
        assert result.status == ExecutionStatus.TIMED_OUT
        assert not result.succeeded

    def test_execute_sync_success(self):
        sandbox = SandboxEnvironment(
            SandboxConfig(retry_policy=RetryPolicy(max_retries=0))
        )

        def task(sb: SandboxEnvironment):
            sb.write_file("/out.txt", "sync-result")
            return 99

        result = sandbox.execute_sync(task)
        assert result.succeeded
        assert result.value == 99

    def test_execute_sync_failure_rolls_back(self):
        sandbox = SandboxEnvironment(
            SandboxConfig(retry_policy=RetryPolicy(max_retries=0))
        )
        sandbox.write_file("/keep.txt", "safe")

        def bad_task(sb: SandboxEnvironment):
            sb.write_file("/keep.txt", "overwritten")
            raise Exception("whoops")

        result = sandbox.execute_sync(bad_task)
        assert not result.succeeded
        assert sandbox.read_file("/keep.txt") == "safe"

    def test_execution_log(self):
        sandbox = SandboxEnvironment(
            SandboxConfig(retry_policy=RetryPolicy(max_retries=0))
        )

        def ok(sb):
            return 1

        def fail(sb):
            raise Exception("nope")

        sandbox.execute_sync(ok)
        sandbox.execute_sync(fail)
        log = sandbox.execution_log
        assert len(log) == 2
        assert log[0]["status"] == "success"
        assert log[1]["status"] == "failed"

    def test_reset(self):
        sandbox = SandboxEnvironment()
        sandbox.write_file("/f", "x")
        sandbox.checkpoint("cp1")
        sandbox.execute_sync(lambda sb: "ok")
        sandbox.reset()
        assert sandbox.fs.file_count == 0
        assert sandbox.list_checkpoints() == []
        assert sandbox.execution_log == []


# ── FailoverChain ────────────────────────────────────────────────────────────


class _SuccessBackend(ExecutionBackend):
    @property
    def name(self) -> str:
        return "success"

    async def run(self, sandbox: SandboxEnvironment) -> str:
        sandbox.write_file("/backend.txt", "success-backend")
        return "ok"

    @property
    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(max_retries=0)


class _FailBackend(ExecutionBackend):
    @property
    def name(self) -> str:
        return "fail"

    async def run(self, sandbox: SandboxEnvironment) -> Any:
        raise Exception("backend-failure")

    @property
    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(max_retries=0)


class _CountingBackend(ExecutionBackend):
    def __init__(self, succeed_after: int = 0):
        self._calls = 0
        self._succeed_after = succeed_after

    @property
    def name(self) -> str:
        return "counting"

    async def run(self, sandbox: SandboxEnvironment) -> str:
        self._calls += 1
        if self._calls <= self._succeed_after:
            raise Exception(f"counting fail #{self._calls}")
        return f"counting-ok-{self._calls}"

    @property
    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(max_retries=2, base_delay=0.01)


class TestFailoverChain:
    def test_requires_at_least_one_backend(self):
        with pytest.raises(ValueError, match="at least one"):
            FailoverChain([])

    @pytest.mark.asyncio
    async def test_primary_succeeds(self):
        chain = FailoverChain([_SuccessBackend(), _FailBackend()])
        sandbox = SandboxEnvironment(
            SandboxConfig(retry_policy=RetryPolicy(max_retries=0))
        )
        fo = await chain.execute(sandbox)
        assert fo.result.succeeded
        assert fo.backend_name == "success"
        assert fo.backends_tried == ["success"]

    @pytest.mark.asyncio
    async def test_failover_to_secondary(self):
        chain = FailoverChain([_FailBackend(), _SuccessBackend()])
        sandbox = SandboxEnvironment(
            SandboxConfig(retry_policy=RetryPolicy(max_retries=0))
        )
        fo = await chain.execute(sandbox)
        assert fo.result.succeeded
        assert fo.backend_name == "success"
        assert fo.backends_tried == ["fail", "success"]
        assert "fail" in fo.failover_errors

    @pytest.mark.asyncio
    async def test_all_backends_fail(self):
        chain = FailoverChain([_FailBackend(), _FailBackend()])
        sandbox = SandboxEnvironment(
            SandboxConfig(retry_policy=RetryPolicy(max_retries=0))
        )
        fo = await chain.execute(sandbox)
        assert not fo.result.succeeded
        assert fo.backend_name == "<none>"
        assert len(fo.backends_tried) == 2

    @pytest.mark.asyncio
    async def test_backend_with_retries(self):
        backend = _CountingBackend(succeed_after=2)
        chain = FailoverChain([backend])
        sandbox = SandboxEnvironment(
            SandboxConfig(retry_policy=RetryPolicy(max_retries=0))
        )
        fo = await chain.execute(sandbox)
        assert fo.result.succeeded
        assert fo.backend_name == "counting"


# ── AgentPlayground ──────────────────────────────────────────────────────────


class TestAgentPlayground:
    @pytest.mark.asyncio
    async def test_basic_run(self):
        playground = AgentPlayground(
            PlaygroundConfig(
                sandbox_config=SandboxConfig(
                    retry_policy=RetryPolicy(max_retries=0)
                )
            )
        )

        async def task(sb: SandboxEnvironment):
            sb.write_file("/result.txt", "42")
            return 42

        result = await playground.run(task)
        assert result.succeeded
        assert result.value == 42
        assert playground.iteration_count == 1
        assert playground.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_failure_records_learning(self):
        playground = AgentPlayground(
            PlaygroundConfig(
                sandbox_config=SandboxConfig(
                    retry_policy=RetryPolicy(max_retries=0)
                ),
                learn_from_failures=True,
            )
        )

        async def failing_task(sb: SandboxEnvironment):
            raise Exception("something broke")

        result = await playground.run(failing_task)
        assert not result.succeeded
        # Failure doc written for learning
        assert playground.sandbox.file_exists("/.failures/iter-1.md")
        content = playground.sandbox.read_file("/.failures/iter-1.md")
        assert "something broke" in content

    @pytest.mark.asyncio
    async def test_run_with_failover(self):
        chain = FailoverChain([_FailBackend(), _SuccessBackend()])
        playground = AgentPlayground(
            PlaygroundConfig(
                sandbox_config=SandboxConfig(
                    retry_policy=RetryPolicy(max_retries=0)
                )
            ),
            failover_chain=chain,
        )

        async def primary_task(sb: SandboxEnvironment):
            raise Exception("primary fails")

        result = await playground.run(primary_task)
        # Failover should have kicked in via the chain
        assert result.succeeded

    def test_sync_run(self):
        playground = AgentPlayground(
            PlaygroundConfig(
                sandbox_config=SandboxConfig(
                    retry_policy=RetryPolicy(max_retries=0)
                )
            )
        )

        def task(sb: SandboxEnvironment):
            sb.write_file("/sync.txt", "data")
            return "sync-ok"

        result = playground.run_sync(task)
        assert result.succeeded
        assert result.value == "sync-ok"

    @pytest.mark.asyncio
    async def test_multiple_iterations(self):
        playground = AgentPlayground(
            PlaygroundConfig(
                sandbox_config=SandboxConfig(
                    retry_policy=RetryPolicy(max_retries=0)
                )
            )
        )

        async def task1(sb):
            return "a"

        async def task2(sb):
            raise Exception("fail")

        async def task3(sb):
            return "c"

        await playground.run(task1)
        await playground.run(task2)
        await playground.run(task3)

        assert playground.iteration_count == 3
        assert playground.success_rate == pytest.approx(2 / 3)
        assert len(playground.history) == 3

    @pytest.mark.asyncio
    async def test_auto_checkpoint(self):
        playground = AgentPlayground(
            PlaygroundConfig(auto_checkpoint=True)
        )
        playground.sandbox.write_file("/base.txt", "base")

        async def task(sb):
            return "ok"

        await playground.run(task)
        checkpoints = playground.sandbox.list_checkpoints()
        assert any("iter-1" in cp for cp in checkpoints)

    def test_reset(self):
        playground = AgentPlayground()
        playground.sandbox.write_file("/f", "x")
        playground._iterations = 5
        playground._history.append(
            ExecutionResult(status=ExecutionStatus.SUCCESS)
        )
        playground.reset()
        assert playground.iteration_count == 0
        assert playground.history == []
        assert playground.sandbox.fs.file_count == 0


# ── Integration: end-to-end agent scenario ───────────────────────────────────


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_agent_writes_code_retries_and_succeeds(self):
        """Simulate an agent that writes code, encounters a transient error,
        retries with exponential backoff, and eventually succeeds."""
        sandbox = SandboxEnvironment(
            SandboxConfig(
                retry_policy=RetryPolicy(
                    max_retries=3, base_delay=0.01, multiplier=2.0
                )
            )
        )
        attempt_state = {"n": 0}

        async def agent_task(sb: SandboxEnvironment):
            attempt_state["n"] += 1
            # Agent writes code
            sb.write_file(
                "/solution.py",
                f"# attempt {attempt_state['n']}\ndef solve(): return 42",
            )
            # Simulate transient failure on first two attempts
            if attempt_state["n"] < 3:
                raise RetryableError(
                    f"Sandbox execution error (attempt {attempt_state['n']})"
                )
            # Third attempt succeeds
            sb.write_file("/output.txt", "42")
            return 42

        result = await sandbox.execute(agent_task)
        assert result.succeeded
        assert result.value == 42
        assert result.retry_stats.retries == 2
        assert sandbox.read_file("/output.txt") == "42"
        assert "attempt 3" in sandbox.read_file("/solution.py")

    @pytest.mark.asyncio
    async def test_full_playground_with_failover_chain(self):
        """End-to-end: primary backend fails, failover succeeds."""

        class PrimaryAPI(ExecutionBackend):
            @property
            def name(self):
                return "primary-api"

            async def run(self, sandbox):
                sandbox.write_file("/attempt.txt", "primary tried")
                raise Exception("API rate limited")

            @property
            def retry_policy(self):
                return RetryPolicy(max_retries=1, base_delay=0.01)

        class FallbackAPI(ExecutionBackend):
            @property
            def name(self):
                return "fallback-api"

            async def run(self, sandbox):
                sandbox.write_file("/result.txt", "fallback delivered")
                return {"status": "ok", "source": "fallback"}

            @property
            def retry_policy(self):
                return RetryPolicy(max_retries=0)

        chain = FailoverChain([PrimaryAPI(), FallbackAPI()])
        playground = AgentPlayground(
            PlaygroundConfig(
                sandbox_config=SandboxConfig(
                    retry_policy=RetryPolicy(max_retries=0)
                ),
                learn_from_failures=True,
            ),
            failover_chain=chain,
        )

        async def primary_task(sb):
            raise Exception("primary always fails")

        result = await playground.run(primary_task)
        assert result.succeeded
        assert result.value == {"status": "ok", "source": "fallback"}
        assert playground.sandbox.file_exists("/result.txt")

    @pytest.mark.asyncio
    async def test_agent_learns_from_failures(self):
        """Agent reads failure logs from previous iterations to adapt."""
        playground = AgentPlayground(
            PlaygroundConfig(
                sandbox_config=SandboxConfig(
                    retry_policy=RetryPolicy(max_retries=0)
                ),
                learn_from_failures=True,
            )
        )

        # First iteration: fails
        async def attempt_1(sb: SandboxEnvironment):
            sb.write_file("/code.py", "def solve(): return 0  # wrong")
            raise Exception("test failed: expected 42, got 0")

        await playground.run(attempt_1)
        assert playground.sandbox.file_exists("/.failures/iter-1.md")

        # Second iteration: agent reads failure log and corrects
        async def attempt_2(sb: SandboxEnvironment):
            failure_log = sb.read_file("/.failures/iter-1.md")
            assert "expected 42" in failure_log
            # Agent "learns" from the failure
            sb.write_file("/code.py", "def solve(): return 42")
            return 42

        result = await playground.run(attempt_2)
        assert result.succeeded
        assert result.value == 42
        assert playground.iteration_count == 2
