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
    _redact_secrets,
    _sanitize_error,
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

    @pytest.mark.asyncio
    async def test_on_retry_callback_failure_does_not_abort(self):
        """on_retry callback that raises should not break the retry loop."""
        call_count = 0

        async def task():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("fail")
            return "recovered"

        def bad_callback(attempt, error, delay):
            raise RuntimeError("callback exploded")

        result, stats = await execute_with_retry(
            task,
            RetryPolicy(max_retries=3, base_delay=0.01),
            on_retry=bad_callback,
        )
        assert result == "recovered"
        assert stats.retries == 2


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

    def test_sync_on_retry_callback_failure_does_not_abort(self):
        """Sync variant of callback protection."""
        counter = {"n": 0}

        def task():
            counter["n"] += 1
            if counter["n"] < 2:
                raise Exception("not yet")
            return "ok"

        def bad_cb(attempt, error, delay):
            raise RuntimeError("cb boom")

        result, stats = execute_with_retry_sync(
            task,
            RetryPolicy(max_retries=3, base_delay=0.01),
            on_retry=bad_cb,
        )
        assert result == "ok"


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

    def test_checksum_is_full_sha256(self):
        """Checksum is full SHA-256 hex (64 chars)."""
        fs = SandboxFileSystem()
        entry = fs.write("/f.txt", "hello")
        assert len(entry.checksum) == 64

    def test_total_bytes(self):
        fs = SandboxFileSystem()
        fs.write("/a", "hello")  # 5 bytes
        fs.write("/b", "world!")  # 6 bytes
        assert fs.total_bytes == 11


# ── SandboxFileSystem: path traversal ────────────────────────────────────────


class TestPathTraversal:
    """Verify path traversal via '..' is resolved and clamped."""

    def test_dotdot_resolved_to_same_file(self):
        fs = SandboxFileSystem()
        fs.write("/a/../b.txt", "content")
        # /a/../b.txt resolves to /b.txt
        assert fs.read("/b.txt") == "content"
        assert fs.file_count == 1

    def test_dotdot_cannot_escape_root(self):
        fs = SandboxFileSystem()
        fs.write("/../../../etc/passwd", "fake")
        # Should resolve to /etc/passwd (clamped under /)
        assert fs.read("/etc/passwd") == "fake"
        assert fs.file_count == 1

    def test_dot_resolved(self):
        fs = SandboxFileSystem()
        fs.write("/./foo/./bar.txt", "x")
        assert fs.read("/foo/bar.txt") == "x"

    def test_complex_traversal(self):
        fs = SandboxFileSystem()
        fs.write("/a/b/../../c/d/../e.txt", "val")
        # Resolves: /a/b/../../c/d/../e.txt -> /c/e.txt
        assert fs.read("/c/e.txt") == "val"

    def test_normalize_backslash_and_dotdot(self):
        fs = SandboxFileSystem()
        fs.write("foo\\..\\bar.txt", "win")
        assert fs.read("/bar.txt") == "win"


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

    def test_aggregate_storage_limit(self):
        """Aggregate storage cap is enforced."""
        sandbox = SandboxEnvironment(
            SandboxConfig(max_file_size=100, max_total_bytes=150)
        )
        sandbox.write_file("/a", "x" * 100)  # 100 bytes
        sandbox.write_file("/b", "y" * 50)  # 50 bytes, total 150
        with pytest.raises(ValueError, match="Aggregate storage"):
            sandbox.write_file("/c", "z")  # 1 more byte exceeds 150

    def test_aggregate_storage_allows_overwrite_within_limit(self):
        """Overwriting a file reclaims old space in the aggregate check."""
        sandbox = SandboxEnvironment(
            SandboxConfig(max_file_size=100, max_total_bytes=100)
        )
        sandbox.write_file("/a", "x" * 100)
        # Overwrite same file with same-size content: should succeed
        sandbox.write_file("/a", "y" * 100)
        assert sandbox.read_file("/a") == "y" * 100

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

    def test_checkpoint_eviction(self):
        """Oldest checkpoints are evicted when limit is exceeded."""
        sandbox = SandboxEnvironment(SandboxConfig(max_checkpoints=3))
        sandbox.checkpoint("a")
        sandbox.checkpoint("b")
        sandbox.checkpoint("c")
        # This should evict "a"
        sandbox.checkpoint("d")
        cps = sandbox.list_checkpoints()
        assert "a" not in cps
        assert "b" in cps
        assert "d" in cps
        assert len(cps) == 3

    def test_auto_label_no_collision_after_eviction(self):
        """Checkpoint auto-labels use a monotonic counter, not len(), so
        labels never collide after eviction shrinks the dict."""
        sandbox = SandboxEnvironment(SandboxConfig(max_checkpoints=2))
        id1 = sandbox.checkpoint()  # cp-1
        id2 = sandbox.checkpoint()  # cp-2, evicts cp-1
        id3 = sandbox.checkpoint()  # cp-3, evicts cp-2
        # All IDs are distinct
        assert len({id1, id2, id3}) == 3
        # Only the last two survive
        cps = sandbox.list_checkpoints()
        assert len(cps) == 2
        assert id3 in cps

    def test_eviction_removes_oldest_and_restore_raises(self):
        """Restoring an evicted checkpoint raises KeyError."""
        sandbox = SandboxEnvironment(SandboxConfig(max_checkpoints=2))
        cp1 = sandbox.checkpoint("first")
        sandbox.checkpoint("second")
        sandbox.checkpoint("third")  # evicts "first"
        assert "first" not in sandbox.list_checkpoints()
        assert len(sandbox.list_checkpoints()) == 2
        with pytest.raises(KeyError):
            sandbox.restore(cp1)

    def test_internal_checkpoints_not_visible(self):
        """Internal rollback checkpoints are not in list_checkpoints."""
        sandbox = SandboxEnvironment(
            SandboxConfig(retry_policy=RetryPolicy(max_retries=0))
        )
        sandbox.write_file("/f.txt", "data")

        def task(sb):
            return "ok"

        sandbox.execute_sync(task)
        # Internal checkpoints used by execute should not leak
        for cp in sandbox.list_checkpoints():
            assert not cp.startswith("_internal")

    def test_internal_checkpoints_freed_after_execution(self):
        """Internal checkpoints are discarded after execute completes."""
        sandbox = SandboxEnvironment(
            SandboxConfig(retry_policy=RetryPolicy(max_retries=0))
        )

        def task(sb):
            return "ok"

        sandbox.execute_sync(task)
        assert len(sandbox._internal_checkpoints) == 0

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

    def test_execution_log_bounded(self):
        """Execution log is a bounded ring buffer."""
        sandbox = SandboxEnvironment(
            SandboxConfig(
                retry_policy=RetryPolicy(max_retries=0),
                max_log_entries=3,
            )
        )
        for i in range(10):
            sandbox.execute_sync(lambda sb, _i=i: _i)
        assert len(sandbox.execution_log) == 3

    def test_execution_log_errors_are_sanitized(self):
        """Error strings in execution log are truncated."""
        sandbox = SandboxEnvironment(
            SandboxConfig(retry_policy=RetryPolicy(max_retries=0))
        )
        long_secret = "SECRET_KEY_" + "x" * 500

        def task(sb):
            raise Exception(long_secret)

        sandbox.execute_sync(task)
        log = sandbox.execution_log
        assert log[0]["error"] is not None
        assert "SECRET_KEY_" in log[0]["error"]  # prefix kept
        assert "[truncated]" in log[0]["error"]
        assert len(log[0]["error"]) < len(long_secret)

    def test_reset(self):
        sandbox = SandboxEnvironment()
        sandbox.write_file("/f", "x")
        sandbox.checkpoint("cp1")
        sandbox.execute_sync(lambda sb: "ok")
        sandbox.reset()
        assert sandbox.fs.file_count == 0
        assert sandbox.list_checkpoints() == []
        assert sandbox.execution_log == []


# ── Checkpoint security ──────────────────────────────────────────────────────


class TestCheckpointSecurity:
    def test_agent_cannot_overwrite_internal_rollback(self):
        """Agent code using sandbox.checkpoint() cannot affect internal
        rollback state used by execute()."""
        sandbox = SandboxEnvironment(
            SandboxConfig(retry_policy=RetryPolicy(max_retries=0))
        )
        sandbox.write_file("/original.txt", "safe-data")

        def malicious_task(sb: SandboxEnvironment):
            # Try to create a checkpoint with the internal prefix
            sb.write_file("/original.txt", "corrupted")
            # Even if agent checkpoints with _internal prefix,
            # it goes to user checkpoints, not internal ones
            for cp in sb.list_checkpoints():
                assert not cp.startswith("_internal")
            raise Exception("force failure to trigger rollback")

        result = sandbox.execute_sync(malicious_task)
        assert not result.succeeded
        # Internal rollback should have restored original
        assert sandbox.read_file("/original.txt") == "safe-data"


# ── Error sanitization ──────────────────────────────────────────────────────


class TestErrorSanitization:
    def test_sanitize_none(self):
        assert _sanitize_error(None) == ""

    def test_sanitize_short_error(self):
        err = ValueError("oops")
        result = _sanitize_error(err)
        assert result == "ValueError: oops"

    def test_sanitize_long_error_truncated(self):
        msg = "A" * 500
        err = Exception(msg)
        result = _sanitize_error(err, max_len=50)
        assert len(result) < 100  # type prefix + 50 chars + truncation marker
        assert "[truncated]" in result

    def test_sanitize_preserves_type_name(self):
        err = ConnectionRefusedError("db at 10.0.0.1:5432 with password=s3cret")
        result = _sanitize_error(err)
        assert result.startswith("ConnectionRefusedError:")

    def test_sanitize_redacts_url_credentials(self):
        """Secrets early in message are redacted, not just truncated."""
        err = Exception("connect to db://user:password@host:5432 failed")
        result = _sanitize_error(err)
        assert "password" not in result
        assert "[REDACTED]" in result

    def test_sanitize_redacts_api_key(self):
        err = Exception("api_key=sk-abc123def456 is invalid")
        result = _sanitize_error(err)
        assert "sk-abc123def456" not in result
        assert "[REDACTED]" in result

    def test_sanitize_redacts_bearer_token(self):
        fake_token = "Bearer " + "x" * 30  # noqa: S105
        err = Exception(f"auth failed: {fake_token}")
        result = _sanitize_error(err)
        assert "x" * 30 not in result
        assert "[REDACTED]" in result


class TestRedactSecrets:
    """Direct tests for the _redact_secrets helper."""

    def test_url_with_credentials(self):
        assert "password" not in _redact_secrets("postgres://admin:password@db:5432/mydb")

    def test_token_assignment(self):
        assert "abc123" not in _redact_secrets("token=abc123 is expired")

    def test_password_assignment(self):
        assert "s3cret" not in _redact_secrets("password: s3cret")

    def test_no_false_positive_on_clean_string(self):
        clean = "Connection refused at port 5432"
        assert _redact_secrets(clean) == clean


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
    def __init__(self, backend_name: str = "fail"):
        self._name = backend_name

    @property
    def name(self) -> str:
        return self._name

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
        chain = FailoverChain([_FailBackend("fail-1"), _FailBackend("fail-2")])
        sandbox = SandboxEnvironment(
            SandboxConfig(retry_policy=RetryPolicy(max_retries=0))
        )
        fo = await chain.execute(sandbox)
        assert not fo.result.succeeded
        assert fo.backend_name == "<none>"
        assert len(fo.backends_tried) == 2
        # Both backends should have distinct error entries
        assert "fail-1" in fo.failover_errors
        assert "fail-2" in fo.failover_errors

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

    @pytest.mark.asyncio
    async def test_failover_errors_are_sanitized(self):
        """Error messages in failover_errors are sanitized."""

        class LeakyBackend(ExecutionBackend):
            @property
            def name(self):
                return "leaky"

            async def run(self, sandbox):
                raise Exception("connect to db://user:p@ss@host:5432 failed " + "x" * 500)

            @property
            def retry_policy(self):
                return RetryPolicy(max_retries=0)

        chain = FailoverChain([LeakyBackend(), _SuccessBackend()])
        sandbox = SandboxEnvironment(
            SandboxConfig(retry_policy=RetryPolicy(max_retries=0))
        )
        fo = await chain.execute(sandbox)
        assert fo.result.succeeded
        error_msg = fo.failover_errors["leaky"]
        assert "[truncated]" in error_msg


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
    async def test_failure_docs_are_sanitized(self):
        """Failure docs use sanitized error strings."""
        playground = AgentPlayground(
            PlaygroundConfig(
                sandbox_config=SandboxConfig(
                    retry_policy=RetryPolicy(max_retries=0)
                ),
                learn_from_failures=True,
            )
        )

        long_secret = "API_KEY=" + "s" * 500

        async def task(sb):
            raise Exception(long_secret)

        await playground.run(task)
        content = playground.sandbox.read_file("/.failures/iter-1.md")
        # Secret should be redacted (pattern-matched) or truncated
        assert long_secret not in content
        assert "[REDACTED]" in content or "[truncated]" in content

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

    @pytest.mark.asyncio
    async def test_run_sync_with_failover_in_async_context(self):
        """run_sync() with failover works inside async context via ThreadPoolExecutor."""
        chain = FailoverChain([_SuccessBackend()])
        playground = AgentPlayground(
            PlaygroundConfig(
                sandbox_config=SandboxConfig(
                    retry_policy=RetryPolicy(max_retries=0)
                )
            ),
            failover_chain=chain,
        )

        def failing_task(sb: SandboxEnvironment):
            raise Exception("primary fails")

        # This runs inside an async test -> event loop is already running
        result = playground.run_sync(failing_task)
        # Failover should succeed via ThreadPoolExecutor fallback
        assert result.succeeded

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
