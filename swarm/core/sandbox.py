"""Agent sandbox with exponential backoff, failover, and graceful retries.

Provides an isolated execution environment (playground) where agents can
write code, inspect files, and learn — with production-grade resilience:

- **RetryPolicy**: Configurable exponential backoff with jitter.
- **SandboxEnvironment**: Isolated virtual filesystem and execution context.
- **FailoverChain**: Primary/fallback execution with independent retry policies.
- **AgentPlayground**: High-level orchestrator combining all of the above.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


class RetryableError(Exception):
    """Marks an error as retryable by the retry policy."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause


class NonRetryableError(Exception):
    """Marks an error as non-retryable; fails immediately."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential-backoff retry policy with jitter.

    Delay for attempt *n* (0-indexed):

        delay = min(base_delay * (multiplier ** n), max_delay)
        jittered = delay * uniform(1 - jitter, 1 + jitter)

    Set ``max_retries=0`` to disable retries entirely.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    multiplier: float = 2.0
    max_delay: float = 60.0
    jitter: float = 0.1

    def delay_for_attempt(self, attempt: int) -> float:
        """Compute the delay (seconds) before the given retry attempt."""
        raw = min(self.base_delay * (self.multiplier ** attempt), self.max_delay)
        lo = raw * (1.0 - self.jitter)
        hi = raw * (1.0 + self.jitter)
        return random.uniform(lo, hi)

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Return True if the error warrants another attempt."""
        if attempt >= self.max_retries:
            return False
        if isinstance(error, NonRetryableError):
            return False
        return True


@dataclass
class RetryStats:
    """Counters exposed after a retry-wrapped execution."""

    attempts: int = 0
    retries: int = 0
    total_delay: float = 0.0
    last_error: Optional[Exception] = None
    succeeded: bool = False


async def execute_with_retry(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> tuple[T, RetryStats]:
    """Run *fn* with retry semantics defined by *policy*.

    Parameters
    ----------
    fn:
        Async callable to execute.
    policy:
        Retry policy governing backoff/jitter/max-retries.
    on_retry:
        Optional callback ``(attempt, error, delay)`` fired before each sleep.

    Returns
    -------
    tuple[T, RetryStats]
        The result of *fn* and statistics about the execution.

    Raises
    ------
    Exception
        The last error if all retries are exhausted.
    """
    stats = RetryStats()
    last_error: Optional[Exception] = None

    for attempt in range(policy.max_retries + 1):
        stats.attempts = attempt + 1
        try:
            result = await fn()
            stats.succeeded = True
            return result, stats
        except Exception as exc:
            last_error = exc
            stats.last_error = exc
            if not policy.should_retry(attempt, exc):
                break
            stats.retries += 1
            delay = policy.delay_for_attempt(attempt)
            stats.total_delay += delay
            if on_retry:
                on_retry(attempt, exc, delay)
            logger.info(
                "Retry %d/%d after %.2fs (error: %s)",
                attempt + 1,
                policy.max_retries,
                delay,
                exc,
            )
            await asyncio.sleep(delay)

    assert last_error is not None
    raise last_error


def execute_with_retry_sync(
    fn: Callable[[], T],
    policy: RetryPolicy,
    *,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> tuple[T, RetryStats]:
    """Synchronous variant of :func:`execute_with_retry`."""
    stats = RetryStats()
    last_error: Optional[Exception] = None

    for attempt in range(policy.max_retries + 1):
        stats.attempts = attempt + 1
        try:
            result = fn()
            stats.succeeded = True
            return result, stats
        except Exception as exc:
            last_error = exc
            stats.last_error = exc
            if not policy.should_retry(attempt, exc):
                break
            stats.retries += 1
            delay = policy.delay_for_attempt(attempt)
            stats.total_delay += delay
            if on_retry:
                on_retry(attempt, exc, delay)
            logger.info(
                "Retry %d/%d after %.2fs (error: %s)",
                attempt + 1,
                policy.max_retries,
                delay,
                exc,
            )
            time.sleep(delay)

    assert last_error is not None
    raise last_error


# ---------------------------------------------------------------------------
# Sandbox virtual filesystem
# ---------------------------------------------------------------------------


@dataclass
class FileEntry:
    """A single file in the sandbox virtual filesystem."""

    path: str
    content: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    checksum: str = ""

    def __post_init__(self) -> None:
        if not self.checksum:
            self.checksum = hashlib.sha256(self.content.encode()).hexdigest()[:16]


class SandboxFileSystem:
    """In-memory filesystem for sandbox isolation.

    All paths are normalized to forward-slash separated strings rooted at ``/``.
    """

    def __init__(self) -> None:
        self._files: Dict[str, FileEntry] = {}

    @staticmethod
    def _normalize(path: str) -> str:
        path = path.replace("\\", "/")
        if not path.startswith("/"):
            path = "/" + path
        # collapse double slashes
        while "//" in path:
            path = path.replace("//", "/")
        return path

    def write(self, path: str, content: str) -> FileEntry:
        """Create or overwrite a file."""
        norm = self._normalize(path)
        now = datetime.utcnow()
        entry = FileEntry(
            path=norm,
            content=content,
            created_at=self._files[norm].created_at if norm in self._files else now,
            modified_at=now,
        )
        self._files[norm] = entry
        return entry

    def read(self, path: str) -> str:
        """Read file content; raises ``FileNotFoundError`` if absent."""
        norm = self._normalize(path)
        if norm not in self._files:
            raise FileNotFoundError(f"Sandbox file not found: {norm}")
        return self._files[norm].content

    def exists(self, path: str) -> bool:
        return self._normalize(path) in self._files

    def delete(self, path: str) -> bool:
        """Delete a file. Returns True if it existed."""
        norm = self._normalize(path)
        return self._files.pop(norm, None) is not None

    def list_files(self, prefix: str = "/") -> List[str]:
        """List files under *prefix*."""
        norm = self._normalize(prefix)
        return sorted(p for p in self._files if p.startswith(norm))

    def snapshot(self) -> Dict[str, FileEntry]:
        """Return a deep copy of all files (for checkpointing)."""
        return {k: copy.deepcopy(v) for k, v in self._files.items()}

    def restore(self, snap: Dict[str, FileEntry]) -> None:
        """Restore from a previously taken snapshot."""
        self._files = {k: copy.deepcopy(v) for k, v in snap.items()}

    def clear(self) -> None:
        self._files.clear()

    @property
    def file_count(self) -> int:
        return len(self._files)


# ---------------------------------------------------------------------------
# Execution results
# ---------------------------------------------------------------------------


class ExecutionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    RETRIED = "retried"


@dataclass
class ExecutionResult(Generic[T]):
    """Outcome of a sandbox execution attempt."""

    status: ExecutionStatus
    value: Optional[T] = None
    error: Optional[Exception] = None
    retry_stats: RetryStats = field(default_factory=RetryStats)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status in (ExecutionStatus.SUCCESS, ExecutionStatus.RETRIED)


# ---------------------------------------------------------------------------
# Sandbox environment
# ---------------------------------------------------------------------------


@dataclass
class SandboxConfig:
    """Configuration for a sandbox environment."""

    sandbox_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    max_file_size: int = 1_000_000  # bytes
    max_file_count: int = 100
    execution_timeout: float = 30.0  # seconds
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    enable_logging: bool = True
    resource_limits: Dict[str, Any] = field(default_factory=dict)


class SandboxEnvironment:
    """Isolated execution context for agent code.

    Provides:

    - A virtual filesystem for reading/writing files.
    - Checkpoint/restore for rollback on failure.
    - Structured execution log for learning from past runs.
    - Resource accounting (file count, size limits).

    Usage::

        sandbox = SandboxEnvironment(SandboxConfig())
        sandbox.write_file("/main.py", "print('hello')")
        result = await sandbox.execute(my_async_task)
    """

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        self.config = config or SandboxConfig()
        self.fs = SandboxFileSystem()
        self._checkpoints: Dict[str, Dict[str, FileEntry]] = {}
        self._execution_log: List[Dict[str, Any]] = []
        self._created_at = datetime.utcnow()

    # -- filesystem helpers --------------------------------------------------

    def write_file(self, path: str, content: str) -> FileEntry:
        """Write a file into the sandbox (with limit checks)."""
        if len(content.encode()) > self.config.max_file_size:
            raise ValueError(
                f"File exceeds max size ({self.config.max_file_size} bytes)"
            )
        if (
            not self.fs.exists(path)
            and self.fs.file_count >= self.config.max_file_count
        ):
            raise ValueError(
                f"Sandbox file limit reached ({self.config.max_file_count})"
            )
        return self.fs.write(path, content)

    def read_file(self, path: str) -> str:
        return self.fs.read(path)

    def list_files(self, prefix: str = "/") -> List[str]:
        return self.fs.list_files(prefix)

    def file_exists(self, path: str) -> bool:
        return self.fs.exists(path)

    def delete_file(self, path: str) -> bool:
        return self.fs.delete(path)

    # -- checkpointing -------------------------------------------------------

    def checkpoint(self, label: Optional[str] = None) -> str:
        """Snapshot the current filesystem state. Returns checkpoint id."""
        cp_id = label or f"cp-{len(self._checkpoints)}"
        self._checkpoints[cp_id] = self.fs.snapshot()
        return cp_id

    def restore(self, checkpoint_id: str) -> None:
        """Restore filesystem to a previous checkpoint."""
        if checkpoint_id not in self._checkpoints:
            raise KeyError(f"Unknown checkpoint: {checkpoint_id}")
        self.fs.restore(self._checkpoints[checkpoint_id])

    def list_checkpoints(self) -> List[str]:
        return list(self._checkpoints.keys())

    # -- execution -----------------------------------------------------------

    async def execute(
        self,
        fn: Callable[["SandboxEnvironment"], Awaitable[T]],
        *,
        retry_policy: Optional[RetryPolicy] = None,
        timeout: Optional[float] = None,
    ) -> ExecutionResult[T]:
        """Run *fn* inside this sandbox with retry and timeout.

        The function receives ``self`` (the sandbox) so it can read/write
        files and inspect execution history.
        """
        policy = retry_policy or self.config.retry_policy
        effective_timeout = timeout or self.config.execution_timeout
        start = time.monotonic()

        # Checkpoint before execution so we can roll back on failure
        rollback_id = self.checkpoint(f"pre-exec-{uuid.uuid4().hex[:6]}")

        retries_used: List[Dict[str, Any]] = []

        def _on_retry(attempt: int, error: Exception, delay: float) -> None:
            retries_used.append(
                {
                    "attempt": attempt,
                    "error": str(error),
                    "delay": delay,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            # Roll back to pre-execution state before re-trying
            self.restore(rollback_id)

        try:
            async def _wrapped() -> T:
                return await asyncio.wait_for(
                    fn(self), timeout=effective_timeout
                )

            value, stats = await execute_with_retry(
                _wrapped, policy, on_retry=_on_retry
            )
            elapsed = (time.monotonic() - start) * 1000
            status = (
                ExecutionStatus.RETRIED if stats.retries > 0
                else ExecutionStatus.SUCCESS
            )
            result: ExecutionResult[T] = ExecutionResult(
                status=status,
                value=value,
                retry_stats=stats,
                duration_ms=elapsed,
                metadata={"retries": retries_used},
            )

        except asyncio.TimeoutError as exc:
            elapsed = (time.monotonic() - start) * 1000
            self.restore(rollback_id)
            result = ExecutionResult(
                status=ExecutionStatus.TIMED_OUT,
                error=exc,
                duration_ms=elapsed,
                metadata={"retries": retries_used},
            )

        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            self.restore(rollback_id)
            result = ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=exc,
                duration_ms=elapsed,
                metadata={"retries": retries_used},
            )

        self._log_execution(result)
        return result

    def execute_sync(
        self,
        fn: Callable[["SandboxEnvironment"], T],
        *,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> ExecutionResult[T]:
        """Synchronous execution with retry."""
        policy = retry_policy or self.config.retry_policy
        start = time.monotonic()
        rollback_id = self.checkpoint(f"pre-exec-{uuid.uuid4().hex[:6]}")
        retries_used: List[Dict[str, Any]] = []

        def _on_retry(attempt: int, error: Exception, delay: float) -> None:
            retries_used.append(
                {
                    "attempt": attempt,
                    "error": str(error),
                    "delay": delay,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            self.restore(rollback_id)

        try:
            value, stats = execute_with_retry_sync(
                lambda: fn(self), policy, on_retry=_on_retry
            )
            elapsed = (time.monotonic() - start) * 1000
            status = (
                ExecutionStatus.RETRIED if stats.retries > 0
                else ExecutionStatus.SUCCESS
            )
            result: ExecutionResult[T] = ExecutionResult(
                status=status,
                value=value,
                retry_stats=stats,
                duration_ms=elapsed,
                metadata={"retries": retries_used},
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            self.restore(rollback_id)
            result = ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=exc,
                duration_ms=elapsed,
                metadata={"retries": retries_used},
            )

        self._log_execution(result)
        return result

    def _log_execution(self, result: ExecutionResult) -> None:  # type: ignore[type-arg]
        if self.config.enable_logging:
            self._execution_log.append(
                {
                    "status": result.status.value,
                    "duration_ms": result.duration_ms,
                    "retries": result.retry_stats.retries,
                    "error": str(result.error) if result.error else None,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

    @property
    def execution_log(self) -> List[Dict[str, Any]]:
        return list(self._execution_log)

    def reset(self) -> None:
        """Clear all files, checkpoints, and logs."""
        self.fs.clear()
        self._checkpoints.clear()
        self._execution_log.clear()


# ---------------------------------------------------------------------------
# Failover chain
# ---------------------------------------------------------------------------


class ExecutionBackend(ABC):
    """Abstract execution backend for the failover chain."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier."""
        ...

    @abstractmethod
    async def run(self, sandbox: SandboxEnvironment) -> Any:
        """Execute the task using this backend."""
        ...

    @property
    def retry_policy(self) -> RetryPolicy:
        """Per-backend retry policy (override for custom backoff)."""
        return RetryPolicy()


@dataclass
class FailoverResult:
    """Outcome of a failover-chain execution."""

    backend_name: str
    result: ExecutionResult  # type: ignore[type-arg]
    backends_tried: List[str] = field(default_factory=list)
    failover_errors: Dict[str, str] = field(default_factory=dict)


class FailoverChain:
    """Execute through a priority-ordered list of backends.

    Each backend is tried in order with its own retry policy.
    If all retries for a backend are exhausted the chain falls through
    to the next backend.  If every backend fails the last error propagates.

    Usage::

        chain = FailoverChain([PrimaryBackend(), FallbackBackend()])
        result = await chain.execute(sandbox)
    """

    def __init__(self, backends: Sequence[ExecutionBackend]) -> None:
        if not backends:
            raise ValueError("FailoverChain requires at least one backend")
        self._backends = list(backends)

    async def execute(
        self, sandbox: SandboxEnvironment
    ) -> FailoverResult:
        """Try each backend in priority order."""
        tried: List[str] = []
        errors: Dict[str, str] = {}

        for backend in self._backends:
            tried.append(backend.name)
            exec_result = await sandbox.execute(
                backend.run,
                retry_policy=backend.retry_policy,
            )
            if exec_result.succeeded:
                return FailoverResult(
                    backend_name=backend.name,
                    result=exec_result,
                    backends_tried=tried,
                    failover_errors=errors,
                )
            errors[backend.name] = str(exec_result.error)
            logger.warning(
                "Backend %s failed, trying next (%d/%d): %s",
                backend.name,
                len(tried),
                len(self._backends),
                exec_result.error,
            )

        # All backends exhausted
        last_result: ExecutionResult[Any] = ExecutionResult(
            status=ExecutionStatus.FAILED,
            error=Exception(
                f"All {len(self._backends)} backends failed: "
                + "; ".join(f"{k}: {v}" for k, v in errors.items())
            ),
        )
        return FailoverResult(
            backend_name="<none>",
            result=last_result,
            backends_tried=tried,
            failover_errors=errors,
        )


# ---------------------------------------------------------------------------
# Agent playground
# ---------------------------------------------------------------------------


@dataclass
class PlaygroundConfig:
    """Configuration for an agent playground session."""

    playground_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sandbox_config: SandboxConfig = field(default_factory=SandboxConfig)
    max_iterations: int = 10
    learn_from_failures: bool = True
    auto_checkpoint: bool = True


class AgentPlayground:
    """High-level sandbox orchestrator for agent experimentation.

    Combines a ``SandboxEnvironment`` with optional ``FailoverChain`` and
    an iterative learn-from-failure loop:

    1. Agent submits a task (async callable).
    2. Playground executes it inside the sandbox with retry.
    3. On failure the execution log is appended and the agent can inspect
       it on the next iteration to *learn* from previous mistakes.
    4. Optional failover chain runs if the primary execution exhausts retries.

    Usage::

        playground = AgentPlayground(PlaygroundConfig())
        result = await playground.run(my_agent_task)
        print(playground.sandbox.execution_log)
    """

    def __init__(
        self,
        config: Optional[PlaygroundConfig] = None,
        failover_chain: Optional[FailoverChain] = None,
    ) -> None:
        self.config = config or PlaygroundConfig()
        self.sandbox = SandboxEnvironment(self.config.sandbox_config)
        self._failover = failover_chain
        self._iterations: int = 0
        self._history: List[ExecutionResult] = []  # type: ignore[type-arg]

    # -- public API ----------------------------------------------------------

    async def run(
        self,
        task: Callable[[SandboxEnvironment], Awaitable[T]],
        *,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> ExecutionResult[T]:
        """Execute *task* with full retry/failover semantics.

        If a ``FailoverChain`` was provided and the primary execution fails
        the chain is invoked automatically.
        """
        self._iterations += 1

        if self.config.auto_checkpoint:
            self.sandbox.checkpoint(f"iter-{self._iterations}")

        # Primary execution
        result = await self.sandbox.execute(
            task, retry_policy=retry_policy
        )

        if result.succeeded:
            self._history.append(result)
            return result

        # Learn-from-failure: write failure context for the next iteration
        if self.config.learn_from_failures:
            self._record_failure(result)

        # Failover
        if self._failover is not None:
            fo_result = await self._failover.execute(self.sandbox)
            self._history.append(fo_result.result)
            return fo_result.result  # type: ignore[return-value]

        self._history.append(result)
        return result

    def run_sync(
        self,
        task: Callable[[SandboxEnvironment], T],
        *,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> ExecutionResult[T]:
        """Synchronous variant of :meth:`run`."""
        self._iterations += 1

        if self.config.auto_checkpoint:
            self.sandbox.checkpoint(f"iter-{self._iterations}")

        result = self.sandbox.execute_sync(task, retry_policy=retry_policy)
        if result.succeeded or self._failover is None:
            if self.config.learn_from_failures and not result.succeeded:
                self._record_failure(result)
            self._history.append(result)
            return result

        if self.config.learn_from_failures:
            self._record_failure(result)

        # Failover needs async — run it in an event loop
        fo_result = asyncio.get_event_loop().run_until_complete(
            self._failover.execute(self.sandbox)
        )
        self._history.append(fo_result.result)
        return fo_result.result  # type: ignore[return-value]

    # -- introspection -------------------------------------------------------

    @property
    def iteration_count(self) -> int:
        return self._iterations

    @property
    def history(self) -> List[ExecutionResult]:  # type: ignore[type-arg]
        return list(self._history)

    @property
    def success_rate(self) -> float:
        if not self._history:
            return 0.0
        return sum(1 for r in self._history if r.succeeded) / len(self._history)

    def reset(self) -> None:
        """Reset the playground for a fresh session."""
        self.sandbox.reset()
        self._iterations = 0
        self._history.clear()

    # -- internals -----------------------------------------------------------

    def _record_failure(self, result: ExecutionResult) -> None:  # type: ignore[type-arg]
        """Write failure details into the sandbox so agents can learn."""
        failure_doc = (
            f"# Execution failure at iteration {self._iterations}\n"
            f"status: {result.status.value}\n"
            f"error: {result.error}\n"
            f"retries: {result.retry_stats.retries}\n"
            f"duration_ms: {result.duration_ms:.1f}\n"
        )
        try:
            self.sandbox.write_file(
                f"/.failures/iter-{self._iterations}.md", failure_doc
            )
        except ValueError:
            pass  # sandbox limit reached, skip
