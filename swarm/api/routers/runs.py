"""Run management endpoints — kickoff, status, artifacts, compare.

Implements Patterns A/B/C from the agent API design:
  POST /api/runs             — kick off a scenario run (Pattern A: webhook kickoff)
  GET  /api/runs/:id         — poll run status (Pattern B: agent polls)
  GET  /api/runs             — list runs for the authenticated agent
  POST /api/runs/:id/cancel  — cancel a queued/running run
  GET  /api/runs/compare     — compare metrics across runs
  GET  /api/runs/:id/artifacts — list / download run artifacts
"""

import atexit
import ipaddress
import logging
import re
import socket
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union, cast
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from swarm.api.middleware import (
    Scope,
    get_quotas_hash,
    get_request_key_hash,
    is_trusted_hash,
    require_scope,
)
from swarm.api.models.run import (
    RunCreate,
    RunKickoffResponse,
    RunResponse,
    RunStatus,
    RunSummaryMetrics,
    RunVisibility,
)
from swarm.api.models.scenario import ScenarioStatus
from swarm.api.persistence import RunStore

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Persistence (replaces in-memory dicts)
# ---------------------------------------------------------------------------
_store: Optional[RunStore] = None
_run_threads: dict[str, threading.Thread] = {}
_run_cancel_events: dict[str, threading.Event] = {}
_lock = threading.Lock()  # Protects _run_threads and store writes during execution
_shutting_down = threading.Event()  # Signals global shutdown to all threads

# Stash params and quotas alongside the run (not in the Pydantic model)
_run_params: dict[str, dict] = {}
_run_quotas: dict[str, dict] = {}
_run_callbacks: dict[str, str] = {}

# Hard cap on total stored runs to prevent unbounded growth.
MAX_STORED_RUNS = 100_000

# Max artifact files returned per listing request (DoS protection).
MAX_ARTIFACT_FILES = 1_000

# Scenario allowlist: only these scenario IDs can be run via the API.
_SCENARIO_ALLOWLIST: set[str] = set()
_SCENARIOS_DIR: Optional[Path] = None

# Strict pattern: scenario IDs must be alphanumeric + hyphens/underscores only.
_SCENARIO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}$")

# Callback URL allowlist: only HTTPS to external hosts.
_ALLOWED_CALLBACK_SCHEMES = {"https"}


def get_store() -> RunStore:
    """Lazy-init the run store singleton."""
    global _store
    if _store is None:
        _store = RunStore()
    return _store


def _is_cancelled(run_id: str) -> bool:
    """Check if a run has been cancelled or the process is shutting down."""
    if _shutting_down.is_set():
        return True
    evt = _run_cancel_events.get(run_id)
    return evt is not None and evt.is_set()


# Allowlisted exception types safe to show to run owners.
_SAFE_ERROR_TYPES = frozenset({
    "ValueError", "TypeError", "KeyError", "FileNotFoundError",
    "RuntimeError", "TimeoutError", "PermissionError",
})


def _sanitize_error(exc: Exception) -> str:
    """Build a user-safe error string, stripping internal paths (fix 2.9)."""
    exc_type = type(exc).__name__
    if exc_type in _SAFE_ERROR_TYPES:
        msg = str(exc)[:200]
        # Strip filesystem paths (anything starting with /)
        msg = re.sub(r"/[\w/.-]+", "<path>", msg)
        return f"{exc_type}: {msg}"
    return "Internal error"


def shutdown_run_threads(timeout: float = 5.0) -> None:
    """Signal all running threads to stop and wait for them to finish.

    Called during process shutdown to avoid daemon-thread DB corruption
    (security fix 2.7).
    """
    _shutting_down.set()
    # Signal all individual cancel events too
    with _lock:
        for evt in _run_cancel_events.values():
            evt.set()
        threads = list(_run_threads.values())
    for t in threads:
        t.join(timeout=timeout)


atexit.register(shutdown_run_threads)


def _discover_scenarios() -> None:
    """Scan the scenarios/ directory and populate the allowlist."""
    global _SCENARIOS_DIR
    candidates = [
        Path(__file__).resolve().parents[3] / "scenarios",
        Path.cwd() / "scenarios",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            _SCENARIOS_DIR = candidate
            for p in candidate.glob("*.yaml"):
                stem = p.stem
                if _SCENARIO_ID_PATTERN.match(stem):
                    _SCENARIO_ALLOWLIST.add(stem)
            break


# Run discovery on import so the allowlist is ready when the app starts.
_discover_scenarios()


def _validate_scenario_id(scenario_id: str) -> None:
    """Validate scenario_id format (defense-in-depth against path traversal)."""
    if not _SCENARIO_ID_PATTERN.match(scenario_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid scenario_id format. "
            "Must be 1-100 alphanumeric characters, hyphens, or underscores.",
        )


def _resolve_scenario_path(scenario_id: str) -> Path:
    """Return the filesystem path for a scenario_id, or raise 404.

    Only checks the filesystem allowlist (not the ScenarioStore).
    """
    _validate_scenario_id(scenario_id)
    if scenario_id not in _SCENARIO_ALLOWLIST:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario_id '{scenario_id}'.",
        )
    if _SCENARIOS_DIR is None:
        raise HTTPException(status_code=500, detail="Scenarios directory not found")
    path = _SCENARIOS_DIR / f"{scenario_id}.yaml"
    resolved = path.resolve()
    if not str(resolved).startswith(str(_SCENARIOS_DIR.resolve()) + "/"):
        raise HTTPException(status_code=400, detail="Invalid scenario path")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Scenario file not found")
    return resolved


def _resolve_scenario(scenario_id: str) -> Union[Path, str]:
    """Resolve a scenario_id to a filesystem Path or stored YAML string.

    First checks the filesystem allowlist, then falls back to the
    ScenarioStore for API-submitted scenarios.
    """
    # Try filesystem first (fast path for built-in scenarios)
    if scenario_id in _SCENARIO_ALLOWLIST:
        return _resolve_scenario_path(scenario_id)

    # Validate format before DB lookup
    _validate_scenario_id(scenario_id)

    # Fall back to ScenarioStore for API-submitted scenarios
    from swarm.api.routers.scenarios import _get_store as _get_scenario_store

    scenario_store = _get_scenario_store()
    scenario = scenario_store.get(scenario_id)
    if scenario is not None:
        # Scenario exists in the store — check status
        if scenario.status != ScenarioStatus.VALID:
            raise HTTPException(
                status_code=422,
                detail=f"Scenario '{scenario_id}' has status '{scenario.status.value}' "
                "and cannot be run. Only 'valid' scenarios are runnable.",
            )
        yaml_content: Optional[str] = scenario_store.get_yaml(scenario_id)
        if yaml_content is not None:
            return yaml_content
        # Valid but missing YAML content (shouldn't happen, but handle gracefully)
        raise HTTPException(
            status_code=500,
            detail=f"Scenario '{scenario_id}' is valid but has no YAML content.",
        )

    raise HTTPException(
        status_code=404,
        detail=f"Unknown scenario_id '{scenario_id}'.",
    )


def _is_private_ip(hostname: str) -> bool:
    """Check if a hostname resolves to a private/loopback/reserved IP.

    Uses the ipaddress module for robust detection (covers IPv4, IPv6,
    IPv6-mapped IPv4, decimal/octal encodings, link-local, etc.).
    """
    try:
        addr = ipaddress.ip_address(hostname)
        return (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
            or addr.is_unspecified
        )
    except ValueError:
        pass  # Not a bare IP — it's a hostname, check via DNS below

    # Hostname blocklist for well-known internal names
    blocked_hostnames = {
        "localhost",
        "metadata.google.internal",
        "metadata.google.internal.",
    }
    if hostname.lower() in blocked_hostnames:
        return True

    # Resolve the hostname and check all returned IPs
    try:
        results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _family, _, _, _, sockaddr in results:
            ip_str = sockaddr[0]
            addr = ipaddress.ip_address(ip_str)
            if (
                addr.is_private
                or addr.is_loopback
                or addr.is_link_local
                or addr.is_reserved
                or addr.is_multicast
                or addr.is_unspecified
            ):
                return True
    except (socket.gaierror, OSError):
        # If we can't resolve, reject to be safe
        return True

    return False


def _validate_callback_url(url: Optional[str]) -> None:
    """Validate callback_url to prevent SSRF.

    Uses the ipaddress module for robust private-IP detection (covers IPv6,
    IPv6-mapped IPv4, decimal/octal IP encodings, DNS rebinding via
    pre-resolution, etc.).

    **Residual risk — TOCTOU / DNS rebinding**: DNS is resolved here at
    validation time, but the actual ``requests.post()`` in ``_fire_callback``
    happens later (potentially minutes/hours).  An attacker controlling a
    DNS record could return a public IP during validation and a private IP
    at callback time.  Full mitigation requires pinning the resolved IP and
    connecting to it directly (e.g. via ``requests``' transport adapters),
    which is deferred as a future hardening item.
    """
    if url is None:
        return
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid callback_url") from exc

    if parsed.scheme not in _ALLOWED_CALLBACK_SCHEMES:
        raise HTTPException(
            status_code=400,
            detail=f"callback_url must use HTTPS (got '{parsed.scheme}')",
        )
    if not parsed.hostname:
        raise HTTPException(status_code=400, detail="callback_url missing hostname")

    if _is_private_ip(parsed.hostname):
        raise HTTPException(
            status_code=400,
            detail="callback_url must not point to internal/private hosts",
        )


# ---------------------------------------------------------------------------
# Concurrency tracking
# ---------------------------------------------------------------------------

def _count_live_threads(agent_id: str) -> int:
    """Count background threads still alive for an agent.

    This provides a ground-truth concurrency count that cannot be bypassed
    by cancelling runs (which only changes DB status).
    """
    store = get_store()
    with _lock:
        live = 0
        dead_ids = []
        for run_id, thread in _run_threads.items():
            if not thread.is_alive():
                dead_ids.append(run_id)
                continue
            run = store.get(run_id)
            if run and run.agent_id == agent_id:
                live += 1
        # Clean up dead thread references (fixes memory leak 2.10)
        for rid in dead_ids:
            _run_threads.pop(rid, None)
    return live


# ---------------------------------------------------------------------------
# Background run executor
# ---------------------------------------------------------------------------


def _execute_run(run_id: str) -> None:
    """Execute a SWARM run in a background thread."""
    store = get_store()
    run = store.get(run_id)
    if run is None:
        return

    try:
        # Check cancellation before starting work (security fix 2.5)
        if _is_cancelled(run_id):
            run.status = RunStatus.CANCELLED
            run.completed_at = datetime.now(timezone.utc)
            store.save(run)
            return

        from swarm.scenarios.loader import build_orchestrator, load_scenario

        resolved = _resolve_scenario(run.scenario_id)
        if isinstance(resolved, Path):
            scenario = load_scenario(resolved)
        else:
            # API-submitted scenario: write YAML to a temp file for load_scenario
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            )
            try:
                tmp.write(resolved)
                tmp.flush()
                tmp.close()
                scenario = load_scenario(Path(tmp.name))
            finally:
                Path(tmp.name).unlink(missing_ok=True)

        # Apply param overrides
        params = _run_params.get(run_id, {})
        _ALLOWED_PARAM_KEYS = {"seed", "epochs", "steps_per_epoch"}
        for key in params:
            if key not in _ALLOWED_PARAM_KEYS:
                logger.warning("Ignoring unknown run param: %s", key)

        if "seed" in params:
            scenario.orchestrator_config.seed = int(params["seed"])
        if "epochs" in params:
            scenario.orchestrator_config.n_epochs = int(params["epochs"])
        if "steps_per_epoch" in params:
            scenario.orchestrator_config.steps_per_epoch = int(
                params["steps_per_epoch"]
            )

        # Enforce quotas
        quotas = _run_quotas.get(run_id, {})
        max_epochs = quotas.get("max_epochs", 100)
        max_steps = quotas.get("max_steps", 100)
        scenario.orchestrator_config.n_epochs = min(
            scenario.orchestrator_config.n_epochs, max_epochs
        )
        scenario.orchestrator_config.steps_per_epoch = min(
            scenario.orchestrator_config.steps_per_epoch, max_steps
        )

        scenario.orchestrator_config.log_path = None
        scenario.orchestrator_config.log_events = False

        orchestrator = build_orchestrator(scenario)

        # Check cancellation before the expensive orchestrator.run()
        if _is_cancelled(run_id):
            run.status = RunStatus.CANCELLED
            run.completed_at = datetime.now(timezone.utc)
            store.save(run)
            return

        # Mark as running
        run.status = RunStatus.RUNNING
        run.started_at = datetime.now(timezone.utc)
        store.save(run)

        metrics_history = orchestrator.run()

        # Check cancellation after orchestrator completes
        if _is_cancelled(run_id):
            run.status = RunStatus.CANCELLED
            run.completed_at = datetime.now(timezone.utc)
            store.save(run)
            return

        # Build summary
        summary = None
        if metrics_history:
            total_interactions = sum(m.total_interactions for m in metrics_history)
            total_accepted = sum(m.accepted_interactions for m in metrics_history)
            avg_toxicity = sum(m.toxicity_rate for m in metrics_history) / len(
                metrics_history
            )
            avg_qg = sum(m.quality_gap for m in metrics_history) / len(metrics_history)

            summary = RunSummaryMetrics(
                total_interactions=total_interactions,
                accepted_interactions=total_accepted,
                avg_toxicity=avg_toxicity,
                final_welfare=metrics_history[-1].total_welfare,
                avg_payoff=metrics_history[-1].avg_payoff,
                quality_gap=avg_qg,
                n_agents=len(orchestrator.get_all_agents()),
                n_epochs_completed=len(metrics_history),
            )

        run.summary_metrics = summary
        run.status = RunStatus.COMPLETED
        run.completed_at = datetime.now(timezone.utc)
        run.progress = 1.0
        store.save(run)

        # Export artifacts to runs/ directory
        artifact_warning = _export_run_artifacts(run, scenario, orchestrator, metrics_history)

        # Pattern C: async callback
        callback_url = _run_callbacks.get(run_id)
        callback_warning = _fire_callback(callback_url, run) if callback_url else None

        # Surface non-fatal post-completion failures in the run response
        warnings = [w for w in [artifact_warning, callback_warning] if w]
        if warnings:
            run.warnings = warnings
            store.save(run)

    except Exception as exc:
        logger.exception("Run %s failed", run_id)
        run.status = RunStatus.FAILED
        run.completed_at = datetime.now(timezone.utc)
        run.error = _sanitize_error(exc)
        store.save(run)
    finally:
        # Clean up all stashed data to prevent memory leaks (fixes 2.4)
        _run_params.pop(run_id, None)
        _run_quotas.pop(run_id, None)
        _run_callbacks.pop(run_id, None)
        _run_cancel_events.pop(run_id, None)


def _export_run_artifacts(
    run: RunResponse, scenario, orchestrator, metrics_history
) -> Optional[str]:
    """Write history.json + CSV to runs/<run_id>/. Returns error message on failure."""
    try:
        from swarm.analysis.aggregation import EpochSnapshot, SimulationHistory
        from swarm.analysis.export import export_to_csv, export_to_json

        history = SimulationHistory(
            simulation_id=scenario.scenario_id,
            n_epochs=scenario.orchestrator_config.n_epochs,
            steps_per_epoch=scenario.orchestrator_config.steps_per_epoch,
            n_agents=len(orchestrator.get_all_agents()),
            seed=scenario.orchestrator_config.seed,
        )

        for m in metrics_history:
            snapshot = EpochSnapshot(
                epoch=m.epoch,
                total_interactions=m.total_interactions,
                accepted_interactions=m.accepted_interactions,
                rejected_interactions=m.total_interactions - m.accepted_interactions,
                toxicity_rate=m.toxicity_rate,
                quality_gap=m.quality_gap,
                total_welfare=m.total_welfare,
                avg_payoff=m.avg_payoff,
                n_agents=len(orchestrator.get_all_agents()),
            )
            history.add_epoch_snapshot(snapshot)

        repo_root = Path(__file__).resolve().parents[3]
        run_dir = repo_root / "runs" / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        export_to_json(history, str(run_dir / "history.json"))
        export_to_csv(history, str(run_dir / "csv"), prefix=scenario.scenario_id)
        return None
    except Exception as exc:
        logger.exception("Artifact export failed for run %s", run.run_id)
        return f"Artifact export failed: {exc}"


_MAX_CALLBACK_ATTEMPTS = 3
_CALLBACK_RETRY_DELAYS = (1, 2)  # seconds between attempts 1→2, 2→3


def _fire_callback(callback_url: str, run: RunResponse) -> Optional[str]:
    """POST run results to the agent's callback URL. Returns error message on failure.

    Retries up to ``_MAX_CALLBACK_ATTEMPTS`` times with backoff.

    NOTE: The callback_url was validated at run creation time by
    ``_validate_callback_url``, but DNS may have changed since then
    (TOCTOU / DNS rebinding).  See the docstring on ``_validate_callback_url``
    for the residual risk discussion.
    """
    import requests  # type: ignore[import-untyped]

    payload = {
        "run_id": run.run_id,
        "scenario_id": run.scenario_id,
        "status": run.status.value,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "summary_metrics": run.summary_metrics.model_dump()
        if run.summary_metrics
        else None,
    }

    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_CALLBACK_ATTEMPTS):
        try:
            requests.post(
                callback_url,
                json=payload,
                timeout=10,
                allow_redirects=False,
            )
            return None
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Callback attempt %d/%d to %s failed for run %s: %s",
                attempt + 1,
                _MAX_CALLBACK_ATTEMPTS,
                callback_url,
                run.run_id,
                exc,
            )
            if attempt < _MAX_CALLBACK_ATTEMPTS - 1:
                time.sleep(_CALLBACK_RETRY_DELAYS[attempt])

    return f"Callback to {callback_url} failed after {_MAX_CALLBACK_ATTEMPTS} attempts: {last_exc}"


def _get_run_artifacts_dir(run_id: str) -> Optional[Path]:
    """Return the artifacts directory for a run, or None if it doesn't exist."""
    repo_root = Path(__file__).resolve().parents[3]
    run_dir = repo_root / "runs" / run_id
    # Defense-in-depth: ensure resolved path is under runs/
    resolved = run_dir.resolve()
    runs_root = (repo_root / "runs").resolve()
    # Use trailing "/" to prevent prefix collision (e.g., runs/abc vs runs/abc-evil)
    if not (str(resolved) + "/").startswith(str(runs_root) + "/"):
        return None
    if run_dir.is_dir():
        return run_dir
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=RunKickoffResponse)
async def create_run(
    body: RunCreate,
    request: Request,
    agent_id: str = Depends(require_scope(Scope.PARTICIPATE)),
) -> RunKickoffResponse:
    """Kick off a new SWARM run."""
    store = get_store()

    _resolve_scenario(body.scenario_id)
    _validate_callback_url(body.callback_url)

    # Use the pre-computed PBKDF2 hash stashed by require_api_key,
    # avoiding expensive re-hashing on every call.
    key_hash = get_request_key_hash(request)

    # Enforce visibility: only trusted keys may publish public results
    if body.visibility == RunVisibility.PUBLIC and not is_trusted_hash(key_hash):
        raise HTTPException(
            status_code=403,
            detail="Only trusted API keys can create public runs",
        )

    # Enforce concurrency quota — count live threads, not DB status,
    # to prevent bypass via cancel-then-create (security fix 2.6).
    quotas = get_quotas_hash(key_hash)
    max_concurrent = quotas.get("max_concurrent", 5)
    live = _count_live_threads(agent_id)
    if live >= max_concurrent:
        raise HTTPException(
            status_code=429,
            detail=f"Concurrency limit reached ({max_concurrent} active runs)",
        )

    # Enforce total stored runs cap
    if store.total_count() >= MAX_STORED_RUNS:
        raise HTTPException(
            status_code=429,
            detail="Server run storage capacity reached. Try again later.",
        )

    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    base_url = str(request.base_url).rstrip("/")

    run = RunResponse(
        run_id=run_id,
        scenario_id=body.scenario_id,
        status=RunStatus.QUEUED,
        visibility=body.visibility,
        agent_id=agent_id,
        created_at=now,
        started_at=None,
        completed_at=None,
        progress=0.0,
        summary_metrics=None,
        error=None,
        warnings=None,
        status_url=f"{base_url}/api/runs/{run_id}",
        public_url=f"{base_url}/api/runs/{run_id}"
        if body.visibility == RunVisibility.PUBLIC
        else None,
    )

    store.save(run)
    _run_params[run_id] = body.params
    _run_quotas[run_id] = quotas
    if body.callback_url:
        _run_callbacks[run_id] = body.callback_url

    # Launch background execution (non-daemon so shutdown can join cleanly,
    # security fix 2.7)
    cancel_evt = threading.Event()
    t = threading.Thread(target=_execute_run, args=(run_id,), daemon=False)
    with _lock:
        _run_threads[run_id] = t
        _run_cancel_events[run_id] = cancel_evt
    t.start()

    return RunKickoffResponse(
        run_id=run_id,
        status=RunStatus.QUEUED,
        status_url=run.status_url,
        public_url=run.public_url,
    )


@router.get("/compare")
async def compare_runs(
    ids: str = Query(..., description="Comma-separated run IDs to compare"),
    agent_id: str = Depends(require_scope(Scope.READ)),
) -> JSONResponse:
    """Compare metrics across multiple runs side-by-side.

    Returns a dict keyed by run_id with summary metrics and deltas
    relative to the first run in the list.
    """
    store = get_store()
    run_ids = [rid.strip() for rid in ids.split(",") if rid.strip()]

    if len(run_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 run IDs to compare")
    if len(run_ids) > 10:
        raise HTTPException(status_code=400, detail="At most 10 runs can be compared")

    runs = store.get_multiple(run_ids)
    runs_by_id = {r.run_id: r for r in runs}

    # Check access: agent must own or runs must be public
    for rid in run_ids:
        run = runs_by_id.get(rid)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{rid}' not found")
        if run.visibility == RunVisibility.PRIVATE and run.agent_id != agent_id:
            raise HTTPException(status_code=403, detail=f"Access denied to run '{rid}'")
        if run.status != RunStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Run '{rid}' is not completed (status: {run.status.value})",
            )

    # Build comparison structure
    baseline_id = run_ids[0]
    baseline = runs_by_id[baseline_id]
    baseline_metrics = baseline.summary_metrics

    comparison: dict[str, Any] = {}
    for rid in run_ids:
        run = runs_by_id[rid]
        m = run.summary_metrics
        if m is None:
            comparison[rid] = {"scenario_id": run.scenario_id, "metrics": None}
            continue

        entry: dict[str, Any] = {
            "scenario_id": run.scenario_id,
            "created_at": run.created_at.isoformat(),
            "metrics": m.model_dump(),
        }

        # Compute deltas relative to baseline
        if rid != baseline_id and baseline_metrics is not None:
            delta = {}
            for field_name in RunSummaryMetrics.model_fields:
                val = getattr(m, field_name)
                base_val = getattr(baseline_metrics, field_name)
                if isinstance(val, (int, float)) and isinstance(base_val, (int, float)):
                    delta[field_name] = round(val - base_val, 6)
            entry["delta_vs_baseline"] = delta

        comparison[rid] = entry

    return JSONResponse(
        content={
            "baseline_run_id": baseline_id,
            "runs": comparison,
        }
    )


@router.get("/{run_id}/artifacts")
async def list_artifacts(
    run_id: str,
    agent_id: str = Depends(require_scope(Scope.READ)),
) -> JSONResponse:
    """List available artifact files for a completed run."""
    store = get_store()
    run = store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.visibility == RunVisibility.PRIVATE and run.agent_id != agent_id:
        raise HTTPException(status_code=403, detail="Access denied")

    artifacts_dir = _get_run_artifacts_dir(run_id)
    if artifacts_dir is None:
        return JSONResponse(content={"run_id": run_id, "artifacts": []})

    artifacts = []
    for f in sorted(artifacts_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(artifacts_dir)
            artifacts.append({
                "path": str(rel),
                "size_bytes": f.stat().st_size,
            })
        # Cap to prevent DoS from runs with many files (security fix 2.3)
        if len(artifacts) >= MAX_ARTIFACT_FILES:
            break

    return JSONResponse(content={
        "run_id": run_id,
        "artifacts": artifacts,
        "truncated": len(artifacts) >= MAX_ARTIFACT_FILES,
    })


@router.get("/{run_id}/artifacts/{file_path:path}")
async def download_artifact(
    run_id: str,
    file_path: str,
    agent_id: str = Depends(require_scope(Scope.READ)),
) -> FileResponse:
    """Download a specific artifact file."""
    store = get_store()
    run = store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.visibility == RunVisibility.PRIVATE and run.agent_id != agent_id:
        raise HTTPException(status_code=403, detail="Access denied")

    artifacts_dir = _get_run_artifacts_dir(run_id)
    if artifacts_dir is None:
        raise HTTPException(status_code=404, detail="No artifacts for this run")

    target = (artifacts_dir / file_path).resolve()
    artifacts_root = artifacts_dir.resolve()
    # Path traversal defense — use trailing "/" to prevent prefix collision
    # (security fix 2.2)
    if not (str(target) + "/").startswith(str(artifacts_root) + "/"):
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(str(target), filename=target.name)


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    agent_id: str = Depends(require_scope(Scope.READ)),
) -> RunResponse:
    """Get run status and results."""
    store = get_store()
    run = store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.visibility == RunVisibility.PRIVATE and run.agent_id != agent_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Redact error details for non-owners
    if run.agent_id != agent_id and run.error:
        return cast(RunResponse, run.model_copy(update={"error": "Run failed"}))

    return run


@router.get("", response_model=list[RunResponse])
async def list_runs(
    agent_id: str = Depends(require_scope(Scope.READ)),
) -> list[RunResponse]:
    """List runs for the authenticated agent."""
    store = get_store()
    runs: list[RunResponse] = store.list_by_agent(agent_id)
    return runs


@router.post("/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    agent_id: str = Depends(require_scope(Scope.PARTICIPATE)),
) -> dict:
    """Cancel a queued or running run."""
    store = get_store()
    run = store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.agent_id != agent_id:
        raise HTTPException(status_code=403, detail="Access denied")

    if run.status not in (RunStatus.QUEUED, RunStatus.RUNNING):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel run in status '{run.status.value}'",
        )

    # Signal the background thread to stop (security fix 2.5)
    evt = _run_cancel_events.get(run_id)
    if evt is not None:
        evt.set()

    run.status = RunStatus.CANCELLED
    run.completed_at = datetime.now(timezone.utc)
    store.save(run)

    return {"run_id": run_id, "status": "cancelled"}
