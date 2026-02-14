"""Run management endpoints — kickoff, status, artifacts.

Implements Patterns A/B/C from the agent API design:
  POST /api/runs         — kick off a scenario run (Pattern A: webhook kickoff)
  GET  /api/runs/:id     — poll run status (Pattern B: agent polls)
  GET  /api/runs         — list runs for the authenticated agent
  POST /api/runs/:id/cancel — cancel a queued/running run
"""

import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request

from swarm.api.middleware import get_quotas, is_trusted, require_api_key
from swarm.api.models.run import (
    RunCreate,
    RunKickoffResponse,
    RunResponse,
    RunStatus,
    RunSummaryMetrics,
    RunVisibility,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory run storage (replace with DB in production)
# ---------------------------------------------------------------------------
_runs: dict[str, RunResponse] = {}
_run_threads: dict[str, threading.Thread] = {}
_lock = threading.Lock()  # Protects _runs and related dicts

# Hard cap on total stored runs to prevent unbounded memory growth.
MAX_STORED_RUNS = 10_000

# Scenario allowlist: only these scenario IDs can be run via the API.
# Populated at startup from the scenarios/ directory.
_SCENARIO_ALLOWLIST: set[str] = set()
_SCENARIOS_DIR: Optional[Path] = None

# Strict pattern: scenario IDs must be alphanumeric + hyphens/underscores only.
# This is defense-in-depth against path traversal even though we also check
# the allowlist.
_SCENARIO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}$")

# Callback URL allowlist: only HTTPS to external hosts.
_ALLOWED_CALLBACK_SCHEMES = {"https"}


def _discover_scenarios() -> None:
    """Scan the scenarios/ directory and populate the allowlist."""
    global _SCENARIOS_DIR
    # Try multiple roots (repo root, package parent, CWD)
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
    """Return the filesystem path for a scenario_id, or raise 404."""
    _validate_scenario_id(scenario_id)
    if scenario_id not in _SCENARIO_ALLOWLIST:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario_id '{scenario_id}'.",
        )
    if _SCENARIOS_DIR is None:
        raise HTTPException(status_code=500, detail="Scenarios directory not found")
    path = _SCENARIOS_DIR / f"{scenario_id}.yaml"
    # Resolve and verify the path stays within the scenarios directory
    # (defense-in-depth against symlink attacks)
    resolved = path.resolve()
    if not str(resolved).startswith(str(_SCENARIOS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid scenario path")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Scenario file not found")
    return resolved


def _validate_callback_url(url: Optional[str]) -> None:
    """Validate callback_url to prevent SSRF.

    Only HTTPS URLs to non-private hosts are allowed.
    """
    if url is None:
        return
    try:
        parsed = urlparse(url)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid callback_url")

    if parsed.scheme not in _ALLOWED_CALLBACK_SCHEMES:
        raise HTTPException(
            status_code=400,
            detail=f"callback_url must use HTTPS (got '{parsed.scheme}')",
        )
    if not parsed.hostname:
        raise HTTPException(status_code=400, detail="callback_url missing hostname")

    # Block private/internal network ranges
    hostname = parsed.hostname.lower()
    blocked_hosts = {
        "localhost", "127.0.0.1", "0.0.0.0", "::1",
        "metadata.google.internal", "169.254.169.254",
    }
    if hostname in blocked_hosts:
        raise HTTPException(
            status_code=400,
            detail="callback_url must not point to internal/private hosts",
        )
    # Block private IP ranges and link-local
    if hostname.startswith(("10.", "192.168.", "172.16.", "172.17.", "172.18.",
                            "172.19.", "172.2", "172.30.", "172.31.",
                            "169.254.", "fc", "fd", "fe80")):
        raise HTTPException(
            status_code=400,
            detail="callback_url must not point to private network addresses",
        )


# ---------------------------------------------------------------------------
# Background run executor
# ---------------------------------------------------------------------------

# Stash params and quotas alongside the run (not in the Pydantic model)
_run_params: dict[str, dict] = {}
_run_quotas: dict[str, dict] = {}
_run_callbacks: dict[str, str] = {}


def _execute_run(run_id: str) -> None:
    """Execute a SWARM run in a background thread.

    Updates the in-memory run record with progress and results.
    If a callback_url was provided, POSTs results when done (Pattern C).
    """
    with _lock:
        run = _runs.get(run_id)
    if run is None:
        return

    try:
        from swarm.analysis.aggregation import EpochSnapshot, SimulationHistory
        from swarm.scenarios.loader import build_orchestrator, load_scenario

        scenario_path = _resolve_scenario_path(run.scenario_id)
        scenario = load_scenario(scenario_path)

        # Apply param overrides (only known safe keys)
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

        # Disable file-based event logging for API runs (write to runs/ instead)
        scenario.orchestrator_config.log_path = None
        scenario.orchestrator_config.log_events = False

        orchestrator = build_orchestrator(scenario)

        # Mark as running (thread-safe update)
        with _lock:
            run.status = RunStatus.RUNNING
            run.started_at = datetime.now(timezone.utc)

        metrics_history = orchestrator.run()

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

        with _lock:
            run.summary_metrics = summary
            run.status = RunStatus.COMPLETED
            run.completed_at = datetime.now(timezone.utc)
            run.progress = 1.0

        # Export artifacts to runs/ directory
        _export_run_artifacts(run, scenario, orchestrator, metrics_history)

        # Pattern C: async callback
        callback_url = _run_callbacks.get(run_id)
        if callback_url:
            _fire_callback(callback_url, run)

    except Exception as exc:
        logger.exception("Run %s failed", run_id)
        with _lock:
            run.status = RunStatus.FAILED
            run.completed_at = datetime.now(timezone.utc)
            # Sanitize error: only include the exception class and a truncated message.
            # Never leak file paths, stack traces, or config details.
            exc_type = type(exc).__name__
            exc_msg = str(exc)[:200]
            run.error = f"{exc_type}: {exc_msg}"
    finally:
        # Clean up stashed data to prevent memory leaks
        _run_params.pop(run_id, None)
        _run_quotas.pop(run_id, None)


def _export_run_artifacts(
    run: RunResponse, scenario, orchestrator, metrics_history
) -> None:
    """Write history.json + CSV to runs/<run_id>/."""
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

        # Determine output directory
        repo_root = Path(__file__).resolve().parents[3]
        run_dir = repo_root / "runs" / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        export_to_json(history, str(run_dir / "history.json"))
        export_to_csv(history, str(run_dir / "csv"), prefix=scenario.scenario_id)
    except Exception:
        logger.exception("Artifact export failed for run %s", run.run_id)


def _fire_callback(callback_url: str, run: RunResponse) -> None:
    """POST run results to the agent's callback URL.

    Only non-sensitive summary fields are included in the payload.
    """
    try:
        import requests

        # Build a sanitized payload — never include internal error details
        payload = {
            "run_id": run.run_id,
            "scenario_id": run.scenario_id,
            "status": run.status.value,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "summary_metrics": run.summary_metrics.model_dump()
            if run.summary_metrics
            else None,
        }
        requests.post(
            callback_url,
            json=payload,
            timeout=10,
            allow_redirects=False,  # Prevent redirect-based SSRF bypass
        )
    except Exception:
        logger.warning("Callback to %s failed for run %s", callback_url, run.run_id)


def _evict_oldest_completed_runs() -> None:
    """Evict the oldest completed/failed/cancelled runs when at capacity.

    Must be called with _lock held.
    """
    if len(_runs) < MAX_STORED_RUNS:
        return

    # Find completed runs sorted by creation time (oldest first)
    terminal_statuses = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
    evictable = [
        (rid, r)
        for rid, r in _runs.items()
        if r.status in terminal_statuses
    ]
    evictable.sort(key=lambda pair: pair[1].created_at)

    # Evict oldest 10%
    to_evict = max(1, len(evictable) // 10)
    for rid, _ in evictable[:to_evict]:
        _runs.pop(rid, None)
        _run_callbacks.pop(rid, None)
        _run_threads.pop(rid, None)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=RunKickoffResponse)
async def create_run(
    body: RunCreate,
    request: Request,
    agent_id: str = Depends(require_api_key),
) -> RunKickoffResponse:
    """Kick off a new SWARM run.

    Validates the scenario against the allowlist, enforces quotas,
    and starts execution in a background thread.
    """
    # Validate scenario_id format and presence in allowlist
    _resolve_scenario_path(body.scenario_id)

    # Validate callback_url (SSRF protection)
    _validate_callback_url(body.callback_url)

    # Enforce visibility: only trusted keys may publish public results
    api_key_header = request.headers.get("Authorization", "")
    token = api_key_header[7:] if api_key_header.startswith("Bearer ") else api_key_header
    if body.visibility == RunVisibility.PUBLIC and not is_trusted(token):
        raise HTTPException(
            status_code=403,
            detail="Only trusted API keys can create public runs",
        )

    # Enforce concurrency quota
    quotas = get_quotas(token)
    max_concurrent = quotas.get("max_concurrent", 5)
    with _lock:
        active = sum(
            1
            for r in _runs.values()
            if r.agent_id == agent_id
            and r.status in (RunStatus.QUEUED, RunStatus.RUNNING)
        )
        if active >= max_concurrent:
            raise HTTPException(
                status_code=429,
                detail=f"Concurrency limit reached ({max_concurrent} active runs)",
            )

        # Enforce total stored runs cap
        _evict_oldest_completed_runs()
        if len(_runs) >= MAX_STORED_RUNS:
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
            progress=0.0,
            status_url=f"{base_url}/api/runs/{run_id}",
            public_url=f"{base_url}/api/runs/{run_id}"
            if body.visibility == RunVisibility.PUBLIC
            else None,
        )

        _runs[run_id] = run
        _run_params[run_id] = body.params
        _run_quotas[run_id] = quotas
        if body.callback_url:
            _run_callbacks[run_id] = body.callback_url

    # Launch background execution
    t = threading.Thread(target=_execute_run, args=(run_id,), daemon=True)
    with _lock:
        _run_threads[run_id] = t
    t.start()

    return RunKickoffResponse(
        run_id=run_id,
        status=RunStatus.QUEUED,
        status_url=run.status_url,
        public_url=run.public_url,
    )


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    agent_id: str = Depends(require_api_key),
) -> RunResponse:
    """Get run status and results.

    Private runs are only visible to the agent that created them.
    Public runs redact the error field for non-owners.
    """
    with _lock:
        if run_id not in _runs:
            raise HTTPException(status_code=404, detail="Run not found")
        run = _runs[run_id]

    if run.visibility == RunVisibility.PRIVATE and run.agent_id != agent_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Redact error details for non-owners (even on public runs)
    if run.agent_id != agent_id and run.error:
        run = run.model_copy(update={"error": "Run failed"})

    return run


@router.get("", response_model=list[RunResponse])
async def list_runs(
    agent_id: str = Depends(require_api_key),
) -> list[RunResponse]:
    """List runs for the authenticated agent."""
    with _lock:
        return [r for r in _runs.values() if r.agent_id == agent_id]


@router.post("/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    agent_id: str = Depends(require_api_key),
) -> dict:
    """Cancel a queued or running run."""
    with _lock:
        if run_id not in _runs:
            raise HTTPException(status_code=404, detail="Run not found")

        run = _runs[run_id]
        if run.agent_id != agent_id:
            raise HTTPException(status_code=403, detail="Access denied")

        if run.status not in (RunStatus.QUEUED, RunStatus.RUNNING):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel run in status '{run.status.value}'",
            )

        run.status = RunStatus.CANCELLED
        run.completed_at = datetime.now(timezone.utc)

    return {"run_id": run_id, "status": "cancelled"}
