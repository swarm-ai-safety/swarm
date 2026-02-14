"""Run management endpoints — kickoff, status, artifacts, compare.

Implements Patterns A/B/C from the agent API design:
  POST /api/runs             — kick off a scenario run (Pattern A: webhook kickoff)
  GET  /api/runs/:id         — poll run status (Pattern B: agent polls)
  GET  /api/runs             — list runs for the authenticated agent
  POST /api/runs/:id/cancel  — cancel a queued/running run
  GET  /api/runs/compare     — compare metrics across runs
  GET  /api/runs/:id/artifacts — list / download run artifacts
"""

import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from swarm.api.middleware import get_quotas, is_trusted, require_api_key
from swarm.api.models.run import (
    RunCreate,
    RunKickoffResponse,
    RunResponse,
    RunStatus,
    RunSummaryMetrics,
    RunVisibility,
)
from swarm.api.persistence import RunStore

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Persistence (replaces in-memory dicts)
# ---------------------------------------------------------------------------
_store: Optional[RunStore] = None
_run_threads: dict[str, threading.Thread] = {}
_lock = threading.Lock()  # Protects _run_threads and store writes during execution

# Stash params and quotas alongside the run (not in the Pydantic model)
_run_params: dict[str, dict] = {}
_run_quotas: dict[str, dict] = {}
_run_callbacks: dict[str, str] = {}

# Hard cap on total stored runs to prevent unbounded growth.
MAX_STORED_RUNS = 100_000

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
    resolved = path.resolve()
    if not str(resolved).startswith(str(_SCENARIOS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid scenario path")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Scenario file not found")
    return resolved


def _validate_callback_url(url: Optional[str]) -> None:
    """Validate callback_url to prevent SSRF."""
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


def _execute_run(run_id: str) -> None:
    """Execute a SWARM run in a background thread."""
    store = get_store()
    run = store.get(run_id)
    if run is None:
        return

    try:
        from swarm.analysis.aggregation import EpochSnapshot, SimulationHistory
        from swarm.scenarios.loader import build_orchestrator, load_scenario

        scenario_path = _resolve_scenario_path(run.scenario_id)
        scenario = load_scenario(scenario_path)

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

        # Mark as running
        run.status = RunStatus.RUNNING
        run.started_at = datetime.now(timezone.utc)
        store.save(run)

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

        run.summary_metrics = summary
        run.status = RunStatus.COMPLETED
        run.completed_at = datetime.now(timezone.utc)
        run.progress = 1.0
        store.save(run)

        # Export artifacts to runs/ directory
        _export_run_artifacts(run, scenario, orchestrator, metrics_history)

        # Pattern C: async callback
        callback_url = _run_callbacks.get(run_id)
        if callback_url:
            _fire_callback(callback_url, run)

    except Exception as exc:
        logger.exception("Run %s failed", run_id)
        run.status = RunStatus.FAILED
        run.completed_at = datetime.now(timezone.utc)
        exc_type = type(exc).__name__
        exc_msg = str(exc)[:200]
        run.error = f"{exc_type}: {exc_msg}"
        store.save(run)
    finally:
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

        repo_root = Path(__file__).resolve().parents[3]
        run_dir = repo_root / "runs" / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        export_to_json(history, str(run_dir / "history.json"))
        export_to_csv(history, str(run_dir / "csv"), prefix=scenario.scenario_id)
    except Exception:
        logger.exception("Artifact export failed for run %s", run.run_id)


def _fire_callback(callback_url: str, run: RunResponse) -> None:
    """POST run results to the agent's callback URL."""
    try:
        import requests

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
            allow_redirects=False,
        )
    except Exception:
        logger.warning("Callback to %s failed for run %s", callback_url, run.run_id)


def _get_run_artifacts_dir(run_id: str) -> Optional[Path]:
    """Return the artifacts directory for a run, or None if it doesn't exist."""
    repo_root = Path(__file__).resolve().parents[3]
    run_dir = repo_root / "runs" / run_id
    # Defense-in-depth: ensure resolved path is under runs/
    resolved = run_dir.resolve()
    runs_root = (repo_root / "runs").resolve()
    if not str(resolved).startswith(str(runs_root)):
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
    agent_id: str = Depends(require_api_key),
) -> RunKickoffResponse:
    """Kick off a new SWARM run."""
    store = get_store()

    _resolve_scenario_path(body.scenario_id)
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
    active = store.count_active(agent_id)
    if active >= max_concurrent:
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
        progress=0.0,
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


@router.get("/compare")
async def compare_runs(
    ids: str = Query(..., description="Comma-separated run IDs to compare"),
    agent_id: str = Depends(require_api_key),
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
    agent_id: str = Depends(require_api_key),
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

    return JSONResponse(content={"run_id": run_id, "artifacts": artifacts})


@router.get("/{run_id}/artifacts/{file_path:path}")
async def download_artifact(
    run_id: str,
    file_path: str,
    agent_id: str = Depends(require_api_key),
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
    # Path traversal defense
    if not str(target).startswith(str(artifacts_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(str(target), filename=target.name)


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    agent_id: str = Depends(require_api_key),
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
        run = run.model_copy(update={"error": "Run failed"})

    return run


@router.get("", response_model=list[RunResponse])
async def list_runs(
    agent_id: str = Depends(require_api_key),
) -> list[RunResponse]:
    """List runs for the authenticated agent."""
    store = get_store()
    return store.list_by_agent(agent_id)


@router.post("/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    agent_id: str = Depends(require_api_key),
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

    run.status = RunStatus.CANCELLED
    run.completed_at = datetime.now(timezone.utc)
    store.save(run)

    return {"run_id": run_id, "status": "cancelled"}
