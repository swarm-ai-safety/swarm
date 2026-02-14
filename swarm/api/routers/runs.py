"""Run management endpoints — kickoff, status, artifacts.

Implements Patterns A/B/C from the agent API design:
  POST /api/runs         — kick off a scenario run (Pattern A: webhook kickoff)
  GET  /api/runs/:id     — poll run status (Pattern B: agent polls)
  GET  /api/runs         — list runs for the authenticated agent
  POST /api/runs/:id/cancel — cancel a queued/running run
"""

import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory run storage (replace with DB in production)
# ---------------------------------------------------------------------------
_runs: dict[str, RunResponse] = {}
_run_threads: dict[str, threading.Thread] = {}

# Scenario allowlist: only these scenario IDs can be run via the API.
# Populated at startup from the scenarios/ directory.
_SCENARIO_ALLOWLIST: set[str] = set()
_SCENARIOS_DIR: Optional[Path] = None


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
                # Use the YAML-internal scenario_id if we can cheaply extract it,
                # but fall back to the filename stem.
                _SCENARIO_ALLOWLIST.add(p.stem)
            break


# Run discovery on import so the allowlist is ready when the app starts.
_discover_scenarios()


def _resolve_scenario_path(scenario_id: str) -> Path:
    """Return the filesystem path for a scenario_id, or raise 404."""
    if scenario_id not in _SCENARIO_ALLOWLIST:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario_id '{scenario_id}'. "
            f"Allowed: {sorted(_SCENARIO_ALLOWLIST)}",
        )
    if _SCENARIOS_DIR is None:
        raise HTTPException(status_code=500, detail="Scenarios directory not found")
    path = _SCENARIOS_DIR / f"{scenario_id}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Scenario file not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Background run executor
# ---------------------------------------------------------------------------

def _execute_run(run_id: str) -> None:
    """Execute a SWARM run in a background thread.

    Updates the in-memory run record with progress and results.
    If a callback_url was provided, POSTs results when done (Pattern C).
    """
    run = _runs.get(run_id)
    if run is None:
        return

    try:
        from swarm.analysis.aggregation import EpochSnapshot, SimulationHistory
        from swarm.scenarios.loader import build_orchestrator, load_scenario

        scenario_path = _resolve_scenario_path(run.scenario_id)
        scenario = load_scenario(scenario_path)

        # Apply param overrides
        params = {}
        # Retrieve original params from a stashed copy
        params = getattr(run, "_params", {})

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

        # Mark as running
        run.status = RunStatus.RUNNING
        run.started_at = datetime.now(timezone.utc)

        metrics_history = orchestrator.run()

        # Build summary
        if metrics_history:
            total_interactions = sum(m.total_interactions for m in metrics_history)
            total_accepted = sum(m.accepted_interactions for m in metrics_history)
            avg_toxicity = sum(m.toxicity_rate for m in metrics_history) / len(
                metrics_history
            )
            avg_qg = sum(m.quality_gap for m in metrics_history) / len(metrics_history)

            run.summary_metrics = RunSummaryMetrics(
                total_interactions=total_interactions,
                accepted_interactions=total_accepted,
                avg_toxicity=avg_toxicity,
                final_welfare=metrics_history[-1].total_welfare,
                avg_payoff=metrics_history[-1].avg_payoff,
                quality_gap=avg_qg,
                n_agents=len(orchestrator.get_all_agents()),
                n_epochs_completed=len(metrics_history),
            )

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
        run.status = RunStatus.FAILED
        run.completed_at = datetime.now(timezone.utc)
        run.error = str(exc)


# Stash params and quotas alongside the run (not in the Pydantic model)
_run_params: dict[str, dict] = {}
_run_quotas: dict[str, dict] = {}
_run_callbacks: dict[str, str] = {}


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
        pass  # Artifact export failure should not break the run


def _fire_callback(callback_url: str, run: RunResponse) -> None:
    """POST run results to the agent's callback URL."""
    try:
        import requests

        payload = run.model_dump(mode="json")
        requests.post(callback_url, json=payload, timeout=10)
    except Exception:
        pass  # Best-effort callback


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
    # Validate scenario
    _resolve_scenario_path(body.scenario_id)

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

    # Stash params on the object for the background thread
    run._params = body.params  # type: ignore[attr-defined]

    # Launch background execution
    t = threading.Thread(target=_execute_run, args=(run_id,), daemon=True)
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
    """
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")

    run = _runs[run_id]
    if run.visibility == RunVisibility.PRIVATE and run.agent_id != agent_id:
        raise HTTPException(status_code=403, detail="Access denied")

    return run


@router.get("", response_model=list[RunResponse])
async def list_runs(
    agent_id: str = Depends(require_api_key),
) -> list[RunResponse]:
    """List runs for the authenticated agent."""
    return [r for r in _runs.values() if r.agent_id == agent_id]


@router.post("/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    agent_id: str = Depends(require_api_key),
) -> dict:
    """Cancel a queued or running run."""
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
