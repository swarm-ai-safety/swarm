# Dashboard

Real-time Streamlit dashboard for monitoring simulation metrics.

## Running the Dashboard

```bash
# Generate and run dashboard
streamlit run src/analysis/streamlit_app.py
```

## Programmatic Usage

```python
from src.analysis import (
    DashboardState,
    MetricSnapshot,
    AgentSnapshot,
    extract_metrics_from_orchestrator,
    extract_agent_snapshots,
    run_dashboard,
)

# Extract metrics from orchestrator
snapshot = extract_metrics_from_orchestrator(orchestrator)

# Create dashboard state
state = DashboardState()
state.update_metrics(snapshot)

# Extract agent snapshots
for agent_snapshot in extract_agent_snapshots(orchestrator):
    state.update_agent(agent_snapshot)

# Export to JSON
json_data = state.export_to_json()

# Run dashboard (opens in browser)
run_dashboard(port=8501)
```

## Dashboard Features

- **Metrics Over Time**: Toxicity rate, quality gap, welfare trends
- **Agent Distribution**: Type breakdown, reputation rankings
- **Governance Metrics**: Costs, audits, frozen agents
- **Boundary Metrics**: Crossings, blocked attempts, leakage events
- **Event Log**: Recent interactions and governance actions
