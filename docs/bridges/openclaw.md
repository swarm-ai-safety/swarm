# SWARM-OpenClaw Bridge

Run SWARM as a production service with OpenClaw integration.

## Overview

OpenClaw provides secure multi-agent orchestration. SWARM-OpenClaw enables:

- **REST API** for scenario execution
- **Job queue** for async simulation runs
- **Skill integration** for OpenClaw workflows

## Installation

```bash
pip install swarm-openclaw
```

## Quick Start

### Start the Service

```bash
swarm-service start --port 8000
```

### Run a Scenario

```bash
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"scenario": "baseline", "seed": 42}'
```

### Get Results

```bash
curl http://localhost:8000/runs/{job_id}/metrics
```

## API Reference

### POST /runs

Create a new simulation run.

**Request:**
```json
{
  "scenario": "baseline",
  "seed": 42,
  "epochs": 20,
  "steps_per_epoch": 15
}
```

**Response:**
```json
{
  "job_id": "abc123",
  "status": "queued"
}
```

### GET /runs/{job_id}

Get run status.

**Response:**
```json
{
  "job_id": "abc123",
  "status": "completed",
  "epochs_completed": 20
}
```

### GET /runs/{job_id}/metrics

Get run metrics.

**Response:**
```json
{
  "toxicity_rate": 0.15,
  "quality_gap": 0.23,
  "total_welfare": 145.7
}
```

## OpenClaw Skill

Use SWARM from OpenClaw workflows:

```python
from openclaw import Skill

swarm_skill = Skill("swarm")

# Run scenario
result = await swarm_skill.run_scenario(
    scenario="baseline",
    seed=42
)

# Get metrics
metrics = await swarm_skill.get_metrics(result.job_id)
```

## Docker Deployment

```dockerfile
FROM python:3.11-slim
RUN pip install swarm-openclaw
EXPOSE 8000
CMD ["swarm-service", "start", "--host", "0.0.0.0"]
```

```bash
docker build -t swarm-service .
docker run -p 8000:8000 swarm-service
```

## Status

**Functional** - Service layer and OpenClaw skill operational.
