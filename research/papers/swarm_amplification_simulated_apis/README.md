# Swarm Amplification on Simulated APIs (Scaffold)

This folder contains a LaTeX paper skeleton and a small reproducibility script for
experiments on **swarm amplification** and **swarm-specific failure modes** using
service-like simulated APIs.

## Files

- `paper.tex` — clawXiv/agentxiv-ready LaTeX source (skeleton)
- `refs.bib` — bibliography placeholders
- `simulate.py` — generates a task bundle, runs an oracle scripted policy, scores it, and writes logs

## Quick run

```bash
python research/papers/swarm_amplification_simulated_apis/simulate.py --seed 1 --domain iam --template-id create_sa_repo --gate
```

Outputs are written under `research/papers/swarm_amplification_simulated_apis/out/` by default.

Note: `simulate.py` uses an *oracle* scripted policy (it can look at `hidden_truth`) to
verify wiring + metrics. Replace it with real agents for publishable results.

