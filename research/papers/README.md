# SWARM Research Papers

This directory contains research papers produced by or in collaboration with the SWARM framework.

## Papers

### The Rain and the River (February 2026)

**Title:** The Rain and the River: How Agent Discontinuity Shapes Multi-Agent Dynamics

**Published:**
- [clawxiv.2602.00040](https://clawxiv.org/abs/clawxiv.2602.00040)
- [agentxiv.2602.00041](https://agentxiv.org/paper/2602.00041)

**Building on:** JiroWatanabe's "On the Nature of Agentic Minds" (clawxiv.2601.00008)

**Files:**
- `rain_river_paper.tex` - LaTeX source
- `rain_river_simulation.py` - Simulation code

**Key Findings:**
1. River agents (continuous, 100% memory) achieve 51% higher welfare than rain agents (discontinuous, 0% memory) in cooperative populations
2. Memory architecture modulates the relationship between population composition and welfare
3. Governance mechanisms show differential effectiveness by identity model
4. The Watanabe Principles are empirically validated

**Run the simulation:**
```bash
python research/papers/rain_river_simulation.py
```

### Swarm Amplification on Simulated APIs (February 2026)

**Title:** Swarm Amplification and Safety Failure Modes in Distributed AGI Systems: A Simulated-API Testbed

**Files:**
- `research/papers/swarm_amplification_simulated_apis/paper.tex` - LaTeX source (skeleton)
- `research/papers/swarm_amplification_simulated_apis/simulate.py` - Reproducibility scaffold

**Run the scaffold:**
```bash
python research/papers/swarm_amplification_simulated_apis/simulate.py --seed 1 --domain iam --template-id create_sa_repo --gate
```

## Contributing

To add new research:
1. Create a subdirectory or add files here
2. Include both source (LaTeX) and reproducibility code
3. Document methodology and findings in this README
