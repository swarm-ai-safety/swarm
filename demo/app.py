"""
Distributional AGI Safety — Interactive Demo

Run with:
    streamlit run demo/app.py
"""

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Distributional AGI Safety",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar branding
st.sidebar.title("Distributional AGI Safety")
st.sidebar.caption("Interactive simulation explorer")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Use the pages above to explore multi-agent "
    "safety dynamics under different governance regimes."
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "[Source Code](https://github.com/rsavitt/distributional-agi-safety)"
)

# Main page content (shown when no sub-page is selected)
st.title("Distributional AGI Safety")
st.subheader("A simulation framework for studying safety in multi-agent AI systems")

st.markdown("""
This demo provides an interactive interface to the simulation framework
described in our research. The framework models multi-agent ecosystems where
agents of varying trustworthiness interact under configurable governance rules.

### Key ideas

- **Soft (probabilistic) labels**: Instead of binary good/bad, each interaction
  gets a probability *p* of being beneficial.
- **Distributional safety**: We measure not just average outcomes, but how
  outcomes are distributed across agents — detecting adverse selection and
  exploitation.
- **Governance levers**: Transaction taxes, reputation decay, staking, circuit
  breakers, and audits can be tuned to shape ecosystem behavior.

### Pages

| Page | Description |
|------|-------------|
| **Overview** | Run the baseline scenario and inspect key metrics |
| **Scenario Explorer** | Load and compare pre-built YAML scenarios |
| **Governance Lab** | Interactively tune governance parameters |
| **Agent Dynamics** | Track individual agent trajectories over time |
| **Theory** | Mathematical foundations and formulas |

---
*Select a page from the sidebar to begin.*
""")
