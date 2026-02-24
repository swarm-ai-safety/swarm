"""SWARM-Mesa ABM Bridge.

Connects SWARM's governance and metrics framework to Mesa, Python's
most popular agent-based modelling (ABM) library.  Each Mesa model step
is mapped to ``SoftInteraction`` objects scored by ``ProxyComputer``,
enabling governance testing on complex emergent ABM dynamics.

Architecture::

    Mesa Model (step loop)
        └── MesaBridge  (this module)
                ├── SwarmMesaObserver   (Mesa reporter → ProxyObservables)
                ├── ProxyComputer       (observables → v_hat → p)
                ├── SoftPayoffEngine    (p → payoffs)
                └── EventLog            (append-only audit trail)

Integration approach::

    bridge = MesaBridge(model=my_mesa_model)
    bridge.step()   # calls model.step() and records interactions
    bridge.run(n_steps=100)

Mesa agent attribute mapping::

    task_progress   ← agent.task_progress (or 1.0 if absent)
    rework_count    ← agent.rework_count  (or 0.0 if absent)
    engagement      ← agent.engagement    (or 0.5 if absent)
    verifier_score  ← agent.verifier_score (or 0.5 if absent)

If a Mesa agent does not have these attributes, sensible defaults are used
so existing Mesa models work without modification.

Requires: ``pip install mesa>=2.0.0``
"""

from swarm.bridges.mesa.bridge import MesaBridge, MesaBridgeError
from swarm.bridges.mesa.config import MesaBridgeConfig

__all__ = [
    "MesaBridge",
    "MesaBridgeConfig",
    "MesaBridgeError",
]
