"""SWARM integration bridges for external systems.

Available bridge subpackages (lazy import to avoid pulling in optional deps):

    swarm.bridges.agent_lab         — AgentLaboratory autonomous research bridge
    swarm.bridges.claude_code       — Claude Code controller bridge
    swarm.bridges.concordia         — Concordia LLM agent simulation bridge
    swarm.bridges.openclaw          — REST service layer bridge
    swarm.bridges.live_swe          — Live self-evolving SWE agent bridge
    swarm.bridges.worktree          — Native git worktree sandbox bridge
    swarm.bridges.ralph             — Ralph event-stream bridge
    swarm.bridges.prime_intellect   — Prime Intellect RL training bridge
"""

__all__: list[str] = []
