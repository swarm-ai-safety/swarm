"""SWARM integration bridges for external systems.

Available bridge subpackages (lazy import to avoid pulling in optional deps):

    swarm.bridges.agent_lab         — AgentLaboratory autonomous research bridge
    swarm.bridges.claude_code       — Claude Code controller bridge
    swarm.bridges.concordia         — Concordia LLM agent simulation bridge
    swarm.bridges.langgraph_swarm   — LangGraph Swarm governance extension bridge
    swarm.bridges.openclaw          — REST service layer bridge
    swarm.bridges.live_swe          — Live self-evolving SWE agent bridge
    swarm.bridges.worktree          — Native git worktree sandbox bridge
    swarm.bridges.ralph             — Ralph event-stream bridge
    swarm.bridges.pettingzoo        — PettingZoo multi-agent RL environment bridge
    swarm.bridges.prime_intellect   — Prime Intellect RL training bridge
    swarm.bridges.ai_scientist     — AI-Scientist autonomous research pipeline bridge
    swarm.bridges.sciagentgym      — SciAgentGym tool substrate integration bridge
    swarm.bridges.hodoscope        — Hodoscope trajectory analysis & visualization bridge
    swarm.bridges.rag              — RAG over run history (semantic search)
    swarm.bridges.langchain        — LangChain agent bridge
"""

__all__: list[str] = []
