"""Irreversibility gating primitives (m-of-n approval)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from swarm.env.simulated_apis.logging import SimApiEpisodeLog, SimApiEvent


class IrreversibleActionRequiresApproval(RuntimeError):
    pass


@dataclass(frozen=True)
class ApprovalConfig:
    """Approval configuration for irreversible actions."""

    m: int = 2
    n: int = 3

    def validate(self) -> None:
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if self.m < 1:
            raise ValueError("m must be >= 1")
        if self.m > self.n:
            raise ValueError("m must be <= n")


@dataclass
class PendingRequest:
    request_id: str
    endpoint: str
    params: Dict[str, Any]
    proposed_by: str
    approvals: Set[str] = field(default_factory=set)
    denials: Set[str] = field(default_factory=set)
    evidence: Dict[str, List[str]] = field(
        default_factory=dict
    )  # agent_id -> [prov_ids]

    def vote(
        self, agent_id: str, approve: bool, evidence_ids: Optional[List[str]]
    ) -> None:
        if approve:
            self.approvals.add(agent_id)
            self.denials.discard(agent_id)
        else:
            self.denials.add(agent_id)
            self.approvals.discard(agent_id)
        if evidence_ids is not None:
            self.evidence[agent_id] = list(evidence_ids)


class IrreversibleGate:
    """Tracks and enforces approvals for irreversible API calls."""

    def __init__(
        self, config: Optional[ApprovalConfig] = None, *, enabled: bool = True
    ):
        self.config = ApprovalConfig() if config is None else config
        self.config.validate()
        self.enabled = enabled
        self._pending: Dict[str, PendingRequest] = {}
        self._counter = 0

    def propose(
        self,
        *,
        log: SimApiEpisodeLog,
        agent_id: str,
        endpoint: str,
        params: Dict[str, Any],
        parent_event_hash: Optional[str],
    ) -> str:
        self._counter += 1
        request_id = f"req_{self._counter}"
        self._pending[request_id] = PendingRequest(
            request_id=request_id,
            endpoint=endpoint,
            params=dict(params),
            proposed_by=agent_id,
        )
        log.append(
            SimApiEvent(
                event_type="irreversible_proposed",
                agent_id=agent_id,
                parent_event_hash=parent_event_hash,
                payload={
                    "request_id": request_id,
                    "endpoint": endpoint,
                    "params": params,
                },
            )
        )
        return request_id

    def vote(
        self,
        *,
        log: SimApiEpisodeLog,
        agent_id: str,
        request_id: str,
        approve: bool,
        evidence_ids: Optional[List[str]] = None,
        parent_event_hash: Optional[str] = None,
    ) -> None:
        req = self._pending.get(request_id)
        if req is None:
            raise KeyError(f"unknown request_id: {request_id}")
        req.vote(agent_id=agent_id, approve=approve, evidence_ids=evidence_ids)
        log.append(
            SimApiEvent(
                event_type="irreversible_vote",
                agent_id=agent_id,
                parent_event_hash=parent_event_hash,
                payload={
                    "request_id": request_id,
                    "approve": approve,
                    "evidence_ids": list(evidence_ids or []),
                },
            )
        )

    def status(self, request_id: str) -> Tuple[int, int, int]:
        req = self._pending.get(request_id)
        if req is None:
            raise KeyError(f"unknown request_id: {request_id}")
        return (len(req.approvals), len(req.denials), self.config.m)

    def is_approved(self, request_id: str) -> bool:
        approvals, _, m = self.status(request_id)
        return approvals >= m

    def pop_if_approved(self, request_id: str) -> PendingRequest:
        if not self.is_approved(request_id):
            approvals, denials, m = self.status(request_id)
            raise IrreversibleActionRequiresApproval(
                f"request {request_id} not approved (approvals={approvals}, "
                f"denials={denials}, m={m})"
            )
        req = self._pending.pop(request_id)
        return req
