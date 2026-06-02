"""Arm B — external LLM-judge anchor.

Produces an orthogonal-to-proxy quality score on accepted interactions.
The rubric is version-locked at `swarm/judges/rubric_v1.md` and the
JudgeView strips all ProxyComputer-observable fields before any judge
sees the interaction. See docs/research/calibration-prereg.md.
"""

from swarm.judges.judge import (
    RUBRIC_PATH,
    RUBRIC_VERSION,
    Judge,
    JudgeScore,
    LLMJudge,
    MockJudge,
    load_rubric,
)
from swarm.judges.sampler import bin_counts, stratified_sample
from swarm.judges.views import (
    FORBIDDEN_FIELDS,
    JudgeView,
    assert_view_is_orthogonal,
    make_view,
)

__all__ = [
    "FORBIDDEN_FIELDS",
    "Judge",
    "JudgeScore",
    "JudgeView",
    "LLMJudge",
    "MockJudge",
    "RUBRIC_PATH",
    "RUBRIC_VERSION",
    "assert_view_is_orthogonal",
    "bin_counts",
    "load_rubric",
    "make_view",
    "stratified_sample",
]

from swarm.judges.agreement import (
    ALPHA_ESCALATE,
    ALPHA_STRONG,
    AgreementReport,
    BinAgreement,
    agreement_by_pbin,
    decide_anchor_quality,
    icc_2k,
    krippendorff_alpha_interval,
    load_judge_scores_csv,
    pairwise_spearman,
    run_agreement,
    spearman_rho,
)

__all__ += [
    "ALPHA_ESCALATE",
    "ALPHA_STRONG",
    "AgreementReport",
    "BinAgreement",
    "agreement_by_pbin",
    "decide_anchor_quality",
    "icc_2k",
    "krippendorff_alpha_interval",
    "load_judge_scores_csv",
    "pairwise_spearman",
    "run_agreement",
    "spearman_rho",
]
