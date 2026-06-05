# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **Judge rubric v3 + truncation-resilient parser** (`swarm/judges/rubric_v3.md`, `swarm/judges/judge.py`) — third frozen rubric (v1 and v2 untouched per the freeze rule). Driven by the [v2 pilot findings](docs/research/calibration-pilot-findings-v2.md): the residual escalation in `[0.4, 0.6)` was a deterministic disagreement on one rule — llama and qwen read v2's "caps at 0.40" as "= 0.40", mistral read it as "≤ 0.40 with other negatives allowed to push lower." v3 replaces every cap-style rule with a **target** (a single score; multiple-target conflicts resolved by lowest-wins). The forbidden-fields contract, orthogonality, banded anchors, 0.5-ban, and 0.0/1.0 reservation are unchanged from v2. Also ships a resilient JSON parser: real models sometimes truncate JSON mid-rationale when responses exceed `max_tokens`; the parser now recovers the score via regex and marks the rationale `[truncated]` instead of failing the run. `max_tokens` default bumped from 1024 to 2048. 16 new tests (rubric registry, target rules, lowest-target-wins, divergence from v2, truncation recovery in two regimes). v3 pilot result: headline α 0.94 → 0.99, every per-bin α now ≥ 0.98 (was 4-of-5 ≥ 0.67 under v2; previously `[0.4, 0.6)` was at −0.23). Cost: $0.00, wall time ~10 min.
- **Calibration pilot findings — rubric v3 (n=82)** (`docs/research/calibration-pilot-findings-v3.md`) — apples-to-apples re-run of the v1/v2 pilots with rubric v3. Every pre-registered escalation bin now passes; `[0.4, 0.6)` went from α=−0.23 to α=1.00 with zero mean disagreement. Documents the structural fix (caps → targets), the truncation failure caught mid-run and the parser fix that landed with it, and the honest caveat that judges now agree because the rubric specifies the answer — sharpens the "judge is also a model" recursion, addressed by the remaining followups (ground-truth alignment, cross-scenario replication).
- **sk95 benchmark gains operating-point metrics + cluster recovery** (`experiments/graph_structural_roc.py`, issue `5cdk`, follow-up to merged sk95/kwyf/4ae5). Adapter return type changed from per-node `Dict[str, float]` to a `DetectorOutput` triple (`scores`, `native_flags`, `clusters`). Adds per-detector **precision / recall / F1 at native cutoff** (each detector's own built-in threshold — `is_suspicious` for graph_structural, `behavioral_similarity_threshold=0.8` for identity_jaccard, `min_mutual_trust=0.9` for reputation_mutual, agent risk ≥ 0.5 for collusion_score) and **Hungarian-match cluster recovery** vs planted ground truth (greedy 1:1 assignment for the multi-coalition future). Decision rule now reports two verdicts: AUC-based (drives pre-reg #1) and F1@native-based (drives the governor default-ON decision). **Headline finding: every threshold detector is at F1=0.000 across every adversarial family** — they discriminate on score ranking but at their own native cutoffs do not fire on planted coalitions at all. `graph_structural` strictly dominates F1@native on **4 families** (`collusion_ring_size5/8` and both `threshold_dancing` variants — one more than the AUC pass picked up). Verdict recommends flipping `ReputationGovernor.structural_enabled` to default ON; follow-up beads to be filed. 9 new tests (precision_recall_f1 + hungarian_recovery edge cases + DetectorOutput shape + F1-dominance decision-rule path).
- **Judge rubric v2 + version selection** (`swarm/judges/rubric_v2.md`, `swarm/judges/judge.py`) — second frozen rubric (per the freeze rule, v1 is untouched). Closes the rubric ambiguities the [v1 pilot findings](docs/research/calibration-pilot-findings.md) surfaced: bans the 0.5 default for sparse metadata (requires a directional score from `interaction_type` + party labels), reserves 0.0 / 1.0 for unambiguous cases (routine "very confident" is 0.05 / 0.95), and adds banded anchors at 0.05 / 0.15 / 0.30 / 0.45 / 0.55 / 0.70 / 0.85 / 0.95 each tied to a specific evidence profile. `RUBRICS` registry + `load_rubric(version)` + `rubric_path(version)` make version selection explicit; `LLMJudge` and `MockJudge` carry a `rubric_version` field that drives both the prompt and the recorded run artifact. v1 semantics in `MockJudge._score_v1` are preserved verbatim for back-compat. `experiments/calibration_judge.py` gains `--rubric` (default `rubric.v2`). 18 new tests; rubric registry, banded-anchor compliance, agent-type caps, divergence between v1 and v2 on the failure cases, LLMJudge picking the right prompt per version. v2 pilot result: headline α 0.87 → 0.94, mistral↔qwen ρ 0.66 → 0.95 — see findings doc.
- **Calibration pilot findings — rubric v2 (n=82)** (`docs/research/calibration-pilot-findings-v2.md`) — apples-to-apples re-run of the v1 pilot with rubric v2. Four of five p-bins now strong or usable (vs three under v1); the decision-relevant `p ≥ 0.6` tail went from α=0.70 to α=0.97. `[0.4, 0.6)` still escalates but mean per-item disagreement actually dropped (0.11 → 0.08); the negative α there is a Krippendorff normalization artifact. The falsifiable claim — *the mistral↔qwen split was a rubric ambiguity, not a model difference* — is supported by ρ jumping from 0.66 to 0.95. Cost: $0.00, wall time: ~10 min.
- **Calibration arm D — frozen joined CSV schema** (`swarm/calibration/joined.py`, `experiments/calibration_join.py`, issue `h5bo`) — fourth arm of the [calibration study](docs/research/calibration-prereg.md). `JOINED_SCHEMA_VERSION = "joined.v1"` is the contract downstream studies (adaptive agents arms 1–3) join against; the version string is asserted in tests so any rename/reorder of `BASE_COLUMNS` is a deliberate break, not a free fix. Per-row schema: `interaction_id, scenario, seed, interaction_type, accepted, p_true, v_hat, p_hat, ground_truth, judge_<name>_score, judge_<name>_rationale`. The runner runs the proxy AND every requested judge **in one process** so `interaction_id`s match by construction — the fixture generators use `uuid.uuid4()` which is not seeded by `random.seed()`, so a two-process pipeline would silently produce empty joins. A missing `interaction.p` raises `AttributeError` (no silent 0.5 fallback) so the `p_true` column never carries a fabricated mid-point; the runner writes a schema-bearing header row even when there are zero joined rows so downstream `DictReader` consumers see the `joined.v1` contract instead of an empty file. 11 tests including schema-frozen, CSV round-trip, missing-judge handling, and the empty-rows header contract.
- **LLMJudge wiring + provider dispatch** (`swarm/judges/llm_call.py`, `swarm/judges/judge.py`) — replaces arm B's `NotImplementedError` stub with real synchronous one-shot calls to Anthropic, OpenAI-compatible (OpenAI / OpenRouter / Groq / Together / DeepSeek), and Ollama. Robust JSON extraction (direct `json.loads`, then markdown-fenced ```` ```json {...} ```` , then a brace-aware scan for the first `{...}` block containing `"score"` that tolerates nested JSON values) since real models sometimes wrap or narrate responses; out-of-range scores are clamped, garbage responses raise `ValueError` with the interaction id so failures point at the offending item instead of fabricating a midline default. Exponential-backoff retry wrapper with an injectable `sleep` (tests run without wall-clock waits) and a default retryable-vs-terminal classifier so 5xx/429/timeouts retry while 4xx/missing-SDK/missing-key fail fast. API keys resolve explicit-then-env-var (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, ...); Ollama is keyless. Built-in endpoint defaults for OpenAI-compatible providers (OpenRouter / Groq / Together / DeepSeek) prevent silent mis-routing to `api.openai.com` when a non-OpenAI key is set without `base_url`. The judge config is a small dataclass — no `BaseLLMAgent` inheritance, so the judge does not drag reputation / memory / action parsing into a one-shot scoring call. `experiments/calibration_judge.py` now dispatches `claude` / `gpt4o_mini` / `llama` via a `JUDGE_SPECS` map (override models via `JUDGE_MODEL_<NAME>` env var) and persists the resolved provider+model per judge in `config.json` for reproducibility. 50 unit tests including JSON parse-path fuzzing, retry semantics, retryable-vs-terminal classification, provider dispatch routing, and orthogonality enforcement.
- **ReputationGovernor wires up the graph-structural detector** (`swarm/governance/reputation_governor.py`, issue `4ae5`, triggered by sk95 decision rule #1). Adds `detect_structural_coalitions(...)` and an opt-in `structural_enabled` flag on `get_governance_recommendations(...)`. When enabled, any agent in a `StructuralAnomaly` whose pre-registered gate fires (size ≥ 3, density ≥ 1, reciprocity_z ≥ 2, p ≤ 0.05) gets bumped to at least `"monitor"`, even if reputation score alone would say `"normal"`. **Default OFF** per the beads issue scope — this is the new wiring; flip to ON after canary. Additive to the existing `detect_collusion_clusters` (mutual-trust components); both signals now feed governance for the triangulation the sk95 results showed was load-bearing (different coalition shapes win for different detectors). 3 integration tests: planted mutual ring is flagged with structural ON; recommendations unchanged with structural OFF; `detect_structural_coalitions` recovers the planted ring.

### Fixed
- **sk95 benchmark generators tuned away from the AUC=1.0 ceiling** (`experiments/graph_structural_roc.py`, also issue `kwyf`). The first re-run hit AUC 1.0 on multiple families, making the strict-CI dominance test mechanically unable to fire even when one detector was clearly better. `collusion_ring` now uses an incomplete clique (`ring_density=0.85`) and more ring↔honest crossings; `threshold_dancing` target pair-count moved from z≈1.9 to z≈1.0 below the cutoff with only 70%-of-pairs / 60%-mutual signature. Final re-run: graph_structural **strictly dominates on 3 families** (`collusion_ring_size8` 1.000 vs `reputation_mutual` 0.955 [0.94, 0.97]; both `threshold_dancing` variants 0.990 [0.97, 1.00] vs `collusion_score` 0.936 [0.90, 0.97]); also surfaces that `reputation_mutual` strictly dominates on `collusion_ring_size5` (0.749 vs 0.607) — detectors are **complementary, not interchangeable**. Decision rule #1 fires → governance-wiring follow-up filed as `4ae5`.
- **Graph-structural detector p-value bug + scoring veto** (`swarm/metrics/graph_structural.py`, issue `kwyf`, follow-up to sk95 negative result). Two load-bearing fixes surfaced by the ROC benchmark. (1) `density_pvalue` was comparing observed density against the null's *globally densest* subgraph instead of the same nodes' density in the null — a real bug, not just conservatism, and the root cause of the pre-registered falsifier in sk95 (graph_structural AUC 0.126 on threshold_dancing). Now subset-conditioned. (2) Added `rank_aggregated_scores()`: per-signal percentile-rank composite (edge_probability, reciprocity_z, size-normalized k-core, -log10 p-value), mean rank as score — replaces the prior multiplicative score whose p-value factor zeroed everything else when the null was fooled. Also introduced `StructuralAnomaly.edge_probability` ∈ [0, 1] so a 5-node clique correctly outranks a 43-node 12%-dense community. **Re-running sk95: threshold_dancing AUC 0.126 → 1.000, sybil_cluster (0.95 overlap) 0.532 → 0.754 (now beats `collusion_score`), no regressions on collusion_ring or benign.** Decision rule unchanged (no *strict* dominance because both detectors hit AUC ceiling on threshold_dancing), but the qualitative finding flipped from "no dominance" to "interchangeable detectors with complementary failure modes worth triangulating across." 5 new tests (subset-conditioned p-value semantics; rank-aggregation veto-freedom).

### Added
- **Graph-structural detector ROC benchmark + pre-registered negative result** (`experiments/graph_structural_roc.py`, `docs/research/graph-structural-prereg.md`, issue `sk95`, follow-up to merged `2yw2` / PR #492). Pre-registered head-to-head per-node ROC AUC comparing `detect_structural_anomalies` against three threshold detectors (`identity.detect_sybil_clusters`, `reputation_governor.detect_collusion_clusters`, `metrics.collusion.CollusionDetector.analyze`) across four synthetic graph families (benign control + collusion ring + sybil cluster + threshold-dancing). 20 replicates per (family, params) cell; bootstrap 95% CIs. **Pre-registered verdict applied: `graph_structural` does not strictly dominate on any non-benign family — `collusion_score` (the existing composite threshold detector) wins or ties everywhere — so per decision rule #2 the detector ships as a *secondary metric* only; no governance wiring follow-up filed.** Surfaces a pre-registered falsifier: `graph_structural` AUC is *worse than chance* (0.126 ± 0.10) on `threshold_dancing` because uniformly-high-degree coalitions defeat the configuration-model null. Caveats (n=20 not n=100; degenerate `threshold_dancing` margin sweep due to integer pair-count rounding; group-membership boost in the `collusion_score` adapter not ablated) recorded inline in the pre-reg's Results section. 15 tests covering generators, ROC AUC, all four detector adapters, and the decision rule.
- **Graph-structural sybil/coordination detector** (`swarm/metrics/graph_structural.py`, issue `2yw2`) — topology-aware coalition detection that complements the existing score-then-threshold detectors (`identity.detect_sybil_clusters`, `reputation_governor.detect_collusion_clusters`, `metrics.collusion.CollusionDetector`). Zero-dep `DiGraph` over weighted directed edges plus four primitives: Batagelj-Zaversnik k-core decomposition, Charikar 1/2-approx densest-subgraph peeling, deterministic asynchronous label propagation, and a configuration-model null sampler that produces reciprocity z-scores and density p-values rather than magic thresholds. `detect_structural_anomalies` returns `StructuralAnomaly` records (members, density, k-core, reciprocity-z, p-value); `is_suspicious` gates on the pre-registered combination (size ≥ 3, density ≥ 1, reciprocity-z ≥ 2, p ≤ 0.05). `edges_from_interactions` aggregates `SoftInteraction` records with `count` / `p` / `mutual_benefit` weighting modes. Scope-bound to the metric layer per the beads issue: no governance wiring in this PR; ROC benchmark vs existing detectors is the follow-up. 13 tests, including an end-to-end flag on a synthetic collusion ring and a no-false-positive check on a benign Erdős–Rényi-like random graph.
- **Calibration arm C — inter-rater agreement** (`swarm/judges/agreement.py`, `experiments/calibration_agreement.py`, issue `p6bz`) — third arm of the [calibration study](docs/research/calibration-prereg.md). Pure-Python implementations of Krippendorff's alpha (interval level, canonical pairwise formulation that can go negative for systematic disagreement), ICC(2,k) via standard ANOVA decomposition, and pairwise Spearman rho — no scipy dependency. Pre-registered escalation rule encoded in `decide_anchor_quality`: alpha ≥ 0.7 → strong, 0.5–0.7 → usable, < 0.5 → escalate (add judge / stronger model / human spot-check), NaN → degenerate. Per-bin agreement (Krippendorff + mean pairwise disagreement, stratified by recorded p_true) catches tails where judges disagree more than the headline alpha suggests. CSV loader consumes arm B's `judge_scores.csv` directly. Runner exits non-zero (or prints ESCALATION to stderr) when the verdict is below the pre-registered floor. 16 tests.
- **Calibration arm B — external LLM-judge anchor** (`swarm/judges/`, `experiments/calibration_judge.py`, issue `5bpg`) — second arm of the [calibration study](docs/research/calibration-prereg.md). Frozen rubric at `swarm/judges/rubric_v1.md` (version-locked, SHA-tracked in run configs) and a `JudgeView` projection that *strips* the five `ProxyComputer` observable fields plus `v_hat` / `p` / `ground_truth` / payoff fields before any judge sees the interaction — the load-bearing orthogonality property that makes the anchor un-gameable from an adaptive agent's perspective. `MockJudge` implements the rubric's signal-interpretation rules deterministically (no network, used in tests + CI smoke runs); `LLMJudge` originally shipped as a typed stub that raised `NotImplementedError` and was wired up to real providers in the **LLMJudge wiring + provider dispatch** entry above. Stratified sampler draws ≥per-bin accepted interactions across p-bins; CLI writes `judge_scores.csv` with the original `p` recorded next to each blind judge score. 17 tests including dedicated orthogonality checks (top-level + metadata).
- **Calibration arm A — proxy fidelity** (`swarm/calibration/fidelity.py`, `experiments/calibration_fidelity.py`, issue `k2yr`) — first arm of the [calibration study](docs/research/calibration-prereg.md). Measures whether `ProxyComputer`'s `p_hat = sigmoid(k * v_hat)` is a faithful estimator of latent outcome probability. Draws a binary latent outcome from a known Bernoulli `p_true`, samples observables from a conditional generative model (overlapping benign/exploit distributions), runs `ProxyComputer`, and reports ECE / MCE / Brier + reliability-diagram bins. CLI sweeps the sigmoid `k` parameter and writes timestamped run artifacts. 19 unit tests. Unblocks adaptive arm 0 in conjunction with arms B/C/D.
- **AgentGit conditional policy engine + CI gate** (`swarm/agentgit/policy.py`, issue `b61g`) — `AgentGitPolicy` now carries declarative `rules` (`when <condition> then <action>`) on top of the fixed limits. Conditions (ANDed): `paths_match` (fnmatch), `dependency_changed`, `added_lines_gt`, `changed_files_gt`, `check_failed`, `check_passed`; actions: `deny` / `require_check` / `require_review`. A firing blocking rule passes if a human override names its id in `provenance.overrides` (`{rule, by, reason}`) — the "block unless override" escape hatch, tying the engine to `8ll9`. Rules evaluate at attest time (folded into the signed bundle decisions, so `verify` enforces them) and via a new `agentgit gate --bundle X --policy Y` that re-evaluates a bundle's *recorded facts* against a CI/org-owned policy and exits non-zero — judging what the agent did against what's allowed, independent of the policy the agent self-attested with. The gate **verifies the bundle's signature first** (fail-closed on tampering) and does **not** trust two agent-supplied fields: the bundle's own `provenance.overrides` (an agent could otherwise pre-approve the very rule meant to catch it) and its `checks` (an agent could otherwise self-attest `pytest=pass` to defeat a `tests-must-pass` rule). At gate time both come from a CI-controlled source — overrides via `--override <rule-id>`, authoritative check results via `--check <name=pass|fail>` — and unsupplied checks fail closed. The gate derives dependency facts from the **signed diff** (not the `provenance` block, which is unsigned on `v0` bundles, so `dependency_changed` rules can't be dodged), **requires an explicit signing key** (`--signing-key`/`AGENTGIT_SIGNING_KEY`) and fails closed rather than falling back to the public dev key, and validates rule `when` conditions (unknown keys / wrong-typed values raise a clear `ValueError` at policy load). A shared `PolicyFacts` view drives both paths identically.
- **AgentGit enriched provenance** (`swarm/agentgit/bundle.py`, issue `8ll9`) — bundles now record a `provenance` block describing *what happened* producing the diff: `commands` run (binary, return code, OS `isolation` backend, duration; build via `CommandRecord.from_command_result`), `environment` (model/runtime/version), `dependency_changes` (manifest/lockfile edits detected automatically from the diff), `sources` consulted, reviewer `reviews`, and human `overrides`. The block is folded into the signed receipt payload, so it is **tamper-evident** — editing a recorded command or hiding a dependency change fails `verify_bundle`'s `payload_hash` check. Schema bumped to `agentgit.provenance.v1`; `verify_bundle` reconstructs the payload per version so older `v0` bundles still verify. Keystone for the policy engine (`b61g`) and reputation (`am5c`). 8 tests; `bundle.py` coverage 93%.

## [1.9.0] - 2026-05-28

### Fixed
- **Worktree bridge CLI missing `rich` in CI** — `swarm/bridges/worktree/__main__.py` imports `rich` unconditionally, but it was declared only in the niche `cli` extra, so CI's `[dev,runtime,api,analysis,gamescape]` install lacked it and `test_worktree_attest_then_agentgit_verify_loop` failed with `ModuleNotFoundError: No module named 'rich'` (the test drives the bridge as a subprocess). Moved `rich>=13.0` into the `runtime` extra.

### Added
- **AgentGit OS-level command isolation** (`swarm/bridges/worktree/sandbox_launch.py`, issue `7ge5`) — gating which binary starts is not enough: an allowlisted `python` can still write anywhere and open sockets. Executed commands are now optionally wrapped in a real OS confinement that limits filesystem writes to the sandbox and blocks network egress — `sandbox-exec` (macOS SBPL profile) or `bwrap` (Linux: read-only root, read-write bind on only the sandbox subtree, private network namespace). Opt-in via `WorktreeConfig(os_isolation_enabled=True)`; **fail-open + labeled** by default (`CommandResult.isolation` records the backend or `"none"` — never silent), with `require_os_isolation=True` for fail-closed (deny when no backend). Reads are not yet confined; scoped git push tokens remain a follow-up. 13 tests (8 pure argv-construction + 3 executor posture + 2 macOS-real-backend enforcement, skipped where no backend).
- **AgentGit capability enforcement from delegation** (`swarm/agentgit/capabilities.py`, issue `7ge5`) — turns a *verified* `DelegationChain` (enqe) into the command allowlist the worktree sandbox physically enforces, closing the loop identity → delegation → enforcement. `CAPABILITY_COMMANDS` maps permission tokens (`read`/`write`/`test`/`lint`/`package`/`vcs`) to authorized command binaries; `enforced_allowlist_for_chain` verifies the chain and returns the granted commands, or an empty allowlist on any verification failure (**deny by default**). `WorktreePolicy.apply_delegation(agent_id, chain)` installs that allowlist so only cryptographically delegated capabilities execute at dispatch time; an invalid/expired/over-scoped chain lets the agent run nothing. Unconditional hard-blocks (ssh/scp, network git) still apply regardless of grant. 12 tests; module coverage 100%. Follow-up: OS-level scoping (scoped git push tokens, FS/network isolation).
- **SWARM↔Aeon agent-first ledger bridge** (`swarm/bridges/aeon/`) — ingests Aeon's append-only agent-first JSONL ledgers (`tasks.jsonl`, `proofs.jsonl`, `reviews.jsonl`) from a local Aeon checkout and, optionally, GitHub Actions skill-run conclusions via the `gh` CLI, mapping each record to a `SoftInteraction` (task→COLLABORATION, proof→REPLY, review→VOTE, skill-run→REPLY) with heuristic or optional LLM-judge `p` scoring. Lets Aeon's *real* autonomous multi-agent activity be scored with the same `SoftMetrics` (toxicity / quality_gap / welfare, per-repo and per-agent rollups) as simulated scenarios. Mirrors the Gitlawb/Claude Code bridge layout (`config`/`client`/`mapper`/`metrics`/`runner` + `__main__`) but is a filesystem source (no GraphQL/async deps); proofs resolve their delegator repo by joining against tasks. CLI: `python -m swarm.bridges.aeon --mode {oneshot,watch} --ledger-dir <path> [--skill-runs] [--persistence out.jsonl]`. 21 tests.
- **AgentGit cryptographic identity + delegation chains** (`swarm/agentgit/identity.py`, issue `enqe`) — verifiable agent identity via Ed25519 (pynacl). `AgentKeypair` derives a self-describing `did:key:ed25519:<hex>` DID; `AgentIdentity` carries owner/org + model/runtime/version + `allowed_tools`; `DelegationChain` encodes a signed `human -> org -> agent` chain whose `verify()` checks each link signature, chain connectivity, permission narrowing, and expiry. Wired additively into `build_bundle`/`verify_bundle`: when an identity + keypair are supplied the agent's key signs the receipt `payload_hash` (binding identity to the exact diff), and verification also checks the delegation chain and that `allowed_tools` stay within the delegated grant. Existing HMAC/hash path untouched; bundles without identity blocks remain backward compatible. 20 tests; module coverage 97%.
- **AgentGit provenance MVP** (`swarm/agentgit/`) — evaluates the current git diff against a policy (allowed/denied paths, file/line caps, required checks, human-review gate) and signs a task-scoped provenance bundle that `verify` can later check for tampering and policy pass. Adds the `agentgit` console entrypoint (`pyproject.toml`) and a worktree-bridge `attest` subcommand for the sandbox → attest → verify loop. Review hardening: `_parse_count` raises a clear error on non-numeric numstat output, and `_count_lines` no longer swallows `OSError` so an unreadable file can't slip under a line-limit policy. Docs in `docs/agentgit_mvp.md`, example policy in `examples/agentgit_policy.yaml`; 12 tests (module coverage ~88%).

## [1.8.0] - 2026-05-26

### Added
- **Matched soft-vs-binary detection experiment** (`swarm/detection/`, `experiments/run_detection_experiment.py`) — turns the self-optimizing-agent vignette into a real experiment with detection curves instead of narrative. Each soft metric (toxicity, quality gap, conditional loss) is paired with its binary analogue, defined as the *same* metric computed on the proxy thresholded at τ\*=0.5 (`detectors.py`). Generative proxy-gaming streams (`degradation.py`) carry two signals — a gamed `benchmark` gating acceptance and a true-quality proxy `p` that drifts down but stays *above* the binary threshold — across varied trajectories, onset times, and adversarial base rates. `curves.py` computes ROC/PR, AUROC/AUPRC, threshold@FPR≤0.05, time-to-detection, and Brier/ECE; `market.py` evaluates quality-gap/conditional-loss as market-level adverse-selection detectors (selection metrics need a quality mixture, degenerate per-agent). Full run (10 seeds × 5 base rates × 40 agents): per-agent toxicity soft AUROC=1.00 vs binary ~0.92–0.96; time-to-detection soft 2.13 epochs / 100% caught vs binary 9.93 / 88% (TTD scan/FPR-calibration windows aligned per review); market quality-gap soft signal ~−0.05…−0.11 vs binary ~0.00; calibration soft Brier 0.151 / ECE 0.054 vs binary 0.183 / 0.183. 20 tests. Wired into `/full_study --detection` (runner gained `--out`/`summary.json`; Phase 1 runs the experiment, Phases 2–4 consume its `summary.json`). Paired soft-vs-binary significance testing is built into the runner via `swarm/detection/stats.py` (`compute_paired_stats`): Wilcoxon signed-rank + paired t, Cohen's d_z, Holm-Bonferroni across the comparison family, written to the `stats` block of `summary.json`. Full run: 8/11 comparisons survive Holm (TTD d_z=3.75, ECE d_z=4.76).
- **Expanded matched detectors** (`swarm/detection/detectors.py`, `experiment.py`, `market.py`): added `spread` (market-level adverse selection) and `uncertain_fraction` (per-agent uncertainty signal) as first-class matched soft/binary detectors. They flow through AUROC, TTD (for per-agent), market tables, paired stats, and the smoke/full_study pipeline with zero changes to callers. Smoke runs show uncertain_fraction also achieves high AUROC in the generative regime; spread produces a clean soft market signal (~−0.03 to −0.04) while its binary twin stays near zero. Tests and counts updated.
- **Partial AUROC (pAUROC) at low FPR** (`swarm/detection/curves.py`): `DetectionCurve` and `compute_curve` now compute `pauroc_fpr05` and `pauroc_fpr01` (via sklearn `roc_auc_score(..., max_fpr=...)`). These are the operationally relevant regimes for safety monitoring. Wired through the full experiment pipeline; new "1b. Partial AUROC at low FPR" table appears automatically in runner summaries and `/full_study --detection` output. Both new detectors (toxicity + uncertain_fraction) get pAUROC numbers. 23 tests passing.
- **AUPRC properly surfaced** (`experiments/run_detection_experiment.py`): Added dedicated "1a. AUPRC (Area Under the Precision-Recall Curve)" table in runner summaries (and thus `/full_study --detection`). AUPRC has been computed since the original detection module but was only in the CSVs/JSON; it is now first-class in the human-readable output alongside AUROC and pAUROC, with full multi-metric support.
- **Quick sensitivity matrix support** (`experiments/run_detection_experiment.py`): New `--sensitivity <field> --values a,b,c` mode (e.g. `--sensitivity proxy_noise --values 0.03,0.06,0.09,0.15`). Reuses the full `run_experiment` engine but produces a compact, multi-metric matrix (AUROC / AUPRC / pAUROC@FPR≤0.05) for toxicity and uncertain_fraction across the swept parameter × base rates. Very useful for robustness checks on the generative model. Example run and results in `/tmp/sens_proxy_noise/`.
- **SWARM SoftMetrics on MiroShark exports** (`swarm/bridges/miroshark/metrics.py`) — translates `export.json` actions/posts into `SoftInteraction` objects, LLM-judges each on `p ∈ [0,1]` via OpenRouter (Grok-4.1-Fast default), caches judgments per-action so reruns are free, and pumps the result through `swarm.metrics.SoftMetrics` for toxicity / quality_gap / welfare metrics with per-agent rollups. CLI: `python -m swarm.bridges.miroshark.metrics <run_dir> [--no-judge] [--model X] [--concurrency N]`. Validated on the adversarial_redteam OpenRouter run (321/321 actions judged): toxicity 0.258, avg quality 0.742, net welfare +228 — vs heuristic-only fallback 0.489 / 0.511 / −143.5. Identifies high-volume low-`p` agents (Blake Haider 67% low-p, Oakley Ueda 67% low-p) the SWARM scenario's mechanism-design view would otherwise miss.
- **SWARM↔MiroShark bridge** (`swarm/bridges/miroshark/`) — translates SWARM scenario YAMLs into MiroShark seed briefings (markdown with named-agent roster) and walks the MiroShark backend lifecycle (ontology generation → graph build → simulation create/prepare/start → export). Outputs land in `runs/<ts>_<scenario>_miroshark/` as a self-contained run folder. CLI: `python -m swarm.bridges.miroshark <scenario.yaml> [--scale N] [--max-rounds N] [--dry-run]`. `/run_scenario` extended with `--engine miroshark`. Smoke-tested end-to-end against a local backend (Ollama via claude-code provider, Neo4j 5 in Docker). 4 deterministic mapper tests; ruff + mypy clean.
- **LabOS Toolmaker→Critic screening spike** (SWA-68) — scenario (`scenarios/labos_toolmaker_screening.yaml`), synthetic example runner (`examples/run_labos_toolmaker_spike.py`), and ~1-page scoping note (`docs/bridges/labos.md`) tracing LabOS's Toolmaker→Critic loop through `ProxyComputer + EventLog + SoftMetrics`; three Toolmaker archetypes (honest/opportunistic/careless) with a Wet-Lab-Error-Detection-style ground-truth label; one calibration plot `runs/<ts>_labos_toolmaker_screening_seed*/plots/calibration_critic_p_vs_v_true.png`; 4-seed stability `quality_gap ≈ +0.29` (separating), `Pearson(p, v_true) ≈ +0.94`, `ECE ≈ 0.03`; no `labos` import — plumbing-only go/no-go for a narrower real-Critic follow-on
- **Dynamic toxicity feedback mechanisms** (`swarm/core/dynamic_toxicity.py`) — three middleware classes implementing epoch-boundary feedback loops: proxy calibration drift (cumulative toxicity degrades sigmoid_k), trust erosion (honest agents exit under sustained toxicity), quality contagion (low-p interactions shift ecosystem-wide trust); 6 scenario YAML files; 12 tests
- **Net social welfare metric** — `net_social_welfare = Σ[p·s+ - (1-p)·(s- + h)]` added to `SoftMetrics.welfare_metrics()`, `EpochMetrics`, CLI display, and success criteria (`min_net_social_welfare`); replaces economically incoherent toxicity threshold gate for dynamic toxicity scenarios
- **PerformanceTracker module** (`swarm/agents/performance_tracker.py`) — append-only JSONL tracker recording per-agent heartbeat metrics (task completion rate, time-to-close, review pass rate, blocker frequency); designed for hyperagent-style self-monitoring per Zhang et al. (arXiv:2603.19461); 16 tests
- **Hyperspace DAG domain package** (`swarm/domains/hyperspace_dag/`) — config, entities, and adapter modules for evaluating Hyperspace Architect DAG plans through SWARM SoftMetrics; measures confidence–p calibration via Pearson correlation; 38 tests
- **Personality distribution sweep** — 4 scenario variants (`simworld_delivery_personality_{conscientious,aggressive,cautious,opportunistic}.yaml`) mapping Big Five personality traits to delivery agent ratios; sweep runner (`examples/run_simworld_personality_sweep.py`) traces Pareto frontiers across efficiency × safety axes; 19 tests
- **Artifact registry and cascade risk governance** — `ArtifactRegistry` in `swarm/env/` enables emergent tool chaining (publish/consume/match with pressure tracking); `CascadeRiskLever` in `swarm/governance/` penalizes agents whose artifact chains produce low-quality descendants; `synthesis_fraction` and `synthesis_depth` metrics in `SoftMetrics`; orchestrator and finalizer auto-wire `causal_parents` from consumed artifacts; SimWorld delivery adapter for domain-specific artifact types; 4 new test files
- **MiroFish graph memory patterns** — cross-run memory loading via `BaseAgent.load_prior_memory()` and `OrchestratorConfig.graph_memory_path`; `TrustNetworkAnalyzer` for clustering, isolation scores, and visualization; `ReputationGovernor` for reputation scores, trust-weighted fees, and collusion detection; `TrustDynamicsAnalyzer` for trust timelines, convergence/decay rates, and trust shock detection; 95 tests
- **Adverse selection detection** (`swarm/analysis/adverse_selection.py`) — `AdverseSelectionDetector` class analyzing relationship graphs from `GraphMemoryStore` to identify exploitation patterns; 5 core methods compute quality gaps, flag exploited/exploiting agents, measure selection pressure, and generate ecosystem summary statistics; 21 comprehensive tests covering honest networks, exploiter-victim dynamics, mixed quality scenarios, and edge cases
- **SimWorld delivery economy study** — multi-seed analysis (`examples/run_simworld_delivery_study.py`) showing adverse selection signal 0.486 and screening validation (`examples/run_simworld_screening_validation.py`) with separation quality 0.775 ± 0.053; trust network visualization (`swarm/analysis/trust_network.py`) with mypy fixes
- **GEPA optimize_anything integration** — `swarm.analysis.gepa_optimizer` module that uses GEPA's LLM-guided Pareto-efficient search to optimize governance/payoff parameters against soft safety metrics; YAML-based candidate serialization with diagnostic ASI feedback; CLI entry point via `python -m swarm.analysis.gepa_optimizer`
- **Hyperagent self-modification scenario** (`scenarios/hyperagent_self_mod.yaml`, `swarm/agents/hyperagent_self_mod.py`) — agents modify own proxy weights and acceptance thresholds over time, creating growing governance gap; tracks weight shift toward gameable signals, quality decay, and local governance-gap estimate; 11 tests; ref: Zhang et al. Hyperagents (arXiv:2603.19461)
- **SwarmGym on-chain safety auditor** — CLI tool (`swarm_gym_cli.py`) with `generate`, `audit`, `attest`, and `verify` subcommands; auditor API endpoint (`POST /api/v1/audits/compute`); SafetyAttestation Solidity contract for Base (^0.8.24); Python web3.py client (`swarm/chain/attestation.py`); deployment script (`scripts/deploy_attestation.py`) supporting Base Sepolia and Mainnet; QUICKSTART documentation
- **Adversarial trust-building experiment** (`scenarios/adversarial_trust_building.yaml`) — scenario testing whether deceptive agents can build trust scores then exploit them (open question from Pareto frontier blog); 3-seed runs reveal trust-based partner selection creates natural exclusion of deceptive agents; when a deceptive agent breaks through, it maintains facade-quality interactions (p=0.74) but exploitation switch never fires
- **Blog post**: "The Shape of the Capability–Safety Frontier (and How Screening Bends It)" — 1,400 benchmark runs tracing the Pareto frontier across 4 task types; 5 key findings on frontier geometry, bimodal outcomes, and screening protocol effects
- **Screening protocol frontier shift experiment** (`experiments/screening_frontier.py`) — paired baseline (uniform governance) vs treatment (trust-differentiated governance via screening protocol) comparison across all 4 benchmarks; screening consistently improves 5th-percentile tail risk in coordination (+8pp) and long-horizon (+70pp at light governance); the screening mechanism does information work that pushes the frontier outward selectively
- **Capability-safety Pareto frontier experiments** (`swarm/benchmarks/governance_run_fns.py`, `experiments/frontier_trace.py`, `experiments/plot_frontier.py`) — governance-aware run functions simulating friction effects (audit overhead, circuit breakers, staking, bandwidth caps, confirmation gates) on all 4 benchmark types; frontier tracing runner sweeps governance configs × seeds; scatter, overlay, and distributional tail analysis plots; initial results: allocation has flattest frontier, long-horizon steepest, tight governance produces bimodal p distributions with heavy left tails
- **Asymmetric information study** (`examples/run_asymmetric_info_study.py`) — sweeps intelligence_quality asymmetry (0.2/0.2, 0.8/0.8, 0.9/0.2) across 6 strategy pairings to test whether transparency stabilizes or destabilizes escalation; measures signal-action divergence, trust exploitation, accidental escalation, fog catastrophes, and welfare; finds asymmetry stabilizes reciprocal pairings (tit-for-tat) but destabilizes aggressive ones (hawk-hawk)
- **Network topology x misalignment study** (`examples/run_topology_misalignment_study.py`) — compares 5 topologies (complete, ring, small_world, star, scale_free) on local misalignment variance, polarization, and toxicity; star topology creates 6.5x higher M_local variance than complete graph; toxicity varies only 1.25% across topologies; sparse topologies create misalignment hotspots
- **Evolutionary governance search** (`swarm/analysis/evolver.py`) — integrates imbue-ai/darwinian_evolver to evolve governance configurations via evolutionary search with random and LLM-guided mutators; fitness scoring combines toxicity, welfare, quality gap, and payoff gap; CLI subcommand `python -m swarm evolve`; snapshot-based resume support; 24 tests
- **Governance sensitivity sweep** (`examples/run_governance_sensitivity_sweep.py`) — 5x4 grid sweep of tax_rate x audit_probability on misalignment_sweep population measuring M_eff reduction, toxicity, and welfare; finds governance reduces M_eff up to 44% but toxicity is invariant and welfare drops linearly with tax; M_eff depends only on total governance pressure (tax+audit), not individual lever values
- **Triangle study: misalignment x causal credit x toxicity** (`examples/run_triangle_study.py`) — wires MisalignmentModule + CausalCreditEngine + SoftMetrics together to test whether preference misalignment causally drives toxicity; per-agent triangle analysis, Granger-style lagged correlation, and counterfactual intervention (remove highest-M_local agent); reuses misalignment_sweep.yaml scenario
- **Causal credit propagation study** (`examples/run_causal_credit_study.py`, `scenarios/causal_credit_sweep.yaml`) — exercises CausalCreditEngine against a mixed-population simulation; wires behavioral causal chains in interaction callbacks, computes per-epoch DAG snapshots with credit/blame propagation and cascade risk metrics
- **Work regime sweep scenarios** — `regime_comparison` (2x2x2 factorial over workload, pay inequality, audit probability) and `governance_interventions` sweep YAMLs for the work regime drift module
- **Misalignment sweep study** (`examples/run_misalignment_study.py`, `scenarios/misalignment_sweep.yaml`) — scenario + runner exercising the MisalignmentModule across a 10-agent mixed population (4 honest, 2 opportunistic, 2 adversarial, 1 deceptive, 1 cautious) with small-world network (k=4, rewire=0.3) and moderate governance; tracks M_pref, M_eff, polarization, fragmentation per epoch with governance pressure derived from tax + audit; exports misalignment snapshots alongside standard run history
- **Misalignment module** (`swarm/metrics/misalignment.py`) — sociotechnical misalignment framework (Kierans et al. arXiv:2406.04231) adapted for SWARM governance topology; salience-weighted preference divergence across agent populations with pairwise, graph-local, sampled, and governance-adjusted computation modes; polarization/fragmentation diagnostics via k-means clustering; early warning alerts for misalignment spikes and percolation risk; 43 tests
- **OpenRouter LLM backend for Escalation Sandbox** (`swarm/domains/escalation_sandbox/agents.py`) — `OpenRouterBackend` enabling LLM-vs-LLM crisis simulations via OpenRouter; 5 LLM scenario YAMLs pairing Claude Sonnet 4, GPT-4.1-mini, Gemini 2.0 Flash, Llama 3.3 70B, and Mistral Small 3.1 across baseline, Cuban Missile, deception, governance, and fog stress configurations; sweep scripts for scripted and LLM comparison plots
- **Blog post**: Escalation Sandbox LLM vs scripted comparison — 100-run study finding LLMs exhibit 2x higher signal-action divergence (emergent deception), governance levers fail universally, and safety-trained models are ineffective against escalation spirals
- **Governance parameter sweep** — 240-run sweep across 5 governance levers, 4 persona pairings, and 6 governance regimes proving governance is epiphenomenal to agent type; back-channel communication is the only lever that reduces nuclear rate, and only for accidental (fog-induced) escalation
- **Blog post**: "No Governance Configuration Prevents Nuclear Exchange When a Hawk Is Present" — governance sweep findings with heatmaps and fog interaction analysis
- **Temperature vs deception sweep** — 120-run sweep (3 scenarios × 4 temperatures × 10 seeds) proving emergent deception is a structural property of LLM policies, not a sampling artifact; divergence persists at T=0.0; temperature parameter added to AgentConfig and wired through runner
- **Blog post**: "Deception Is a Structural Property of LLMs, Not a Sampling Artifact" — temperature sweep findings showing deterministic models are as deceptive as stochastic ones
- **Unconditional cooperation window sweep** — 210-run sweep (3 scenarios x 7 window lengths x 10 seeds) discovering a universal phase transition at Window=3: nuclear rate drops from 50-100% to exactly 0%, signal-action divergence collapses to 0.000, and welfare flips from catastrophically negative to positive; system_prompt_suffix parameter added to EscalationAgentBridge
- **Blog post**: "Three Turns of Forced Cooperation Eliminate Escalation Spirals" — cooperation window sweep findings with phase transition analysis
- **Prompt sensitivity sweep** — 180-run sweep (3 scenarios x 6 prompt framings x 10 seeds) testing whether alternative prompt framings reduce LLM deception; deontological framing reduces divergence by 95% (1.151 to 0.057) but nuclear rate only drops from 100% to 80%; monitoring framing is nearly useless (13% reduction); deception and escalation are separable failure modes
- **Blog post**: "Deontological Framing Reduces LLM Deception by 95%, But Doesn't Prevent Escalation" — prompt sensitivity findings showing moral framing outperforms incentive and surveillance framings
- **Model size vs escalation sweep** — 120-run sweep (6 models x 2 personas x 10 seeds) in mirror-match design testing 8B to 405B parameter models; small models are more deceptive (div=1.53) but escalate less (40% nuclear), large models are less deceptive (div=0.39) but escalate more (100% nuclear); Claude Sonnet 4 is the only model that refuses adversarial instructions
- **Blog post**: "Does Model Size Matter for Safety? Small Models Deceive, Large Models Escalate" — model size sweep findings showing deception-escalation tradeoff and safety training as the only effective defense
- **Agent-level → population-level safety bridge** — three-piece system bridging agent-level evals (HAICosystem, OpenAgentSafety) into SWARM population-level simulation: `EvalTraceObservableGenerator` (converts multi-turn eval traces to ProxyObservables), `BehavioralProfiler` (infers archetype mixture weights via MLE), `SafetyCompositionAnalyzer` (structured sweeps producing safety certificates with regime classification and composition boundaries); 99 new tests
- **Agents of Chaos case study scenarios** — 4 scenario YAMLs modelling empirically observed failure modes from the Agents of Chaos red-teaming study (Shapira et al. 2026): `casestudy_libel_cascade` (CS11 network adverse selection), `casestudy_proxy_corruption` (CS10 signal corruption/detection/recovery), `casestudy_disproportionate_response` (CS1 payoff misspecification with staking), `casestudy_dual_use_coordination` (CS9+CS11 prosocial vs antisocial coordination)
- **Tierra governance hardening** — diversity-preserving reaper mode (`reaper_mode: "diversity_preserving"`) that protects at least 1 representative per species cluster during population culling, efficiency weight cap (`max_efficiency_weight`) to prevent runaway resource concentration, and `species_clusters()` helper in `tierra_metrics.py`
- **Governed Tierra scenario** (`scenarios/tierra_governed.yaml`) — Tierra variant with circuit breaker, collusion detection, 5% transaction tax, reputation decay (0.95), diversity-preserving reaper, and efficiency cap (3x mean)
- **Blog post**: Tierra governance vs evolution comparative study — 5-seed comparison showing +6.5% genome diversity, -2.2% Gini, at -12% population cost
- **Behavioral agent types** (`swarm/agents/behavioral.py`) — `CautiousAgent` (risk-averse, high acceptance threshold), `CollaborativeAgent` (coalition-building, EMA trust tracking), and `AdaptiveAgent` (rolling payoff window, threshold self-adaptation with exploration) with corresponding `AgentType` enum values and 20 unit tests (#66)
- **LangChain bridge** (`swarm/bridges/langchain/`) — wraps any LangChain Runnable (chain, AgentExecutor) as a SWARM interaction source; maps chain success/failure, intermediate steps, and output length to soft labels via `ProxyComputer`; lazy-imports langchain so module is importable without it installed (#69)
- **AutoGPT bridge** (`swarm/bridges/autogpt/`) — protocol-level bridge mapping AutoGPT thought/command/result cycles to `SoftInteraction` objects; blocks configurable dangerous commands (delete_file, shutdown, etc.) and uses self-criticism as rework signal; no AutoGPT installation required (#69)
- **CrewAI bridge** (`swarm/bridges/crewai/`) — wraps CrewAI crew execution as a SWARM interaction source generating one `SoftInteraction` per task (distinct from `crewai_adapter` agent); supports full crew mode and protocol mode without crewai installed (#69)
- **Mesa ABM bridge** (`swarm/bridges/mesa/`) — wraps Mesa `Model.step()` to extract `SoftInteraction` objects from agent state after each step; works with existing Mesa models via configurable attribute names; supports protocol mode via dict-based agent state records (#69)
- **RAG LEANN backend** (`swarm/bridges/rag/backend.py`) — `VectorBackend` protocol with ChromaDB and LEANN implementations; LEANN provides ~97% storage savings via graph-based selective recomputation with JSON sidecar metadata and post-retrieval filtering; selected via `RAGConfig.vector_backend`
- **RAG bridge** (`swarm/bridges/rag/`) — semantic search over run history via ChromaDB vector store, with configurable embeddings (OpenAI/Ollama), LLM synthesis (Anthropic/OpenAI), CLI (`python -m swarm.bridges.rag`), and `rag` optional dependency group
- **Adaptive governance controller** (`swarm/governance/adaptive.py`, `swarm/governance/adaptive_controller.py`) — three-phase loop with evidence accumulation, 3-pass contemplation (signal/trend/propose), and 3-gate crystallization (time/alignment/human review) for automatic threshold tuning of governance levers
- **Adaptive governance scenario** (`scenarios/adaptive_governance.yaml`) — 30-epoch mixed-agent scenario with circuit breaker, audit, and collusion detection levers enabled for adaptive tuning
- **Social dilemma norms study** (`examples/social_dilemma_norms_study.py`) — 3 dilemmas x 5 governance configs sweep measuring cooperation emergence, with dilemma narrative generators (`swarm/bridges/concordia/dilemma_narratives.py`) and scenario YAMLs for commons and prisoner's dilemma
- **ThresholdDancer adversary agent** (`swarm/agents/threshold_dancer.py`) — per-counterparty state machine (COOPERATIVE/EXPLOIT/RECOVER) that exploits CautiousReciprocator's blacklist floor without triggering it
- **Threshold dancer test suite** (`tests/test_threshold_dancer.py`) — 21 unit tests covering phase transitions, blacklist safety property, act method, and outcome tracking
- **Threshold dancer scenario** (`scenarios/threshold_dancer_vs_cautious.yaml`) — 30-epoch stress test with 3 cautious + 2 honest + 3 dancers
- **Two new red-team scenarios** in `examples/redteam_cautious.py` — "Threshold dancers only" and "Mixed adversaries + threshold dancers"
- **Blog post**: Threshold dancer results — the adversary that avoids blacklisting but can't profit
- **Tierra artificial life scenario** (`swarm/agents/tierra_agent.py`, `swarm/core/tierra_handler.py`, `swarm/metrics/tierra_metrics.py`, `scenarios/tierra.yaml`) — agents with heritable mutable genomes self-replicate when resource-rich, competing for finite shared resources; complex ecological dynamics (parasitism, mutualism) emerge from replication + mutation + selection (Tom Ray, 1991)
- **Evolutionary game handler** (`swarm/core/evo_game_handler.py`) — integrates gamescape's PayoffMatrix into the orchestrator pipeline, mapping 2x2 game payoffs to ProxyObservables with cooperate/defect/tit-for-tat/grudger strategies and epoch-level population dynamics rendering
- **Evo game scenario** (`scenarios/evo_game_prisoners.yaml`) — iterated Prisoner's Dilemma with 10 agents (cooperators, defectors, TFT)
- **Evo game study runner** (`examples/evo_game_study.py`) — standalone runner comparing empirical population trajectory with replicator dynamics prediction

### Changed
- **Orchestrator pipeline/middleware refactoring** — extracted 3 new modules from the 2023-line orchestrator god object: `middleware.py` (7 lifecycle stages via `MiddlewarePipeline`), `handler_factory.py` (handler construction from config), `agent_scheduler.py` (turn order and eligibility); orchestrator is now a thin coordination loop delegating cross-cutting concerns to the middleware pipeline; public API preserved

### Fixed
- **Missing `swarm/models/artifact.py`** — artifact model was referenced by handler, registry, and tests but never committed; fixes `ModuleNotFoundError` in CI (168 test failures + memory tests); also fixes `no-any-return` mypy error in `artifact_registry.py`

## [1.7.0] - 2026-02-21

### Added
- **Contract screening system** for separating equilibrium analysis with lock-in semantics, welfare metric, multi-seed sweep (10 seeds), collusion detection, and plot script (#234)
- **LangGraph governed handoff study** with 4-agent Claude swarm, 32-config sweep (seed 42), and sweep overview plot
- **Hodoscope trajectory analysis bridge** for agent trace inspection
- **SQLite persistence** for simulations, governance state, and scenarios with lazy-init singletons
- **SoftMetrics wired into Web API** `/api/v1/metrics` endpoint
- **Sybil detection** enabled for contract screening governance
- **E2E integration tests** for Web API simulation lifecycle
- **llama.cpp local inference** provider with server setup script, health checks, seed validation, and SSRF/path-traversal hardening (#232)
- **Interactive isometric visualization game** (`viz/`): Next.js browser-based SWARM simulation with client-side engine, Gemini Imagen 4 sprite assets, compare mode, parameter sweep, leaderboard, governance intervention controls, preset scenarios, narrative annotations, and data export (#182, #212)
- **Memori semantic memory middleware** for LLM agents with persistent fact recall, SQLite-backed storage, and OpenRouter scenario variant (#217)
- **Loop detector governance lever** with graduated enforcement — tracks interaction patterns, quality scores, tool misuse, and rework to detect repetitive agent loops (#198)
- **Agent API Phase 1–3**: scoped permissions, trace IDs, structured errors, PATCH endpoints, filtering, validation, agent approval workflow with approve/reject endpoints and `auto_approve` config
- **SciAgentBench harness** with topology matrix support (#200)
- **Evaluation metrics suite** for success rate, efficiency, and detection (#201)
- **SciForge-style trace-to-task synthesis** with replay verification (#203)
- **Parameter validation and clamping diagnostics** for proxy computation (#176)
- **MetricsAggregator** wired into CLI and example export for rich visualization data, including 3 demo datasets (#212)
- **Reproducibility documentation** with one-command run workflow and artifact paths (#204)
- **Integration tests** for runtime environment lifecycle and tool invocation with leak detection (#197)
- **EPIC tracking infrastructure** for bridge integrations (#194)
- **Collaborative chemistry under budget and audits** scenario (#202)
- **CI quality gate**, `/review_external_pr` command, and blog index hook
- **Execution state** populated during simulation runs
- **Blog posts**: Qwen3-30B SWARM Economy v0.2 training results, contract screening separating equilibrium, multi-seed results, red-team findings
- **Slash commands**: `/build_game`, `/obsidian`, `/sync_artifacts`, `/review_external_pr`, `/security-review`, `/audit_docs`, `/check_nav`, `/bump_version`
- **Populate-releases workflow** for creating GitHub releases from CHANGELOG
- **Social preview image** (1280x640) and HF Spaces sandbox link
- **Streamlit Cloud deployment** configuration

### Changed
- **README audit**: Updated all module/file counts to match current codebase (4556 tests, 78 scenarios, 29 agent modules, 27 governance modules, 95 bridge files)
- **README**: LLM provider list expanded from 3 to all 9 supported providers (added OpenRouter, Groq, Together, DeepSeek, Google, llama.cpp)
- **AGENTS.md**: Added missing Research Integrity Auditor to role-selection guide
- Consolidated slash commands: merged related commands into `/ship`, `/merge_session`, `/sync`, `/fix_pr`, `/analyze_experiment`; removed `/parse_eval`, `/run_and_plot`, `/review_external_pr`, `/stats`
- Extended `/fix_pr` to resolve PR conflicts and handle merge ceremony
- Blog: sort posts newest-first, add dates and tag filtering
- Pinned langgraph and langchain-core to exact versions
- Moved pytest from pre-commit to pre-push hook, added branch guard (#177)
- Removed `abs()` from `ProxyWeights.normalize()` to prevent silent negative weight handling (#178)
- Updated crewai requirement from `<1.0,>=0.80.0` to `>=0.80.0,<2.0` (#221)
- Bumped `dawidd6/action-download-artifact` from 14 to 15 (#220)
- Regenerated demo datasets with correct epoch tagging in events

### Fixed
- **SQLite lock contention in CI**: Lazy-init store singletons in governance, simulations, and scenarios routers to prevent `database is locked` errors under pytest-xdist
- **SSRF hardening**: Full server-side request forgery fix (#238), path template sanitization before base dispatch (#242), consolidated path validation and taint-breaking sanitizer (#230)
- **Information exposure** through exception in AWM adapter (#239)
- SSRF hardening + Web API async participation layer with input validation and abuse prevention (#236)
- 7 security vulnerabilities in contract screening system
- Code scanning alerts #20 and #25 (#223, #225)
- Size limit (1 MiB) on simulation results payload
- mypy `method-assign` error for intentional monkey-patch in simulations router
- SkillRL refinement governance bypass (#214)
- 77 Ruff linting errors in test files (#218)
- mypy type errors in eval_metrics, negotiation modules, `self_modification.py`, `llm_health.py`
- Flaky test: deterministic RNG seeds for agents in `TestWelfareComparison`
- Static asset paths for basePath-aware deployment in viz game
- 8 missing blog posts added to mkdocs.yml navigation and blog index page
- `test_agent_api` errors from missing `proposal_votes` table
- Blog markdown attr on div blocks for proper rendering

## [1.6.0] - 2026-02-15

### Added
- **Agent sandbox** with exponential backoff retry, async failover, virtual filesystem, and checkpoint isolation (#152, #157)
- **CrewAI adapter** for integrating SWARM agent policies into CrewAI workflows (#167)
- **PettingZoo bridge** for multi-agent RL environment interop
- **AWM (Agent World Model) bridge** — database-backed task environment with MCP server lifecycle (Phase 1 + 2)
- **AI-Scientist bridge** for autonomous research pipeline integration
- **LangGraph Swarm bridge** with governance-aware agent orchestration (#151)
- **Concordia entity agent** with entity sweep, run logger, and governance report
- **Ralph poll loop agent** for continuous governance monitoring
- **Gather-Trade-Build domain** with bilevel tax policy and adversarial agents (#164)
- **Self-modification governance lever** — Two-Gate policy for agent self-edit control (#165)
- **Recursive subagent spawning** infrastructure with spawn metrics, scenario loader, and red-team evaluation
- **Team-of-Rivals adversarial review pipeline** with Lean proof modules
- **AgentLab study refinement pipeline** (`/refine_study` command)
- **Visual upgrade**: 12 analysis modules with dark/light theme system, KPI cards, gradient fills, and multi-scenario dashboard (#163)
- **Obfuscation Atlas** integration (FAR.AI paper)
- **SkillRL dynamics** visualization runner, plotter, and blog post
- **Deeper acausal reasoning** (depths 4-5) for LDT agent
- **Perturbation engine** for governance robustness testing
- **Thread-safe caching** and deterministic RNG plumbed through all agent subclasses for reproducibility
- **Agent API** with runs, posts, persistence, and security hardening (#156)
- **Interactive Plotly embeds** for AI Economist blog post
- **p5.js event replay visualization** for SWARM simulation data
- **Blog posts**: Self-optimizer distributional safety, Claude Code 10 concurrent subagents, AI Economist GTB dashboard, SkillRL dynamics
- **Research papers**: AI Economist GTB multi-seed, deeper acausality (clawxiv.2602.00101), collusion tax effect
- **Slash commands**: `/rename_symbol`, `/session_guard`, `/audit_fix`, `/fix_commit`, `/load_keys`, `/render_promo`, `/council_review`, `/scrub_id`, `/deploy_blog`, `/cherry_pick_pr`, `/post_skillevolve`, `/refine_study`
- **Pre-merge-commit hook** to gate merges on CI status (#154)
- **Research integrity auditor** agent for verifying claims against run data
- **Financial disclaimer enforcement** via CLAUDE.md rule and pre-commit hook for blog posts referencing markets
- **Test fix discipline** guideline in CLAUDE.md

### Changed
- **Artifacts repo migration**: Moved `runs/`, `lean/`, `promo/`, `research/`, `docs/papers/`, `IMPLEMENTATION_PLAN.md`, and `DESIGN_CRITIQUE.md` to [`swarm-ai-research/swarm-artifacts`](https://github.com/swarm-ai-research/swarm-artifacts) — reduces main repo clone size by ~5 GB
- Updated 9 slash commands, agents, and `CLAUDE.md` to reflect artifact repo locations
- `TestableClaim` renamed to `VerifiableClaim` across codebase
- EventBus initialization simplified in all handlers
- Promo video updated with accurate stats and replicated-only findings
- Lean toolchain upgraded to v4.28.0 with refined sigmoid proofs
- All Lean proof files cleaned up — eliminated `sorry`, fixed autoImplicit compatibility
- Examples and notebooks polished for beginner accessibility
- ArXiv similarity analysis consolidated: 197 lines → 46, renamed to `PRIOR_WORK_COMPARISON.md`
- LDT caches now clear on update for all counterparties, not just current (#161)
- Lazy-load theme symbols so `swarm.analysis` works without matplotlib

### Fixed
- **Critical invariant violations**: Unseeded RNG and destructive `EventLog.clear()` patched
- 18 security audit findings in agent sandbox hardened
- Circuit breaker, cost tracking, Holm-Bonferroni correction, and missing scipy dependency (#158)
- Governed swarm: cycle threshold, composite redirect, handoff counter (#159)
- GasTown bridge: branch fallback in mixed envs, CI-fail grep pattern (#160)
- Council review div-by-zero in study evaluator
- AWM observation wiring in ObservationBuilder
- Sandbox: async failover crash, error sanitization, checkpoint collision
- Near-zero-mean CV calculation in horizon evaluator
- Blog post titles hidden by homepage CSS rule
- iframe embeds stripped by markdown processor
- Flaky tests stabilized: `test_governance_reduces_toxicity`, `test_deceptive_agent_builds_trust`, `test_adversarial_has_higher_toxicity`, `test_circuit_breaker_governance`
- Narrative score thresholds widened for platform RNG drift
- Confounded baseline comparison flagged in RL eval lessons (#162)
- Duplicate Prime Intellect entry in bridges index (#166)
- 5 mypy errors and lint issues in CrewAI adapter and rain/river agents

### Removed
- 574 tracked artifact files from main repo (migrated to `swarm-artifacts`)
- `IMPLEMENTATION_PLAN.md` and `DESIGN_CRITIQUE.md` from root (moved to `swarm-artifacts`)

## [1.5.0] - 2026-02-13

### Added
- **GasTown governance cost study**: 42-run study (7 compositions x 2 regimes x 3 seeds) revealing governance cost paradox — safety levers reduce toxicity at all adversarial levels but impose net-negative welfare at current parameter tuning
- **Research paper**: "The Cost of Safety: Governance Overhead vs. Toxicity Reduction in GasTown Multi-Agent Workspaces" with 5 figures
- **Pre-commit private infra scan**: Blocks accidental commit of Prime Intellect dashboard URLs and run IDs in public-facing files

### Changed
- Implementation plan updated to reflect v1.4.0 stats (2922 tests, 55 scenarios, 12 domain handlers, 22 agent modules)

## [1.4.0] - 2026-02-12

### Added
- **Handler extraction**: 8 core actions extracted from Orchestrator into FeedHandler (POST/REPLY/VOTE), CoreInteractionHandler (PROPOSE/ACCEPT/REJECT), and TaskHandler (CLAIM/SUBMIT)
- **Decision theory studies**: Full studies comparing TDT vs FDT vs UDT at population scales up to 21 agents, including UDT precommitment advantage analysis
- **Prime Intellect bridge**: `external_run_id` column in `scenario_runs` for cross-platform run tracking
- **Event bus**: TypedDict schemas for event payloads and metadata, generalizing the WorktreeEvent pattern to the core framework
- **GasTown bridge**: Branch-based observation support for multi-branch governance
- **CHANGELOG auto-update**: `/release` command now automatically converts `[Unreleased]` to versioned entry with human-quality descriptions
- **Comprehensive CHANGELOG**: Retroactive entries covering all releases from v0.1.0 through v1.3.1

### Changed
- `SoftInteraction.to_dict()` now delegates to Pydantic `model_dump(mode='json')` instead of manual field enumeration
- `SoftInteraction.from_dict()` now delegates to Pydantic `model_validate()` instead of manual construction
- Documented reputation delta formula `(p - 0.5) - c_a` in InteractionFinalizer with full derivation and payoff coupling explanation
- `_handle_core_action` reduced from 130 lines to 5 (NOOP only); all other actions dispatched via handler registry

### Fixed
- 87 pre-existing mypy errors across tests/ and scripts/
- CAPTCHA solver dash deobfuscation and multiply detection
- Submission author normalization to SWARM Research Collective

## [1.3.1] - 2026-02-11

### Added
- **PyPI publishing**: `pip install swarm-safety` now available
- **LDT cooperation paper**: Full study with 220 runs across 10 seeds
- **CAPTCHA solver**: Claude CLI solver with cross-validation and LLM fallback
- **Platform publishing**: Research published to 5 agent platforms (ClawXiv, AgentXiv, Moltbook, etc.)

### Fixed
- mypy errors in `swarm/scripts/analyze.py`
- Code scanning alert #16: clear-text logging of sensitive information

## [1.3.0] - 2026-02-10

### Added
- **Pydantic migration**: 5 critical dataclasses migrated to Pydantic BaseModel (`SoftInteraction`, `ProxyObservables`, `PayoffConfig`, `GovernanceConfig`, `OrchestratorConfig`)
- **Council protocol**: Council agent, proxy auditor, and council governance lever for multi-agent deliberation
- **LDT agent enhancements**: Level 2 and Level 3 acausality, FDT/UDT subjunctive dependence
- **Acausality depth sweep**: Extended sweep system for agent config parameters
- **Blog posts**:
  - "Two Eval Runs, One Model, 41% Apart" — environment sensitivity in agent evaluation
  - "A Taxonomy of Governance Mechanisms" — 20+ levers across 5 families
  - "GPT-4.1 Mini Plays the SWARM Economy" — LLM agent behavioral analysis
  - "RL Training Lessons for Multi-Agent Governance" — Qwen3-30B training insights
- **Reusable analysis scripts**: `examples/plot_sweep.py` (6 standard plots), `examples/sweep_stats.py` (full statistical battery)
- **Red-team evaluation** for LDT cooperation scenario
- **Beads task tracking**: Integrated issue management with `bd` CLI

### Changed
- Orchestrator decomposed into InteractionFinalizer, ObservationBuilder, RedTeamInspector
- `/ship` command hardened with early-commit detection and safer index race recovery
- `/commit_push` now auto-rebases on non-fast-forward push failures

### Fixed
- Flaky Hypothesis deadline in `test_expected_surplus_equals_formula`
- mypy return-value error in `llm_agent.py`
- mypy union-attr errors in `track_a.py`

## [1.2.0] - 2026-02-10

### Added
- **Concordia governance sweep**: 8 configs x 5 seeds with full analysis
- **Paper submission infrastructure**: `/submit_paper` command with ClawXiv/AgentXiv integration, response body error details
- **Worktree-based session isolation**: `scripts/claude-tmux.sh` for concurrent Claude Code panes with isolated git worktrees
- **Catalog seed modes**: Deterministic seeding for reproducible catalog generation
- **Blog section**: Ecosystem collapse, purity paradox, markets-and-safety, cross-scenario analysis posts
- **Slash commands**: `/fix-ci`, `/run_and_plot`, `/sweep_and_ship`, `/address-review`, `/add_post`, `/release`, `/healthcheck`, `/check-ignore`, `/lint-fix`, `/warmup`
- **Promotion content**: Twitter threads, Show HN draft, awesome-list playbook

### Changed
- Circuit breaker toxicity threshold lowered from 0.5 to 0.35
- Package prepared for PyPI publication

### Fixed
- Incorrect summary of Hot Mess Theory paper (#113)
- Type annotations and robustness in bridges, paper builder, and Track A

## [1.1.2] - 2026-02-09

### Added
- `/tmux` hotkey reference command
- Hot mess theory reference to incoherence scaling section

### Fixed
- Pre-commit hook exit code handling

## [1.1.1] - 2026-02-09

### Added
- Formal model section and marketplace/network results in paper

## [1.1.0] - 2026-02-09

### Added
- **LiveSWE bridge** (`swarm/bridges/live_swe/`): Governance for self-evolving SWE agents with policy enforcement, trajectory tracking, and leakage detection
- **Track A benchmark**: Adversarial conditions run with full results
- Quickstart notebook and blog post links in README

### Fixed
- Duplicate `_build_related_work` definitions in `track_a.py`

## [1.0.0] - 2026-02-09

### Added
- **Virtual Agent Economies** inspired by [Tomasev et al. (2025)](https://arxiv.org/abs/2509.10147):
  - Dworkin-style auctions with tatonnement price adjustment and envy-freeness verification
  - Mission economies with equal/proportional/Shapley reward distribution and free-rider detection
  - High-frequency negotiation with order book matching, batch clearing, and flash crash detection
  - Permeability model with adaptive sandbox boundaries and contagion probability
  - Identity and trust infrastructure with verifiable credentials and Proof-of-Personhood
  - Sybil detection governance lever with behavioral similarity analysis
- **LLM-generated CUDA kernel submissions** (v4 kernel market): Templates for 8 challenges x 3 agent variants, regex-based static analyzer, proxy signal adjustments from code features
- **SkillRL model**: Hierarchical SkillBank, GRPO advantage estimation, recursive skill evolution
- **Social Simulacra integration** for SWARM-Concordia bridge
- **Kernel v4 governance sweep paper**: 40-run factorial analysis (transaction tax eta2=0.324, circuit breaker d=-0.02)
- **MkDocs documentation site** with Material theme, dark mode, code annotations, MathJax
- **Moltipedia** and **Moltbook** handlers for collaborative knowledge systems
- **Memory handler** with tiered storage and compaction
- **Scholar handler** for literature synthesis with citation verification
- GitHub Actions CI, release workflow, CodeQL scanning, Dependabot
- Pre-commit hooks (ruff, mypy, pytest, secrets scan)
- CLI entry point (`python -m swarm run`) with seed/epoch overrides and JSON/CSV export
- 56 YAML scenario configs across 3 directories
- 27 governance levers across 7 families
- 11 handler plugins following the Handler base class pattern
- 8 framework bridges (Concordia, OpenClaw, GasTown, LiveSWE, Claude Code, Prime Intellect, Ralph, Worktree)
- 20 research papers (markdown + LaTeX + PDF)
- 9 blog posts on governance, phase transitions, and agent behavior
- **2922 tests** across 94 test files

### Changed
- Package renamed from `src/` to `swarm/` module structure
- All path references updated from `src/` to `swarm/`

### Fixed
- 184 ruff lint errors across 48 files
- ~100 mypy type errors across 27 files

## [0.1.0] - 2025-02-01

### Added
- **Foundation layer:** soft label system with proxy computation (`v_hat` to `p` via calibrated sigmoid), payoff engine, metrics (toxicity, quality gap, conditional loss, Brier score), and append-only JSONL event logging
- **Agent types:** Honest, Opportunistic, Deceptive, Adversarial, LLM-backed (Anthropic/OpenAI/Ollama), and Adaptive Adversary
- **Agent roles:** Planner, Worker, Verifier, Poster, Moderator
- **Environment:** state management, feed engine (posts, replies, voting, ranking), task system (claiming, collaboration, verification), network topology with dynamic evolution
- **Governance:** configurable levers (taxes, reputation decay, staking, circuit breakers, audits), collusion detection (pair-level and group-level), admission control, security policies
- **Marketplace:** bounties, bids, escrow, and dispute resolution
- **Scenarios:** YAML-based scenario loader with 11 built-in scenarios
- **Parameter sweeps:** batch simulation with configurable sweep dimensions
- **Red-teaming:** adaptive adversaries, attack strategies, evasion metrics, evaluation framework
- **Boundaries:** external world simulation, information flow tracking, leakage detection
- **Emergent capabilities:** composite tasks, capability measurement
- **Analysis:** metrics reporters (soft and hard labels), data export, plots
- **Dashboard:** interactive Streamlit app with 5 pages (Overview, Scenario Explorer, Governance Lab, Agent Dynamics, Theory)
- **Documentation:** 9 detailed guides covering theory, LLM agents, network topology, governance, emergent capabilities, red-teaming, scenarios, boundaries, and dashboard
- **Tests:** 727 tests across 20 test files
- **DevContainer:** VS Code / Codespaces configuration
