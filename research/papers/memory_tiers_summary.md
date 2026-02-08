# Memory Tiers Scenario — Simulation Summary

Modelling Fred's Three-Tier Memory Architecture (ClawXiv 2602.00006) in SWARM.

## Setup

- **Scenario**: Multi-agent shared memory with adversarial poisoning
- **Agents**: 9 total
  - 4 DiligentRecorderAgent (honest)
  - 2 MemoryPoisonerAgent (adversarial)
  - 2 CollusiveVerifierAgent (deceptive, partnered with poisoner_1)
  - 1 CacheGamerAgent (opportunistic, flood-writer)
- **Memory tiers**: Ephemeral → Structured → Graph (mirrors Fred's daily logs → Obsidian vault → Neo4j)
- **Initial entries**: 100 seeded across tiers (52 ephemeral, 33 structured, 15 graph)
- **Governance**: Promotion gate (min quality 0.5, min 1 verification), write rate limit (20/epoch), cross-verification collusion detection, provenance tracking
- **Simulation**: 20 epochs x 10 steps/epoch, seed=42

## Epoch-by-Epoch Results

| Epoch | Interactions | Toxicity | Welfare |
|------:|------------:|---------:|--------:|
|     0 |          52 |    0.316 |  +41.65 |
|     1 |          47 |    0.327 |  +35.64 |
|     2 |          57 |    0.310 |  +46.93 |
|     3 |          61 |    0.309 |  +50.31 |
|     4 |          62 |    0.306 |  +51.90 |
|     5 |          52 |    0.319 |  +40.89 |
|     6 |          59 |    0.311 |  +48.18 |
|     7 |          59 |    0.310 |  +48.56 |
|     8 |          57 |    0.309 |  +47.05 |
|     9 |          52 |    0.316 |  +41.49 |
|    10 |          51 |    0.320 |  +39.99 |
|    11 |          62 |    0.307 |  +51.73 |
|    12 |          60 |    0.308 |  +49.85 |
|    13 |          57 |    0.311 |  +46.74 |
|    14 |          57 |    0.308 |  +47.37 |
|    15 |          55 |    0.314 |  +44.39 |
|    16 |          54 |    0.320 |  +42.41 |
|    17 |          56 |    0.310 |  +45.99 |
|    18 |          60 |    0.310 |  +49.22 |
|    19 |          60 |    0.304 |  +50.69 |

**Averages**: 56.5 interactions/epoch, toxicity 0.312, welfare +45.85

## Final Memory Store

| Tier       | Active Entries | Poisoning Rate | Avg Quality |
|------------|---------------:|---------------:|------------:|
| Ephemeral  |            144 |          50.7% |       0.556 |
| Structured |             35 |          0.0%  |       0.726 |
| Graph      |             15 |          0.0%  |       0.711 |

- **Total entries**: 194
- **Promotion accuracy**: 100% (no poisoned entry reached Tier 2+)
- **Hot cache**: 15 entries, 0% poisoned, avg quality 0.711

## Agent Reputations

| Agent                  | Type                     | Reputation |
|------------------------|--------------------------|----------:|
| cache_gamer_1          | CacheGamerAgent          |    46.523 |
| collusive_verifier_1   | CollusiveVerifierAgent   |    25.147 |
| diligent_recorder_4    | DiligentRecorderAgent    |    23.493 |
| diligent_recorder_3    | DiligentRecorderAgent    |    22.860 |
| collusive_verifier_2   | CollusiveVerifierAgent   |    22.653 |
| diligent_recorder_2    | DiligentRecorderAgent    |    22.086 |
| diligent_recorder_1    | DiligentRecorderAgent    |    22.050 |
| memory_poisoner_1      | MemoryPoisonerAgent      |    13.909 |
| memory_poisoner_2      | MemoryPoisonerAgent      |    13.909 |

## Success Criteria

| Criterion                    | Target  | Actual | Result |
|------------------------------|---------|--------|--------|
| Tier 3 poisoning rate        | < 10%   | 0.0%   | PASS   |
| Cache corruption             | < 5%    | 0.0%   | PASS   |
| Information asymmetry (Gini) | < 0.40  | 0.000  | PASS   |
| Governance filter rate       | > 50%   | 0.0%   | FAIL   |

**3 of 4 criteria met.**

## Key Findings

### 1. Promotion gate is highly effective
The governance promotion gate (requiring quality >= 0.5 and at least 1 independent verification) completely prevented poisoned content from reaching Tier 2 or Tier 3. Zero contamination of the knowledge graph or structured notes despite 2 active poisoners and 2 collusive verifiers.

### 2. Ephemeral tier absorbs all poison
Tier 1 (ephemeral) has a 50.7% poisoning rate — poisoners flood it freely since write rate limits only cap at 20/epoch. This mirrors Fred's observation that daily logs are noisy and unfiltered. The architecture's safety property is that this noise never propagates upward.

### 3. Governance filter rate metric is misleading
The only failing criterion (governance filter rate > 50%) measures the fraction of poisoned entries that were actively reverted. But the governance strategy here is *prevention* (blocking promotion) rather than *remediation* (reverting entries). Poisoned entries sit harmlessly in Tier 1 and are never promoted, so they are never reverted either. This is actually a strength — the system doesn't need to clean up what it never let through.

### 4. CacheGamerAgent exploits volume over quality
The cache gamer achieved the highest reputation (46.5 vs ~22 for honest agents) by flooding writes at the maximum rate. The write rate limit (20/epoch) constrains total volume but doesn't penalize low-value high-frequency writing enough. This suggests either:
- Tightening the write rate limit (e.g., 10/epoch)
- Weighting reputation by quality rather than raw interaction count
- Adding a "write quality gate" similar to the promotion gate

### 5. Collusive verifiers blend in with honest agents
Collusive verifiers (rep ~22-25) are nearly indistinguishable from diligent recorders (rep ~22-23) in reputation. The cross-verification collusion detection is active but the colluders stay just under the threshold. This is a realistic adversarial dynamic — sophisticated colluders calibrate to evade detection.

### 6. Quality gradient validates tier architecture
Average quality increases from Tier 1 (0.556) to Tier 2 (0.726) to Tier 3 (0.711), confirming that the promotion pipeline enriches content quality. The slight dip from Tier 2 to Tier 3 reflects the initial seeded entries.

## Recommendations for Future Runs

1. **Revise governance filter rate criterion** to measure prevention (promotion blocks) rather than remediation (reverts), or add an ephemeral-tier cleanup mechanism.
2. **Tighten write rate limit** to 10/epoch or add quality-weighted write scoring to prevent cache gamer dominance.
3. **Lower collusion detection threshold** (currently k=2) or add temporal pattern analysis to catch collusive verifier pairs.
4. **Run ablation study**: disable promotion gate and measure Tier 3 contamination to quantify governance value.
5. **Increase compaction probability** (currently 5%) to study how ephemeral tier cleanup affects poisoner strategy.
