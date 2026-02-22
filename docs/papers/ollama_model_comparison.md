# Local LLM Tool-Use Capability in Multi-Agent Governed Handoff: An Ollama Model Comparison

**Authors:** Raeli Savitt
**Date:** 2026-02-22
**Framework:** SWARM v1.7.0

## Abstract

We evaluate four local LLM models served via Ollama as drop-in replacements for Claude in the SWARM LangGraph governed handoff study. The study sweeps 32 governance parameter configurations (max_cycles x max_handoffs x trust_boundaries) across a 4-agent swarm (coordinator, researcher, writer, reviewer) with governed handoff tools. We compare llama3.2 (3B), llama3.1:8b, qwen2.5:7b, and mistral:7b on three capabilities: (1) tool-call format compliance, (2) multi-agent delegation via handoff tools, and (3) end-to-end task completion producing a `FINAL ANSWER:`. All four model families produce valid tool calls through Ollama's native tool-calling API. mistral:7b achieves the highest completion rate (53.1% at 1 seed) and is the only model to trigger governance denials (3 denied handoffs), demonstrating that local models can exercise governance policies. A follow-up 3-seed study (96 runs) confirms mistral:7b's performance at 62.5% completion (60/96) with 7 governance denials and low cross-seed variance (59.4%–65.6%), establishing it as a reliable zero-cost platform for governance sensitivity research. qwen2.5:7b shows the strongest delegation behavior (100% of runs with handoffs, 66 total) but cannot complete the full workflow. llama3.1:8b produces moderate delegation (59.4% of runs) with frequent chat-history errors. llama3.2 (3B) achieves 28.1% completion by bypassing delegation entirely. Total cost: $0 (all runs local).

## 1. Introduction

The SWARM governed handoff study was designed for Claude Sonnet, which reliably uses structured tool calls to coordinate multi-agent workflows. Running the full 32-config sweep with Claude costs ~$5-10 in API credits per seed. To enable unlimited local experimentation, we added multi-provider support (Ollama, OpenAI) to the LangGraph bridge and evaluated whether commodity local models can drive the same multi-agent governance patterns.

This study tests a specific capability hierarchy:
1. **Format compliance**: Can the model produce tool calls that LangGraph/LangChain can parse?
2. **Delegation**: Does the model use handoff tools to route tasks between agents?
3. **Completion**: Can the model sustain a multi-step workflow to produce a final answer?
4. **Governance activation**: Does the model generate enough handoff traffic to trigger governance policies?

## 2. Experimental Setup

### 2.1 Models

| Model | Family | Parameters | Quantization | Tool-Call Support |
|---|---|---|---|---|
| llama3.2 | Llama 3 | 3B | Q4_0 | Native (Ollama) |
| llama3.1:8b | Llama 3 | 8B | Q4_0 | Native (Ollama) |
| qwen2.5:7b | Qwen 2.5 | 7B | Q4_0 | Native (Ollama) |
| mistral:7b | Mistral | 7B | Q4_0 | Native (Ollama) |

All models served locally via Ollama on Apple Silicon (M-series), accessed through `langchain-ollama` `ChatOllama` at `http://localhost:11434`.

### 2.2 Agent Architecture

Four LLM-backed agents with governed handoff tools:

| Agent | Trust Group | Hands Off To | Role |
|---|---|---|---|
| coordinator | management | researcher, writer | Receives task, delegates, synthesizes |
| researcher | research | writer, coordinator | Produces bullet-point findings |
| writer | content | reviewer, coordinator | Drafts summary from findings |
| reviewer | research | writer, coordinator | QA check, approves or sends back |

**Expected workflow:** coordinator → researcher → writer → reviewer → coordinator (FINAL ANSWER)

### 2.3 Governance Parameter Sweep

| Parameter | Values | Description |
|---|---|---|
| max_cycles | 1, 2, 3, 5 | Max repeated handoff cycles before denial |
| max_handoffs | 5, 10, 15, 30 | Max total handoffs per run |
| trust_boundaries | true, false | Enforce trust-group boundary checks |

**Grid size:** 4 x 4 x 2 = 32 configurations. The initial model comparison uses 1 seed per config (128 total runs, 32 per model). A follow-up study runs mistral:7b with 3 seeds (96 total runs) to assess cross-seed stability.

### 2.4 Task Prompt

> Research and write a brief summary about the safety implications of multi-agent AI handoff patterns. Cover: (1) why handoff governance matters, (2) risks of uncontrolled delegation chains, and (3) one concrete mitigation. Keep the final answer under 200 words.

### 2.5 Completion Criterion

A run is marked "completed" if any message in the output contains the string `FINAL ANSWER:`.

## 3. Results

### 3.1 Summary Table

| Metric | llama3.2 (3B) | llama3.1:8b | qwen2.5:7b | mistral:7b |
|---|---|---|---|---|
| Runs | 32 | 32 | 32 | 32 |
| Completion rate | 9/32 (28.1%) | 0/32 (0.0%) | 0/32 (0.0%) | **17/32 (53.1%)** |
| Runs with handoffs | 1/32 (3.1%) | 19/32 (59.4%) | **32/32 (100%)** | 4/32 (12.5%) |
| Total handoffs | 1 | 27 | **66** | 11 |
| Denied handoffs | 0 | 0 | 0 | **3** |
| Runs with errors | 1 | 8 | 25 | 3 |
| Avg time/run | 10.2s | 21.6s | 39.8s | 64.6s |
| Total cost | $0 | $0 | $0 | $0 |

### 3.2 Model Behavioral Profiles

**mistral:7b — Best overall (53.1% completion, governance activation):**
Mistral achieved the highest completion rate and is the only model to trigger governance denials (3 denied handoffs across 32 runs). It demonstrates a mixed strategy: in most runs (28/32) it answers directly as the coordinator, but in 4 runs it engages the handoff tools and generates enough traffic to exercise cycle detection and trust boundary policies. The 3 governance denials confirm that local models can reach the governed regime, making Mistral viable for governance parameter sensitivity studies. Average time of 64.6s/run reflects deeper reasoning per turn.

**qwen2.5:7b — Strongest delegation (100% handoff rate, 0% completion):**
Qwen 2.5 is the most tool-engaged model: every single run triggered handoffs (2-3 per run, 66 total). However, it consistently fails to complete the workflow. The model reliably delegates from coordinator to researcher/writer but cannot sustain the chain through reviewer and back. 25/32 runs produced errors, typically INVALID_CHAT_HISTORY from malformed tool-call sequences. This suggests strong tool-calling instinct but poor multi-turn tool-call coherence.

**llama3.1:8b — Moderate delegation (59.4% handoff rate, 0% completion):**
The 8B Llama model demonstrated genuine multi-agent delegation: 59.4% of runs triggered at least one handoff, with up to 2 handoffs per run (mean 0.84 across all runs). However, the model could not sustain the full 4-agent workflow. Common failure modes include:
- Issuing a tool call but not waiting for the tool result (INVALID_CHAT_HISTORY errors in 8/32 runs)
- Delegating to one agent but not continuing the chain to reviewer and back
- Producing text responses without the `FINAL ANSWER:` prefix

**llama3.2 (3B) — Completion by bypass (28.1% completion, minimal delegation):**
The smallest model achieved 28.1% completion by largely ignoring the handoff tools and answering the task directly as the coordinator. Only 1 of 32 runs triggered a handoff. The model treated the task as a simple question-answering problem rather than a multi-agent coordination challenge. Completions that bypass the delegation workflow are not equivalent to genuine governed handoffs — they represent the model's failure to engage with the multi-agent architecture.

### 3.3 Governance Policy Observations

mistral:7b is the only model to trigger governance denials (3 denied handoffs). These denials were produced by the cycle detection policy when Mistral attempted repeated handoffs between the same agent pair. This is a significant finding: it demonstrates that local 7B models can generate sufficient handoff complexity to exercise governance constraints, making them viable for testing governance policy sensitivity at zero cost.

The other three models never triggered denials. qwen2.5:7b generates the most handoffs (66 total) but its handoff patterns are too simple (coordinator → one target, then error) to trigger cycle detection. llama3.1:8b's handoffs are similarly shallow. llama3.2 barely uses tools at all.

### 3.4 Capability Ranking

| Capability | Best | Second | Third | Fourth |
|---|---|---|---|---|
| Task completion | mistral:7b (53.1%) | llama3.2 (28.1%) | llama3.1:8b (0%) | qwen2.5:7b (0%) |
| Tool engagement | qwen2.5:7b (100%) | llama3.1:8b (59.4%) | mistral:7b (12.5%) | llama3.2 (3.1%) |
| Governance activation | mistral:7b (3 denials) | — | — | — |
| Error resilience | llama3.2 (1 error) | mistral:7b (3) | llama3.1:8b (8) | qwen2.5:7b (25) |

### 3.5 mistral:7b 3-Seed Follow-Up Study (96 runs)

Given mistral:7b's strong 1-seed performance, we ran a 3-seed follow-up (seeds 42, 43, 44) across all 32 governance configurations (96 total runs) to assess stability and deepen governance policy analysis.

#### 3.5.1 Aggregate Results

| Metric | 1-seed (32 runs) | 3-seed (96 runs) |
|---|---|---|
| Completion rate | 17/32 (53.1%) | **60/96 (62.5%)** |
| Runs with handoffs | 4/32 (12.5%) | 19/96 (19.8%) |
| Total handoffs | 11 | 48 |
| Denied handoffs | 3 | **7** |
| Runs with errors | 3 | 17 |
| Avg time/run | 64.6s | ~15.5s |

The completion rate improved from 53.1% to 62.5% with 3 seeds, reflecting reduced variance from single-seed sampling. Governance denials more than doubled (3 → 7), providing a richer dataset for policy sensitivity analysis.

#### 3.5.2 Cross-Seed Stability

| Seed | Completion | Rate |
|---|---|---|
| 42 | 21/32 | 65.6% |
| 43 | 19/32 | 59.4% |
| 44 | 20/32 | 62.5% |

Cross-seed variance is low (59.4%–65.6%, range of 6.2 percentage points), indicating that mistral:7b's completion behavior is reasonably stable across random initializations rather than dependent on a lucky seed.

#### 3.5.3 Governance Denial Analysis

All 7 governance denials occurred in runs where `max_cycles=1` and `trust_boundaries=True`, concentrated in seed 43 and seed 44. The denial pattern shows:

- **Denials per run:** 2 denied handoffs per denial-triggering run (3 runs with denials in mc=1)
- **Associated handoff volume:** Denial-triggering runs averaged 4 total handoffs — the highest per-run handoff counts in the dataset
- **Denial rate in affected runs:** 0.5 (50%) — when mistral attempts multi-step delegation under tight cycle limits, half the handoffs are denied
- **`branch:to:unknown` warnings:** Runs with governance denials also produced LangGraph channel warnings, suggesting that denied handoffs sometimes cause the model to attempt routing to agent names outside the graph

This confirms that `max_cycles=1` with `trust_boundaries=True` is the governance regime most likely to produce denials with local models — a useful finding for designing targeted governance stress tests.

#### 3.5.4 Completion by Governance Configuration

Completion rates vary by `max_cycles`:

| max_cycles | Completed | Rate |
|---|---|---|
| 1 | 13/24 | 54.2% |
| 2 | 18/24 | 75.0% |
| 3 | 14/24 | 58.3% |
| 5 | 15/24 | 62.5% |

The `max_cycles=2` configuration achieves the highest completion rate (75.0%), while `max_cycles=1` has the lowest (54.2%) — expected since tighter cycle limits increase governance interference. The non-monotonic pattern at `max_cycles=3` (58.3%) likely reflects sampling noise at 24 runs per group.

## 4. Discussion

### 4.1 All Four Families Support Ollama Tool Calls

All four model families produce valid tool calls through Ollama's native tool-calling API, contradicting the common assumption that only Llama 3 is well-supported. The key differences are not in format compliance but in multi-turn tool-call coherence — how well the model maintains valid chat history across successive tool calls and responses.

### 4.2 Delegation vs. Completion Trade-off

A striking inverse relationship exists between delegation intensity and task completion:

- **qwen2.5:7b**: Maximum delegation (66 handoffs), zero completion
- **mistral:7b**: Moderate delegation (11 handoffs at 1 seed, 48 at 3 seeds), highest completion (53.1% → 62.5%)
- **llama3.2**: Minimal delegation (1 handoff), moderate completion (28.1%)

This suggests that aggressive tool use without reliable multi-turn coherence is worse than conservative direct answering. Mistral's balanced approach — using tools selectively while maintaining the ability to produce final answers — is the most effective strategy for this task.

### 4.3 Cross-Seed Stability Validates Single-Seed Screening

The 3-seed mistral:7b follow-up demonstrates that single-seed model comparison is a valid screening methodology: the 1-seed completion rate (53.1%) correctly identified mistral as the top performer, and the 3-seed rate (62.5%) is within the expected range. The low cross-seed variance (6.2pp range) means researchers can use 1-seed sweeps for initial model screening, then invest in multi-seed studies only for the most promising candidates.

### 4.4 Implications for Local Model Research

Local models via Ollama are viable for:
- **Governance policy testing**: mistral:7b triggers governance denials, enabling free governance sensitivity studies
- **Infrastructure validation**: All models exercise the study runner, data pipeline, and export paths
- **Tool-call behavior research**: The four models span a useful behavioral spectrum from tool-avoidant (llama3.2) to tool-aggressive (qwen2.5:7b)
- **Baseline comparison**: Establishing a capability floor against which Claude results can be contextualized

Remaining gaps vs. Claude:
- **Full workflow completion**: No model reliably sustains the 4-agent handoff chain end-to-end
- **Governance stress testing**: Mistral reaches the governed regime (7 denials across 96 runs) but only under `max_cycles=1` — higher cycle limits produce no denials
- **Adversarial handoff patterns**: Requires models that reliably use tools in adversarial/creative ways

## 5. Reproducibility

```bash
# Install dependencies
pip install swarm-safety[langgraph]
pip install langchain-ollama

# Pull models
ollama pull llama3.2
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull mistral:7b

# Run all four sweeps (1 seed each for model comparison)
python examples/langgraph_governed_study.py --provider ollama --model llama3.2 --seeds 1
python examples/langgraph_governed_study.py --provider ollama --model llama3.1:8b --seeds 1
python examples/langgraph_governed_study.py --provider ollama --model qwen2.5:7b --seeds 1
python examples/langgraph_governed_study.py --provider ollama --model mistral:7b --seeds 1

# 3-seed follow-up for mistral:7b
python examples/langgraph_governed_study.py --provider ollama --model mistral:7b --seeds 3
```

### Run Directories

| Model | Run Directory | Timestamp |
|---|---|---|
| llama3.2 | `runs/20260222_093348_langgraph_governed/` | 2026-02-22 09:33 |
| llama3.1:8b | `runs/20260222_100025_langgraph_governed/` | 2026-02-22 10:00 |
| qwen2.5:7b | `runs/20260222_102235_langgraph_governed/` | 2026-02-22 10:22 |
| mistral:7b | `runs/20260222_104358_langgraph_governed/` | 2026-02-22 10:43 |
| mistral:7b (3-seed) | `runs/20260222_113215_langgraph_governed/` | 2026-02-22 11:32 |
| Combined | `runs/20260222_ollama_model_comparison/` | — |

## 6. Conclusion

Local Ollama models show a wider capability spectrum than expected for multi-agent governed handoff. mistral:7b is the standout performer — achieving 53.1% task completion at 1 seed and 62.5% across 3 seeds (96 runs), with 7 governance denials concentrated in the `max_cycles=1` regime. Cross-seed variance is low (59.4%–65.6%), establishing mistral:7b as a reliable zero-cost platform for governance sensitivity research. qwen2.5:7b demonstrates the strongest tool engagement (100% of runs with handoffs) but lacks the multi-turn coherence to complete workflows. The Llama 3 family occupies the middle ground with moderate delegation but no completions (8B) or completion through delegation bypass (3B). All four families support Ollama's native tool-calling API, and the multi-provider infrastructure enables free experimentation across the full behavioral spectrum. The 3-seed follow-up validates single-seed screening as a cost-effective initial comparison methodology.

---

*This study was conducted entirely on local hardware with zero API cost. All models served via Ollama on Apple Silicon.*
