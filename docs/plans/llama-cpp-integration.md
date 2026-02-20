# Plan: llama.cpp Integration for Local CPU Inference

## Goal

Enable SWARM to use local LLaMA models via [llama.cpp](https://github.com/ggml-org/llama.cpp) for CPU-only inference (e.g. on a MacBook), without requiring any cloud API keys.

## Architecture Decision

**Primary path (Option A):** `llama-server` as an OpenAI-compatible HTTP endpoint.

SWARM already has `_call_openai_compatible_async` in `LLMAgent` — llama-server exposes `/v1/chat/completions` with the same schema. This is a near-zero-friction integration.

**Secondary path (Option B):** In-process `llama-cpp-python` bindings. Useful for maximum determinism (pinned threads, KV cache, seeds) and zero network overhead. Implemented as a separate code path behind the same `LLMProvider.LLAMA_CPP` enum value, selected via config.

```
┌─────────────────────────────────────────────────────┐
│ SWARM Scenario YAML                                  │
│   provider: llama_cpp                                │
│   model: llama-3.2-3b-instruct                       │
│   base_url: http://localhost:8080/v1  (Option A)     │
│   OR                                                 │
│   model_path: ./models/model.gguf     (Option B)     │
└────────────────────────┬────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
    Option A (HTTP)            Option B (in-process)
          │                             │
  ┌───────▼────────┐          ┌────────▼─────────┐
  │ _call_openai_  │          │ _call_llama_cpp_ │
  │ compatible_    │          │ direct_async()   │
  │ async()        │          │                  │
  │ (existing)     │          │ llama-cpp-python │
  └───────┬────────┘          └────────┬─────────┘
          │                             │
  ┌───────▼────────┐          ┌────────▼─────────┐
  │ llama-server   │          │ llama_cpp.Llama  │
  │ :8080/v1/...   │          │ (in-process)     │
  └────────────────┘          └──────────────────┘
```

## Implementation Steps

### Step 1: Add `LLAMA_CPP` provider to `LLMProvider` enum

**File:** `swarm/agents/llm_config.py`

- Add `LLAMA_CPP = "llama_cpp"` to the `LLMProvider` enum.
- Add new optional fields to `LLMConfig`:
  - `model_path: Optional[str] = None` — path to a local `.gguf` file (Option B only).
  - `n_ctx: int = 4096` — context window size for in-process mode.
  - `n_threads: Optional[int] = None` — CPU thread count (defaults to auto-detect).
  - `llama_seed: int = -1` — sampling seed for determinism (`-1` = random).
- In `__post_init__`, set default `base_url = "http://localhost:8080/v1"` when provider is `LLAMA_CPP` and no `base_url` is provided.
- Skip API key validation for `LLAMA_CPP` (like Ollama).
- Add `LLAMA_CPP` cost entries to `LLMUsageStats` (all `0.0` since local).

### Step 2: Wire routing in `LLMAgent._call_llm_async`

**File:** `swarm/agents/llm_agent.py`

- Add `LLMProvider.LLAMA_CPP` case in the provider dispatch:
  - If `model_path` is set → call new `_call_llama_cpp_direct_async()` (Option B).
  - Otherwise → call existing `_call_openai_compatible_async()` (Option A).
- Add `_call_llama_cpp_direct_async()` method:
  - Lazy-load `llama_cpp.Llama` model instance (one per agent, cached on `self`).
  - Use `create_chat_completion()` for inference.
  - Extract text + token counts from response.
  - Run in thread pool (`loop.run_in_executor`) since llama-cpp-python is blocking.
- In `_get_api_key_from_env()`, return a dummy key for `LLAMA_CPP` (llama-server ignores it, but the OpenAI client requires a non-empty string).

### Step 3: Add health check utility

**File:** `swarm/agents/llm_health.py` (new, small)

A tiny helper that SWARM can call before a run to verify the llama-server is reachable:

```python
def check_llama_server(base_url: str = "http://localhost:8080") -> bool:
    """Ping llama-server /health endpoint. Returns True if ready."""
```

Integrated into the orchestrator startup when provider is `LLAMA_CPP` + Option A, so users get a clear error instead of a timeout mid-simulation.

### Step 4: Add optional dependency

**File:** `pyproject.toml`

```toml
[project.optional-dependencies]
llama_cpp = [
    "llama-cpp-python>=0.3.0",
]
```

Also add to the `llm` extras group so `pip install -e ".[llm]"` pulls it in. The `openai` package (already in `llm` extras) is needed for Option A.

### Step 5: Create example scenario YAML

**File:** `scenarios/llm_llama_cpp.yaml`

Two variant blocks showing both options:

```yaml
# Option A: llama-server (recommended)
agents:
  - type: llm
    count: 3
    llm:
      provider: llama_cpp
      model: llama-3.2-3b-instruct   # name passed to llama-server
      base_url: http://localhost:8080/v1
      temperature: 0.2
      max_tokens: 512
      seed: 42

# Option B: in-process (uncomment to use)
#  - type: llm
#    count: 3
#    llm:
#      provider: llama_cpp
#      model_path: ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
#      n_ctx: 4096
#      n_threads: 8
#      temperature: 0.2
#      max_tokens: 512
#      seed: 42
```

### Step 6: Helper script for model download + server launch

**File:** `scripts/llama-server-setup.sh`

Bash script that:

1. Checks if `llama-server` binary exists; if not, prints build instructions (or downloads a release binary).
2. Downloads a recommended small GGUF model if not present (e.g., `Llama-3.2-3B-Instruct-Q4_K_M.gguf` from HuggingFace — ~2 GB, runs well on MacBook CPU).
3. Launches `llama-server -m <model> --port 8080 --ctx-size 4096 --threads <auto>`.
4. Waits for `/health` to return OK.

Documented usage:

```bash
# One-time setup
./scripts/llama-server-setup.sh download   # fetch model
./scripts/llama-server-setup.sh start      # start server

# Then run SWARM
python -m swarm run scenarios/llm_llama_cpp.yaml --seed 42
```

### Step 7: Tests

**File:** `tests/test_llama_cpp_provider.py`

- **Unit tests** (no server required):
  - `LLMConfig` with `provider=llama_cpp` sets correct defaults.
  - Routing dispatches to `_call_openai_compatible_async` (Option A) or `_call_llama_cpp_direct_async` (Option B) based on `model_path`.
  - Health check returns False when server is down (mocked).
- **Integration test** (marked `@pytest.mark.slow`, skipped without server):
  - Start llama-server in fixture, send one chat completion, verify response structure.

### Step 8: Documentation

**File:** `docs/guides/llama-cpp-local-inference.md`

Covers:
- Prerequisites (llama.cpp build or binary, GGUF model).
- Option A walkthrough (server mode).
- Option B walkthrough (in-process mode).
- Recommended models for CPU (3B Q4_K_M for light, 8B Q4_K_M for quality).
- Determinism controls (seed, temperature 0, fixed threads).
- Throughput tips (batching, `--parallel` flag on llama-server for multi-agent).
- Observability: prompt audit logging (already built in via `prompt_audit_path`).

## Files Changed (Summary)

| File | Change |
|---|---|
| `swarm/agents/llm_config.py` | Add `LLAMA_CPP` enum + config fields |
| `swarm/agents/llm_agent.py` | Add routing + `_call_llama_cpp_direct_async` |
| `swarm/agents/llm_health.py` | New — health check utility |
| `pyproject.toml` | Add `llama_cpp` optional dep |
| `scenarios/llm_llama_cpp.yaml` | New — example scenario |
| `scripts/llama-server-setup.sh` | New — model download + server launcher |
| `tests/test_llama_cpp_provider.py` | New — unit + integration tests |
| `docs/guides/llama-cpp-local-inference.md` | New — user guide |

## Design Notes

- **Why not just use the existing Ollama provider?** Ollama wraps llama.cpp but adds its own HTTP layer, model management, and overhead. Direct llama-server gives lower latency, better determinism control, and avoids Ollama as a dependency. Users who prefer Ollama can already use `provider: ollama`.
- **Why both options?** Option A (HTTP) is simpler, supports multi-process, and matches the existing OpenAI-compatible pattern. Option B (in-process) is needed for tight determinism (pinned seeds, threads, KV cache) in reproducible experiments — a core SWARM requirement.
- **No new heavy dependencies for Option A.** The `openai` Python package (already in `[llm]` extras) is the only runtime dependency. The user just needs `llama-server` binary running externally.
- **Model files are gitignored.** The `models/` directory (GGUF files) should be added to `.gitignore` — these are multi-GB binaries.
