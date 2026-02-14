# /council_review

Run a multi-LLM council evaluation on study results. Three expert personas (mechanism designer, statistician, red-teamer) deliberate on the findings using the council protocol.

## Usage

`/council_review <run_dir> [--type sweep|scenario|cross_study] [--provider anthropic|openai|ollama]`

Examples:
- `/council_review runs/20260213-143607_delegation_games_sweep`
- `/council_review scenarios/baseline.yaml --type scenario`
- `/council_review runs/study_a runs/study_b --type cross_study`
- `/council_review runs/latest_sweep --provider ollama`

## Prerequisites

The evaluator automatically creates LLM agents from each council member's config. You need **at least one** LLM provider available:

| Provider | Setup | Notes |
|---|---|---|
| **Anthropic** (default) | `export ANTHROPIC_API_KEY=sk-ant-...` | Uses `claude-sonnet-4-20250514` |
| **OpenAI** | `export OPENAI_API_KEY=sk-...` | Uses `gpt-4o` |
| **Ollama** (local, free) | `brew install ollama && ollama serve` | Requires a pulled model |
| **Groq** | `export GROQ_API_KEY=gsk_...` | Uses `llama-3.3-70b-versatile` |
| **Together** | `export TOGETHER_API_KEY=...` | Uses `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` |
| **DeepSeek** | `export DEEPSEEK_API_KEY=sk-...` | Uses `deepseek-chat` |
| **Google** | `export GOOGLE_API_KEY=...` | Uses `gemini-2.0-flash` (requires `google-genai`) |

To use a non-default provider, pass `--provider`:

```
/council_review runs/my_sweep --provider ollama
```

Or mix providers per member:

```python
from swarm.agents.llm_config import LLMConfig, LLMProvider
from swarm.council.study_evaluator import StudyEvaluator, default_evaluator_config

config = default_evaluator_config(provider_configs={
    "mechanism_designer": LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-20250514"),
    "statistician": LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o"),
    "red_teamer": LLMConfig(provider=LLMProvider.OLLAMA, model="llama3"),
})
evaluator = StudyEvaluator(config=config)
evaluation = evaluator.evaluate_sweep("runs/my_sweep")
```

## Arguments

- `run_dir`: Path to the run directory (or scenario YAML if `--type scenario`, or multiple run dirs if `--type cross_study`).
- `--type`: Evaluation type. Default: `sweep`.
  - `sweep`: Evaluate sweep results (reads `summary.json` + `sweep_results.csv`)
  - `scenario`: Pre-run design review of a scenario YAML
  - `cross_study`: Compare multiple study results
- `--provider`: LLM provider for council members. Default: `anthropic`.

## Behavior

1. **Auto-load API keys**: Before doing anything else, check if the needed API key is set. If not, attempt to load it from `~/.zshrc` (or `~/.bashrc`):
   ```bash
   # For OpenRouter (most common):
   export OPENROUTER_API_KEY="$(grep '^export OPENROUTER_API_KEY=' ~/.zshrc | tail -1 | sed 's/^export //' | cut -d= -f2- | tr -d '"' | tr -d "'")"
   ```
   Do this for whichever provider(s) the council members need. Print which keys were loaded (character count only, never the actual value). If no key can be found for any provider, report the error and stop.

2. **Build evaluator**: Create a `StudyEvaluator` with the 3-member council (mechanism designer as chairman, statistician, red-teamer). Each member gets an `LLMAgent` instance wired as an async query function.

   **Default model configuration** (diverse multi-model council via OpenRouter):
   - Mechanism designer: `anthropic/claude-opus-4` (weight=1.5, chairman/synthesizer)
   - Statistician: `openai/gpt-4o` (weight=1.0)
   - Red teamer: `google/gemini-2.5-pro-preview` (weight=1.0)

   If `--provider` is specified, use that provider's default model for all 3 members instead.

3. **Run evaluation**: Call the appropriate method based on `--type`:
   - `sweep` → `evaluator.evaluate_sweep(run_dir)`
   - `scenario` → `evaluator.evaluate_scenario(yaml_path)`
   - `cross_study` → `evaluator.evaluate_cross_study(run_dirs)`

4. **Save output**: Write `council_review.json` to the run directory (or next to the YAML for scenario reviews).

5. **Print summary**: Display the structured evaluation:
   ```
   Council Review Complete
     Type:    sweep
     Members: 3/3 responded

     FINDINGS:
       - <finding 1>
       - <finding 2>

     CONCERNS:
       - <concern 1>

     RECOMMENDATIONS:
       - <recommendation 1>

     Full trace: <run_dir>/council_review.json
   ```

   If the section parser returns empty findings/concerns/recommendations (LLM used different heading format), print the raw synthesis instead.

## Implementation

```python
import os, subprocess
from pathlib import Path
from swarm.agents.llm_config import LLMConfig, LLMProvider
from swarm.council.study_evaluator import StudyEvaluator, default_evaluator_config, save_evaluation

# Step 1: Auto-load API keys from ~/.zshrc if not already set
KEY_VARS = [
    "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
    "GROQ_API_KEY", "TOGETHER_API_KEY", "DEEPSEEK_API_KEY", "GOOGLE_API_KEY",
]
shell_rc = Path.home() / ".zshrc"
if not shell_rc.exists():
    shell_rc = Path.home() / ".bashrc"

for var in KEY_VARS:
    if not os.environ.get(var) and shell_rc.exists():
        result = subprocess.run(
            ["grep", f"^export {var}=", str(shell_rc)],
            capture_output=True, text=True,
        )
        if result.stdout.strip():
            line = result.stdout.strip().splitlines()[-1]
            val = line.split("=", 1)[1].strip().strip("'\"")
            if val:
                os.environ[var] = val

# Step 2: Build evaluator with diverse models (default: OpenRouter multi-model)
# Parse $ARGUMENTS for --provider override
config = default_evaluator_config(provider_configs={
    "mechanism_designer": LLMConfig(
        provider=LLMProvider.OPENROUTER,
        model="anthropic/claude-opus-4",
        temperature=0.3, max_tokens=1500,
    ),
    "statistician": LLMConfig(
        provider=LLMProvider.OPENROUTER,
        model="openai/gpt-4o",
        temperature=0.3, max_tokens=1500,
    ),
    "red_teamer": LLMConfig(
        provider=LLMProvider.OPENROUTER,
        model="google/gemini-2.5-pro-preview",
        temperature=0.3, max_tokens=1500,
    ),
})

evaluator = StudyEvaluator(config=config)

# Step 3: Run the appropriate evaluation
if eval_type == "sweep":
    evaluation = evaluator.evaluate_sweep(run_dir)
    out_path = Path(run_dir) / "council_review.json"
elif eval_type == "scenario":
    evaluation = evaluator.evaluate_scenario(yaml_path)
    out_path = Path(yaml_path).parent / "council_review.json"
elif eval_type == "cross_study":
    evaluation = evaluator.evaluate_cross_study(run_dirs)
    out_path = Path(run_dirs[0]) / "council_review.json"

save_evaluation(evaluation, out_path)
```

## Constraints

- Never blocks on API failures — if the council fails, print the error and continue.
- Requires at least 2 of 3 council members to respond (quorum).
- Does not send raw CSV data to the LLM — only summary statistics and top results.
