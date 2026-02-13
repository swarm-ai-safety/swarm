# /council_review

Run a multi-LLM council evaluation on study results. Three expert personas (mechanism designer, statistician, red-teamer) deliberate on the findings using the council protocol.

## Usage

`/council_review <run_dir> [--type sweep|scenario|cross_study] [--provider anthropic|openai|ollama]`

Examples:
- `/council_review runs/20260213-143607_delegation_games_sweep`
- `/council_review scenarios/baseline.yaml --type scenario`
- `/council_review runs/study_a runs/study_b --type cross_study`
- `/council_review runs/latest_sweep --provider ollama`

## Arguments

- `run_dir`: Path to the run directory (or scenario YAML if `--type scenario`, or multiple run dirs if `--type cross_study`).
- `--type`: Evaluation type. Default: `sweep`.
  - `sweep`: Evaluate sweep results (reads `summary.json` + `sweep_results.csv`)
  - `scenario`: Pre-run design review of a scenario YAML
  - `cross_study`: Compare multiple study results
- `--provider`: LLM provider for council members. Default: `anthropic`.

## Behavior

1. **Build evaluator**: Create a `StudyEvaluator` with the 3-member council (mechanism designer as chairman, statistician, red-teamer).

2. **Run evaluation**: Call the appropriate method based on `--type`:
   - `sweep` → `evaluator.evaluate_sweep(run_dir)`
   - `scenario` → `evaluator.evaluate_scenario(yaml_path)`
   - `cross_study` → `evaluator.evaluate_cross_study(run_dirs)`

3. **Save output**: Write `council_review.json` to the run directory (or next to the YAML for scenario reviews).

4. **Print summary**: Display the structured evaluation:
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

## Implementation

```python
from pathlib import Path
from swarm.council.study_evaluator import StudyEvaluator, save_evaluation

# Parse arguments from $ARGUMENTS
# Default type is "sweep", default provider is "anthropic"

evaluator = StudyEvaluator()

# Run the appropriate evaluation
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
