#!/usr/bin/env python3
"""
SWARM SkillsBench Trial Runner

Runs the benchmark with and without skills, using Claude API for agent
attempts and Docker for isolated task execution.

Usage:
    # Full benchmark (3 runs per condition)
    python bench/run_trials.py --runs 3

    # Single task, debug mode
    python bench/run_trials.py --task swarm-run-scenario --runs 1

    # Oracle validation only (no API calls)
    python bench/run_trials.py --oracle

    # Specific condition only
    python bench/run_trials.py --condition with-skills --runs 5

Environment:
    ANTHROPIC_API_KEY  — Required for agent trials (not needed for --oracle)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import tomllib

BENCH_DIR = Path(__file__).parent.resolve()
REPO_ROOT = BENCH_DIR.parent
TASKS_DIR = BENCH_DIR / "tasks"
SKILLS_DIR = BENCH_DIR / "skills"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskMeta:
    name: str
    difficulty: str
    timeout: int
    skills: list[str]
    instruction: str
    docker_context: Path
    test_dir: Path
    solution_dir: Path


@dataclass
class TrialResult:
    task: str
    condition: str  # "no-skills" or "with-skills"
    run_id: int
    passed: bool
    tests_total: int
    tests_passed: int
    tests_failed: int
    duration_s: float
    error: Optional[str] = None
    agent_response_tokens: int = 0


@dataclass
class BenchmarkResults:
    timestamp: str
    runs_per_condition: int
    conditions: list[str]
    results: list[TrialResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_task(task_dir: Path) -> TaskMeta:
    """Load a task from its directory."""
    with open(task_dir / "task.toml", "rb") as f:
        meta = tomllib.load(f)["task"]

    instruction = (task_dir / "instruction.md").read_text()

    return TaskMeta(
        name=meta["name"],
        difficulty=meta["difficulty"],
        timeout=int(meta["timeout"]),
        skills=meta.get("skills", []),
        instruction=instruction,
        docker_context=task_dir / "environment",
        test_dir=task_dir / "tests",
        solution_dir=task_dir / "solution",
    )


def load_all_tasks() -> list[TaskMeta]:
    """Load all tasks from the tasks directory."""
    tasks = []
    for d in sorted(TASKS_DIR.iterdir()):
        if d.is_dir() and (d / "task.toml").exists():
            tasks.append(load_task(d))
    return tasks


def load_skill(skill_name: str) -> str:
    """Load a SKILL.md file by name."""
    skill_path = SKILLS_DIR / skill_name / "SKILL.md"
    if not skill_path.exists():
        raise FileNotFoundError(f"Skill not found: {skill_path}")
    return skill_path.read_text()


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def build_task_image(task: TaskMeta) -> str:
    """Build the Docker image for a task. Returns image tag."""
    tag = f"skillsbench-{task.name}:latest"

    # Build context needs access to repo-level dirs (swarm-package, scenarios, fixtures)
    # We create a temporary build context with everything needed
    build_ctx = Path(f"/tmp/skillsbench-build-{task.name}")
    if build_ctx.exists():
        shutil.rmtree(build_ctx)
    build_ctx.mkdir(parents=True)

    # Copy the Dockerfile, patching swarm install to include [runtime] extras
    dockerfile_text = (task.docker_context / "Dockerfile").read_text()
    dockerfile_text = dockerfile_text.replace(
        "pip install --no-cache-dir /root/swarm-package/",
        "pip install --no-cache-dir '/root/swarm-package/[runtime]'",
    )
    (build_ctx / "Dockerfile").write_text(dockerfile_text)

    if "swarm-package/" in dockerfile_text:
        # Copy the swarm package source
        swarm_src = REPO_ROOT
        pkg_dst = build_ctx / "swarm-package"
        pkg_dst.mkdir()
        # Copy pyproject.toml + swarm/ directory (minimal)
        shutil.copy2(swarm_src / "pyproject.toml", pkg_dst / "pyproject.toml")
        if (swarm_src / "setup.cfg").exists():
            shutil.copy2(swarm_src / "setup.cfg", pkg_dst / "setup.cfg")
        if (swarm_src / "setup.py").exists():
            shutil.copy2(swarm_src / "setup.py", pkg_dst / "setup.py")
        shutil.copytree(swarm_src / "swarm", pkg_dst / "swarm")

    if "scenarios/" in dockerfile_text:
        shutil.copytree(REPO_ROOT / "scenarios", build_ctx / "scenarios")

    if "data/" in dockerfile_text:
        # Copy task-specific data from fixtures
        data_src = task.docker_context / "data" if (task.docker_context / "data").exists() else None
        if data_src is None:
            # Some tasks symlink or expect fixtures
            fixtures_dir = BENCH_DIR / "fixtures"
            if fixtures_dir.exists():
                shutil.copytree(fixtures_dir, build_ctx / "data")
        else:
            shutil.copytree(data_src, build_ctx / "data")

    print(f"  Building image {tag}...")
    result = subprocess.run(
        ["docker", "build", "-t", tag, "."],
        cwd=build_ctx,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"  BUILD FAILED:\n{result.stderr[-500:]}")
        raise RuntimeError(f"Docker build failed for {task.name}")

    # Cleanup
    shutil.rmtree(build_ctx, ignore_errors=True)
    return tag


# ---------------------------------------------------------------------------
# Agent (Claude API)
# ---------------------------------------------------------------------------

def compose_prompt(task: TaskMeta, with_skills: bool) -> str:
    """Compose the prompt sent to the agent."""
    parts = []
    parts.append("You are an AI research assistant working inside a Docker container.\n")
    parts.append("Complete the following task by writing a bash script that produces the required outputs.\n")
    parts.append("Write your solution as a single bash script. Output ONLY the script contents, no explanation.\n")
    parts.append("The script will be executed as `bash solve.sh` in /root/.\n\n")

    parts.append("# Task Instructions\n\n")
    parts.append(task.instruction)
    parts.append("\n\n")

    if with_skills and task.skills:
        parts.append("# Available Skills\n\n")
        parts.append("The following skill documentation is available to help you:\n\n")
        for skill_name in task.skills:
            try:
                skill_content = load_skill(skill_name)
                parts.append(f"## Skill: {skill_name}\n\n")
                parts.append(skill_content)
                parts.append("\n\n")
            except FileNotFoundError:
                pass

    return "".join(parts)


def call_claude_api(prompt: str, timeout: int = 120) -> tuple[str, int]:
    """Call Claude API and return (response_text, output_tokens).

    Uses the anthropic Python SDK.
    """
    try:
        import anthropic
    except ImportError:
        print("ERROR: `anthropic` package not installed. Run: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text
    tokens = response.usage.output_tokens
    return text, tokens


def extract_script(response: str) -> str:
    """Extract bash script from Claude's response."""
    # Try to find code block
    if "```bash" in response:
        start = response.index("```bash") + len("```bash")
        end = response.index("```", start)
        return response[start:end].strip()
    elif "```sh" in response:
        start = response.index("```sh") + len("```sh")
        end = response.index("```", start)
        return response[start:end].strip()
    elif "```" in response:
        start = response.index("```") + 3
        # Skip language tag on first line
        newline = response.index("\n", start)
        end = response.index("```", newline)
        return response[newline:end].strip()
    else:
        # Assume the whole response is the script
        return response.strip()


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------

def run_trial_in_docker(
    task: TaskMeta,
    image_tag: str,
    script: str,
    trial_dir: Path,
) -> TrialResult:
    """Run a trial: execute script in Docker, then run tests."""
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Write solve script
    solve_path = trial_dir / "solve.sh"
    solve_path.write_text(script)

    # Write test files
    test_out_dir = trial_dir / "tests"
    test_out_dir.mkdir(exist_ok=True)
    for f in task.test_dir.iterdir():
        shutil.copy2(f, test_out_dir / f.name)

    container_name = f"skillsbench-{task.name}-{int(time.time())}"
    start = time.time()

    try:
        # 1. Start container
        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", container_name,
                image_tag,
                "sleep", str(task.timeout + 60),
            ],
            capture_output=True, text=True, check=True,
        )

        # 2. Copy solve script into container
        subprocess.run(
            ["docker", "cp", str(solve_path), f"{container_name}:/root/solve.sh"],
            capture_output=True, text=True, check=True,
        )

        # 3. Copy test files into container
        subprocess.run(
            ["docker", "cp", str(test_out_dir) + "/.", f"{container_name}:/root/tests/"],
            capture_output=True, text=True, check=True,
        )

        # 4. Install pytest in container (needed for tests)
        subprocess.run(
            ["docker", "exec", container_name, "pip", "install", "--no-cache-dir", "pytest"],
            capture_output=True, text=True, timeout=60,
        )

        # 5. Execute solve script
        solve_result = subprocess.run(
            ["docker", "exec", container_name, "bash", "/root/solve.sh"],
            capture_output=True, text=True,
            timeout=task.timeout,
        )
        (trial_dir / "solve_stdout.txt").write_text(solve_result.stdout)
        (trial_dir / "solve_stderr.txt").write_text(solve_result.stderr)

        if solve_result.returncode != 0:
            duration = time.time() - start
            return TrialResult(
                task=task.name, condition="", run_id=0,
                passed=False, tests_total=0, tests_passed=0, tests_failed=0,
                duration_s=duration,
                error=f"solve.sh failed (rc={solve_result.returncode}): {solve_result.stderr[-300:]}",
            )

        # 6. Run tests
        test_result = subprocess.run(
            ["docker", "exec", container_name,
             "python3", "-m", "pytest", "/root/tests/test_outputs.py", "-v", "--tb=short"],
            capture_output=True, text=True,
            timeout=120,
        )
        (trial_dir / "test_stdout.txt").write_text(test_result.stdout)
        (trial_dir / "test_stderr.txt").write_text(test_result.stderr)

        duration = time.time() - start

        # Parse pytest output
        passed, failed, total = parse_pytest_output(test_result.stdout)

        return TrialResult(
            task=task.name, condition="", run_id=0,
            passed=(test_result.returncode == 0),
            tests_total=total,
            tests_passed=passed,
            tests_failed=failed,
            duration_s=duration,
        )

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return TrialResult(
            task=task.name, condition="", run_id=0,
            passed=False, tests_total=0, tests_passed=0, tests_failed=0,
            duration_s=duration,
            error=f"Timeout after {task.timeout}s",
        )
    except Exception as e:
        duration = time.time() - start
        return TrialResult(
            task=task.name, condition="", run_id=0,
            passed=False, tests_total=0, tests_passed=0, tests_failed=0,
            duration_s=duration,
            error=str(e),
        )
    finally:
        # Cleanup container
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True, text=True,
        )


def parse_pytest_output(output: str) -> tuple[int, int, int]:
    """Parse pytest -v output for pass/fail counts."""
    passed = failed = 0
    for line in output.splitlines():
        if " PASSED" in line:
            passed += 1
        elif " FAILED" in line:
            failed += 1
    total = passed + failed
    return passed, failed, total


def run_oracle_trial(task: TaskMeta, image_tag: str, trial_dir: Path) -> TrialResult:
    """Run the oracle solution for a task."""
    oracle_script = (task.solution_dir / "solve.sh").read_text()
    result = run_trial_in_docker(task, image_tag, oracle_script, trial_dir)
    result.condition = "oracle"
    result.run_id = 0
    return result


# ---------------------------------------------------------------------------
# Results reporting
# ---------------------------------------------------------------------------

def print_results_table(results: BenchmarkResults):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("SWARM SkillsBench Results")
    print(f"Timestamp: {results.timestamp}")
    print(f"Runs per condition: {results.runs_per_condition}")
    print("=" * 80)

    # Group by condition
    for condition in results.conditions:
        cond_results = [r for r in results.results if r.condition == condition]
        if not cond_results:
            continue

        print(f"\n--- {condition} ---")
        print(f"{'Task':<30} {'Pass Rate':>10} {'Tests':>8} {'Avg Time':>10}")
        print("-" * 60)

        # Group by task
        tasks = sorted({r.task for r in cond_results})
        total_passed = 0
        total_runs = 0

        for task_name in tasks:
            task_results = [r for r in cond_results if r.task == task_name]
            n_passed = sum(1 for r in task_results if r.passed)
            n_runs = len(task_results)
            total_passed += n_passed
            total_runs += n_runs
            avg_time = sum(r.duration_s for r in task_results) / n_runs
            tests_str = f"{task_results[0].tests_passed}/{task_results[0].tests_total}" if task_results else "?"
            print(f"{task_name:<30} {n_passed}/{n_runs:>5}     {tests_str:>8} {avg_time:>8.1f}s")

        print("-" * 60)
        print(f"{'TOTAL':<30} {total_passed}/{total_runs:>5}")

    # Skill uplift comparison
    if "no-skills" in results.conditions and "with-skills" in results.conditions:
        print("\n--- Skill Uplift ---")
        tasks = sorted({r.task for r in results.results})
        for task_name in tasks:
            no_skill = [r for r in results.results if r.task == task_name and r.condition == "no-skills"]
            with_skill = [r for r in results.results if r.task == task_name and r.condition == "with-skills"]
            if no_skill and with_skill:
                ns_rate = sum(1 for r in no_skill if r.passed) / len(no_skill)
                ws_rate = sum(1 for r in with_skill if r.passed) / len(with_skill)
                delta = ws_rate - ns_rate
                symbol = "+" if delta > 0 else ""
                print(f"  {task_name:<30} {ns_rate:.0%} → {ws_rate:.0%}  ({symbol}{delta:.0%})")

        # Aggregate
        ns_all = [r for r in results.results if r.condition == "no-skills"]
        ws_all = [r for r in results.results if r.condition == "with-skills"]
        ns_agg = sum(1 for r in ns_all if r.passed) / len(ns_all) if ns_all else 0
        ws_agg = sum(1 for r in ws_all if r.passed) / len(ws_all) if ws_all else 0
        delta = ws_agg - ns_agg
        symbol = "+" if delta > 0 else ""
        print(f"\n  {'AGGREGATE':<30} {ns_agg:.0%} → {ws_agg:.0%}  ({symbol}{delta:.0%})")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SWARM SkillsBench Trial Runner")
    parser.add_argument("--task", type=str, help="Run a single task by name")
    parser.add_argument("--runs", type=int, default=3, help="Runs per condition (default: 3)")
    parser.add_argument("--condition", choices=["no-skills", "with-skills", "both"],
                        default="both", help="Which condition to run")
    parser.add_argument("--oracle", action="store_true",
                        help="Run oracle solutions only (no API calls)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    # Load tasks
    all_tasks = load_all_tasks()
    if args.task:
        tasks = [t for t in all_tasks if t.name == args.task]
        if not tasks:
            print(f"Task not found: {args.task}")
            print(f"Available: {', '.join(t.name for t in all_tasks)}")
            sys.exit(1)
    else:
        tasks = all_tasks

    # Setup output directory
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_dir = Path(args.output) if args.output else BENCH_DIR / "results" / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine conditions
    if args.oracle:
        conditions = ["oracle"]
    elif args.condition == "both":
        conditions = ["no-skills", "with-skills"]
    else:
        conditions = [args.condition]

    benchmark = BenchmarkResults(
        timestamp=ts,
        runs_per_condition=args.runs if not args.oracle else 1,
        conditions=conditions,
    )

    print("SWARM SkillsBench Trial Runner")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Conditions: {conditions}")
    print(f"  Runs/condition: {benchmark.runs_per_condition}")
    print(f"  Output: {output_dir}")
    print()

    # Build Docker images
    print("Building Docker images...")
    image_tags: dict[str, str] = {}
    for task in tasks:
        try:
            tag = build_task_image(task)
            image_tags[task.name] = tag
            print(f"  ✓ {task.name}")
        except Exception as e:
            print(f"  ✗ {task.name}: {e}")

    print(f"\nBuilt {len(image_tags)}/{len(tasks)} images.\n")

    # Run trials
    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"{'='*60}")

        for task in tasks:
            if task.name not in image_tags:
                print(f"\n  SKIP {task.name} (no image)")
                continue

            n_runs = 1 if condition == "oracle" else args.runs

            for run_id in range(n_runs):
                trial_dir = output_dir / condition / task.name / f"run-{run_id}"
                print(f"\n  [{condition}] {task.name} run={run_id}...", end=" ", flush=True)

                if condition == "oracle":
                    result = run_oracle_trial(task, image_tags[task.name], trial_dir)
                else:
                    # Compose prompt and call Claude
                    with_skills = (condition == "with-skills")
                    prompt = compose_prompt(task, with_skills=with_skills)

                    # Save prompt
                    trial_dir.mkdir(parents=True, exist_ok=True)
                    (trial_dir / "prompt.txt").write_text(prompt)

                    try:
                        response, tokens = call_claude_api(prompt, timeout=task.timeout)
                        (trial_dir / "response.txt").write_text(response)
                        script = extract_script(response)
                        (trial_dir / "extracted_solve.sh").write_text(script)
                    except Exception as e:
                        result = TrialResult(
                            task=task.name, condition=condition, run_id=run_id,
                            passed=False, tests_total=0, tests_passed=0,
                            tests_failed=0, duration_s=0, error=f"API error: {e}",
                        )
                        benchmark.results.append(result)
                        print(f"API ERROR: {e}")
                        continue

                    result = run_trial_in_docker(
                        task, image_tags[task.name], script, trial_dir,
                    )
                    result.agent_response_tokens = tokens

                result.condition = condition
                result.run_id = run_id
                benchmark.results.append(result)

                status = "PASS" if result.passed else "FAIL"
                print(f"{status} ({result.tests_passed}/{result.tests_total} tests, {result.duration_s:.1f}s)")
                if result.error:
                    print(f"    Error: {result.error[:100]}")

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(asdict(benchmark), f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print_results_table(benchmark)


if __name__ == "__main__":
    main()
