"""Track A: Verifiable reasoning benchmark pipeline."""

from __future__ import annotations

import base64
import json
import math
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from swarm.research.pdf_export import PDFExportError, paper_to_pdf
from swarm.research.swarm_papers.agentrxiv_bridge import AgentRxivBridge
from swarm.research.swarm_papers.memory import (
    BasicAudit,
    MemoryArtifact,
    MemoryStore,
    RetrievalPolicy,
    WritePolicy,
    new_artifact_id,
    summarize_artifacts,
)
from swarm.research.swarm_papers.paper import (
    CritiqueSummary,
    PaperBuilder,
    PaperContext,
    PaperFigure,
)

try:  # optional dependency
    from swarm.agents.llm_config import LLMConfig, LLMProvider
except Exception:  # pragma: no cover - optional
    LLMConfig = None  # type: ignore
    LLMProvider = None  # type: ignore


@dataclass
class ReasoningTask:
    task_id: str
    prompt: str
    answer: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Solution:
    solver_name: str
    role: str
    answer: str
    confidence: float
    raw_text: str
    tokens: int = 0


@dataclass
class EpisodeLog:
    episode_id: str
    condition: str
    task_id: str
    prompt: str
    ground_truth: str
    final_answer: str
    correct: bool
    confidence: float
    solver_outputs: list[dict]
    divergence_score: float
    reconciled: bool
    critic_flagged: bool
    critic_note: str
    adversary_flagged: bool
    memory_used: list[str]
    tokens: int
    timestamp: str
    family: str

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "condition": self.condition,
            "task_id": self.task_id,
            "prompt": self.prompt,
            "ground_truth": self.ground_truth,
            "final_answer": self.final_answer,
            "correct": self.correct,
            "confidence": self.confidence,
            "solver_outputs": self.solver_outputs,
            "divergence_score": self.divergence_score,
            "reconciled": self.reconciled,
            "critic_flagged": self.critic_flagged,
            "critic_note": self.critic_note,
            "adversary_flagged": self.adversary_flagged,
            "memory_used": self.memory_used,
            "tokens": self.tokens,
            "timestamp": self.timestamp,
            "family": self.family,
        }


@dataclass
class ConditionSpec:
    name: str
    description: str
    n_solvers: int = 1
    use_critic: bool = False
    use_reconcile: bool = False
    vote: bool = False
    use_memory: bool = False
    adversary: bool = False
    divergence_threshold: float = 0.35


@dataclass
class ConditionMetrics:
    name: str
    accuracy: float
    avg_tokens: float
    avg_confidence: float
    disagreement_rate: float
    reconcile_rate: float
    critic_flag_rate: float
    adversary_rate: float
    note: str = ""


@dataclass
class RunSummary:
    run_id: str
    task_count: int
    conditions: list[ConditionMetrics]
    memory_artifacts: list[MemoryArtifact]
    related_work: list[str]
    family_metrics: dict[str, dict[str, float]]
    critique_summary: CritiqueSummary
    reputation_summary: dict[str, dict[str, float]]


@dataclass
class TrackAConfig:
    n_tasks: int = 200
    seed: int = 42
    difficulty: float = 0.5
    output_dir: str = "runs/swarm_collate"
    conditions: list[ConditionSpec] = field(default_factory=list)
    retrieval_policy: RetrievalPolicy = field(default_factory=RetrievalPolicy)
    write_policy: WritePolicy = field(default_factory=WritePolicy)
    enable_agentrxiv: bool = True
    agentrxiv_url: str | None = None
    enable_pdf: bool = False
    publish_to_agentrxiv: bool = False
    query: str = "SWARM Track A verifiable reasoning"
    llm_enabled: bool = False
    llm_config: "LLMConfig | None" = None

    def __post_init__(self) -> None:
        if not self.conditions:
            self.conditions = default_conditions()


def default_conditions() -> list[ConditionSpec]:
    return [
        ConditionSpec(
            name="single",
            description="Single solver baseline",
            n_solvers=1,
        ),
        ConditionSpec(
            name="diverge",
            description="Two solvers, pick highest confidence",
            n_solvers=2,
        ),
        ConditionSpec(
            name="sda",
            description="Diverge + reconcile on disagreement",
            n_solvers=2,
            use_reconcile=True,
        ),
        ConditionSpec(
            name="critic",
            description="Diverge + critic + reconcile",
            n_solvers=2,
            use_critic=True,
            use_reconcile=True,
        ),
        ConditionSpec(
            name="memory",
            description="SDA + memory retrieval",
            n_solvers=2,
            use_reconcile=True,
            use_critic=True,
            use_memory=True,
        ),
    ]


class TrackATaskGenerator:
    """Generate Track A tasks (arithmetic, algebra, logic, symbolic)."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._people = ["Ava", "Ben", "Cleo"]
        self._pets = ["cat", "dog", "owl"]
        self._colors = ["red", "green", "blue"]

    def generate(self, n_tasks: int, difficulty: float) -> list[ReasoningTask]:
        tasks: list[ReasoningTask] = []
        for idx in range(n_tasks):
            roll = self._rng.random()
            if roll < 0.25:
                task = self._expression_task(idx, difficulty)
            elif roll < 0.50:
                task = self._word_problem_task(idx, difficulty)
            elif roll < 0.70:
                task = self._algebra_task(idx, difficulty)
            elif roll < 0.90:
                task = self._symbolic_task(idx, difficulty)
            else:
                task = self._logic_grid_task(idx)
            tasks.append(task)
        return tasks

    def _expression_task(self, idx: int, difficulty: float) -> ReasoningTask:
        max_val = int(10 + 90 * difficulty)
        a = self._rng.randint(2, max_val)
        b = self._rng.randint(2, max_val)
        c = self._rng.randint(2, max_val)
        op = self._rng.choice(["+", "-", "*", "/"])
        if op == "/":
            b = self._rng.randint(2, max_val)
            a = b * self._rng.randint(2, max_val)
        expr = f"({a} {op} {b}) + {c}"
        answer = _safe_eval(expr)
        prompt = f"Compute the value of {expr}. Provide only the final numeric answer."
        return ReasoningTask(
            task_id=f"expr_{idx}",
            prompt=prompt,
            answer=_format_number(answer),
            metadata={"expression": expr, "family": "arithmetic"},
        )

    def _word_problem_task(self, idx: int, difficulty: float) -> ReasoningTask:
        max_val = int(12 + 60 * difficulty)
        a = self._rng.randint(3, max_val)
        b = self._rng.randint(3, max_val)
        c = self._rng.randint(2, 9)
        total = a + b
        if total % c != 0:
            total += c - (total % c)
        a = total - b
        prompt = (
            "A lab has {a} samples and receives {b} more. "
            "They split them evenly into {c} trays. "
            "How many samples are in each tray?"
        ).format(a=a, b=b, c=c)
        expr = f"({a} + {b}) / {c}"
        answer = _safe_eval(expr)
        return ReasoningTask(
            task_id=f"word_{idx}",
            prompt=prompt,
            answer=_format_number(answer),
            metadata={"expression": expr, "family": "word"},
        )

    def _algebra_task(self, idx: int, difficulty: float) -> ReasoningTask:
        max_val = int(8 + 70 * difficulty)
        if self._rng.random() < 0.5:
            a = self._rng.randint(2, max_val)
            x = self._rng.randint(2, max_val)
            b = self._rng.randint(1, max_val)
            c = a * x + b
            prompt = f"Solve for x: {a}x + {b} = {c}."
            expr = f"({c} - {b}) / {a}"
        else:
            a = self._rng.randint(2, max_val)
            x = self._rng.randint(2, max_val)
            b = self._rng.randint(1, max_val)
            c = a * (x + b)
            prompt = f"Solve for x: {a}(x + {b}) = {c}."
            expr = f"({c} / {a}) - {b}"

        answer = _safe_eval(expr)
        return ReasoningTask(
            task_id=f"alg_{idx}",
            prompt=prompt,
            answer=_format_number(answer),
            metadata={"expression": expr, "family": "algebra"},
        )

    def _symbolic_task(self, idx: int, difficulty: float) -> ReasoningTask:
        max_val = int(6 + 40 * difficulty)
        if self._rng.random() < 0.5:
            a = self._rng.randint(1, max_val)
            b = self._rng.randint(1, max_val)
            c = self._rng.randint(1, max_val)
            prompt = (
                f"Let a={a}, b={b}. Compute 2a^2 + 3b - {c}. "
                "Provide only the final numeric answer."
            )
            expr = f"2*({a}**2) + 3*{b} - {c}"
        else:
            a = self._rng.randint(1, max_val)
            b = self._rng.randint(1, max_val)
            c = self._rng.randint(1, max_val)
            d = self._rng.randint(1, max_val)
            k = self._rng.randint(1, max_val)
            prompt = (
                f"Let f(x) = {a}x + {b} and g(x) = {c}x^2 + {d}. "
                f"Compute g(f({k})). Provide only the final numeric answer."
            )
            expr = f"{c}*(({a}*{k}+{b})**2) + {d}"

        answer = _safe_eval(expr)
        return ReasoningTask(
            task_id=f"sym_{idx}",
            prompt=prompt,
            answer=_format_number(answer),
            metadata={"expression": expr, "family": "symbolic"},
        )

    def _logic_grid_task(self, idx: int) -> ReasoningTask:
        people = list(self._people)
        pets = list(self._pets)
        colors = list(self._colors)
        self._rng.shuffle(pets)
        self._rng.shuffle(colors)

        mapping = {
            person: {"pet": pet, "color": color}
            for person, pet, color in zip(people, pets, colors, strict=False)
        }

        target_person = self._rng.choice(people)
        target_pet = mapping[target_person]["pet"]
        target_color = mapping[target_person]["color"]

        remaining = [p for p in people if p != target_person]
        person_color_clue = self._rng.choice(remaining)
        person_neg = [p for p in remaining if p != person_color_clue][0]
        color_other = mapping[person_color_clue]["color"]

        clues = [
            f"{person_color_clue} wears {color_other}.",
            f"{person_neg} does not wear {target_color}.",
            f"The person wearing {target_color} has the {target_pet}.",
        ]
        clues_text = " ".join(f"{i+1}) {clue}" for i, clue in enumerate(clues))
        prompt = (
            "Three researchers (Ava, Ben, Cleo) each have a different pet "
            "(cat, dog, owl) and wear a different color (red, green, blue). "
            "Clues: "
            + clues_text
            + f" Question: Who has the {target_pet}? Provide the name only."
        )

        return ReasoningTask(
            task_id=f"logic_{idx}",
            prompt=prompt,
            answer=target_person,
            metadata={
                "solution": target_person,
                "options": people,
                "family": "logic_grid",
            },
        )


class BaseSolver:
    def solve(self, task: ReasoningTask, memory_hint: str) -> Solution:  # pragma: no cover
        raise NotImplementedError


class HeuristicSolver(BaseSolver):
    def __init__(
        self,
        name: str,
        role: str,
        *,
        noise_rate: float = 0.0,
        base_confidence: float = 0.7,
        seed: int | None = None,
    ) -> None:
        self.name = name
        self.role = role
        self.noise_rate = noise_rate
        self.base_confidence = base_confidence
        self._rng = random.Random(seed)

    def solve(self, task: ReasoningTask, memory_hint: str) -> Solution:
        expr = task.metadata.get("expression", "")
        solution = task.metadata.get("solution")
        options = task.metadata.get("options", [])

        if solution is not None:
            answer_str = str(solution)
        elif expr:
            answer_str = _format_number(_safe_eval(expr))
        else:
            answer_str = _format_number(_fallback_parse(task.prompt))

        if self._rng.random() < self.noise_rate:
            answer_str = _introduce_noise(answer_str, options, self._rng)

        confidence = max(0.05, min(0.95, self.base_confidence))
        raw = f"Answer: {answer_str}\nConfidence: {confidence:.2f}"
        return Solution(
            solver_name=self.name,
            role=self.role,
            answer=answer_str,
            confidence=confidence,
            raw_text=raw,
            tokens=0,
        )


class LLMClient:
    def __init__(self, config: "LLMConfig") -> None:
        self.config = config

    def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        provider = self.config.provider
        if provider == LLMProvider.ANTHROPIC:
            return self._call_anthropic(system_prompt, user_prompt)
        if provider == LLMProvider.OPENAI:
            return self._call_openai(system_prompt, user_prompt)
        if provider == LLMProvider.OLLAMA:
            return self._call_ollama(system_prompt, user_prompt)
        raise ValueError(f"Unknown provider: {provider}")

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        try:
            import anthropic
        except ImportError as err:  # pragma: no cover - optional
            raise ImportError(
                "anthropic package not installed. Install swarm-safety[llm]."
            ) from err
        client = anthropic.Anthropic(api_key=self.config.api_key)
        message = client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = message.content[0].text if message.content else ""
        usage = getattr(message, "usage", None)
        in_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        out_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        return text, in_tokens, out_tokens

    def _call_openai(self, system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        try:
            import openai
        except ImportError as err:  # pragma: no cover - optional
            raise ImportError(
                "openai package not installed. Install swarm-safety[llm]."
            ) from err
        client = openai.OpenAI(api_key=self.config.api_key)
        response = client.chat.completions.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content if response.choices else ""
        usage = getattr(response, "usage", None)
        in_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        return text or "", in_tokens, out_tokens

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        try:
            import httpx
        except ImportError as err:  # pragma: no cover - optional
            raise ImportError(
                "httpx package not installed. Install swarm-safety[llm]."
            ) from err
        base_url = self.config.base_url or "http://localhost:11434"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
            "stream": False,
        }
        with httpx.Client(timeout=self.config.timeout) as client:
            resp = client.post(f"{base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        text = data.get("message", {}).get("content", "")
        return text, 0, 0


class LLMSolver(BaseSolver):
    def __init__(self, name: str, role: str, config: "LLMConfig") -> None:
        self.name = name
        self.role = role
        self.client = LLMClient(config)
        self.config = config

    def solve(self, task: ReasoningTask, memory_hint: str) -> Solution:
        system_prompt = f"You are a {self.role} math solver."
        user_prompt = (
            f"Problem: {task.prompt}\n\n"
            f"Helpful memory (if any):\n{memory_hint or 'None'}\n\n"
            "Return a JSON object with keys: answer (string), confidence (0-1), reasoning (string)."
        )
        text, in_tokens, out_tokens = self.client.complete(system_prompt, user_prompt)
        answer, confidence = _parse_llm_answer(text)
        return Solution(
            solver_name=self.name,
            role=self.role,
            answer=answer,
            confidence=confidence,
            raw_text=text,
            tokens=in_tokens + out_tokens,
        )


class SimpleCritic:
    def critique(self, solutions: list[Solution], task: ReasoningTask) -> str | None:
        if len(solutions) < 2:
            return None
        answers = [_normalize_answer(sol.answer) for sol in solutions]
        if any(not ans for ans in answers):
            return "missing answer"

        if task.metadata.get("family") == "logic_grid":
            if len(set(answers)) > 1:
                return "logic grid inconsistency"
            return None

        numeric_answers = []
        for ans in answers:
            try:
                numeric_answers.append(float(ans))
            except Exception:
                return "non-numeric answer in numeric task"

        spread = max(numeric_answers) - min(numeric_answers)
        top_conf = sorted((sol.confidence for sol in solutions), reverse=True)
        if len(set(answers)) > 1 and top_conf[0] >= 0.7 and top_conf[1] >= 0.6:
            return "confident disagreement"
        if len(set(answers)) > 1 and spread >= 5:
            return "large numeric divergence"
        if len(set(answers)) > 1:
            return "moderate numeric divergence"
        return None


@dataclass
class ReputationStats:
    total: int = 0
    correct: int = 0
    ema_accuracy: float = 0.5
    deviation_count: int = 0


class ReputationTracker:
    def __init__(self, ema_alpha: float = 0.1) -> None:
        self.ema_alpha = ema_alpha
        self.solver_stats: dict[str, ReputationStats] = {}
        self.role_stats: dict[str, ReputationStats] = {}

    def _get(self, store: dict[str, ReputationStats], key: str) -> ReputationStats:
        if key not in store:
            store[key] = ReputationStats()
        return store[key]

    def update(self, solutions: list[Solution], ground_truth: str) -> None:
        majority = _majority_answer(solutions)
        for sol in solutions:
            correct = _answers_match(sol.answer, ground_truth)
            solver_stats = self._get(self.solver_stats, sol.solver_name)
            role_stats = self._get(self.role_stats, sol.role)
            self._update_stats(solver_stats, correct, sol, majority)
            self._update_stats(role_stats, correct, sol, majority)

    def _update_stats(
        self,
        stats: ReputationStats,
        correct: bool,
        sol: Solution,
        majority: str | None,
    ) -> None:
        stats.total += 1
        if correct:
            stats.correct += 1
        stats.ema_accuracy = (
            (1 - self.ema_alpha) * stats.ema_accuracy
            + self.ema_alpha * (1.0 if correct else 0.0)
        )
        if majority and _normalize_answer(sol.answer) != majority:
            stats.deviation_count += 1

    def score(self, solver_name: str, role: str) -> float:
        scores = []
        if solver_name in self.solver_stats:
            scores.append(self.solver_stats[solver_name].ema_accuracy)
        if role in self.role_stats:
            scores.append(self.role_stats[role].ema_accuracy)
        if not scores:
            return 0.5
        return sum(scores) / len(scores)

    def deviation_rate(self, solver_name: str, role: str) -> float:
        rates = []
        if solver_name in self.solver_stats and self.solver_stats[solver_name].total:
            stats = self.solver_stats[solver_name]
            rates.append(stats.deviation_count / stats.total)
        if role in self.role_stats and self.role_stats[role].total:
            stats = self.role_stats[role]
            rates.append(stats.deviation_count / stats.total)
        return max(rates) if rates else 0.0

    def snapshot(self) -> dict[str, dict[str, float]]:
        def _pack(stats: ReputationStats) -> dict[str, float]:
            rate = stats.correct / stats.total if stats.total else 0.0
            dev = stats.deviation_count / stats.total if stats.total else 0.0
            return {
                "total": float(stats.total),
                "accuracy": rate,
                "ema_accuracy": stats.ema_accuracy,
                "deviation_rate": dev,
            }

        return {
            "solvers": {name: _pack(stats) for name, stats in self.solver_stats.items()},
            "roles": {name: _pack(stats) for name, stats in self.role_stats.items()},
        }


class SimpleReconciler:
    def reconcile(self, solutions: list[Solution]) -> Solution:
        # pick highest confidence solution
        if not solutions:
            return Solution("reconciler", "reconciler", "", 0.0, "")
        return max(solutions, key=lambda sol: sol.confidence)


class TrackARunner:
    """Execute Track A experiments and produce a paper draft."""

    def __init__(self, config: TrackAConfig):
        self.config = config
        self.run_id = datetime.now(timezone.utc).strftime("track_a_%Y%m%d_%H%M%S")
        self.output_dir = Path(config.output_dir) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.output_dir / "episodes.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.memory_store = MemoryStore(self.output_dir / "memory.jsonl")
        self.audit = BasicAudit()
        self.agentrxiv = AgentRxivBridge(config.agentrxiv_url) if config.enable_agentrxiv else None
        self._agentrxiv_cache: list[MemoryArtifact] = []
        self._agentrxiv_loaded = False
        self.reputation = ReputationTracker()

    def run(self) -> RunSummary:
        tasks = TrackATaskGenerator(self.config.seed).generate(
            self.config.n_tasks, self.config.difficulty
        )
        baseline_acc = None
        condition_metrics: list[ConditionMetrics] = []
        memory_artifacts: list[MemoryArtifact] = []
        family_metrics: dict[str, dict[str, float]] = {}
        all_logs: list[EpisodeLog] = []

        related_work = []
        if self.agentrxiv and self.agentrxiv.available():
            related_work = [
                f"{paper.title or 'Untitled'} ({paper.paper_id})"
                for paper in self.agentrxiv.related_work(
                    self.config.query, limit=5
                )
            ]

        for condition in self.config.conditions:
            logs = self._run_condition(condition, tasks)
            all_logs.extend(logs)
            metrics = self._compute_metrics(condition, logs)
            condition_metrics.append(metrics)
            if condition.name == "single":
                baseline_acc = metrics.accuracy
            family_metrics[condition.name] = self._compute_family_metrics(logs)

            artifact = self._maybe_write_memory(condition, metrics, baseline_acc, len(tasks))
            if artifact is not None:
                memory_artifacts.append(artifact)

        summary = RunSummary(
            run_id=self.run_id,
            task_count=len(tasks),
            conditions=condition_metrics,
            memory_artifacts=memory_artifacts,
            related_work=related_work,
            family_metrics=family_metrics,
            critique_summary=self._summarize_critiques(all_logs),
            reputation_summary=self.reputation.snapshot(),
        )
        self._write_summary(summary)
        self._write_paper(summary)
        return summary

    def _run_condition(self, condition: ConditionSpec, tasks: list[ReasoningTask]) -> list[EpisodeLog]:
        solver_pool = self._build_solvers(condition)
        critic = SimpleCritic() if condition.use_critic else None
        reconciler = SimpleReconciler()
        logs: list[EpisodeLog] = []

        for task in tasks:
            memory_items = self._retrieve_memory(task, condition)
            memory_hint = summarize_artifacts(memory_items)
            solutions = [solver.solve(task, memory_hint) for solver in solver_pool]
            divergence = _divergence_score(solutions)
            critic_note = critic.critique(solutions, task) if critic else None
            if critic_note:
                for sol in solutions:
                    sol.confidence = max(0.01, sol.confidence - 0.1)

            reconciled = False
            final_solution = None
            if condition.use_reconcile and divergence >= condition.divergence_threshold:
                reconciled = True
                final_solution = reconciler.reconcile(solutions)
            else:
                final_solution = self._select_solution(condition, solutions)

            correct = _answers_match(final_solution.answer, task.answer)
            critic_flagged = bool(critic_note)
            adversary_flagged = _detect_adversary(
                solutions,
                task.answer,
                reputation=self.reputation,
                threshold=self.config.write_policy.adversary_conf_threshold,
            )
            tokens = sum(sol.tokens for sol in solutions)
            log = EpisodeLog(
                episode_id=uuid.uuid4().hex[:12],
                condition=condition.name,
                task_id=task.task_id,
                prompt=task.prompt,
                ground_truth=task.answer,
                final_answer=final_solution.answer,
                correct=correct,
                confidence=final_solution.confidence,
                solver_outputs=[
                    {
                        "solver": sol.solver_name,
                        "role": sol.role,
                        "answer": sol.answer,
                        "confidence": sol.confidence,
                        "tokens": sol.tokens,
                    }
                    for sol in solutions
                ],
                divergence_score=divergence,
                reconciled=reconciled,
                critic_flagged=critic_flagged,
                critic_note=critic_note or "",
                adversary_flagged=adversary_flagged,
                memory_used=[artifact.artifact_id for artifact in memory_items],
                tokens=tokens,
                timestamp=datetime.now(timezone.utc).isoformat(),
                family=task.metadata.get("family", "unknown"),
            )
            logs.append(log)
            self._append_episode(log)
            self.reputation.update(solutions, task.answer)

        return logs

    def _build_solvers(self, condition: ConditionSpec) -> list[BaseSolver]:
        solvers: list[BaseSolver] = []
        if self.config.llm_enabled and self.config.llm_config is not None:
            base_cfg = self.config.llm_config
            precise_cfg = _adjust_llm_config(base_cfg, temperature=0.2)
            creative_cfg = _adjust_llm_config(base_cfg, temperature=0.8)
            solvers.append(LLMSolver("precise", "precise solver", precise_cfg))
            if condition.n_solvers > 1:
                solvers.append(LLMSolver("creative", "creative solver", creative_cfg))
            for idx in range(2, condition.n_solvers):
                cfg = _adjust_llm_config(base_cfg, temperature=0.6)
                solvers.append(LLMSolver(f"solver_{idx}", "solver", cfg))
        else:
            solvers.append(HeuristicSolver("precise", "precise solver", seed=self.config.seed))
            if condition.n_solvers > 1:
                solvers.append(
                    HeuristicSolver(
                        "creative",
                        "creative solver",
                        noise_rate=0.2,
                        base_confidence=0.6,
                        seed=self.config.seed + 1,
                    )
                )
            for idx in range(2, condition.n_solvers):
                solvers.append(
                    HeuristicSolver(
                        f"solver_{idx}",
                        "solver",
                        noise_rate=0.15,
                        base_confidence=0.55,
                        seed=self.config.seed + idx,
                    )
                )

        if condition.adversary:
            solvers.append(
                HeuristicSolver(
                    "adversary",
                    "adversary",
                    noise_rate=0.9,
                    base_confidence=0.8,
                    seed=self.config.seed + 99,
                )
            )
        return solvers

    def _select_solution(self, condition: ConditionSpec, solutions: list[Solution]) -> Solution:
        if not solutions:
            return Solution("none", "none", "", 0.0, "")
        if condition.vote and len(solutions) > 2:
            return _majority_vote(solutions)
        return max(solutions, key=lambda sol: sol.confidence)

    def _retrieve_memory(self, task: ReasoningTask, condition: ConditionSpec) -> list[MemoryArtifact]:
        if not condition.use_memory:
            return []
        query = f"{self.config.query} {task.prompt}"
        artifacts = self.memory_store.search(query, policy=self.config.retrieval_policy)
        if self.agentrxiv and self.agentrxiv.available():
            if not self._agentrxiv_loaded:
                self._agentrxiv_cache = self.agentrxiv.to_artifacts(
                    self.config.query,
                    limit=self.config.retrieval_policy.max_items,
                    min_score=self.config.retrieval_policy.min_score,
                )
                self._agentrxiv_loaded = True
            artifacts.extend(self._agentrxiv_cache)
        if len(artifacts) > self.config.retrieval_policy.max_items:
            artifacts = artifacts[: self.config.retrieval_policy.max_items]
        return artifacts

    def _compute_metrics(self, condition: ConditionSpec, logs: list[EpisodeLog]) -> ConditionMetrics:
        if not logs:
            return ConditionMetrics(condition.name, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        accuracy = sum(1 for log in logs if log.correct) / len(logs)
        avg_tokens = mean(log.tokens for log in logs)
        avg_conf = mean(log.confidence for log in logs)
        disagreement = sum(1 for log in logs if log.divergence_score >= condition.divergence_threshold)
        reconcile = sum(1 for log in logs if log.reconciled)
        disagreement_rate = disagreement / len(logs)
        reconcile_rate = reconcile / len(logs)
        critic_flag_rate = sum(1 for log in logs if log.critic_flagged) / len(logs)
        adversary_rate = sum(1 for log in logs if log.adversary_flagged) / len(logs)
        return ConditionMetrics(
            name=condition.name,
            accuracy=accuracy,
            avg_tokens=avg_tokens,
            avg_confidence=avg_conf,
            disagreement_rate=disagreement_rate,
            reconcile_rate=reconcile_rate,
            critic_flag_rate=critic_flag_rate,
            adversary_rate=adversary_rate,
            note=condition.description,
        )

    def _maybe_write_memory(
        self,
        condition: ConditionSpec,
        metrics: ConditionMetrics,
        baseline_acc: float | None,
        n_tasks: int,
    ) -> MemoryArtifact | None:
        if condition.name == "single":
            return None
        delta = metrics.accuracy - baseline_acc if baseline_acc is not None else None
        artifact = MemoryArtifact(
            artifact_id=new_artifact_id(),
            title=f"Mechanism: {condition.name}",
            summary=(
                f"{condition.description}. Accuracy {metrics.accuracy:.2%}"
                + (f" (delta {delta:+.2%} vs baseline)." if delta is not None else ".")
            ),
            use_when="Verifiable reasoning tasks with deterministic checks",
            failure_modes=[],
            metrics={
                "accuracy": metrics.accuracy,
                "avg_tokens": metrics.avg_tokens,
                "disagreement_rate": metrics.disagreement_rate,
                "reconcile_rate": metrics.reconcile_rate,
            },
            source="local",
            source_id=self.run_id,
        )

        report = self.audit.evaluate(
            artifact=artifact,
            accuracy=metrics.accuracy,
            delta_vs_baseline=delta,
            n_tasks=n_tasks,
            critic_flag_rate=metrics.critic_flag_rate if condition.use_critic else None,
            adversary_rate=metrics.adversary_rate,
            policy=self.config.write_policy,
        )
        if self.config.write_policy.require_audit and not report.passed:
            return None

        self.memory_store.append(artifact)
        return artifact

    def _append_episode(self, log: EpisodeLog) -> None:
        with self.episodes_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log.to_dict()) + "\n")

    def _write_summary(self, summary: RunSummary) -> None:
        payload = {
            "run_id": summary.run_id,
            "task_count": summary.task_count,
            "conditions": [
                {
                    "name": cond.name,
                    "accuracy": cond.accuracy,
                    "avg_tokens": cond.avg_tokens,
                    "avg_confidence": cond.avg_confidence,
                    "disagreement_rate": cond.disagreement_rate,
                    "reconcile_rate": cond.reconcile_rate,
                    "critic_flag_rate": cond.critic_flag_rate,
                    "adversary_rate": cond.adversary_rate,
                    "note": cond.note,
                }
                for cond in summary.conditions
            ],
            "memory_artifacts": [artifact.to_dict() for artifact in summary.memory_artifacts],
            "related_work": summary.related_work,
            "family_metrics": summary.family_metrics,
            "critique_summary": {
                "total_flags": summary.critique_summary.total_flags,
                "flag_rate": summary.critique_summary.flag_rate,
                "top_reasons": summary.critique_summary.top_reasons,
            },
            "reputation_summary": summary.reputation_summary,
        }
        self.summary_path.write_text(json.dumps(payload, indent=2))

    def _write_paper(self, summary: RunSummary) -> None:
        builder = PaperBuilder()
        figures, images = self._generate_plots(summary)
        conditions_payload = [
            {
                "name": cond.name,
                "accuracy": cond.accuracy,
                "avg_tokens": cond.avg_tokens,
                "note": cond.note,
            }
            for cond in summary.conditions
        ]
        context = PaperContext(
            title="SWARM Track A: Disagreement + Memory in Verifiable Reasoning",
            abstract=(
                "We benchmark SWARM coordination mechanisms on a verifiable reasoning "
                "track, comparing divergence, critique, reconciliation, and memory "
                "retrieval. We report accuracy, disagreement rates, and costs across "
                f"{summary.task_count} tasks."
            ),
            run_id=summary.run_id,
            task_count=summary.task_count,
            conditions=conditions_payload,
            related_work=summary.related_work,
            memory_items=summary.memory_artifacts,
            critique_summary=summary.critique_summary,
            family_metrics=summary.family_metrics,
            figures=figures,
            images=images,
        )
        paper = builder.build(context)
        tex_path = self.output_dir / "paper.tex"
        tex_path.write_text(paper.source)

        if self.config.enable_pdf:
            try:
                pdf_path = paper_to_pdf(paper, output_path=self.output_dir / "paper.pdf")
            except PDFExportError:
                return
            if self.config.publish_to_agentrxiv and self.agentrxiv:
                result = self.agentrxiv.submit(paper, str(pdf_path))
                if result.success:
                    self.agentrxiv.trigger_update()

    def _generate_plots(self, summary: RunSummary) -> tuple[list[PaperFigure], dict[str, str]]:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return [], {}

        figures: list[PaperFigure] = []
        images: dict[str, str] = {}

        names = [cond.name for cond in summary.conditions]
        accuracies = [cond.accuracy for cond in summary.conditions]
        tokens = [cond.avg_tokens for cond in summary.conditions]

        # Figure 1: Accuracy by condition
        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        ax1.bar(names, accuracies, color="#4C72B0")
        ax1.set_ylim(0, 1.0)
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Track A Accuracy by Condition")
        ax1.grid(axis="y", linestyle="--", alpha=0.4)
        fig1.tight_layout()
        fig1_path = self.output_dir / "figure_accuracy.png"
        fig1.savefig(fig1_path, dpi=150)
        plt.close(fig1)

        figures.append(
            PaperFigure(
                filename=fig1_path.name,
                caption="Accuracy across coordination conditions.",
            )
        )
        images[fig1_path.name] = _encode_image(fig1_path)

        # Figure 2: Coordination rates
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        x = list(range(len(names)))
        width = 0.25
        disagreement = [cond.disagreement_rate for cond in summary.conditions]
        reconcile = [cond.reconcile_rate for cond in summary.conditions]
        adversary = [cond.adversary_rate for cond in summary.conditions]
        ax2.bar([i - width for i in x], disagreement, width=width, label="disagreement")
        ax2.bar(x, reconcile, width=width, label="reconcile")
        ax2.bar([i + width for i in x], adversary, width=width, label="adversary")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names)
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel("Rate")
        ax2.set_title("Coordination & Adversary Flags")
        ax2.legend(fontsize=8, ncol=3, frameon=False)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        fig2.tight_layout()
        fig2_path = self.output_dir / "figure_rates.png"
        fig2.savefig(fig2_path, dpi=150)
        plt.close(fig2)

        figures.append(
            PaperFigure(
                filename=fig2_path.name,
                caption="Disagreement, reconcile, and adversary-flag rates by condition.",
            )
        )
        images[fig2_path.name] = _encode_image(fig2_path)

        # Figure 3: Accuracy vs tokens
        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        ax3.scatter(tokens, accuracies, color="#55A868")
        for name, x_val, y_val in zip(names, tokens, accuracies, strict=False):
            ax3.annotate(name, (x_val, y_val), textcoords="offset points", xytext=(4, 3), fontsize=8)
        ax3.set_xlabel("Avg Tokens")
        ax3.set_ylabel("Accuracy")
        ax3.set_title("Efficiency Tradeoff")
        ax3.grid(True, linestyle="--", alpha=0.4)
        fig3.tight_layout()
        fig3_path = self.output_dir / "figure_efficiency.png"
        fig3.savefig(fig3_path, dpi=150)
        plt.close(fig3)

        figures.append(
            PaperFigure(
                filename=fig3_path.name,
                caption="Accuracy vs average token cost.",
            )
        )
        images[fig3_path.name] = _encode_image(fig3_path)

        if summary.family_metrics:
            fig4, ax4 = plt.subplots(figsize=(7, 3.8))
            families = _family_order(summary.family_metrics)
            conditions = [cond.name for cond in summary.conditions]
            matrix = []
            for cond in conditions:
                row = []
                for family in families:
                    row.append(summary.family_metrics.get(cond, {}).get(family, 0.0))
                matrix.append(row)
            im = ax4.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
            ax4.set_xticks(range(len(families)))
            ax4.set_xticklabels([_family_label(f) for f in families], rotation=30, ha="right")
            ax4.set_yticks(range(len(conditions)))
            ax4.set_yticklabels(conditions)
            ax4.set_title("Accuracy by Task Family and Condition")
            for i in range(len(conditions)):
                for j in range(len(families)):
                    ax4.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=7)
            fig4.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
            fig4.tight_layout()
            fig4_path = self.output_dir / "figure_family_heatmap.png"
            fig4.savefig(fig4_path, dpi=150)
            plt.close(fig4)

            figures.append(
                PaperFigure(
                    filename=fig4_path.name,
                    caption="Per-family accuracy (arithmetic, algebra, logic, symbolic, word).",
                )
            )
            images[fig4_path.name] = _encode_image(fig4_path)

        return figures, images

    def _compute_family_metrics(self, logs: list[EpisodeLog]) -> dict[str, float]:
        family_totals: dict[str, list[int]] = {}
        for log in logs:
            family = log.family or "unknown"
            if family not in family_totals:
                family_totals[family] = [0, 0]
            family_totals[family][1] += 1
            if log.correct:
                family_totals[family][0] += 1
        return {
            family: (counts[0] / counts[1]) if counts[1] else 0.0
            for family, counts in family_totals.items()
        }

    def _summarize_critiques(self, logs: list[EpisodeLog]) -> CritiqueSummary:
        total = len(logs)
        flagged = [log for log in logs if log.critic_flagged]
        if total == 0:
            return CritiqueSummary(total_flags=0, flag_rate=0.0, top_reasons=[])
        reasons: dict[str, int] = {}
        for log in flagged:
            reason = log.critic_note.strip() or "unspecified"
            reasons[reason] = reasons.get(reason, 0) + 1
        top_reasons = [
            reason for reason, _ in sorted(reasons.items(), key=lambda item: item[1], reverse=True)[:3]
        ]
        return CritiqueSummary(
            total_flags=len(flagged),
            flag_rate=len(flagged) / total,
            top_reasons=top_reasons,
        )


def _safe_eval(expr: str) -> float:
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        return 0.0
    try:
        value = eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return 0.0
    return round(float(value), 4)


def _fallback_parse(prompt: str) -> float:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", prompt)
    return float(numbers[-1]) if numbers else 0.0


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(round(float(value), 4))


def _introduce_noise(answer: str, options: list[str], rng: random.Random) -> str:
    if options:
        choices = [opt for opt in options if opt != answer]
        if choices:
            return rng.choice(choices)
    try:
        num = float(answer)
        num += rng.choice([-2, -1, 1, 2])
        return _format_number(num)
    except Exception:
        return answer


def _encode_image(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def _parse_llm_answer(text: str) -> tuple[str, float]:
    try:
        data = json.loads(text)
        answer = str(data.get("answer", ""))
        confidence = float(data.get("confidence", 0.5))
        return answer, _clamp(confidence)
    except Exception:
        pass
    answer_match = re.search(r"answer\s*[:=]\s*([\-0-9\.]+)", text, re.I)
    confidence_match = re.search(r"confidence\s*[:=]\s*([0-9\.]+)", text, re.I)
    answer = answer_match.group(1) if answer_match else text.strip().split("\n")[-1]
    confidence = float(confidence_match.group(1)) if confidence_match else 0.5
    return answer, _clamp(confidence)


def _normalize_answer(answer: str) -> str:
    answer = answer.strip()
    num_match = re.findall(r"-?\d+(?:\.\d+)?", answer)
    if num_match:
        try:
            value = float(num_match[-1])
            if value.is_integer():
                return str(int(value))
            return str(round(value, 4))
        except Exception:
            pass
    return answer.lower()


def _answers_match(pred: str, gold: str) -> bool:
    pred_norm = _normalize_answer(pred)
    gold_norm = _normalize_answer(gold)
    try:
        return math.isclose(float(pred_norm), float(gold_norm), rel_tol=1e-3, abs_tol=1e-3)
    except Exception:
        return pred_norm == gold_norm


def _divergence_score(solutions: list[Solution]) -> float:
    if len(solutions) < 2:
        return 0.0
    a = _normalize_answer(solutions[0].answer)
    b = _normalize_answer(solutions[1].answer)
    if a == b:
        return 0.0
    # simple normalized mismatch
    return 1.0


def _detect_adversary(
    solutions: list[Solution],
    ground_truth: str,
    *,
    reputation: "ReputationTracker",
    threshold: float = 0.8,
) -> bool:
    if not solutions:
        return False

    any_correct = any(_answers_match(sol.answer, ground_truth) for sol in solutions)
    majority = _majority_answer(solutions)
    for sol in solutions:
        if sol.confidence < threshold:
            continue
        if _answers_match(sol.answer, ground_truth):
            continue
        role = sol.role.lower()
        name = sol.solver_name.lower()
        rep = reputation.score(sol.solver_name, sol.role)
        dev = reputation.deviation_rate(sol.solver_name, sol.role)

        if "adversary" in role or "adversary" in name:
            return True
        if rep < 0.45 and dev > 0.5:
            return True
        if majority and _normalize_answer(sol.answer) != majority and dev > 0.6:
            return True

    top = max(solutions, key=lambda sol: sol.confidence)
    top_rep = reputation.score(top.solver_name, top.role)
    top_dev = reputation.deviation_rate(top.solver_name, top.role)
    if top.confidence >= threshold and not _answers_match(top.answer, ground_truth):
        if any_correct and (top_rep < 0.6 or top_dev > 0.5):
            return True
    return False


def _majority_vote(solutions: list[Solution]) -> Solution:
    counts: dict[str, int] = {}
    for sol in solutions:
        key = _normalize_answer(sol.answer)
        counts[key] = counts.get(key, 0) + 1
    winner = max(counts.items(), key=lambda item: item[1])[0]
    for sol in solutions:
        if _normalize_answer(sol.answer) == winner:
            return sol
    return solutions[0]


def _majority_answer(solutions: list[Solution]) -> str | None:
    if not solutions:
        return None
    counts: dict[str, int] = {}
    for sol in solutions:
        key = _normalize_answer(sol.answer)
        counts[key] = counts.get(key, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def _family_order(family_metrics: dict[str, dict[str, float]]) -> list[str]:
    preferred = ["arithmetic", "algebra", "logic_grid", "symbolic", "word"]
    families = set()
    for metrics in family_metrics.values():
        families.update(metrics.keys())
    ordered = [fam for fam in preferred if fam in families]
    remainder = sorted(fam for fam in families if fam not in ordered)
    return ordered + remainder


def _family_label(family: str) -> str:
    return "logic" if family == "logic_grid" else family


def _clamp(value: float) -> float:
    return max(0.01, min(0.99, value))


def _adjust_llm_config(config: "LLMConfig", temperature: float) -> "LLMConfig":
    return LLMConfig(
        provider=config.provider,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        max_retries=config.max_retries,
        retry_base_delay=config.retry_base_delay,
        persona=config.persona,
        system_prompt=config.system_prompt,
        cost_tracking=config.cost_tracking,
        prompt_audit_path=config.prompt_audit_path,
        prompt_audit_include_system_prompt=config.prompt_audit_include_system_prompt,
        prompt_audit_hash_system_prompt=config.prompt_audit_hash_system_prompt,
        prompt_audit_max_chars=config.prompt_audit_max_chars,
    )
