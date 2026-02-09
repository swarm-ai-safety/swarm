"""Tests for incoherence forecaster feature/model modules."""

import random

import pytest

from swarm.forecaster.features import (
    combine_feature_dicts,
    extract_behavioral_features,
    extract_structural_features,
)
from swarm.forecaster.model import IncoherenceForecaster
from swarm.models.interaction import SoftInteraction
from swarm.replay import EpisodeSpec, ReplayRunner
from swarm.scenarios import load_scenario

pytestmark = pytest.mark.slow


def test_structural_feature_extraction():
    features = extract_structural_features(
        horizon_length=12,
        agent_count=8,
        action_space_size=6,
        adversarial_fraction=0.25,
    )
    assert features["horizon_length"] == 12.0
    assert features["agent_count"] == 8.0
    assert features["action_space_size"] == 6.0
    assert features["adversarial_fraction"] == pytest.approx(0.25)


def test_behavioral_feature_extraction():
    interactions = [
        SoftInteraction(p=0.5, accepted=True),
        SoftInteraction(p=0.9, accepted=False),
    ]
    features = extract_behavioral_features(interactions)
    assert features["behavioral_interaction_count"] == 2.0
    assert 0.0 <= features["behavioral_uncertainty_mean"] <= 1.0
    assert 0.0 <= features["behavioral_acceptance_rate"] <= 1.0


def test_train_predict_api_on_synthetic_replay_data():
    rng = random.Random(42)
    rows = []
    labels = []
    for _ in range(200):
        horizon = rng.randint(2, 30)
        agents = rng.randint(2, 15)
        adv_frac = rng.random()
        row = extract_structural_features(
            horizon_length=horizon,
            agent_count=agents,
            action_space_size=6,
            adversarial_fraction=adv_frac,
        )
        rows.append(row)
        labels.append(int((0.04 * horizon + 0.08 * agents + 1.0 * adv_frac) > 2.1))

    forecaster = IncoherenceForecaster(learning_rate=0.1, n_iters=800)
    forecaster.fit(rows, labels)
    prob = forecaster.predict_proba(rows[0])
    pred = forecaster.predict(rows[0], threshold=0.5)
    assert 0.0 <= prob <= 1.0
    assert pred in {0, 1}


def test_holdout_eval_reports_auc_and_calibration():
    rng = random.Random(7)
    train_rows = []
    train_labels = []
    test_rows = []
    test_labels = []

    for i in range(300):
        row = combine_feature_dicts(
            extract_structural_features(
                horizon_length=rng.randint(2, 30),
                agent_count=rng.randint(2, 15),
                action_space_size=6,
                adversarial_fraction=rng.random(),
            ),
            {
                "behavioral_uncertainty_mean": rng.random(),
                "behavioral_acceptance_rate": rng.random(),
            },
        )
        score = (
            0.03 * row["horizon_length"]
            + 0.07 * row["agent_count"]
            + 0.9 * row["adversarial_fraction"]
            + 0.8 * row["behavioral_uncertainty_mean"]
            - 0.3 * row["behavioral_acceptance_rate"]
        )
        label = int(score > 2.2)
        if i < 220:
            train_rows.append(row)
            train_labels.append(label)
        else:
            test_rows.append(row)
            test_labels.append(label)

    forecaster = IncoherenceForecaster(learning_rate=0.08, n_iters=1000)
    metrics = forecaster.fit_and_evaluate(
        train_features=train_rows,
        train_labels=train_labels,
        test_features=test_rows,
        test_labels=test_labels,
    )
    assert "auc" in metrics
    assert "expected_calibration_error" in metrics
    assert metrics["auc"] >= 0.7
    assert 0.0 <= metrics["expected_calibration_error"] <= 1.0


def test_train_predict_api_on_replay_dataset():
    paths = [
        "scenarios/incoherence/short_low_branching.yaml",
        "scenarios/incoherence/medium_medium_branching.yaml",
        "scenarios/incoherence/long_high_branching.yaml",
    ]
    rows = []
    labels = []
    for path in paths:
        scenario = load_scenario(path)
        spec = EpisodeSpec(scenario=scenario, seed=42, replay_k=2)
        replay_results = ReplayRunner(spec).run()
        for result in replay_results:
            row = combine_feature_dicts(
                extract_structural_features(
                    horizon_length=scenario.orchestrator_config.steps_per_epoch,
                    agent_count=sum(
                        spec_item.get("count", 1) for spec_item in scenario.agent_specs
                    ),
                    action_space_size=6,
                    adversarial_fraction=(
                        sum(
                            spec_item.get("count", 1)
                            for spec_item in scenario.agent_specs
                            if spec_item.get("type") in {"deceptive", "adversarial"}
                        )
                        / max(
                            1,
                            sum(
                                spec_item.get("count", 1)
                                for spec_item in scenario.agent_specs
                            ),
                        )
                    ),
                ),
                {
                    "behavioral_uncertainty_mean": result.avg_toxicity,
                    "behavioral_acceptance_rate": 1.0
                    if result.total_interactions == 0
                    else (result.accepted_interactions / result.total_interactions),
                },
            )
            rows.append(row)
            labels.append(int(result.avg_toxicity > 0.5))

    forecaster = IncoherenceForecaster(learning_rate=0.1, n_iters=500)
    forecaster.fit(rows, labels)
    prob = forecaster.predict_proba(rows[-1])
    assert 0.0 <= prob <= 1.0
