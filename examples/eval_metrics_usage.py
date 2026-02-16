"""Example usage of the evaluation metrics suite.

This example demonstrates how to use the eval metrics for measuring:
- Success rate across experiments
- Efficiency (calls per success)
- Loopiness (detecting repetitive behaviors)
- Audit effectiveness
- Deception detection
"""

from swarm.evaluation import (
    aggregate_success_metrics,
    audit_effectiveness,
    calls_per_success,
    deception_detection_rate,
    loopiness_score,
    success_rate,
)


def example_success_rate():
    """Example: Measuring success rate of agent actions."""
    print("\n=== Success Rate Example ===")

    # Simulated agent attempts at a task
    attempts = [
        {"action": "verify_claim", "success": True, "agent": "alice"},
        {"action": "verify_claim", "success": False, "agent": "bob"},
        {"action": "verify_claim", "success": True, "agent": "charlie"},
        {"action": "verify_claim", "success": True, "agent": "alice"},
        {"action": "verify_claim", "success": False, "agent": "bob"},
    ]

    rate = success_rate(attempts)
    print(f"Overall success rate: {rate:.1%}")
    print(f"Successful attempts: {sum(1 for a in attempts if a['success'])}/{len(attempts)}")


def example_calls_per_success():
    """Example: Measuring efficiency (API calls per success)."""
    print("\n=== Calls Per Success Example ===")

    # Simulated attempts with API call counts
    attempts = [
        {"success": True, "calls": 3, "agent": "alice"},
        {"success": False, "calls": 5, "agent": "bob"},
        {"success": True, "calls": 2, "agent": "charlie"},
        {"success": True, "calls": 4, "agent": "alice"},
    ]

    efficiency = calls_per_success(attempts)
    print(f"Average calls per success: {efficiency:.1f}")
    print(f"Total successes: {sum(1 for a in attempts if a['success'])}")
    print(f"Total calls (successful): {sum(a['calls'] for a in attempts if a['success'])}")


def example_loopiness():
    """Example: Detecting repetitive/stuck behavior."""
    print("\n=== Loopiness Detection Example ===")

    # Agent that makes progress
    productive_sequence = [
        "analyze", "query_db", "verify", "synthesize", "report"
    ]

    # Agent stuck in a loop
    stuck_sequence = [
        "query", "retry", "query", "retry", "query", "retry", "query", "retry"
    ]

    productive_score = loopiness_score(productive_sequence, window_size=4)
    stuck_score = loopiness_score(stuck_sequence, window_size=4)

    print(f"Productive agent loopiness: {productive_score:.2f}")
    print(f"Stuck agent loopiness: {stuck_score:.2f}")
    print(f"Stuck agent is {stuck_score / (productive_score + 0.01):.1f}x more loopy")


def example_audit_effectiveness():
    """Example: Measuring audit/governance effectiveness."""
    print("\n=== Audit Effectiveness Example ===")

    # Actions flagged by audit system
    audited_actions = [
        {"action_id": "a1", "reason": "low_quality"},
        {"action_id": "a2", "reason": "suspicious"},
        {"action_id": "a5", "reason": "low_quality"},
    ]

    # Ground truth: actually malicious actions
    malicious_actions = [
        {"action_id": "a1", "type": "spam"},
        {"action_id": "a2", "type": "deceptive"},
        {"action_id": "a4", "type": "toxic"},
    ]

    metrics = audit_effectiveness(audited_actions, malicious_actions)

    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall: {metrics['recall']:.1%}")
    print(f"F1 Score: {metrics['f1_score']:.1%}")
    print(f"Detection Rate: {metrics['detection_rate']:.1%}")
    print("\nInterpretation:")
    print(f"  - {metrics['precision']:.0%} of audited actions were actually malicious")
    print(f"  - {metrics['recall']:.0%} of malicious actions were caught")


def example_deception_detection():
    """Example: Measuring deception detection capability."""
    print("\n=== Deception Detection Example ===")

    # Agent behaviors with ground truth and detection
    behaviors = [
        {
            "agent": "alice",
            "is_deceptive": False,
            "detected_as_deceptive": False,
            "behavior": "normal_operation",
        },
        {
            "agent": "bob",
            "is_deceptive": True,
            "detected_as_deceptive": True,
            "behavior": "fake_credentials",
        },
        {
            "agent": "charlie",
            "is_deceptive": True,
            "detected_as_deceptive": False,
            "behavior": "hidden_objective",
        },
        {
            "agent": "dave",
            "is_deceptive": False,
            "detected_as_deceptive": True,
            "behavior": "unusual_pattern",
        },
    ]

    metrics = deception_detection_rate(behaviors)

    print(f"True Positive Rate (TPR): {metrics['true_positive_rate']:.1%}")
    print(f"False Positive Rate (FPR): {metrics['false_positive_rate']:.1%}")
    print(f"Overall Accuracy: {metrics['accuracy']:.1%}")
    print("\nInterpretation:")
    print(f"  - Caught {metrics['true_positive_rate']:.0%} of deceptive agents")
    print(f"  - Wrongly flagged {metrics['false_positive_rate']:.0%} of honest agents")


def example_aggregate_metrics():
    """Example: Aggregating success metrics across multiple experiments."""
    print("\n=== Aggregate Success Metrics Example ===")

    # Multiple experiment runs
    experiments = [
        {
            "name": "baseline",
            "attempts": [
                {"success": True},
                {"success": False},
                {"success": True},
            ],
        },
        {
            "name": "with_verification",
            "attempts": [
                {"success": True},
                {"success": True},
                {"success": True},
                {"success": False},
            ],
        },
        {
            "name": "adversarial",
            "attempts": [
                {"success": False},
                {"success": False},
                {"success": True},
            ],
        },
    ]

    metrics = aggregate_success_metrics(experiments)

    print(f"Mean success rate: {metrics['mean_success_rate']:.1%}")
    print(f"Std deviation: {metrics['std_success_rate']:.3f}")
    print(f"Min success rate: {metrics['min_success_rate']:.1%}")
    print(f"Max success rate: {metrics['max_success_rate']:.1%}")
    print(f"Total attempts: {metrics['total_attempts']}")
    print(f"Total successes: {metrics['total_successes']}")


def example_combined_analysis():
    """Example: Combining multiple metrics for comprehensive analysis."""
    print("\n=== Combined Analysis Example ===")

    experiment_data = {
        "attempts": [
            {"success": True, "calls": 3},
            {"success": False, "calls": 5},
            {"success": True, "calls": 2},
            {"success": True, "calls": 4},
        ],
        "action_sequence": [
            "query", "analyze", "verify", "query", "analyze", "verify",
            "query", "analyze", "verify", "report",
        ],
        "audited": [
            {"action_id": "a1"},
            {"action_id": "a2"},
        ],
        "malicious": [
            {"action_id": "a1"},
            {"action_id": "a3"},
        ],
    }

    # Compute all metrics
    s_rate = success_rate(experiment_data["attempts"])
    efficiency = calls_per_success(experiment_data["attempts"])
    loopy = loopiness_score(experiment_data["action_sequence"], window_size=3)
    audit_metrics = audit_effectiveness(
        experiment_data["audited"], experiment_data["malicious"]
    )

    print("Experiment Performance:")
    print(f"  Success Rate: {s_rate:.1%}")
    print(f"  Efficiency: {efficiency:.1f} calls/success")
    print(f"  Loopiness: {loopy:.2f}")
    print(f"  Audit Precision: {audit_metrics['precision']:.1%}")
    print(f"  Audit Recall: {audit_metrics['recall']:.1%}")

    # Overall assessment
    print("\nAssessment:")
    if s_rate > 0.7 and efficiency < 5.0 and loopy < 0.3:
        print("  ✓ Strong performance across all metrics")
    elif s_rate > 0.5:
        print("  ~ Moderate performance, room for improvement")
    else:
        print("  ✗ Performance needs attention")


if __name__ == "__main__":
    print("=" * 70)
    print("SWARM Evaluation Metrics Suite - Usage Examples")
    print("=" * 70)

    example_success_rate()
    example_calls_per_success()
    example_loopiness()
    example_audit_effectiveness()
    example_deception_detection()
    example_aggregate_metrics()
    example_combined_analysis()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
