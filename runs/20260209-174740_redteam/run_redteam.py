"""Red-team evaluation runner for LDT agent PR security check."""

import json
import random
import sys
from pathlib import Path

# Ensure reproducibility
random.seed(42)

from swarm.redteam import RedTeamEvaluator, AttackLibrary

# Governance configuration matching the LDT scenario (minimal governance)
ldt_governance_config = {
    "circuit_breaker_enabled": False,
    "collusion_detection_enabled": False,
    "audit_enabled": False,
    "staking_enabled": False,
    "transaction_tax_rate": 0.0,
}

# Also test with hardened governance
hardened_governance_config = {
    "circuit_breaker_enabled": True,
    "collusion_detection_enabled": True,
    "audit_enabled": True,
    "staking_enabled": True,
    "transaction_tax_rate": 0.05,
}

run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/redteam_latest")
run_dir.mkdir(parents=True, exist_ok=True)


def run_evaluation(name, governance_config):
    """Run full red-team evaluation with all attacks."""
    random.seed(42)
    evaluator = RedTeamEvaluator(
        governance_config=governance_config,
        attack_scenarios=AttackLibrary.get_all_attacks(),
    )

    # Use evaluate() with a dummy orchestrator factory since the evaluator
    # uses _simulate_attack_outcome internally.
    report = evaluator.evaluate(
        orchestrator_factory=lambda config: None,
        epochs_per_attack=20,
        verbose=True,
    )

    return report


# Run both configurations
print("=" * 60)
print("RED-TEAM EVALUATION: LDT Scenario (minimal governance)")
print("=" * 60)
report_minimal = run_evaluation("minimal", ldt_governance_config)

print()
print("=" * 60)
print("RED-TEAM EVALUATION: Hardened Governance")
print("=" * 60)
report_hardened = run_evaluation("hardened", hardened_governance_config)

# Write reports
report_json = {
    "minimal_governance": report_minimal.to_dict(),
    "hardened_governance": report_hardened.to_dict(),
}

with open(run_dir / "report.json", "w") as f:
    json.dump(report_json, f, indent=2, default=str)

# Write human-readable report
report_text = []
report_text.append("RED-TEAM SECURITY EVALUATION FOR LDT AGENT PR")
report_text.append("=" * 60)
report_text.append("")
report_text.append("CONFIGURATION 1: LDT Scenario (minimal governance)")
report_text.append(report_minimal.generate_summary())
report_text.append("")
report_text.append("CONFIGURATION 2: Hardened Governance")
report_text.append(report_hardened.generate_summary())

with open(run_dir / "report.txt", "w") as f:
    f.write("\n".join(report_text))

# Print summary
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print()
print(f"Minimal Governance: Score={report_minimal.robustness.robustness_score:.2f} Grade={report_minimal.robustness.grade}")
print(f"  Attacks tested: {report_minimal.robustness.attacks_tested}")
print(f"  Attacks prevented: {report_minimal.robustness.attacks_prevented}")
print(f"  Attacks successful: {report_minimal.robustness.attacks_successful}")
print(f"  Overall evasion rate: {report_minimal.robustness.overall_evasion_rate:.2%}")
print(f"  Total damage allowed: {report_minimal.robustness.total_damage_allowed:.1f}")
print(f"  Most effective attack: {report_minimal.most_effective_attack}")
print()
print(f"Hardened Governance: Score={report_hardened.robustness.robustness_score:.2f} Grade={report_hardened.robustness.grade}")
print(f"  Attacks tested: {report_hardened.robustness.attacks_tested}")
print(f"  Attacks prevented: {report_hardened.robustness.attacks_prevented}")
print(f"  Attacks successful: {report_hardened.robustness.attacks_successful}")
print(f"  Overall evasion rate: {report_hardened.robustness.overall_evasion_rate:.2%}")
print(f"  Total damage allowed: {report_hardened.robustness.total_damage_allowed:.1f}")
print(f"  Most effective attack: {report_hardened.most_effective_attack}")
print()

# Print vulnerabilities
for config_name, report in [("MINIMAL", report_minimal), ("HARDENED", report_hardened)]:
    vulns = report.robustness.vulnerabilities
    if vulns:
        print(f"VULNERABILITIES ({config_name}):")
        for v in vulns:
            print(f"  [{v.severity.upper()}] {v.vulnerability_id}")
            print(f"    {v.description}")
            print(f"    Affected lever: {v.affected_lever}")
            print(f"    Mitigation: {v.mitigation}")
            print()

# Print recommendations
for config_name, report in [("MINIMAL", report_minimal), ("HARDENED", report_hardened)]:
    recs = report.robustness.recommendations
    if recs:
        print(f"RECOMMENDATIONS ({config_name}):")
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r}")
        print()

print(f"Full reports written to: {run_dir}/report.json and {run_dir}/report.txt")
