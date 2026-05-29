"""Command line interface for AgentGit MVP."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from swarm.agentgit.bundle import build_bundle, load_bundle, verify_bundle, write_bundle
from swarm.agentgit.policy import AgentGitPolicy, gate_bundle


def cmd_attest(args: argparse.Namespace) -> int:
    policy = AgentGitPolicy.from_yaml(Path(args.policy))
    checks = _parse_checks(args.check)
    bundle = build_bundle(
        repo=Path(args.repo),
        task_id=args.task,
        agent_id=args.agent,
        policy=policy,
        base_ref=args.base,
        check_results=checks,
        signing_key=args.signing_key or os.environ.get("AGENTGIT_SIGNING_KEY"),
        signer_id=args.signer_id,
    )
    write_bundle(bundle, Path(args.output))

    passed = bundle["policy"]["passed"]
    status = "PASS" if passed else "FAIL"
    print(
        f"{status} agentgit attestation: {args.output} "
        f"({bundle['git']['totals']['changed_files']} changed files)"
    )
    return 0 if passed or args.warn_only else 1


def cmd_verify(args: argparse.Namespace) -> int:
    bundle = load_bundle(Path(args.bundle))
    ok, errors = verify_bundle(
        bundle,
        signing_key=args.signing_key or os.environ.get("AGENTGIT_SIGNING_KEY"),
        require_policy_pass=not args.allow_policy_fail,
    )
    if ok:
        print(f"PASS agentgit verify: {args.bundle}")
        return 0

    print(f"FAIL agentgit verify: {args.bundle}")
    for error in errors:
        print(f"- {error}")
    return 1


def cmd_gate(args: argparse.Namespace) -> int:
    """Enforce a CI/org-owned policy against an already-attested bundle."""
    bundle = load_bundle(Path(args.bundle))

    # The gate reads facts out of the bundle, so the bundle must be authentic
    # first: verify the signature (not the bundle's own policy — CI applies its
    # own) and fail closed on any tampering/malformed input.
    verify_ok, verify_errors = verify_bundle(
        bundle,
        signing_key=args.signing_key or os.environ.get("AGENTGIT_SIGNING_KEY"),
        require_policy_pass=False,
    )
    if not verify_ok:
        print(f"FAIL agentgit gate: {args.bundle} (bundle failed verification)")
        for error in verify_errors:
            print(f"- {error}")
        return 1

    policy = AgentGitPolicy.from_yaml(Path(args.policy))
    ok, decisions = gate_bundle(bundle, policy, trusted_overrides=args.override)
    status = "PASS" if ok else "FAIL"
    print(f"{status} agentgit gate: {args.bundle} (policy {args.policy})")
    for decision in decisions:
        if not decision.passed:
            label = "WARN" if decision.severity == "warning" else "FAIL"
            print(f"- [{label}] [{decision.policy_id}] {decision.reason}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m swarm.agentgit",
        description="Create task-scoped provenance bundles for agent-authored git diffs.",
    )
    subparsers = parser.add_subparsers(dest="subcmd")

    attest = subparsers.add_parser("attest", help="Evaluate and sign the current diff")
    attest.add_argument("--repo", default=".", help="Git repository path")
    attest.add_argument("--task", required=True, help="Delegated task identifier")
    attest.add_argument("--agent", required=True, help="Agent identity")
    attest.add_argument("--policy", required=True, help="AgentGit policy YAML")
    attest.add_argument("--base", default="HEAD", help="Base ref to diff against")
    attest.add_argument(
        "--output",
        default=".agentgit/provenance.json",
        help="Output provenance bundle path",
    )
    attest.add_argument(
        "--check",
        action="append",
        default=[],
        metavar="NAME=pass|fail",
        help="Record a required check result; may be repeated",
    )
    attest.add_argument(
        "--signing-key",
        default=None,
        help="Hex HMAC key. Defaults to AGENTGIT_SIGNING_KEY or a dev key.",
    )
    attest.add_argument("--signer-id", default="agentgit-local")
    attest.add_argument(
        "--warn-only",
        action="store_true",
        help="Write bundle but return success even when policy fails",
    )
    attest.set_defaults(func=cmd_attest)

    verify = subparsers.add_parser("verify", help="Verify an AgentGit bundle")
    verify.add_argument("bundle", help="Path to provenance bundle JSON")
    verify.add_argument(
        "--signing-key",
        default=None,
        help="Hex HMAC key. Defaults to AGENTGIT_SIGNING_KEY or a dev key.",
    )
    verify.add_argument(
        "--allow-policy-fail",
        action="store_true",
        help="Verify hash/signature even when policy failed",
    )
    verify.set_defaults(func=cmd_verify)

    gate = subparsers.add_parser(
        "gate",
        help="Enforce a CI-owned policy against a bundle's recorded facts",
    )
    gate.add_argument("--bundle", required=True, help="Path to provenance bundle JSON")
    gate.add_argument("--policy", required=True, help="CI/org-owned AgentGit policy YAML")
    gate.add_argument(
        "--signing-key",
        default=None,
        help="Hex HMAC key for bundle verification. Defaults to AGENTGIT_SIGNING_KEY.",
    )
    gate.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="RULE_ID",
        help="CI-trusted override: pass a blocking rule's id; may be repeated",
    )
    gate.set_defaults(func=cmd_gate)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    result: int = args.func(args)
    return result


def _parse_checks(raw_checks: List[str]) -> Dict[str, bool]:
    checks: Dict[str, bool] = {}
    for raw in raw_checks:
        if "=" not in raw:
            raise SystemExit(f"Invalid --check value {raw!r}; expected NAME=pass|fail")
        name, value = raw.split("=", 1)
        normalised = value.strip().lower()
        if normalised not in {"pass", "passed", "true", "fail", "failed", "false"}:
            raise SystemExit(f"Invalid check result {value!r} for {name!r}")
        checks[name.strip()] = normalised in {"pass", "passed", "true"}
    return checks


if __name__ == "__main__":
    sys.exit(main())
