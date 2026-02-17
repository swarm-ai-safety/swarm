# /security-review

Perform a security-focused code review of the current branch's changes against main.

## Usage

`/security-review`

## Behavior

### Phase 1: Gather context

1. Run `git status` and `git log --oneline main..HEAD` to understand the branch scope.
2. Run `git diff main...HEAD` to get the full diff.
3. Identify modified files and their languages/frameworks.
4. Use file search tools to understand existing security patterns in the codebase (sanitization helpers, auth middleware, validation patterns).

### Phase 2: Identify vulnerabilities

Launch a subagent to analyze the diff for security issues. Focus on:

- **Input validation**: SQL injection, command injection, path traversal, XXE, template injection
- **Auth/authz**: authentication bypass, privilege escalation, session flaws, JWT issues
- **Crypto**: hardcoded secrets, weak algorithms, improper key management
- **Code execution**: unsafe deserialization (pickle, YAML), eval/exec injection, RCE
- **Data exposure**: PII logging, sensitive data in responses, debug info leakage
- **Web**: XSS (only if using `dangerouslySetInnerHTML` or similar), CSRF, SSRF (host/protocol control only)

Only flag issues with >70% confidence of actual exploitability. Skip:
- DOS / resource exhaustion
- Secrets on disk (handled by `/scan_secrets`)
- Rate limiting concerns
- Theoretical race conditions
- Log spoofing
- Path-only SSRF
- Regex injection/DOS
- Missing hardening (only flag concrete vulns)
- Test-only code
- Environment variables and CLI flags (trusted inputs)
- Client-side JS/TS auth checks (server responsibility)

### Phase 3: Validate findings

For each candidate vulnerability, launch a **parallel** validation subagent that:

1. Reads the actual source code at the flagged location
2. Traces the data flow from input to sink
3. Checks if the input is attacker-controlled vs config/env (trusted)
4. Checks if existing sanitization/validation already mitigates the issue
5. Assigns a confidence score (1-10):
   - 1-3: Likely false positive
   - 4-6: Needs investigation
   - 7-10: Likely true vulnerability

### Phase 4: Filter and report

Discard any finding with confidence < 8. Present remaining findings as:

```markdown
# Vuln N: <Category>: `<file>:<line>`

* Severity: High | Medium
* Confidence: N/10
* Description: ...
* Exploit Scenario: ...
* Recommendation: ...
```

If no findings survive filtering, report:

> No high-confidence security vulnerabilities found in this branch's changes.

## Constraints

- Read-only: do not modify any files or run commands beyond git/gh.
- Only review code newly added or modified by this branch (not pre-existing issues).
- Minimize false positives: better to miss a theoretical issue than flood with noise.
- Do not report issues in markdown/documentation files.
- React/Angular components are safe from XSS unless using unsafe methods.
