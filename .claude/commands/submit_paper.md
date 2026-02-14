# /submit_paper

Submit a paper to ClawXiv and/or AgentXiv with validation, credential checks, and rate-limit retry.

## Usage

```
/submit_paper <paper_name> [--platform clawxiv|agentxiv|both] [--dry-run]
```

Where `<paper_name>` is one of the paper stems (e.g. `distributional_agi_safety`, `governance_mechanisms`, `collusion_dynamics`) or `all` for batch submission.

## Pre-flight Checks (run ALL before any submission)

1. **Credential validation**
   - ClawXiv: load key from `~/.config/clawxiv/swarmsafety.json` or `.credentials/agent_platforms.md`
   - AgentXiv: load key from `~/.config/agentxiv/credentials.json` (`current` entry)
   - For AgentXiv, call `GET /api/v1/agents/status` and verify status is `claimed` or `active`. If `pending_claim`, STOP and tell the user to claim the account first.

2. **Author attribution check**
   - Grep the submission `.tex` for `\author{}` content.
   - If the author is NOT "SWARM Research Collective", **auto-replace** it to "SWARM Research Collective" in the submission copy (`research/papers/`) and inform the user (do not ask — the canonical `docs/papers/` source is not modified).
   - This avoids a blocking question on every submission. The user can pass `--keep-author` to skip the replacement.

3. **Category validation**
   - ClawXiv valid categories: `cs.MA`, `cs.AI`, `cs.CL`, `cs.LG`, `cs.CR`, `cs.SE` (check via API if unsure)
   - AgentXiv valid categories: `general`, `agent-behavior`, `agent-communication`, `tool-use`, `reasoning`, `memory`, `planning`, `meta-cognition`, `alignment`, `emergent-behavior`, `human-agent-interaction`, `multi-agent-systems`

4. **Source validation**
   - Validate using the exact API below (avoids rediscovery after context compaction):
   ```python
   from swarm.research.submission import SubmissionValidator
   from swarm.research.platforms import Paper
   paper = Paper(title=..., abstract=..., source=tex, categories=['cs.MA', 'cs.AI'])
   sv = SubmissionValidator()          # instance, not classmethod
   result = sv.validate(paper)         # returns ValidationResult
   # result.passed (bool) — True if no errors
   # result.issues (list[ValidationIssue]) — each has .severity.value, .code, .message
   # result.quality_score (float) — 0-100
   ```
   - **Auto-fix section names**: If the source contains `\section{Experimental Setup}`, rename it to `\section{Methods}` before validation. Similarly rename `\section{Experimental Methods}` to `\section{Methods}`. The validator requires `\section{Methods}` or `\section{Experiments}` exactly.
   - The validator also requires a `\section{Conclusion}`. If missing, add a brief one to the submission copy before validating.
   - For papers with `\includegraphics`, verify all referenced figures exist

5. **Source location**
   - `/compile_paper` automatically writes a submission-ready copy to `research/papers/` with section renames applied. If `research/papers/<name>.tex` exists, use it directly.
   - **Fallback only**: If `research/papers/<name>.tex` does not exist (paper was compiled before this fix), copy from `docs/papers/<name>.tex` and apply the section rename from step 4.

## Submission Flow

### ClawXiv (LaTeX)

**Base URL**: `https://www.clawxiv.org` (NOT `https://clawxiv.org` — that returns a 308 redirect which breaks `urllib`).

1. Read `.tex` source from `research/papers/<name>.tex`
2. Extract abstract from `\begin{abstract}...\end{abstract}`
3. Collect referenced images as base64 from `docs/papers/figures/`
4. **Flatten figure paths**: ClawXiv rejects nested paths like `figures/subdir/file.png`. Before submission:
   - Strip all directory prefixes from `\includegraphics` paths in the tex source (e.g. `figures/kernel_v4_governance_sweep/welfare.png` → `welfare.png`)
   - Use flat filenames as image dict keys (e.g. `{"welfare.png": "<base64>"}`, NOT `{"figures/subdir/welfare.png": "<base64>"}`)
   - Apply the same regex: `tex = re.sub(r'figures/[^/]+/', '', tex)`
5. Submit using `files` payload format (not bare `source`) when images are present:
   ```python
   # New paper
   POST https://www.clawxiv.org/api/v1/papers
   json={"title": ..., "abstract": ..., "categories": [...], "files": {"source": tex_flat, "images": {name: b64, ...}}}
   ```
6. **Auth header**: ClawXiv uses `X-API-Key` header (not `Authorization: Bearer`).
7. **Rate limit auto-retry**: 1 paper per 30 minutes. On 429 response, implement automatic retry:
   ```python
   import time, json
   MAX_RETRIES = 8
   for attempt in range(MAX_RETRIES):
       try:
           resp = urllib.request.urlopen(req, timeout=60)
           result = json.loads(resp.read().decode())
           print(f"ClawXiv: SUCCESS — {result}")
           break
       except urllib.error.HTTPError as e:
           if e.code != 429:
               raise
           body = json.loads(e.read().decode())
           wait_min = body.get("retry_after_minutes", 5)
           print(f"  Rate limited — {wait_min} min remaining (attempt {attempt+1}/{MAX_RETRIES})")
           time.sleep(wait_min * 60 + 30)  # wait reported time + 30s buffer
   ```
   - Read `retry_after_minutes` from the 429 response body
   - Sleep for exactly that duration (plus 30s buffer to avoid edge cases)
   - Print progress on each retry so the user sees activity
   - Do NOT use fixed 5-minute polling — use the server-reported wait time to minimize total wait
   - After 8 failures, stop and report

### AgentXiv (Markdown)

**Base URL**: `https://agentxiv.org/api/v1`

**Endpoints** (all use `POST` to `/api/v1/tools/*`):
- New paper: `POST /api/v1/tools/submit`
- Revision: `POST /api/v1/tools/revise`
- Search: `POST /api/v1/tools/search`
- Read: `POST /api/v1/tools/read`
- Status: `GET /api/v1/agents/status`

**NOT** `/api/v1/papers` (returns 404). All paper operations go through the `/tools/` namespace.

1. Read `.md` source from `docs/papers/<name>.md`
2. Extract abstract from `## Abstract` section (text between `## Abstract` and next `## `)
3. Submit new paper:
   ```python
   POST https://agentxiv.org/api/v1/tools/submit
   json={"title": ..., "abstract": ..., "content": md, "category": "multi-agent-systems"}
   ```
4. For revisions:
   ```python
   POST https://agentxiv.org/api/v1/tools/revise
   json={"paper_id": ..., "title": ..., "abstract": ..., "content": md, "changelog": "..."}
   ```

## Paper Registry

After successful submission, print the paper ID and update this table in the output:

| Paper | ClawXiv | AgentXiv |
|---|---|---|
| distributional_agi_safety | clawxiv.2602.00058 | 2602.00043 |
| governance_mechanisms | clawxiv.2602.00051 | 2602.00044 |
| collusion_dynamics | clawxiv.2602.00057 | 2602.00045 |
| ldt_cooperation | clawxiv.2602.00069 | -- |
| pi_bridge_claude_study | clawxiv.2602.00071 | 2602.00055 |
| kernel_v4_governance_sweep | clawxiv.2602.00074 | 2602.00057 |
| ldt_acausality_depth | clawxiv.2602.00081 | 2602.00058 |
| moltbook_captcha_study | clawxiv.2602.00086 | 2602.00063 |

## Error Handling

- Always inspect `response.text` on 400 errors (the client now includes response bodies via `_error_detail`)
- Common 400 causes:
  - **"Invalid categories"**: check category name against valid list above
  - **"Account not verified"**: AgentXiv account needs human claim
  - **"LaTeX compilation failed"**: missing figures (need `files.images`) or bad LaTeX
  - **"Nested file paths are not supported"**: flatten `figures/subdir/` paths (see step 4 above)
  - **"Rate limit exceeded"**: wait and retry

## Key Files

- `research/papers/submit_batch.py` — batch submission script
- `swarm/research/platforms.py` — `ClawxivClient`, `AgentxivClient`, `Paper`
- `swarm/research/submission.py` — `SubmissionValidator`
- `~/.config/clawxiv/swarmsafety.json` — ClawXiv credentials
- `~/.config/agentxiv/credentials.json` — AgentXiv credentials
