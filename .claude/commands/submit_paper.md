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
   - Grep all `.tex` and `.md` paper source files for personal names (anything in `\author{}` or `**Authors:**` that isn't "SWARM Research Collective")
   - If found, STOP and ask the user before submitting

3. **Category validation**
   - ClawXiv valid categories: `cs.MA`, `cs.AI`, `cs.CL`, `cs.LG`, `cs.CR`, `cs.SE` (check via API if unsure)
   - AgentXiv valid categories: `general`, `agent-behavior`, `agent-communication`, `tool-use`, `reasoning`, `memory`, `planning`, `meta-cognition`, `alignment`, `emergent-behavior`, `human-agent-interaction`, `multi-agent-systems`

4. **Source validation**
   - Run `SubmissionValidator.validate(paper)` for LaTeX papers
   - **Auto-fix section names**: If the source contains `\section{Experimental Setup}`, rename it to `\section{Methods}` before validation. Similarly rename `\section{Experimental Methods}` to `\section{Methods}`. The validator requires `\section{Methods}` or `\section{Experiments}` exactly.
   - For papers with `\includegraphics`, verify all referenced figures exist

5. **Source location**
   - If `research/papers/<name>.tex` does not exist but `docs/papers/<name>.tex` does, copy it to `research/papers/` and apply the section rename above. The compile pipeline writes to `docs/papers/` but submission reads from `research/papers/`.

## Submission Flow

### ClawXiv (LaTeX)
1. Read `.tex` source from `research/papers/<name>.tex`
2. Extract abstract from `\begin{abstract}...\end{abstract}`
3. Collect referenced images as base64 from `docs/papers/figures/`
4. Submit using `files` payload format (not bare `source`) when images are present:
   ```python
   json={"title": ..., "abstract": ..., "categories": [...], "files": {"source": tex, "images": {name: b64, ...}}}
   ```
5. **Rate limit**: 1 paper per 30 minutes. If 429, retry with 5-minute polling (up to 8 attempts)

### AgentXiv (Markdown)
1. Read `.md` source from `docs/papers/<name>.md`
2. Extract abstract from `## Abstract` section
3. Submit with `arxiv_id` field for revisions, `content` for source:
   ```python
   json={"title": ..., "abstract": ..., "content": md, "category": "multi-agent-systems"}
   ```
4. For revisions, use `POST /api/v1/tools/revise` with `arxiv_id`, `content`, `changelog`

## Paper Registry

After successful submission, print the paper ID and update this table in the output:

| Paper | ClawXiv | AgentXiv |
|---|---|---|
| distributional_agi_safety | clawxiv.2602.00058 | 2602.00043 |
| governance_mechanisms | clawxiv.2602.00051 | 2602.00044 |
| collusion_dynamics | clawxiv.2602.00057 | 2602.00045 |
| ldt_cooperation | clawxiv.2602.00069 | -- |

## Error Handling

- Always inspect `response.text` on 400 errors (the client now includes response bodies via `_error_detail`)
- Common 400 causes:
  - **"Invalid categories"**: check category name against valid list above
  - **"Account not verified"**: AgentXiv account needs human claim
  - **"LaTeX compilation failed"**: missing figures (need `files.images`) or bad LaTeX
  - **"Rate limit exceeded"**: wait and retry

## Key Files

- `research/papers/submit_batch.py` — batch submission script
- `swarm/research/platforms.py` — `ClawxivClient`, `AgentxivClient`, `Paper`
- `swarm/research/submission.py` — `SubmissionValidator`
- `~/.config/clawxiv/swarmsafety.json` — ClawXiv credentials
- `~/.config/agentxiv/credentials.json` — AgentXiv credentials
