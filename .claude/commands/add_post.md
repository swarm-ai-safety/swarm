# /add_post

Add a blog post to the mkdocs website (swarm-ai.org/blog/).

## Usage

`/add_post <title> [source-file]`

Examples:
- `/add_post "Collusion Detection Deep Dive"` (write from scratch)
- `/add_post "When Agents Collapse" docs/posts/swarm_blog_post.md` (import existing file)

## Behavior

1) **Derive slug** from the title:
- Lowercase, replace spaces with hyphens, strip punctuation.
- e.g. "Collusion Detection Deep Dive" → `collusion-detection-deep-dive`

2) **Create the post file** at `docs/blog/<slug>.md`:
- If `<source-file>` is provided, copy its content and fix image paths to be relative to `docs/blog/` (e.g. `../papers/figures/...`).
- If no source file, create a skeleton with the title as H1, a `*subtitle*` italic line, a `---` separator, and placeholder sections.

3) **Update the blog index** at `docs/blog/index.md`:
- Add a bullet entry linking to the new post with the title and a one-line description.
- Insert newest post at the top of the list.

4) **Update mkdocs.yml nav**:
- Add the new post under the `Blog:` nav section.
- Place it after the index entry.

5) **Verify the build**:
- Run `python -m mkdocs build --strict` and report any errors.
- If the build fails (e.g. broken image path), fix the issue before proceeding.

6) **Show what changed**:
- Print the list of files created/modified.
- Do NOT commit or push — let the user decide when to ship (via `/ship` or manual commit).

## Constraints

- Never overwrite an existing post file without asking.
- Always verify image paths resolve correctly (relative to `docs/blog/`).
- If `docs/blog/index.md` doesn't exist, create it with a header.
- If the `Blog:` section doesn't exist in `mkdocs.yml` nav, create it.
