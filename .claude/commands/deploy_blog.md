# /deploy_blog

Build and deploy the SWARM blog/docs site to Vercel production.

## Usage

`/deploy_blog` or `/deploy_blog --verify [URL ...]`

## Behavior

1. Run `mkdocs build` locally to verify the site builds without errors.
   - Warnings about missing image targets are acceptable (pre-existing).
   - Actual build errors (syntax, missing plugins) must be fixed before deploying.

2. Deploy to Vercel production:
   ```bash
   vercel --prod
   ```

3. Verify deployment succeeded by checking the output for `Production:` URL.

4. **Render-verify** any page that has client-side JavaScript (charts, live
   dashboards, anything that fetches data after load). HTTP 200 is NOT proof a
   page renders — `curl`/WebFetch do not execute JS, so a deploy can report
   READY while the page is visually broken. Use the headless-Chrome verifier:

   ```bash
   scripts/render-verify.sh "https://www.swarm-ai.org/bridges/gitlawb/" \
     --id-nonempty gl-count \
     --expect-text "Quality Distribution"
   ```

   The script renders the page in headless Chrome (running its JS), then:
   - flags any **runaway `<canvas>`** (backing dims past a sane cap) — the
     failure mode that silently blanked the Gitlawb dashboard in May 2026;
   - asserts JS-driven elements left their placeholder (`--id-nonempty ID`
     fails if text is empty / `0` / `--` / `Connecting...`);
   - asserts expected/forbidden text (`--expect-text` / `--reject-text`);
   - saves a screenshot for a human to eyeball.

   Run it for every JS-bearing page touched by the deploy. A non-zero exit
   means the page is broken even though the deploy "succeeded" — investigate
   before declaring the deploy done. Pure static Markdown pages don't need it.

5. Print the production URL, timestamp, and the render-verify summary.

### `--verify` mode

`/deploy_blog --verify [URL ...]` runs **only** step 4 against the given URLs
(or the dashboard pages under `docs/bridges/` if none are given) without
rebuilding or redeploying. Use it to re-check an already-live page, or after
someone else deploys.

## Important Notes

- The site is hosted on **Vercel**, NOT GitHub Pages. Do not use `mkdocs gh-deploy`.
- The `.vercelignore` file excludes `.beads/`, `.git/`, `external/`, `runs/`, and cache directories. If Vercel errors on a socket or large file, check this file first.
- The `vercel.json` build command installs mkdocs-material and builds via `mkdocs build`, outputting to `site/`.
- Vercel auto-deploys on push to `main` if connected, but this command forces an immediate production deploy from the local working tree.

## Prerequisites

- `vercel` CLI installed and authenticated (`npm i -g vercel`)
- `mkdocs-material` and `pymdown-extensions` installed locally for the build check
- Google Chrome or Chromium installed for `--verify` / step 4 (the verifier
  auto-detects the macOS Chrome app or a `google-chrome`/`chromium` on PATH)

## Why render-verification exists (step 4)

In May 2026 the SWARM-Gitlawb dashboard deployed cleanly — `vercel --prod`
returned READY, every URL returned HTTP 200, and the snapshot JSON was correct.
But the **Quality Distribution chart was invisible in browsers**: `drawBarChart`
set `canvas.height = canvas.offsetHeight * 2` with no CSS height cap, so the
canvas doubled in size on every redraw and, across ~40 backfill redraws, blew
past the browser's maximum bitmap size and rendered as a blank white box that
hid the rest of the page.

None of `curl`, WebFetch, or "HTTP 200" could have caught this, because they
don't run JavaScript. Only a headless render does. `scripts/render-verify.sh`
encodes that lesson — its runaway-canvas check fails on exactly this bug
(verified: it flags the reproduced canvas at 67M×67M px). Treat a green deploy
as unverified until JS-bearing pages pass render-verify.

## Constraints

- Do NOT use `--strict` flag with mkdocs build — pre-existing broken image links in papers/ will cause it to fail.
- Always deploy from the repo root directory.
