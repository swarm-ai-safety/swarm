# /deploy_blog

Build and deploy the SWARM blog/docs site to Vercel production.

## Usage

`/deploy_blog`

## Behavior

1. Run `mkdocs build` locally to verify the site builds without errors.
   - Warnings about missing image targets are acceptable (pre-existing).
   - Actual build errors (syntax, missing plugins) must be fixed before deploying.

2. Deploy to Vercel production:
   ```bash
   vercel --prod
   ```

3. Verify deployment succeeded by checking the output for `Production:` URL.

4. Print the production URL and timestamp.

## Important Notes

- The site is hosted on **Vercel**, NOT GitHub Pages. Do not use `mkdocs gh-deploy`.
- The `.vercelignore` file excludes `.beads/`, `.git/`, `external/`, `runs/`, and cache directories. If Vercel errors on a socket or large file, check this file first.
- The `vercel.json` build command installs mkdocs-material and builds via `mkdocs build`, outputting to `site/`.
- Vercel auto-deploys on push to `main` if connected, but this command forces an immediate production deploy from the local working tree.

## Prerequisites

- `vercel` CLI installed and authenticated (`npm i -g vercel`)
- `mkdocs-material` and `pymdown-extensions` installed locally for the build check

## Constraints

- Do NOT use `--strict` flag with mkdocs build — pre-existing broken image links in papers/ will cause it to fail.
- Always deploy from the repo root directory.
