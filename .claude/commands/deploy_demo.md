# /deploy_demo

Deploy (or update) the Streamlit interactive demo to Hugging Face Spaces.

## Usage

`/deploy_demo` — full rebuild and push
`/deploy_demo --check` — just check if the Space is running and on the latest code

## Context

The demo app lives at `examples/demo/` in the main repo but the HF Space at `Swarm-AI-Research/swarm-sandbox` uses a **flat layout** (`/app/` root). This command handles the path rewriting that breaks every time you push manually.

**Space URL**: https://Swarm-AI-Research-swarm-sandbox.hf.space
**Space repo**: https://huggingface.co/spaces/Swarm-AI-Research/swarm-sandbox

## Behavior

### `--check` mode

1. Check health endpoint:
   ```bash
   curl -s -o /dev/null -w "%{http_code}" https://Swarm-AI-Research-swarm-sandbox.hf.space/_stcore/health
   ```

2. Check runtime status and deployed SHA:
   ```bash
   curl -s -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
     "https://huggingface.co/api/spaces/Swarm-AI-Research/swarm-sandbox/runtime"
   ```

3. Report: stage (RUNNING/BUILDING/ERROR), deployed SHA, health status.

### Full deploy (default)

#### Step 1: Prepare staging directory

```bash
STAGE=$(mktemp -d)
```

Copy the required files from the main repo:

```bash
cp examples/demo/app.py "$STAGE/app.py"
cp -r examples/demo/pages "$STAGE/pages"
cp -r examples/demo/utils "$STAGE/utils"
cp -r swarm "$STAGE/swarm"
cp -r scenarios "$STAGE/scenarios"
cp pyproject.toml "$STAGE/pyproject.toml"
mkdir -p "$STAGE/.streamlit"
cp .streamlit/config.toml "$STAGE/.streamlit/config.toml"
```

Clean pycache:
```bash
find "$STAGE" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find "$STAGE" -name "*.pyc" -delete 2>/dev/null
```

#### Step 2: Rewrite paths for flat layout

The main repo has `examples/demo/app.py` (3 levels deep) but the Space has `app.py` at root. Fix `PROJECT_ROOT` in all files:

**`app.py`**: `Path(__file__).resolve().parent` (1 parent — app.py is at Space root)

**`utils/simulation.py`**: `Path(__file__).resolve().parent.parent` (2 parents — utils/ is one level down)

**`pages/*.py`**: `Path(__file__).resolve().parent.parent` (2 parents — pages/ is one level down)

Apply with sed:
```bash
# app.py: any chain of .parent → just .parent
sed -i '' 's/Path(__file__).resolve().parent.parent.parent/Path(__file__).resolve().parent/' "$STAGE/app.py"

# utils/simulation.py: 4 parents → 2
sed -i '' 's/Path(__file__).resolve().parent.parent.parent.parent/Path(__file__).resolve().parent.parent/' "$STAGE/utils/simulation.py"

# pages/*.py: 3 parents → 2 (also handle non-resolve variants)
sed -i '' 's/Path(__file__).resolve().parent.parent.parent/Path(__file__).resolve().parent.parent/' "$STAGE/pages/"*.py
sed -i '' 's/Path(__file__).parent.parent.parent/Path(__file__).resolve().parent.parent/' "$STAGE/pages/"*.py
```

#### Step 3: Rewrite imports

```bash
sed -i '' 's/from demo\.utils\./from utils./g' "$STAGE/pages/"*.py
```

Verify no `demo.` references remain:
```bash
grep -r "from demo\." "$STAGE/pages/" "$STAGE/utils/" && echo "ERROR: demo. imports remain" && exit 1
```

#### Step 4: Write Dockerfile and requirements.txt

**`Dockerfile`**:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir .

EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=7860", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
```

**`requirements.txt`**:
```
streamlit>=1.30
plotly>=5.0
numpy>=1.24
pydantic>=2.0
pyyaml>=6.0
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
```

#### Step 5: Clone HF Space, replace tree, commit, push

The previous `git init` + `git pull --rebase 2>/dev/null || true` pattern fails on every deploy after the first: when the staging dir already has files that exist in `origin/main`, the post-init pull aborts with "untracked working tree files would be overwritten by merge", leaves HEAD unset, and the subsequent commit+push creates an orphan history that origin rejects as non-fast-forward.

Instead, clone the live Space (which gives a working copy with proper history), wipe its tracked files, drop our staged content in place, and commit on top.

```bash
WORK=$(mktemp -d)
git clone https://huggingface.co/spaces/Swarm-AI-Research/swarm-sandbox "$WORK"
cd "$WORK"
# Remove all tracked files except README.md (HF Space card) and .gitattributes (LFS rules)
git ls-files | grep -vE "^(README\.md|\.gitattributes)$" | xargs rm -f
# Drop now-empty directories
find . -type d -empty -not -path "./.git*" -delete 2>/dev/null
# Copy staged content into the working clone (preserves README.md and .gitattributes)
cp -r "$STAGE"/. ./
# Clean macOS / pycache artifacts that may have come along
find . -name ".DS_Store" -delete 2>/dev/null
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
git add -A
git commit -m "Update SWARM demo ($(date +%Y-%m-%d))"
git push origin main
```

#### Step 6: Poll until running

Wait up to 5 minutes, checking every 30 seconds. Verify both `stage=RUNNING` AND that the deployed `sha` matches what we just pushed — otherwise the API will report `RUNNING` for the previous container while the new build is still in progress.

```bash
EXPECTED_SHA=$(git -C "$WORK" rev-parse HEAD)
for i in $(seq 1 10); do
  sleep 30
  RESP=$(curl -s -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
    "https://huggingface.co/api/spaces/Swarm-AI-Research/swarm-sandbox/runtime")
  STATUS=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('stage','UNKNOWN'))")
  SHA=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('sha',''))")
  echo "Attempt $i: stage=$STATUS sha=${SHA:0:10}"
  if [ "$STATUS" = "RUNNING" ] && [ "$SHA" = "$EXPECTED_SHA" ]; then
    echo "Space is live with new SHA: https://Swarm-AI-Research-swarm-sandbox.hf.space"
    break
  fi
  case "$STATUS" in
    *ERROR*) echo "FAILED: stage=$STATUS"; break ;;
  esac
done
```

#### Step 7: Verify health

```bash
HTTP=$(curl -s -o /dev/null -w "%{http_code}" https://Swarm-AI-Research-swarm-sandbox.hf.space/_stcore/health)
if [ "$HTTP" = "200" ]; then
  echo "Health check: PASS"
else
  echo "Health check: FAIL (HTTP $HTTP)"
fi
```

#### Step 8: Cleanup

```bash
rm -rf "$STAGE" "$WORK"
```

## Error Handling

- If HF auth fails: prompt user to run `hf auth login --token <TOKEN> --add-to-git-credential`
- If git push is rejected (non-fast-forward): in the clone-based flow this means someone pushed to the HF Space remote between our `git clone` and `git push` — re-run from Step 5 (re-clone, re-stage, re-push) rather than force-pushing
- If build fails after push: check logs with `curl -s -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" "https://huggingface.co/api/spaces/Swarm-AI-Research/swarm-sandbox/runtime"` and report the `stage` field

## Prerequisites

- HF CLI authenticated (`~/.cache/huggingface/token` must exist)
- Git credential helper configured for huggingface.co

## Relation to Other Commands

- `/healthcheck --hf-space` — quick status check (subset of `--check`)
- `/deploy_blog` — similar pattern but for the blog site
