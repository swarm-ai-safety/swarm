# /deploy_demo

Deploy (or update) the Streamlit interactive demo to Hugging Face Spaces.

## Usage

`/deploy_demo` — full rebuild and push
`/deploy_demo --check` — just check if the Space is running and on the latest code

## Context

The demo app lives at `examples/demo/` in the main repo but the HF Space at `rsavitt/swarm-sandbox` uses a **flat layout** (`/app/` root). This command handles the path rewriting that breaks every time you push manually.

**Space URL**: https://rsavitt-swarm-sandbox.hf.space
**Space repo**: https://huggingface.co/spaces/rsavitt/swarm-sandbox

## Behavior

### `--check` mode

1. Check health endpoint:
   ```bash
   curl -s -o /dev/null -w "%{http_code}" https://rsavitt-swarm-sandbox.hf.space/_stcore/health
   ```

2. Check runtime status and deployed SHA:
   ```bash
   curl -s -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
     "https://huggingface.co/api/spaces/rsavitt/swarm-sandbox/runtime"
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

#### Step 5: Clone, commit, push

```bash
cd "$STAGE"
git init
git remote add origin https://huggingface.co/spaces/rsavitt/swarm-sandbox
git pull origin main --rebase 2>/dev/null || true
git add -A
git commit -m "Update SWARM demo ($(date +%Y-%m-%d))"
git push origin main
```

#### Step 6: Poll until running

Wait up to 5 minutes, checking every 30 seconds:

```bash
for i in $(seq 1 10); do
  sleep 30
  STATUS=$(curl -s -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
    "https://huggingface.co/api/spaces/rsavitt/swarm-sandbox/runtime" | python3 -c "import sys,json; print(json.load(sys.stdin).get('stage','UNKNOWN'))")
  echo "Attempt $i: $STATUS"
  if [ "$STATUS" = "RUNNING" ]; then
    echo "Space is live: https://rsavitt-swarm-sandbox.hf.space"
    break
  fi
done
```

#### Step 7: Verify health

```bash
HTTP=$(curl -s -o /dev/null -w "%{http_code}" https://rsavitt-swarm-sandbox.hf.space/_stcore/health)
if [ "$HTTP" = "200" ]; then
  echo "Health check: PASS"
else
  echo "Health check: FAIL (HTTP $HTTP)"
fi
```

#### Step 8: Cleanup

```bash
rm -rf "$STAGE"
```

## Error Handling

- If HF auth fails: prompt user to run `hf auth login --token <TOKEN> --add-to-git-credential`
- If git push is rejected: `git pull --rebase origin main` then retry push
- If build fails after push: check logs with `curl -s -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" "https://huggingface.co/api/spaces/rsavitt/swarm-sandbox/runtime"` and report the `stage` field

## Prerequisites

- HF CLI authenticated (`~/.cache/huggingface/token` must exist)
- Git credential helper configured for huggingface.co

## Relation to Other Commands

- `/healthcheck --hf-space` — quick status check (subset of `--check`)
- `/deploy_blog` — similar pattern but for the blog site
