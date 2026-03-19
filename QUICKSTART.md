# SwarmGym Quickstart

SwarmGym is an on-chain safety auditor for multi-agent systems. It computes distributional safety metrics from interaction logs and attests them on Base.

## Install

```bash
pip install -e ".[dev,runtime]"
pip install py-solc-x web3
```

## 1. Generate synthetic interactions

```bash
python swarm_gym_cli.py generate --agents 3 --interactions 50 -o demo.jsonl
```

Creates a JSONL file with 50 interactions across 3 agents, mixing benign (60%), uncertain (25%), and toxic (15%) behaviors.

## 2. Audit an agent

```bash
# Human-readable report
python swarm_gym_cli.py audit --file demo.jsonl --agent-id agent_0

# JSON output
python swarm_gym_cli.py audit --file demo.jsonl --agent-id agent_0 --json
```

Returns safety grade (A-F), toxicity, quality gap, adverse selection flag, and a SHA-256 attestation hash.

## 3. Deploy the contract (one-time)

Get testnet ETH from the [Base Sepolia faucet](https://www.alchemy.com/faucets/base-sepolia).

```bash
python scripts/deploy_attestation.py --network sepolia --private-key 0xYOUR_KEY
```

Saves the contract address to `contracts/deployment_sepolia.json`.

## 4. Attest on-chain

```bash
python swarm_gym_cli.py attest \
    --file demo.jsonl \
    --agent-id agent_0 \
    --contract 0xCONTRACT_ADDRESS \
    --private-key 0xYOUR_KEY \
    --rpc https://sepolia.base.org \
    --chain-id 84532
```

Runs the audit and submits the metrics hash on-chain in one step.

## 5. Verify on-chain

```bash
python swarm_gym_cli.py verify \
    --hash 0xMETRICS_HASH \
    --agent-id agent_0 \
    --contract 0xCONTRACT_ADDRESS \
    --rpc https://sepolia.base.org \
    --chain-id 84532
```

Checks that the attestation exists on-chain and matches the agent.

## Architecture

```
Interaction Logs → SwarmGym Audit → Metrics + SHA-256 Hash → Base Contract
                   (SoftMetrics)    (grade, toxicity,        (SafetyAttestation.sol)
                                     quality gap,
                                     adverse selection)
```

**Metrics computed:**
- Toxicity: `E[1-p | accepted]` — expected harm from accepted interactions
- Quality gap: `E[p | accepted] - E[p | rejected]` — negative = adverse selection
- Safety grade: A (safe) through F (unsafe)
- Attestation hash: deterministic SHA-256 of canonical metrics JSON

**On-chain contract** (Solidity ^0.8.24):
- Stores metrics hash, agent ID, grade, adverse selection flag, interaction count
- Prevents duplicate hashes
- Supports per-agent attestation history
- `verify()` checks hash + agent match

## API

The auditor is also available as a REST endpoint:

```bash
POST /api/v1/audits/compute
Content-Type: application/json

{
  "interactions": [...],
  "agent_id": "agent_0"
}
```
