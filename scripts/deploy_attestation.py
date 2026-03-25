#!/usr/bin/env python3
"""Deploy SafetyAttestation contract to Base Sepolia or Mainnet.

Usage:
    # Deploy to Base Sepolia (testnet)
    python scripts/deploy_attestation.py --network sepolia --private-key 0x...

    # Deploy to Base Mainnet
    python scripts/deploy_attestation.py --network mainnet --private-key 0x...

    # Use env var for private key
    PRIVATE_KEY=0x... python scripts/deploy_attestation.py --network sepolia

Requires: py-solc-x, web3
    pip install py-solc-x web3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from solcx import compile_source, install_solc, set_solc_version
from web3 import Web3

NETWORKS: dict[str, dict[str, str | int]] = {
    "sepolia": {
        "rpc": "https://sepolia.base.org",
        "chain_id": 84532,
        "explorer": "https://sepolia.basescan.org",
    },
    "mainnet": {
        "rpc": "https://mainnet.base.org",
        "chain_id": 8453,
        "explorer": "https://basescan.org",
    },
}

SOLC_VERSION = "0.8.24"


def compile_contract(sol_path: Path) -> tuple[list, str]:
    """Compile the Solidity contract and return (abi, bytecode)."""
    install_solc(SOLC_VERSION)
    set_solc_version(SOLC_VERSION)

    source = sol_path.read_text()
    compiled = compile_source(source, output_values=["abi", "bin"])

    # Extract the main contract
    key = next(k for k in compiled if k.endswith(":SafetyAttestation"))
    abi = compiled[key]["abi"]
    bytecode = compiled[key]["bin"]
    return abi, bytecode


def deploy(
    abi: list,
    bytecode: str,
    rpc_url: str,
    chain_id: int,
    private_key: str,
) -> tuple[str, str]:
    """Deploy contract and return (contract_address, tx_hash)."""
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print(f"ERROR: Cannot connect to {rpc_url}", file=sys.stderr)
        sys.exit(1)

    account = w3.eth.account.from_key(private_key)
    print(f"  Deployer:  {account.address}")

    balance = w3.eth.get_balance(account.address)
    balance_eth = w3.from_wei(balance, "ether")
    print(f"  Balance:   {balance_eth} ETH")

    if balance == 0:
        print("ERROR: Zero balance. Get testnet ETH from a faucet.", file=sys.stderr)
        sys.exit(1)

    contract = w3.eth.contract(abi=abi, bytecode=bytecode)

    tx = contract.constructor().build_transaction({
        "from": account.address,
        "nonce": w3.eth.get_transaction_count(account.address),
        "gas": 1_500_000,
        "chainId": chain_id,
    })

    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    tx_hash_hex = "0x" + tx_hash.hex()
    print(f"  Tx Hash:   {tx_hash_hex}")
    print("  Waiting for confirmation...")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    if receipt["status"] != 1:
        print("ERROR: Deployment transaction failed!", file=sys.stderr)
        sys.exit(1)

    contract_address: str = str(receipt["contractAddress"])
    block = receipt["blockNumber"]
    gas_used = receipt["gasUsed"]
    print(f"  Confirmed in block {block} (gas used: {gas_used})")
    return contract_address, tx_hash_hex


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy SafetyAttestation contract")
    parser.add_argument(
        "--network",
        choices=["sepolia", "mainnet"],
        default="sepolia",
        help="Target network (default: sepolia)",
    )
    parser.add_argument("--private-key", default=None, help="Deployer private key (or PRIVATE_KEY env)")
    parser.add_argument(
        "--contract-path",
        default="contracts/SafetyAttestation.sol",
        help="Path to Solidity source",
    )
    args = parser.parse_args()

    private_key = args.private_key or os.environ.get("PRIVATE_KEY")
    if not private_key:
        print("ERROR: --private-key or PRIVATE_KEY env var required", file=sys.stderr)
        sys.exit(1)

    net = NETWORKS[args.network]
    sol_path = Path(args.contract_path)
    if not sol_path.exists():
        print(f"ERROR: Contract not found at {sol_path}", file=sys.stderr)
        sys.exit(1)

    print()
    print("SafetyAttestation Deployment")
    print("=" * 50)
    print(f"  Network:   Base {args.network.title()}")
    print(f"  Chain ID:  {net['chain_id']}")
    print(f"  RPC:       {net['rpc']}")
    print()

    # Compile
    print("Compiling contract...")
    abi, bytecode = compile_contract(sol_path)
    print(f"  ABI entries: {len(abi)}")
    print(f"  Bytecode:    {len(bytecode)} chars")
    print()

    # Deploy
    print("Deploying...")
    contract_address, tx_hash = deploy(
        abi=abi,
        bytecode=bytecode,
        rpc_url=str(net["rpc"]),
        chain_id=int(net["chain_id"]),
        private_key=private_key,
    )

    explorer_url = f"{net['explorer']}/address/{contract_address}"

    print()
    print("DEPLOYMENT COMPLETE")
    print("=" * 50)
    print(f"  Contract:  {contract_address}")
    print(f"  Explorer:  {explorer_url}")
    print(f"  Tx:        {net['explorer']}/tx/{tx_hash}")
    print()

    # Save deployment info
    deploy_info = {
        "network": args.network,
        "chain_id": net["chain_id"],
        "contract_address": contract_address,
        "deployer_tx": tx_hash,
        "explorer_url": explorer_url,
        "abi_entries": len(abi),
    }
    deploy_file = Path(f"contracts/deployment_{args.network}.json")
    deploy_file.write_text(json.dumps(deploy_info, indent=2) + "\n")
    print(f"Deployment info saved to {deploy_file}")

    # Print next steps
    print()
    print("Next steps:")
    print("  python swarm_gym_cli.py generate -o demo.jsonl")
    print("  python swarm_gym_cli.py audit --file demo.jsonl --agent-id agent_0")
    print("  python swarm_gym_cli.py attest --file demo.jsonl --agent-id agent_0 \\")
    print(f"      --contract {contract_address} --private-key $PRIVATE_KEY \\")
    if args.network == "sepolia":
        print(f"      --rpc {net['rpc']} --chain-id {net['chain_id']}")
    print()


if __name__ == "__main__":
    main()
