# council_glue.py - wires policy, council, and vendor proposals with Qdrant
from pathlib import Path
import json, requests, os
from twilio.rest import Client
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, CollectionDescription

from ...utils.policy_engine import PolicyEngine  # Adjusted for src/utils
from ...utils.council_adapter import CouncilAdapter  # Adjusted for src/utils
from ...utils.wallet_budget_guard import WalletBudgetGuard  # Adjusted for src/utils
import vendor_adapters as vendors

BASE_DIR = "C:/Projects/LillithNew"
CONFIG_DIR = "<YOUR_CONFIG_DIR>"  # Replace with path, e.g., C:/Lillith-Evolution/config

def run_council(policy_path: str, ledger_path: str, budget_usd: float, context: dict, loki_endpoint: str = "http://loki:3100", qdrant_endpoint: str = "http://localhost:6333", qdrant_api_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.lLaMFz2dmAHeLzqzaxBIRX1a-ZBQvD2raPrKpJD0Aj4"):
    pe = PolicyEngine(policy_path)
    snapshot = pe.snapshot()
    
    # Load merge config and apply
    merge_config = json.load(open(f"{CONFIG_DIR}/council_merge.template.json"))
    weights = json.load(open(f"{CONFIG_DIR}/council_roster.defaults.json"))["weights"]
    atlas = json.load(open(f"{CONFIG_DIR}/llm_atlas.json"))
    for model in atlas:
        delegate = merge_config["delegate_map_hint"].get(model["model"])
        if delegate and delegate not in weights:
            weights[delegate] = 0.05  # Default weight for new delegates
    
    redlines = snapshot["redlines"]
    adapter = CouncilAdapter(weights, redlines, mcp_config=json.load(open(f"{CONFIG_DIR}/config.json")))
    wallet = WalletBudgetGuard(daily_cap_usd=budget_usd, ledger_path=ledger_path)

    # Initialize Qdrant client
    qdrant = QdrantClient(url=qdrant_endpoint, api_key=qdrant_api_key)
    try:
        qdrant.get_collection("council_log")
    except:
        qdrant.create_collection(
            collection_name="council_log",
            vectors_config={"size": 128, "distance": "Cosine"}
        )

    proposals = {
        "lillith": vendors.local_bert_tinyllama_proposal(context),
        "guardian": {"action":"safety_review","score":0.55,"tags":[]},
        "planner": {"action":"optimize","score":0.5,"tags":[]},
    }

    vendor_endpoints = json.load(open(f"{CONFIG_DIR}/vendor_endpoints.json"))
    for vendor, config in vendor_endpoints.items():
        if wallet.try_spend(0.05, tags=[f"advisor:{vendor}"]):
            if pe.check_net_egress(config["endpoint"]):
                try:
                    proposal = getattr(vendors, f"{vendor}_proposal", vendors.local_bert_tinyllama_proposal)(context)
                    proposals[vendor] = proposal
                    weights[vendor] = weights.get(vendor, weights.get("planner", 0.0) * 0.9)
                    # Log proposal to Qdrant
                    point = PointStruct(
                        id=str(int(time.time() * 1e9)),
                        vector=[random.random() for _ in range(128)],  # Placeholder vector
                        payload={
                            "delegate": vendor,
                            "action": proposal["action"],
                            "score": proposal["
