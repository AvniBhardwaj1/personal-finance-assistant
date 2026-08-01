"""Run a set of predefined prompts against the project (mock or real agent).

This script tries to import the agent (if available) and fallbacks to the
MockReActLLM. It writes a JSONL file for each run in logs/experiments.jsonl.

Usage:
    python scripts/run_experiments.py --mock 1
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'logs'
LOGS.mkdir(exist_ok=True)
OUT = LOGS / 'experiments.jsonl'

from tools import load_data, get_stock_price, get_client_profile
from mock_llm import MockReActLLM


def build_mock():
    tools = {
        'GetClientProfile': lambda uid: get_client_profile(load_data(), uid),
        'GetLiveStockPrice': lambda t: get_stock_price(t)
    }
    return MockReActLLM(tools=tools)


SAMPLE_PROMPTS = [
    "What are the savings for U00001?",
    "Tell me U00002's monthly income and expenses.",
    "What is the price of AAPL?",
    "Based on U00003's income and expenses, recommend a conservative $500 investment plan.",
]


def main(use_mock: bool = True):
    runner = None
    if use_mock:
        runner = build_mock()
    else:
        # Attempt to use the real agent if available in app.py (importing app may have side effects)
        try:
            from app import agent_executor
            runner = None
        except Exception:
            runner = build_mock()

    rows = []
    for p in SAMPLE_PROMPTS:
        if runner is not None:
            out = runner.run(p)
            rows.append(out)
        else:
            # fallback: call mock
            out = build_mock().run(p)
            rows.append(out)

    # append to JSONL
    with open(OUT, 'a') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    print(f"Wrote {len(rows)} experiment rows to {OUT}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mock', action='store_true', help='Force use of mock runner')
    args = parser.parse_args()
    main(use_mock=args.mock)
