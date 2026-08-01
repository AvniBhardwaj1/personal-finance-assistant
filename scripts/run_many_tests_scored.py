"""Load the 100-case mock run log, compute deterministic expected answers, score semantic similarity
with sentence-transformers, and save scored outputs and summary.

Usage:
    python3 scripts/run_many_tests_scored.py --input logs/experiments_100.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import load_data, get_client_profile

# lazy import sentence-transformers inside main to allow early failure handling

def derive_expected(prompt, df):
    import re
    parts = []
    # find user ids like U00001
    uids = re.findall(r"\bU\d{5}\b", prompt.upper())
    for uid in uids:
        profile = get_client_profile(df, uid)
        if isinstance(profile, dict) and 'message' not in profile:
            savings = profile.get('savings_usd', 'N/A')
            income = profile.get('monthly_income_usd', 'N/A')
            parts.append(f"Client {uid} savings: {savings}; income: {income}.")
        else:
            parts.append(profile.get('message', 'Client not found.'))
    # find tickers (simple heuristic: uppercase letters 1-5 not starting with U)
    tickers = re.findall(r"\b([A-Z]{1,5})\b", prompt)
    # filter out Uxxxxx
    tickers = [t for t in tickers if not (t.startswith('U') and len(t) > 1 and t[1:].isdigit())]
    # common tickers only
    common = {'AAPL','MSFT','GOOG','AMZN','TSLA','NVDA','META'}
    for t in tickers:
        if t in common:
            parts.append(f"The current (mock) price for {t} is $123.45.")
    if not parts:
        # fallback generic expected answer
        parts.append("I can help with financial advice. Please ask about a client (e.g., U00001) or a stock ticker (e.g., AAPL).")
    return " \n".join(parts)


def cosine_sim(v1, v2):
    import numpy as np
    v1 = np.array(v1)
    v2 = np.array(v2)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def main(input_path: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    df_data = load_data()
    inp = Path(input_path)
    if not inp.exists():
        print('Input file not found:', inp)
        sys.exit(1)
    rows = [json.loads(l) for l in inp.read_text().strip().splitlines()]
    scored = []
    sims = []
    sims_client = []
    sims_stock = []
    for r in rows:
        prompt = r.get('prompt','')
        final = r.get('final_answer','')
        expected = derive_expected(prompt, df_data)
        # encode
        emb_final = model.encode(final)
        emb_expected = model.encode(expected)
        sim = cosine_sim(emb_final, emb_expected)
        r['expected_answer'] = expected
        r['similarity'] = sim
        scored.append(r)
        sims.append(sim)
        # classify prompt type
        if 'Client' in expected or 'savings' in expected.lower():
            sims_client.append(sim)
        else:
            sims_stock.append(sim)

    out_path = inp.parent / (inp.stem + '_scored.jsonl')
    with open(out_path, 'w') as f:
        for r in scored:
            f.write(json.dumps(r) + '\n')

    summary = {
        'count': len(scored),
        'avg_similarity': sum(sims)/len(sims) if sims else 0.0,
        'avg_similarity_client': sum(sims_client)/len(sims_client) if sims_client else None,
        'avg_similarity_stock': sum(sims_stock)/len(sims_stock) if sims_stock else None,
    }
    summary_path = inp.parent / (inp.stem + '_scored_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('Wrote scored results to', out_path)
    print('Summary:', summary)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='logs/experiments_100.jsonl')
    args = parser.parse_args()
    main(args.input)
