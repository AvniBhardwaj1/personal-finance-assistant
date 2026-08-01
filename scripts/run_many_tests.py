"""Generate N test prompts and run them through the MockReActLLM with deterministic stock prices.

Usage:
    python3 scripts/run_many_tests.py --count 100

Outputs:
  logs/experiments_100.jsonl
  logs/experiments_100_summary.json
  logs/experiments_100_summary.txt
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# ensure project root is on sys.path so imports work when script is run from scripts/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import load_data, get_client_profile
from mock_llm import MockReActLLM
LOGS = ROOT / 'logs'
LOGS.mkdir(exist_ok=True)
OUT = LOGS / 'experiments_100.jsonl'
SUMMARY_JSON = LOGS / 'experiments_100_summary.json'
SUMMARY_TXT = LOGS / 'experiments_100_summary.txt'


def build_deterministic_tools():
    df = load_data()
    def profile_tool(uid):
        return get_client_profile(df, uid)
    def price_tool(t):
        # deterministic mock price
        return f"The current (mock) price for {t} is $123.45."
    return {'GetClientProfile': profile_tool, 'GetLiveStockPrice': price_tool}


def generate_prompts(df, count=100):
    prompts = []
    # collect user ids from df
    uids = []
    if not df.empty:
        uid_col = next((c for c in df.columns if 'user' in c and 'id' in c), df.columns[0])
        uids = df[uid_col].astype(str).unique().tolist()
    # list of common tickers
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META']

    # templates
    client_templates = [
        "What are the savings for {uid}?",
        "Tell me {uid}'s monthly income and expenses.",
        "Based on {uid}'s income and expenses, recommend a conservative $500 investment plan.",
        "What is {uid}'s debt-to-income ratio and credit score?",
        "How much can {uid} save in 6 months if expenses are unchanged?",
        "Assess {uid}'s financial health and give a short recommendation."
    ]
    stock_templates = [
        "What is the current price of {ticker}?",
        "Give me the recent close price for {ticker}.",
        "Is {ticker} a good buy today? Provide only the current price: {ticker}.",
    ]

    # populate
    i = 0
    uid_index = 0
    while i < count:
        if i % 3 == 0 and uids:
            uid = uids[uid_index % len(uids)]
            tpl = client_templates[i % len(client_templates)]
            prompts.append(tpl.format(uid=uid))
            uid_index += 1
        else:
            t = tickers[i % len(tickers)]
            tpl = stock_templates[i % len(stock_templates)]
            prompts.append(tpl.format(ticker=t))
        i += 1
    return prompts


def run(prompts, use_real=False):
    rows = []
    if use_real:
        # try to import real agent executor from app
        try:
            # Ensure project root on path
            import sys
            ROOT = Path(__file__).resolve().parents[1]
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from app import agent_executor
            real_available = agent_executor is not None
        except Exception as e:
            print('Real agent not available:', e)
            real_available = False

        if real_available:
            import subprocess, shlex
            for p in prompts:
                try:
                    # use ollama CLI to run the model for each prompt
                    # prefer JSON output if model supports it; otherwise capture text
                    cmd = f"ollama run my-finance-bot --format json {shlex.quote(p)}"
                    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
                    stdout = proc.stdout.strip()
                    if stdout:
                        # try parse json
                        try:
                            j = json.loads(stdout)
                            # attempt to extract a sensible text field
                            if isinstance(j, dict) and 'output' in j:
                                text = j.get('output')
                            else:
                                text = stdout
                        except Exception:
                            text = stdout
                    else:
                        text = proc.stderr.strip() or 'No output from model'
                    out = {'prompt': p, 'final_answer': text}
                except Exception as e:
                    out = {'prompt': p, 'final_answer': f'Agent error: {e}'}
                rows.append(out)
        else:
            # fallback to mock
            tools = build_deterministic_tools()
            mock = MockReActLLM(tools=tools)
            for p in prompts:
                out = mock.run(p)
                rows.append(out)
    else:
        tools = build_deterministic_tools()
        mock = MockReActLLM(tools=tools)
        for p in prompts:
            out = mock.run(p)
            rows.append(out)

    # write JSONL
    with open(OUT, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return rows

def summarize(rows):
    total = len(rows)
    trace_lengths = [len(r.get('trace', [])) for r in rows]
    avg_trace = sum(trace_lengths) / total if total else 0
    tool_calls = {}
    client_found = 0
    for r in rows:
        for t in r.get('trace', []):
            if t.get('type') == 'action':
                name = t.get('text')
                tool_calls[name] = tool_calls.get(name, 0) + 1
        # detect client-found by final_answer containing 'savings' or 'Client'
        if 'Client' in r.get('final_answer', '') or 'savings' in r.get('final_answer', '').lower():
            client_found += 1
    summary = {
        'total': total,
        'avg_trace_length': avg_trace,
        'tool_calls': tool_calls,
        'client_found_rate': client_found / total if total else 0
    }
    with open(SUMMARY_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    with open(SUMMARY_TXT, 'w') as f:
        f.write('Experiments 100 test summary\n')
        f.write(json.dumps(summary, indent=2))
    return summary


def main(count=100):
    df = load_data()
    prompts = generate_prompts(df, count=count)
    rows = run(prompts)
    summary = summarize(rows)
    print('Wrote', OUT, 'and summary to', SUMMARY_JSON)
    print('Summary:', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--real', action='store_true', help='Attempt to run prompts through the real agent (Ollama)')
    args = parser.parse_args()
    # generate prompts and run
    df = load_data()
    prompts = generate_prompts(df, count=args.count)
    rows = run(prompts, use_real=args.real)
    summary = summarize(rows)
    print('Summary:', summary)
