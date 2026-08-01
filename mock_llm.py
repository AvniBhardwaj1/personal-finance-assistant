"""Advanced Mock LLM for testing agent-tool interactions.

This mock LLM simulates ReAct-style behavior by parsing the input and calling
provided tool callables. It records a trace of Thoughts/Actions/Observations
and returns a final textual answer plus a structured trace that tests can assert on.

Usage:
    from mock_llm import MockReActLLM
    mock = MockReActLLM(tools={'GetClientProfile': get_client_profile, 'GetLiveStockPrice': get_stock_price})
    out = mock.run("What are savings for U00001 and what's AAPL price?")
    print(out['final_answer'])
    print(out['trace'])
"""
from typing import Callable, Dict, Any, List
import re

class MockReActLLM:
    def __init__(self, tools: Dict[str, Callable[[str], Any]]):
        """tools: mapping from tool name to callable taking a single string input."""
        self.tools = tools

    def _find_user_ids(self, text: str) -> List[str]:
        return re.findall(r"\bU\d{5}\b", text.upper())

    def _find_tickers(self, text: str) -> List[str]:
        # simplistic: uppercase consecutive letters of length 1-5 not starting with U followed by digits
        cand = re.findall(r"\b([A-Z]{1,5})\b", text)
        # filter out things that look like user ids or common words
        tickers = [c for c in cand if not re.match(r"U\d{5}", c) and len(c) <=5]
        # prefer known tickers like AAPL, MSFT, GOOG
        return tickers

    def run(self, prompt: str) -> Dict[str, Any]:
        trace = []
        final_answer_parts = []

        # Step 1: look for user ids and fetch profiles
        uids = self._find_user_ids(prompt)
        for uid in uids:
            thought = f"I should look up client {uid} in the dataset."
            trace.append({'type': 'thought', 'text': thought})
            action = 'GetClientProfile'
            trace.append({'type': 'action', 'text': action, 'input': uid})
            obs = self.tools.get(action, lambda x: {'message': 'tool missing'})(uid)
            trace.append({'type': 'observation', 'text': str(obs)})
            if isinstance(obs, dict):
                if 'message' in obs:
                    final_answer_parts.append(obs['message'])
                else:
                    savings = obs.get('savings_usd', 'N/A')
                    income = obs.get('monthly_income_usd', 'N/A')
                    final_answer_parts.append(f"Client {uid} savings: {savings}; income: {income}.")

        # Step 2: look for tickers and fetch prices
        tickers = self._find_tickers(prompt)
        for t in tickers:
            thought = f"I should fetch price for {t}."
            trace.append({'type': 'thought', 'text': thought})
            action = 'GetLiveStockPrice'
            trace.append({'type': 'action', 'text': action, 'input': t})
            obs = self.tools.get(action, lambda x: f"No tool for {action}")(t)
            trace.append({'type': 'observation', 'text': str(obs)})
            final_answer_parts.append(str(obs))

        if not final_answer_parts:
            trace.append({'type': 'thought', 'text': 'No tool calls needed; respond generically.'})
            final_answer = "I can help with financial advice. Please ask about a client (e.g., U00001) or a stock ticker (e.g., AAPL)."
        else:
            final_answer = " \n".join(final_answer_parts)

        return {
            'final_answer': final_answer,
            'trace': trace,
            'prompt': prompt
        }
