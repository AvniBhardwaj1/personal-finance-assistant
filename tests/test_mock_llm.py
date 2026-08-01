from mock_llm import MockReActLLM
from tools import load_data, get_client_profile, get_stock_price


def test_mock_react_trace():
    df = load_data()
    tools = {
        'GetClientProfile': lambda uid: get_client_profile(df, uid),
        'GetLiveStockPrice': lambda t: get_stock_price(t)
    }
    mock = MockReActLLM(tools=tools)
    out = mock.run('What are the savings for U00001 and price of AAPL?')
    assert 'final_answer' in out
    assert 'trace' in out
    # trace should contain both a GetClientProfile action and GetLiveStockPrice action
    actions = [t for t in out['trace'] if t.get('type') == 'action']
    names = [a.get('text') for a in actions]
    assert any('GetClientProfile' in n for n in names) or any('GetClientProfile' == n for n in names)
    assert any('GetLiveStockPrice' in n for n in names) or any('GetLiveStockPrice' == n for n in names)
