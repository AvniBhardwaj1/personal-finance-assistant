from tools import load_data, get_client_profile, get_stock_price


def test_load_data():
    df = load_data()
    assert df is not None
    # If dataset present, expect specific columns (lowercase normalized)
    if not df.empty:
        assert 'user_id' in df.columns or 'user id' in ' '.join(df.columns)


def test_get_client_profile():
    df = load_data()
    if df.empty:
        return
    # pick first client id
    uid = str(df.iloc[0, 0])
    p = get_client_profile(df, uid)
    assert isinstance(p, dict)
    assert 'message' in p or 'savings_usd' in p or 'monthly_income_usd' in p


def test_get_stock_price():
    # do a small smoke test for a common ticker; network access needed
    msg = get_stock_price('AAPL')
    assert isinstance(msg, str)
