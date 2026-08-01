import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Dict

_DATAFILE = Path("synthetic_personal_finance_dataset.csv")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase underscore style for robust access."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_data(path: Path = _DATAFILE) -> pd.DataFrame:
    """Load the CSV dataset and normalize columns. Returns empty DataFrame on error."""
    try:
        df = pd.read_csv(path)
        df = normalize_columns(df)
        return df
    except Exception:
        return pd.DataFrame()


def get_stock_price(ticker: str) -> str:
    """Get a recent stock price for ticker using yfinance. Returns string message."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return "No ticker provided."
    try:
        t = yf.Ticker(ticker)
        # prefer history close price (more robust than .info which can be rate-limited)
        hist = t.history(period="5d")
        if hist is not None and not hist.empty:
            price = float(hist['Close'].iloc[-1])
            return f"The current (recent close) price for {ticker} is ${price:.2f}."
        # fallback to info
        info = getattr(t, 'info', {}) or {}
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        if price:
            return f"The current market price for {ticker} is ${float(price):.2f}."
        return f"Could not fetch a price for {ticker}."
    except Exception as e:
        return f"Error fetching price for {ticker}: {e}"


def get_client_profile(df: pd.DataFrame, user_id: str) -> Dict:
    """Return client row as dict. Matches on user_id-like columns (user_id or user).

    Returns dict with message key if not found.
    """
    if df is None or df.empty:
        return {"message": "Dataset is empty or not loaded."}
    uid = (user_id or "").strip()
    if not uid:
        return {"message": "No user_id provided."}

    # try common column names
    candidates = [c for c in df.columns if 'user' in c and 'id' in c or c == 'user_id']
    # fallback to first column
    if not candidates:
        candidates = [df.columns[0]]

    for col in candidates:
        matches = df[df[col].astype(str) == uid]
        if not matches.empty:
            row = matches.iloc[0].to_dict()
            # convert numpy types to python native
            clean = {k: (v.item() if hasattr(v, 'item') else v) for k, v in row.items()}
            return clean

    return {"message": f"Client {uid} not found."}
