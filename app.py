import streamlit as st
import pandas as pd
import numpy as np
import requests
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ------------------------------------------------
# Page setup
# ------------------------------------------------
st.set_page_config(
    page_title="Binance Futures Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Binance Futures Scanner (Futures Data)")

BASE_URL = "https://fapi.binance.com"  # Binance Futures API base

# ------------------------------------------------
# Indicators
# ------------------------------------------------
def sma(series, period: int):
    return series.rolling(window=period, min_periods=period).mean()

def ema(series, period: int):
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def rsi(series, period: int = 14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)

def macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


# ------------------------------------------------
# Fetch Binance Futures Market Data via REST
# ------------------------------------------------
def get_top_futures_symbols(limit=10):
    url = f"{BASE_URL}/fapi/v1/ticker/24hr"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    usdt_pairs = [t for t in data if t["symbol"].endswith("USDT")]
    sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x["quoteVolume"]), reverse=True)
    return sorted_pairs[:limit]

def get_klines(symbol, interval="15m", limit=200):
    url = f"{BASE_URL}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data, columns=[
        "timestamp","open","high","low","close","volume",
        "close_time","qav","trades","taker_base","taker_quote","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


# ------------------------------------------------
# Main App Logic
# ------------------------------------------------
st.info("Fetching Binance Futures top markets…")

symbols = get_top_futures_symbols(10)

results = []
for sym in symbols:
    symbol = sym["symbol"]
    last_price = float(sym["lastPrice"])
    df = get_klines(symbol, "15m", 200)

    if df.empty:
        continue

    rsi_vals = rsi(df["close"])
    macd_line, signal_line = macd(df["close"])

    results.append({
        "symbol": symbol,
        "last_price": last_price,
        "RSI": round(float(rsi_vals.iloc[-1]), 2),
        "MACD": round(float(macd_line.iloc[-1] - signal_line.iloc[-1]), 4),
    })

st.dataframe(pd.DataFrame(results))
