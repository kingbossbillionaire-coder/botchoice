import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import time
from datetime import datetime
import warnings

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

st.title("📊 Binance Futures Scanner (with API Key)")

# ------------------------------------------------
# Load Binance API key/secret from Streamlit Secrets
# ------------------------------------------------
try:
    api_key = st.secrets["BINANCE_API_KEY"]
    api_secret = st.secrets["BINANCE_API_SECRET"]
except Exception:
    st.error("⚠️ Please add your Binance API Key/Secret in Streamlit secrets.")
    st.stop()

# ------------------------------------------------
# Setup Binance Futures via ccxt
# ------------------------------------------------
exchange = ccxt.binance({
    "apiKey": api_key,
    "secret": api_secret,
    "enableRateLimit": True,
    "options": {"defaultType": "future"}  # forces Futures market
})


# ------------------------------------------------
# Indicators
# ------------------------------------------------
def sma(series: pd.Series, period: int):
    return series.rolling(window=period, min_periods=period).mean()

def ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def rsi(series: pd.Series, period: int = 14):
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

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# ------------------------------------------------
# Fetch Futures symbols & OHLCV with ccxt
# ------------------------------------------------
def get_top_futures_symbols(limit=10):
    markets = exchange.load_markets()
    usdt_markets = [m for m in markets if m.endswith("/USDT")]
    # Sort by volume (descending)
    tickers = exchange.fetch_tickers(usdt_markets)
    sorted_pairs = sorted(tickers.items(), key=lambda x: x[1]["quoteVolume"] if "quoteVolume" in x[1] else 0, reverse=True)
    top = sorted_pairs[:limit]
    return [t[0] for t in top]

def get_ohlcv(symbol, timeframe="1h", limit=200):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        st.error(f"Error fetching OHLCV for {symbol}: {e}")
        return pd.DataFrame()


# ------------------------------------------------
# Main App Logic
# ------------------------------------------------
refresh = st.sidebar.selectbox("Refresh Interval", [60, 120, 300], format_func=lambda x: f"{x//60} min")

st.info("Fetching Binance Futures markets with your API key...")

symbols = get_top_futures_symbols(10)

results = []
for sym in symbols:
    df = get_ohlcv(sym, "15m", 200)
    if df.empty:
        continue

    rsi_vals = rsi(df["close"])
    macd_line, signal_line, _ = macd(df["close"])

    results.append({
        "symbol": sym,
        "last_price": df["close"].iloc[-1],
        "RSI": float(rsi_vals.iloc[-1]),
        "MACD": float(macd_line.iloc[-1] - signal_line.iloc[-1]),
    })

st.dataframe(pd.DataFrame(results))
