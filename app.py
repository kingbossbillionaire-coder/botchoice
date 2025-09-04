import streamlit as st
import pandas as pd
import numpy as np
import ccxt
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
# Setup Binance Futures via ccxt RAW API
# ------------------------------------------------
exchange = ccxt.binance({
    "apiKey": api_key,
    "secret": api_secret,
    "enableRateLimit": True,
    "options": {"defaultType": "future"}  # force Futures only
})


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
# Fetch Futures Market Data
# ------------------------------------------------
def get_top_futures_symbols(limit=10):
    """Fetch top futures pairs by volume."""
    tickers = exchange.fapiPublic_get_ticker_24hr()
    usdt_pairs = [t for t in tickers if t["symbol"].endswith("USDT")]
    sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)
    return sorted_pairs[:limit]

def get_klines(symbol, interval="15m", limit=200):
    """Raw Binance futures kline fetch."""
    ohlcv = exchange.fapiPublic_get_klines({
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    })
    df = pd.DataFrame(ohlcv, columns=[
        "timestamp","open","high","low","close","volume",
        "close_time","quote_asset_volume","trades",
        "taker_base","taker_quote","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


# ------------------------------------------------
# Main App Logic
# ------------------------------------------------
st.info("Fetching Binance Futures data with your API key…")

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
