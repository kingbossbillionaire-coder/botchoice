import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
# Setup Binance (ccxt client)
# ------------------------------------------------
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}  # use Futures API
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

def bbands(series: pd.Series, period=20, std_dev=2):
    mid = sma(series, period)
    rolling_std = series.rolling(window=period, min_periods=period).std()
    upper = mid + std_dev * rolling_std
    lower = mid - std_dev * rolling_std
    return upper, mid, lower

# Two-pole oscillator
def two_pole_oscillator(df, length=20):
    if df.empty or len(df) < 30:
        return np.zeros(len(df)), np.zeros(len(df)), [], [], 0.0

    close = df["close"]
    sma1 = sma(close, 25)
    close_sma_diff = close - sma1
    sma_diff = sma(close_sma_diff, 25)
    std_diff = close_sma_diff.rolling(25, min_periods=25).std()
    sma_n1 = (close_sma_diff - sma_diff) / std_diff.replace(0, np.nan)
    sma_n1 = sma_n1.fillna(0.0).values

    alpha = 2.0 / (length + 1)
    smooth1, smooth2 = np.zeros_like(sma_n1), np.zeros_like(sma_n1)
    for i in range(len(sma_n1)):
        if i == 0:
            smooth1[i] = sma_n1[i]
            smooth2[i] = smooth1[i]
        else:
            smooth1[i] = (1 - alpha) * smooth1[i-1] + alpha * sma_n1[i]
            smooth2[i] = (1 - alpha) * smooth2[i-1] + alpha * smooth1[i]

    two_p = smooth2
    two_pp = np.roll(two_p, 4)

    buy_signals, sell_signals = [], []
    for i in range(1, len(two_p)):
        if two_p[i] > two_pp[i] and two_p[i-1] <= two_pp[i-1] and two_p[i] < 0:
            buy_signals.append(i)
        elif two_p[i] < two_pp[i] and two_p[i-1] >= two_pp[i-1] and two_p[i] > 0:
            sell_signals.append(i)

    return two_p, two_pp, buy_signals, sell_signals, float(two_p[-1])


# ------------------------------------------------
# Fetch Futures Market Data
# ------------------------------------------------
def get_top_futures_symbols(limit=10):
    tickers = exchange.fapiPublic_get_ticker_24hr()
    usdt_pairs = [t for t in tickers if t['symbol'].endswith("USDT")]
    sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
    return sorted_pairs[:limit]

def get_klines(symbol, timeframe="1h", limit=200):
    try:
        ohlcv = exchange.fapiPublic_get_klines({
            "symbol": symbol,
            "interval": timeframe,
            "limit": limit
        })
        df = pd.DataFrame(ohlcv, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        st.error(f"Error fetching klines: {e}")
        return pd.DataFrame()


# ------------------------------------------------
# Main App Logic
# ------------------------------------------------
refresh = st.sidebar.selectbox("Refresh Interval", [60, 120, 300], format_func=lambda x: f"{x//60} min")

st.info("Fetching Binance Futures data with your API key...")

symbols = get_top_futures_symbols(10)
results = []
for s in symbols:
    sym = s["symbol"]
    last_price = float(s["lastPrice"])
    df = get_klines(sym, "15m", 200)
    if df.empty:
        continue

    rsi_vals = rsi(df["close"])
    macd_line, signal_line, _ = macd(df["close"])
    upper, mid, lower = bbands(df["close"])
    two_p, two_pp, buys, sells, osc_val = two_pole_oscillator(df)

    results.append({
        "symbol": sym,
        "last_price": last_price,
        "RSI": float(rsi_vals.iloc[-1]),
        "MACD": float(macd_line.iloc[-1] - signal_line.iloc[-1]),
        "Oscillator": osc_val
    })

st.dataframe(pd.DataFrame(results))
