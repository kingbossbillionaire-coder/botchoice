import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Crypto Futures Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Indicator helpers ----------
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

# ---------- Data Source (CoinGecko instead of Binance) ----------
class CoinGeckoScanner:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"

    def get_top_symbols(self, limit=30):
        url = f"{self.base_url}/coins/markets"
        params = {"vs_currency": "usd", "order": "volume_desc", "per_page": limit, "page": 1, "sparkline": False}
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            results = []
            for item in data:
                results.append({
                    "symbol": item["symbol"].upper() + "USDT",
                    "id": item["id"],
                    "lastPrice": item["current_price"],
                    "volume_24h": item["total_volume"],
                    "price_change_24h": item["price_change_percentage_24h"] or 0
                })
            return results
        except Exception as e:
            st.error(f"Error fetching symbols from CoinGecko: {e}")
            return []

    def get_ohlc(self, coin_id, days=1):
        url = f"{self.base_url}/coins/{coin_id}/ohlc"
        params = {"vs_currency": "usd", "days": days}  # days=1 → 1d OHLC
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                return pd.DataFrame()
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except Exception as e:
            st.error(f"Error fetching OHLC for {coin_id}: {e}")
            return pd.DataFrame()

# ---------- Technical Analyzer ----------
class TechnicalAnalyzer:
    @staticmethod
    def two_pole_oscillator(df, length=20):
        if df.empty or len(df) < 30:
            n = len(df)
            arr = np.zeros(n)
            return {"oscillator": arr, "oscillator_prev": np.roll(arr, 4), "buy_signals": [], "sell_signals": [], "current_value": 0.0}

        close = df["close"]
        sma1 = sma(close, 25)
        close_sma_diff = close - sma1
        sma_diff = sma(close_sma_diff, 25)
        std_diff = close_sma_diff.rolling(25, min_periods=25).std()
        sma_n1 = (close_sma_diff - sma_diff) / std_diff.replace(0, np.nan)
        sma_n1 = sma_n1.fillna(0.0).values

        alpha = 2.0 / (length + 1)
        smooth1 = np.zeros_like(sma_n1)
        smooth2 = np.zeros_like(sma_n1)
        for i in range(len(sma_n1)):
            if i == 0:
                smooth1[i] = sma_n1[i]
                smooth2[i] = smooth1[i]
            else:
                smooth1[i] = (1 - alpha) * smooth1[i-1] + alpha * sma_n1[i]
                smooth2[i] = (1 - alpha) * smooth2[i-1] + alpha * smooth1[i]
        two_p = smooth2
        two_pp = np.roll(two_p, 4)

        buy_signals = [i for i in range(1, len(two_p)) if two_p[i] > two_pp[i] and two_p[i-1] <= two_pp[i-1] and two_p[i] < 0]
        sell_signals = [i for i in range(1, len(two_p)) if two_p[i] < two_pp[i] and two_p[i-1] >= two_pp[i-1] and two_p[i] > 0]

        return {"oscillator": two_p, "oscillator_prev": two_pp, "buy_signals": buy_signals, "sell_signals": sell_signals, "current_value": float(two_p[-1])}

# ---------- Main App ----------
def main():
    st.title("📊 Crypto Futures Scanner (CoinGecko data)")
    scanner = CoinGeckoScanner()

    with st.sidebar:
        refresh_interval = st.selectbox("Refresh Interval", [60, 120, 300], format_func=lambda x: f"{x//60} min")
        st.write("Bot Types: (analysis logic unchanged)")
        show_long = st.checkbox("Long Bots", True)
        show_short = st.checkbox("Short Bots", True)
        show_range = st.checkbox("Range Bots", True)
        if st.button("Force Refresh"):
            st.rerun()

    st.info("Fetching top markets from CoinGecko...")

    top_symbols = scanner.get_top_symbols(10)  # fetch top 10 for quicker testing
    if not top_symbols:
        st.error("No symbols fetched!")
        return

    results = []
    for sym in top_symbols:
        df = scanner.get_ohlc(sym["id"], days=1)
        if df.empty:
            continue
        df = df.tail(200)  # keep last few candles
        tp = TechnicalAnalyzer.two_pole_oscillator(df)

        results.append({
            "symbol": sym["symbol"],
            "price": sym["lastPrice"],
            "24h change": sym["price_change_24h"],
            "oscillator": tp["current_value"]
        })

    df_results = pd.DataFrame(results)
    st.dataframe(df_results)

if __name__ == "__main__":
    main()
