import streamlit as st
import pandas as pd
import numpy as np
import requests
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Crypto Futures Scanner", layout="wide")
st.title("📊 Futures Scanner (CryptoCompare API)")

API_KEY = ""  # optional: create free key at https://www.cryptocompare.com/cryptopian/api-keys

BASE = "https://min-api.cryptocompare.com"

def sma(series, period):
    return series.rolling(window=period, min_periods=period).mean()

def ema(series, period):
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def get_top_symbols(limit=10):
    url = f"{BASE}/data/top/totalvolfull"
    params = {"tsym":"USDT", "limit":limit, "api_key":API_KEY}
    r = requests.get(url, params=params)
    data = r.json()
    coins = []
    for d in data["Data"]:
        coin = d["CoinInfo"]["Name"]
        price = d["RAW"]["USDT"]["PRICE"]
        vol = d["RAW"]["USDT"]["TOTALVOLUME24HTO"]
        coins.append({"symbol": coin+"USDT", "price": price, "volume":vol})
    return coins


def get_ohlcv(symbol="BTC", limit=200):
    url = f"{BASE}/data/v2/histominute"
    params = {"fsym":symbol, "tsym":"USDT", "limit":limit, "api_key":API_KEY}
    r = requests.get(url, params=params)
    data = r.json()
    df = pd.DataFrame(data["Data"]["Data"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"close":"close","open":"open","high":"high","low":"low","volumefrom":"volume"}, inplace=True)
    return df


st.info("Fetching top markets from CryptoCompare…")

symbols = get_top_symbols(10)
results = []

for sym in symbols:
    base = sym["symbol"].replace("USDT","")
    df = get_ohlcv(base, 200)
    if df.empty: 
        continue
    rsi_val = rsi(df["close"]).iloc[-1]
    macd_line, signal_line = macd(df["close"])
    macd_val = macd_line.iloc[-1] - signal_line.iloc[-1]

    results.append({
        "symbol": sym["symbol"],
        "price": round(sym["price"],4),
        "volume": round(sym["volume"],0),
        "RSI": round(rsi_val,2),
        "MACD": round(macd_val,4)
    })

st.dataframe(pd.DataFrame(results))
