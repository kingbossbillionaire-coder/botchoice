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
    page_title="Binance Futures Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Indicator helpers (pure NumPy/Pandas) ----------
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

# Custom CSS for dark mode
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: var(--background-color);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        margin: 0.5rem 0;
    }
    .profit-positive { color: #00ff88; font-weight: bold; }
    .profit-negative { color: #ff4444; font-weight: bold; }
    .bot-recommendation {
        padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid;
    }
    .long-bot { border-left-color: #00ff88; background-color: rgba(0, 255, 136, 0.1); }
    .short-bot { border-left-color: #ff4444; background-color: rgba(255, 68, 68, 0.1); }
    .range-bot { border-left-color: #ffaa00; background-color: rgba(255, 170, 0, 0.1); }
    </style>
    """, unsafe_allow_html=True)

class BinanceScanner:
    def __init__(self):
        # Use Futures API base
        self.base_url = "https://fapi.binance.com"
        self.session = requests.Session()

    def get_top_futures_symbols(self, limit=30):
        try:
            url = f"{self.base_url}/fapi/v1/ticker/24hr"
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            usdt_pairs = [item for item in data if item.get('symbol','').endswith('USDT')]
            sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x.get('quoteVolume', 0) or 0), reverse=True)
            return sorted_pairs[:limit]
        except Exception as e:
            st.error(f"Error fetching symbols: {e}")
            return []

    def get_klines(self, symbol, interval, limit=100):
        try:
            url = f"{self.base_url}/fapi/v1/klines"
            params = {'symbol': symbol, 'interval': interval, 'limit': limit}
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):
                return pd.DataFrame()
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.dropna().reset_index(drop=True)
            return df
        except Exception as e:
            st.error(f"Error fetching klines for {symbol}: {e}")
            return pd.DataFrame()

class TechnicalAnalyzer:
    @staticmethod
    def two_pole_oscillator(df, length=20):
        if df.empty or len(df) < 30:
            n = len(df)
            arr = np.zeros(n)
            return {
                'oscillator': arr, 'oscillator_prev': np.roll(arr, 4),
                'buy_signals': [], 'sell_signals': [], 'current_value': 0.0
            }
        close = df['close']
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

        buy_signals, sell_signals = [], []
        for i in range(1, len(two_p)):
            if two_p[i] > two_pp[i] and two_p[i-1] <= two_pp[i-1] and two_p[i] < 0:
                buy_signals.append(i)
            elif two_p[i] < two_pp[i] and two_p[i-1] >= two_pp[i-1] and two_p[i] > 0:
                sell_signals.append(i)

        return {
            'oscillator': two_p,
            'oscillator_prev': two_pp,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'current_value': float(two_p[-1]) if len(two_p) else 0.0
        }

    @staticmethod
    def calculate_support_resistance(df, window=20):
        if df.empty or len(df) < window:
            return np.nan, np.nan
        highs = df['high'].rolling(window=window, min_periods=window).max()
        lows = df['low'].rolling(window=window, min_periods=window).min()
        return float(lows.iloc[-1]), float(highs.iloc[-1])

    @staticmethod
    def calculate_rsi(df, period=14):
        return rsi(df['close'], period)

    @staticmethod
    def calculate_macd(df):
        return macd(df['close'])

    @staticmethod
    def calculate_bollinger_bands(df, period=20, std=2):
        return bbands(df['close'], period, std)

class TradingBotAnalyzer:
    @staticmethod
    def analyze_trend_direction(df_1h, df_15m, df_5m, df_1m):
        timeframes = {'1h': df_1h, '15m': df_15m, '5m': df_5m, '1m': df_1m}
        predictions = {}
        for tf, df in timeframes.items():
            if df.empty or len(df) < 12:
                predictions[tf] = 'Unknown'
                continue
            rsi_vals = TechnicalAnalyzer.calculate_rsi(df)
            macd_line, signal_line, _ = TechnicalAnalyzer.calculate_macd(df)
            tp = TechnicalAnalyzer.two_pole_oscillator(df)

            price_trend = 1 if df['close'].iloc[-1] > df['close'].iloc[-10] else -1
            rsi_trend = 1 if rsi_vals.iloc[-1] > 50 else -1
            macd_trend = 1 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1
            oscillator_trend = 1 if tp['current_value'] > 0 else -1
            total_score = price_trend + rsi_trend + macd_trend + oscillator_trend

            if total_score >= 2:
                predictions[tf] = 'Upward'
            elif total_score <= -2:
                predictions[tf] = 'Downward'
            else:
                predictions[tf] = 'Ranging'
        return predictions

    @staticmethod
    def recommend_bot_strategy(df, symbol, current_price):
        if df.empty:
            return None
        rsi_vals = TechnicalAnalyzer.calculate_rsi(df)
        upper_bb, middle_bb, lower_bb = TechnicalAnalyzer.calculate_bollinger_bands(df)
        support, resistance = TechnicalAnalyzer.calculate_support_resistance(df)
        two_pole = TechnicalAnalyzer.two_pole_oscillator(df)

        current_rsi = float(rsi_vals.iloc[-1]) if not pd.isna(rsi_vals.iloc[-1]) else 50
        mid = float(middle_bb.iloc[-1]) if not pd.isna(middle_bb.iloc[-1]) else current_price
        current_upper_bb = float(upper_bb.iloc[-1]) if not pd.isna(upper_bb.iloc[-1]) else current_price * 1.02
        current_lower_bb = float(lower_bb.iloc[-1]) if not pd.isna(lower_bb.iloc[-1]) else current_price * 0.98
        oscillator_value = two_pole['current_value']

        recs = []

        if (current_rsi < 40 and oscillator_value < -0.3 and
            current_price < mid and len(two_pole['buy_signals']) > 0):
            entry_price = current_price
            upper_range = min(resistance if not pd.isna(resistance) else current_upper_bb, current_upper_bb)
            stop_loss = max(support if not pd.isna(support) else current_price * 0.97, current_price * 0.97)
            expected_profit = ((upper_range - entry_price) / entry_price) * 100
            recs.append({
                'type': 'LONG',
                'confidence': min(90, max(60, 100 - current_rsi + abs(oscillator_value) * 20)),
                'entry_range': f"{entry_price:.4f}",
                'upper_range': f"{upper_range:.4f}",
                'lower_range': f"{entry_price * 0.995:.4f}",
                'stop_loss': f"{stop_loss:.4f}",
                'expected_profit': f"{expected_profit:.2f}%"
            })

        if (current_rsi > 60 and oscillator_value > 0.3 and
            current_price > mid and len(two_pole['sell_signals']) > 0):
            entry_price = current_price
            lower_range = max(support if not pd.isna(support) else current_lower_bb, current_lower_bb)
            stop_loss = min(resistance if not pd.isna(resistance) else current_price * 1.03, current_price * 1.03)
            expected_profit = ((entry_price - lower_range) / entry_price) * 100
            recs.append({
                'type': 'SHORT',
                'confidence': min(90, max(60, current_rsi - 10 + abs(oscillator_value) * 20)),
                'entry_range': f"{entry_price:.4f}",
                'upper_range': f"{entry_price * 1.005:.4f}",
                'lower_range': f"{lower_range:.4f}",
                'stop_loss': f"{stop_loss:.4f}",
                'expected_profit': f"{expected_profit:.2f}%"
            })

        range_size = 0.0
        if not pd.isna(resistance) and not pd.isna(support) and current_price > 0:
            range_size = (resistance - support) / current_price
        if (0.02 < range_size < 0.08 and abs(oscillator_value) < 0.5 and 30 < current_rsi < 70):
            recs.append({
                'type': 'RANGE',
                'confidence': min(85, max(50, 70 + (range_size * 500))),
                'entry_range': f"{current_price:.4f}",
                'upper_range': f"{(resistance if not pd.isna(resistance) else current_price*1.02):.4f}",
                'lower_range': f"{(support if not pd.isna(support) else current_price*0.98):.4f}",
                'stop_loss': f"{current_price * 0.95:.4f}",
                'expected_profit': f"{(range_size * 50):.2f}%"
            })

        if recs:
            return max(recs, key=lambda x: x['confidence'])
        return {
            'type': 'HOLD',
            'confidence': 30,
            'entry_range': f"{current_price:.4f}",
            'upper_range': f"{current_price * 1.01:.4f}",
            'lower_range': f"{current_price * 0.99:.4f}",
            'stop_loss': f"{current_price * 0.95:.4f}",
            'expected_profit': "0.50%"
        }

def init_session_state():
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()
    if 'scanner_data' not in st.session_state:
        st.session_state.scanner_data = []

def apply_theme():
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: #fafafa; }
        .metric-card { background-color: #262730; border-color: #464853; }
        </style>
        """, unsafe_allow_html=True)

def main():
    init_session_state()
    load_css()
    apply_theme()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("🌙" if not st.session_state.dark_mode else "☀️"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    with col2:
        st.markdown('<h1 class="main-header">📈 Binance Futures Scanner</h1>', unsafe_allow_html=True)
    with col3:
        auto_refresh = st.checkbox("Auto Refresh (1min)", value=True)

    scanner = BinanceScanner()
    analyzer = TradingBotAnalyzer()

    with st.sidebar:
        st.header("⚙️ Settings")
        refresh_interval = st.selectbox(
            "Refresh Interval", [60, 120, 300],
            format_func=lambda x: f"{x//60} minute{'s' if x//60 > 1 else ''}"
        )
        min_confidence = st.slider("Minimum Confidence %", 0, 100, 60)
        show_charts = st.checkbox("Show Charts", value=False)

        st.header("📊 Bot Types")
        show_long = st.checkbox("Long Bots", value=True)
        show_short = st.checkbox("Short Bots", value=True)
        show_range = st.checkbox("Range Bots", value=True)

        if st.button("🔄 Force Refresh"):
            st.session_state.last_update = datetime.now() - timedelta(minutes=2)

    current_time = datetime.now()
    if auto_refresh and (current_time - st.session_state.last_update).seconds >= refresh_interval:
        st.session_state.last_update = current_time
        st.rerun()

    with st.spinner("🔍 Scanning top 30 Binance futures..."):
        top_symbols = scanner.get_top_futures_symbols(30)
        if not top_symbols:
            st.error("Failed to fetch data from Binance Futures API")
            return

        scanner_results = []
        progress_bar = st.progress(0)

        for i, symbol_data in enumerate(top_symbols):
            symbol = symbol_data['symbol']
            try:
                current_price = float(symbol_data.get('lastPrice') or 0)
            except:
                current_price = 0.0
            price_change = float(symbol_data.get('priceChangePercent') or 0.0)
            volume_24h = float(symbol_data.get('quoteVolume') or 0.0)

            df_1h = scanner.get_klines(symbol, '1h', 300)
            df_15m = scanner.get_klines(symbol, '15m', 300)
            df_5m = scanner.get_klines(symbol, '5m', 300)
            df_1m = scanner.get_klines(symbol, '1m', 300)

            if not df_1h.empty:
                trend_predictions = analyzer.analyze_trend_direction(df_1h, df_15m, df_5m, df_1m)
                bot_recommendation = analyzer.recommend_bot_strategy(df_1h, symbol, current_price)
                support, resistance = TechnicalAnalyzer.calculate_support_resistance(df_1h)

                scanner_results.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    'price_change_24h': price_change,
                    'volume_24h': volume_24h,
                    'bot_type': bot_recommendation['type'],
                    'confidence': bot_recommendation['confidence'],
                    'entry_range': bot_recommendation['entry_range'],
                    'upper_range': bot_recommendation['upper_range'],
                    'lower_range': bot_recommendation['lower_range'],
                    'stop_loss': bot_recommendation['stop_loss'],
                    'expected_profit': bot_recommendation['expected_profit'],
                    'support': support,
                    'resistance': resistance,
                    'trend_1h': trend_predictions.get('1h', 'Unknown'),
                    'trend_15m': trend_predictions.get('15m', 'Unknown'),
                    'trend_5m': trend_predictions.get('5m', 'Unknown'),
                    'trend_1m': trend_predictions.get('1m', 'Unknown'),
                    'df_1h': df_1h
                })

            progress_bar.progress((i + 1) / len(top_symbols))

        scanner_results.sort(key=lambda x: x['confidence'], reverse=True)
        st.session_state.scanner_data = scanner_results

    st.header("🎯 Top Trading Opportunities")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_opps = len([r for r in scanner_results if r['confidence'] >= min_confidence])
        st.metric("Total Opportunities", total_opps)
    with col2:
        st.metric("Long Bots", len([r for r in scanner_results if r['bot_type'] == 'LONG' and r['confidence'] >= min_confidence]))
    with col3:
        st.metric("Short Bots", len([r for r in scanner_results if r['bot_type'] == 'SHORT' and r['confidence'] >= min_confidence]))
    with col4:
        st.metric("Range Bots", len([r for r in scanner_results if r['bot_type'] == 'RANGE' and r['confidence'] >= min_confidence]))

    filtered_results = []
    for result in scanner_results:
        if result['confidence'] < min_confidence:
            continue
        if not show_long and result['bot_type'] == 'LONG':
            continue
        if not show_short and result['bot_type'] == 'SHORT':
            continue
        if not show_range and result['bot_type'] == 'RANGE':
            continue
        filtered_results.append(result)

    for i, result in enumerate(filtered_results[:15]):
        with st.container():
            bot_class = f"{result['bot_type'].lower()}-bot"
            st.markdown(f"""
            <div class="bot-recommendation {bot_class}">
                <h3>#{i+1} {result['symbol']} - {result['bot_type']} Bot</h3>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Current Price", f"${result['current_price']:.4f}", f"{result['price_change_24h']:.2f}%")
                st.metric("Confidence", f"{result['confidence']:.0f}%")
            with c2:
                st.write("**Price Ranges:**")
                st.write(f"Entry: {result['entry_range']}")
                st.write(f"Upper: {result['upper_range']}")
                st.write(f"Lower: {result['lower_range']}")
                st.write(f"Stop Loss: {result['stop_loss']}")
            with c3:
                st.write("**Support/Resistance:**")
                spt = result['support']
                rst = result['resistance']
                st.write(f"Support: {spt:.4f}" if pd.notna(spt) else "Support: N/A")
                st.write(f"Resistance: {rst:.4f}" if pd.notna(rst) else "Resistance: N/A")
                try:
                    p = float(str(result['expected_profit']).replace('%',''))
                    profit_class = "profit-positive" if p > 0 else "profit-negative"
                except:
                    profit_class = "profit-positive"
                st.markdown(f"<span class='{profit_class}'>Expected Profit: {result['expected_profit']}</span>", unsafe_allow_html=True)
            with c4:
                st.write("**Trend Analysis:**")
                st.write(f"1h: {result['trend_1h']}")
                st.write(f"15m: {result['trend_15m']}")
                st.write(f"5m: {result['trend_5m']}")
                st.write(f"1m: {result['trend_1m']}")

            if st.checkbox(f"Show {result['symbol']} chart", key=f"chart_{i}") and not result['df_1h'].empty:
                df = result['df_1h'].tail(80)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                    subplot_titles=[f"{result['symbol']} Price", "Two-Pole Oscillator"],
                                    row_width=[0.7, 0.3])

                fig.add_trace(
                    go.Candlestick(
                        x=df['timestamp'], open=df['open'], high=df['high'],
                        low=df['low'], close=df['close'], name='Price'
                    ), row=1, col=1
                )
                if pd.notna(result['support']):
                    fig.add_hline(y=result['support'], line_dash="dash", line_color="green", row=1, col=1)
                if pd.notna(result['resistance']):
                    fig.add_hline(y=result['resistance'], line_dash="dash", line_color="red", row=1, col=1)

                tp = TechnicalAnalyzer.two_pole_oscillator(df)
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=tp['oscillator'], mode='lines', name='Two-Pole', line=dict(color='blue')),
                    row=2, col=1
                )
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

    st.info(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

    if auto_refresh:
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()
