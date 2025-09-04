import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import talib
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Binance Futures Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .profit-positive {
        color: #00ff88;
        font-weight: bold;
    }
    .profit-negative {
        color: #ff4444;
        font-weight: bold;
    }
    .bot-recommendation {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .long-bot {
        border-left-color: #00ff88;
        background-color: rgba(0, 255, 136, 0.1);
    }
    .short-bot {
        border-left-color: #ff4444;
        background-color: rgba(255, 68, 68, 0.1);
    }
    .range-bot {
        border-left-color: #ffaa00;
        background-color: rgba(255, 170, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

class BinanceScanner:
    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.session = requests.Session()

    def get_top_futures_symbols(self, limit=30):
        """Get top futures symbols by 24h volume"""
        try:
            url = f"{self.base_url}/fapi/v1/ticker/24hr"
            response = self.session.get(url, timeout=10)
            data = response.json()

            # Filter USDT pairs and sort by volume
            usdt_pairs = [item for item in data if item['symbol'].endswith('USDT')]
            sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)

            return sorted_pairs[:limit]
        except Exception as e:
            st.error(f"Error fetching symbols: {e}")
            return []

    def get_klines(self, symbol, interval, limit=100):
        """Get kline data for a symbol"""
        try:
            url = f"{self.base_url}/fapi/v1/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            response = self.session.get(url, params=params, timeout=10)
            data = response.json()

            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            st.error(f"Error fetching klines for {symbol}: {e}")
            return pd.DataFrame()

class TechnicalAnalyzer:
    @staticmethod
    def two_pole_oscillator(df, length=20):
        """Implementation of the Two-Pole Oscillator from Pine Script"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # Calculate SMA and normalized values
        sma1 = talib.SMA(close, 25)
        close_sma_diff = close - sma1
        sma_diff = talib.SMA(close_sma_diff, 25)
        std_diff = talib.STDDEV(close_sma_diff, 25)
        sma_n1 = (close_sma_diff - sma_diff) / std_diff

        # Two-pole filter implementation
        alpha = 2.0 / (length + 1)
        smooth1 = np.zeros_like(sma_n1)
        smooth2 = np.zeros_like(sma_n1)

        for i in range(len(sma_n1)):
            if i == 0:
                smooth1[i] = sma_n1[i] if not np.isnan(sma_n1[i]) else 0
                smooth2[i] = smooth1[i]
            else:
                smooth1[i] = (1 - alpha) * smooth1[i-1] + alpha * sma_n1[i] if not np.isnan(sma_n1[i]) else smooth1[i-1]
                smooth2[i] = (1 - alpha) * smooth2[i-1] + alpha * smooth1[i]

        two_p = smooth2
        two_pp = np.roll(two_p, 4)  # two_p[4] in Pine Script

        # Generate signals
        buy_signals = []
        sell_signals = []

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
            'current_value': two_p[-1] if len(two_p) > 0 else 0
        }

    @staticmethod
    def calculate_support_resistance(df, window=20):
        """Calculate support and resistance levels"""
        highs = df['high'].rolling(window=window).max()
        lows = df['low'].rolling(window=window).min()

        resistance = highs.iloc[-1]
        support = lows.iloc[-1]

        return support, resistance

    @staticmethod
    def calculate_rsi(df, period=14):
        """Calculate RSI"""
        return talib.RSI(df['close'].values, timeperiod=period)

    @staticmethod
    def calculate_macd(df):
        """Calculate MACD"""
        macd, signal, histogram = talib.MACD(df['close'].values)
        return macd, signal, histogram

    @staticmethod
    def calculate_bollinger_bands(df, period=20, std=2):
        """Calculate Bollinger Bands"""
        upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=period, nbdevup=std, nbdevdn=std)
        return upper, middle, lower

class TradingBotAnalyzer:
    @staticmethod
    def analyze_trend_direction(df_1h, df_15m, df_5m, df_1m):
        """Analyze trend direction across multiple timeframes"""
        timeframes = {
            '1h': df_1h,
            '15m': df_15m,
            '5m': df_5m,
            '1m': df_1m
        }

        predictions = {}

        for tf, df in timeframes.items():
            if df.empty:
                predictions[tf] = 'Unknown'
                continue

            # Calculate indicators
            rsi = TechnicalAnalyzer.calculate_rsi(df)
            macd, signal, _ = TechnicalAnalyzer.calculate_macd(df)
            two_pole = TechnicalAnalyzer.two_pole_oscillator(df)

            # Trend analysis
            price_trend = 1 if df['close'].iloc[-1] > df['close'].iloc[-10] else -1
            rsi_trend = 1 if rsi[-1] > 50 else -1
            macd_trend = 1 if macd[-1] > signal[-1] else -1
            oscillator_trend = 1 if two_pole['current_value'] > 0 else -1

            # Combine signals
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
        """Recommend bot strategy based on technical analysis"""
        if df.empty:
            return None

        # Calculate indicators
        rsi = TechnicalAnalyzer.calculate_rsi(df)
        upper_bb, middle_bb, lower_bb = TechnicalAnalyzer.calculate_bollinger_bands(df)
        support, resistance = TechnicalAnalyzer.calculate_support_resistance(df)
        two_pole = TechnicalAnalyzer.two_pole_oscillator(df)

        # Get latest values
        current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
        current_upper_bb = upper_bb[-1] if not np.isnan(upper_bb[-1]) else current_price * 1.02
        current_lower_bb = lower_bb[-1] if not np.isnan(lower_bb[-1]) else current_price * 0.98
        oscillator_value = two_pole['current_value']

        # Bot recommendations
        recommendations = []

        # Long Bot Analysis
        if (current_rsi < 40 and oscillator_value < -0.3 and 
            current_price < middle_bb[-1] and len(two_pole['buy_signals']) > 0):

            entry_price = current_price
            upper_range = min(resistance, current_upper_bb)
            stop_loss = max(support, current_price * 0.97)
            expected_profit = ((upper_range - entry_price) / entry_price) * 100

            recommendations.append({
                'type': 'LONG',
                'confidence': min(90, max(60, 100 - current_rsi + abs(oscillator_value) * 20)),
                'entry_range': f"{entry_price:.4f}",
                'upper_range': f"{upper_range:.4f}",
                'lower_range': f"{entry_price * 0.995:.4f}",
                'stop_loss': f"{stop_loss:.4f}",
                'expected_profit': f"{expected_profit:.2f}%"
            })

        # Short Bot Analysis
        if (current_rsi > 60 and oscillator_value > 0.3 and 
            current_price > middle_bb[-1] and len(two_pole['sell_signals']) > 0):

            entry_price = current_price
            lower_range = max(support, current_lower_bb)
            stop_loss = min(resistance, current_price * 1.03)
            expected_profit = ((entry_price - lower_range) / entry_price) * 100

            recommendations.append({
                'type': 'SHORT',
                'confidence': min(90, max(60, current_rsi - 10 + abs(oscillator_value) * 20)),
                'entry_range': f"{entry_price:.4f}",
                'upper_range': f"{entry_price * 1.005:.4f}",
                'lower_range': f"{lower_range:.4f}",
                'stop_loss': f"{stop_loss:.4f}",
                'expected_profit': f"{expected_profit:.2f}%"
            })

        # Range Bot Analysis
        range_size = (resistance - support) / current_price
        if (range_size > 0.02 and range_size < 0.08 and 
            abs(oscillator_value) < 0.5 and 30 < current_rsi < 70):

            recommendations.append({
                'type': 'RANGE',
                'confidence': min(85, max(50, 70 + (range_size * 500))),
                'entry_range': f"{current_price:.4f}",
                'upper_range': f"{resistance:.4f}",
                'lower_range': f"{support:.4f}",
                'stop_loss': f"{current_price * 0.95:.4f}",
                'expected_profit': f"{(range_size * 50):.2f}%"
            })

        # Return best recommendation
        if recommendations:
            return max(recommendations, key=lambda x: x['confidence'])

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
    """Initialize session state variables"""
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()
    if 'scanner_data' not in st.session_state:
        st.session_state.scanner_data = []

def apply_theme():
    """Apply dark/light theme"""
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        .metric-card {
            background-color: #262730;
            border-color: #464853;
        }
        </style>
        """, unsafe_allow_html=True)

def main():
    init_session_state()
    load_css()
    apply_theme()

    # Header with dark mode toggle
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("🌙" if not st.session_state.dark_mode else "☀️"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    with col2:
        st.markdown('<h1 class="main-header">📈 Binance Futures Scanner</h1>', unsafe_allow_html=True)

    with col3:
        auto_refresh = st.checkbox("Auto Refresh (1min)", value=True)

    # Initialize scanner
    scanner = BinanceScanner()
    analyzer = TradingBotAnalyzer()

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")

        refresh_interval = st.selectbox(
            "Refresh Interval",
            [60, 120, 300],
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

    # Auto refresh logic
    current_time = datetime.now()
    if auto_refresh and (current_time - st.session_state.last_update).seconds >= refresh_interval:
        st.session_state.last_update = current_time
        st.rerun()

    # Main scanning logic
    with st.spinner("🔍 Scanning top 30 Binance futures..."):
        top_symbols = scanner.get_top_futures_symbols(30)

        if not top_symbols:
            st.error("Failed to fetch data from Binance API")
            return

        scanner_results = []
        progress_bar = st.progress(0)

        for i, symbol_data in enumerate(top_symbols):
            symbol = symbol_data['symbol']
            current_price = float(symbol_data['lastPrice'])
            volume_24h = float(symbol_data['quoteVolume'])
            price_change = float(symbol_data['priceChangePercent'])

            # Get kline data for different timeframes
            df_1h = scanner.get_klines(symbol, '1h', 100)
            df_15m = scanner.get_klines(symbol, '15m', 100)
            df_5m = scanner.get_klines(symbol, '5m', 100)
            df_1m = scanner.get_klines(symbol, '1m', 100)

            if not df_1h.empty:
                # Analyze trends
                trend_predictions = analyzer.analyze_trend_direction(df_1h, df_15m, df_5m, df_1m)

                # Get bot recommendation
                bot_recommendation = analyzer.recommend_bot_strategy(df_1h, symbol, current_price)

                # Calculate support/resistance
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

        # Sort by confidence
        scanner_results.sort(key=lambda x: x['confidence'], reverse=True)
        st.session_state.scanner_data = scanner_results

    # Display results
    st.header("🎯 Top Trading Opportunities")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_opportunities = len([r for r in scanner_results if r['confidence'] >= min_confidence])
        st.metric("Total Opportunities", total_opportunities)

    with col2:
        long_bots = len([r for r in scanner_results if r['bot_type'] == 'LONG' and r['confidence'] >= min_confidence])
        st.metric("Long Bots", long_bots)

    with col3:
        short_bots = len([r for r in scanner_results if r['bot_type'] == 'SHORT' and r['confidence'] >= min_confidence])
        st.metric("Short Bots", short_bots)

    with col4:
        range_bots = len([r for r in scanner_results if r['bot_type'] == 'RANGE' and r['confidence'] >= min_confidence])
        st.metric("Range Bots", range_bots)

    # Filter results
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

    # Display trading opportunities
    for i, result in enumerate(filtered_results[:15]):  # Show top 15
        with st.container():
            # Bot type styling
            bot_class = f"{result['bot_type'].lower()}-bot"

            st.markdown(f"""
            <div class="bot-recommendation {bot_class}">
                <h3>#{i+1} {result['symbol']} - {result['bot_type']} Bot</h3>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Current Price", 
                    f"${result['current_price']:.4f}",
                    f"{result['price_change_24h']:.2f}%"
                )
                st.metric("Confidence", f"{result['confidence']:.0f}%")

            with col2:
                st.write("**Price Ranges:**")
                st.write(f"Entry: {result['entry_range']}")
                st.write(f"Upper: {result['upper_range']}")
                st.write(f"Lower: {result['lower_range']}")
                st.write(f"Stop Loss: {result['stop_loss']}")

            with col3:
                st.write("**Support/Resistance:**")
                st.write(f"Support: {result['support']:.4f}")
                st.write(f"Resistance: {result['resistance']:.4f}")
                profit_class = "profit-positive" if "%" in result['expected_profit'] and float(result['expected_profit'].replace('%', '')) > 0 else "profit-negative"
                st.markdown(f"<span class='{profit_class}'>Expected Profit: {result['expected_profit']}</span>", unsafe_allow_html=True)

            with col4:
                st.write("**Trend Analysis:**")
                st.write(f"1h: {result['trend_1h']}")
                st.write(f"15m: {result['trend_15m']}")
                st.write(f"5m: {result['trend_5m']}")
                st.write(f"1m: {result['trend_1m']}")

            # Optional chart display
            if show_charts and not result['df_1h'].empty:
                with st.expander(f"📊 {result['symbol']} Chart"):
                    df = result['df_1h'].tail(50)

                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=[f'{result["symbol"]} Price', 'Two-Pole Oscillator'],
                        row_width=[0.7, 0.3]
                    )

                    # Candlestick chart
                    fig.add_trace(
                        go.Candlestick(
                            x=df['timestamp'],
                            open=df['open'],
                            high=df['high'],
                            low=df['low'],
                            close=df['close'],
                            name='Price'
                        ),
                        row=1, col=1
                    )

                    # Support/Resistance lines
                    fig.add_hline(y=result['support'], line_dash="dash", line_color="green", row=1, col=1)
                    fig.add_hline(y=result['resistance'], line_dash="dash", line_color="red", row=1, col=1)

                    # Two-pole oscillator
                    two_pole_data = TechnicalAnalyzer.two_pole_oscillator(df)
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=two_pole_data['oscillator'],
                            mode='lines',
                            name='Two-Pole Oscillator',
                            line=dict(color='blue')
                        ),
                        row=2, col=1
                    )

                    fig.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            st.divider()

    # Last update info
    st.info(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

    # Auto refresh
    if auto_refresh:
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()