import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Tuple, Optional
import streamlit as st

class DataProcessor:
    """Utility class for data processing and validation"""

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> bool:
        """Validate if dataframe has required columns and data"""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        return not df.empty and all(col in df.columns for col in required_cols)

    @staticmethod
    def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate price data"""
        if df.empty:
            return df

        # Remove any rows with NaN values
        df = df.dropna()

        # Ensure positive prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df[col] = df[col].abs()

        # Validate OHLC logic
        df = df[
            (df['high'] >= df['low']) & 
            (df['high'] >= df['open']) & 
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) & 
            (df['low'] <= df['close'])
        ]

        return df

    @staticmethod
    def calculate_volatility(df: pd.DataFrame, period: int = 20) -> float:
        """Calculate price volatility"""
        if df.empty or len(df) < period:
            return 0.0

        returns = df['close'].pct_change().dropna()
        return returns.std() * np.sqrt(period) * 100

class SignalGenerator:
    """Generate trading signals based on multiple indicators"""

    @staticmethod
    def generate_composite_signal(df: pd.DataFrame) -> Dict:
        """Generate composite trading signal"""
        if df.empty or len(df) < 50:
            return {'signal': 'HOLD', 'strength': 0}

        signals = []

        # RSI Signal
        rsi = talib.RSI(df['close'].values)
        if rsi[-1] < 30:
            signals.append(('BUY', 2))
        elif rsi[-1] > 70:
            signals.append(('SELL', 2))
        elif rsi[-1] < 40:
            signals.append(('BUY', 1))
        elif rsi[-1] > 60:
            signals.append(('SELL', 1))

        # MACD Signal
        macd, signal, histogram = talib.MACD(df['close'].values)
        if macd[-1] > signal[-1] and macd[-2] <= signal[-2]:
            signals.append(('BUY', 2))
        elif macd[-1] < signal[-1] and macd[-2] >= signal[-2]:
            signals.append(('SELL', 2))

        # Moving Average Signal
        sma_20 = talib.SMA(df['close'].values, 20)
        sma_50 = talib.SMA(df['close'].values, 50)
        if sma_20[-1] > sma_50[-1]:
            signals.append(('BUY', 1))
        else:
            signals.append(('SELL', 1))

        # Calculate composite signal
        buy_strength = sum(weight for signal, weight in signals if signal == 'BUY')
        sell_strength = sum(weight for signal, weight in signals if signal == 'SELL')

        if buy_strength > sell_strength:
            return {'signal': 'BUY', 'strength': min(100, buy_strength * 15)}
        elif sell_strength > buy_strength:
            return {'signal': 'SELL', 'strength': min(100, sell_strength * 15)}
        else:
            return {'signal': 'HOLD', 'strength': 30}

class RiskCalculator:
    """Calculate risk metrics and position sizing"""

    @staticmethod
    def calculate_position_size(account_balance: float, risk_percent: float, 
                              entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        risk_amount = account_balance * (risk_percent / 100)
        price_diff = abs(entry_price - stop_loss)

        if price_diff == 0:
            return 0

        position_size = risk_amount / price_diff
        return round(position_size, 6)

    @staticmethod
    def calculate_risk_reward_ratio(entry_price: float, stop_loss: float, 
                                  take_profit: float) -> float:
        """Calculate risk-reward ratio"""
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)

        if risk == 0:
            return 0

        return round(reward / risk, 2)

class MarketAnalyzer:
    """Advanced market analysis utilities"""

    @staticmethod
    def detect_market_structure(df: pd.DataFrame) -> str:
        """Detect overall market structure"""
        if df.empty or len(df) < 20:
            return "Unknown"

        # Calculate trend strength
        close_prices = df['close'].values
        sma_20 = talib.SMA(close_prices, 20)
        sma_50 = talib.SMA(close_prices, 50)

        # Price position relative to moving averages
        price_above_sma20 = close_prices[-1] > sma_20[-1]
        price_above_sma50 = close_prices[-1] > sma_50[-1]
        sma20_above_sma50 = sma_20[-1] > sma_50[-1]

        # Trend determination
        if price_above_sma20 and price_above_sma50 and sma20_above_sma50:
            return "Strong Uptrend"
        elif price_above_sma20 and sma20_above_sma50:
            return "Uptrend"
        elif not price_above_sma20 and not price_above_sma50 and not sma20_above_sma50:
            return "Strong Downtrend"
        elif not price_above_sma20 and not sma20_above_sma50:
            return "Downtrend"
        else:
            return "Sideways"

    @staticmethod
    def calculate_momentum_score(df: pd.DataFrame) -> float:
        """Calculate momentum score (0-100)"""
        if df.empty or len(df) < 14:
            return 50

        # RSI momentum
        rsi = talib.RSI(df['close'].values)
        rsi_score = rsi[-1] if not np.isnan(rsi[-1]) else 50

        # Price momentum (rate of change)
        roc = talib.ROC(df['close'].values, timeperiod=10)
        roc_score = 50 + (roc[-1] * 2) if not np.isnan(roc[-1]) else 50
        roc_score = max(0, min(100, roc_score))

        # Volume momentum
        volume_sma = talib.SMA(df['volume'].values, 20)
        volume_ratio = df['volume'].iloc[-1] / volume_sma[-1] if volume_sma[-1] > 0 else 1
        volume_score = min(100, volume_ratio * 50)

        # Composite momentum score
        momentum = (rsi_score * 0.4 + roc_score * 0.4 + volume_score * 0.2)
        return round(momentum, 2)

class AlertSystem:
    """Alert system for trading opportunities"""

    @staticmethod
    def check_alert_conditions(result: Dict) -> List[str]:
        """Check if any alert conditions are met"""
        alerts = []

        confidence = result.get('confidence', 0)
        bot_type = result.get('bot_type', 'HOLD')
        expected_profit = result.get('expected_profit', '0%')

        # High confidence alerts
        if confidence >= 85:
            alerts.append(f"🚨 HIGH CONFIDENCE {bot_type} signal for {result['symbol']}")

        # High profit potential
        try:
            profit_value = float(expected_profit.replace('%', ''))
            if profit_value >= 3.0:
                alerts.append(f"💰 HIGH PROFIT potential: {expected_profit} for {result['symbol']}")
        except:
            pass

        # Strong trend alignment
        trends = [result.get(f'trend_{tf}') for tf in ['1h', '15m', '5m', '1m']]
        if len(set(trends)) == 1 and trends[0] in ['Upward', 'Downward']:
            alerts.append(f"📈 STRONG TREND alignment ({trends[0]}) for {result['symbol']}")

        return alerts

def format_currency(value: float, decimals: int = 4) -> str:
    """Format currency values"""
    if value >= 1:
        return f"${value:,.{decimals}f}"
    else:
        return f"${value:.{decimals}f}"

def format_percentage(value: float) -> str:
    """Format percentage values"""
    return f"{value:+.2f}%"

def get_trend_emoji(trend: str) -> str:
    """Get emoji for trend direction"""
    trend_emojis = {
        'Upward': '📈',
        'Downward': '📉',
        'Ranging': '↔️',
        'Unknown': '❓'
    }
    return trend_emojis.get(trend, '❓')

def get_bot_emoji(bot_type: str) -> str:
    """Get emoji for bot type"""
    bot_emojis = {
        'LONG': '🚀',
        'SHORT': '🔻',
        'RANGE': '↔️',
        'HOLD': '⏸️'
    }
    return bot_emojis.get(bot_type, '⏸️')

@st.cache_data(ttl=60)
def cached_api_call(url: str, params: Dict = None) -> Dict:
    """Cached API call to reduce requests"""
    try:
        response = requests.get(url, params=params, timeout=10)
        return response.json()
    except Exception as e:
        st.error(f"API call failed: {e}")
        return {}

def display_performance_metrics(results: List[Dict]):
    """Display overall performance metrics"""
    if not results:
        return

    total_results = len(results)
    high_confidence = len([r for r in results if r['confidence'] >= 80])
    avg_confidence = np.mean([r['confidence'] for r in results])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Analyzed", total_results)
    with col2:
        st.metric("High Confidence", f"{high_confidence}/{total_results}")
    with col3:
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")

def create_summary_chart(results: List[Dict]):
    """Create summary chart of bot recommendations"""
    if not results:
        return

    bot_counts = {}
    for result in results:
        bot_type = result['bot_type']
        bot_counts[bot_type] = bot_counts.get(bot_type, 0) + 1

    if bot_counts:
        fig = px.pie(
            values=list(bot_counts.values()),
            names=list(bot_counts.keys()),
            title="Bot Strategy Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)