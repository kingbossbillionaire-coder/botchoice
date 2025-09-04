# Configuration file for Binance Futures Scanner

# API Configuration
BINANCE_BASE_URL = "https://api.binance.com"
REQUEST_TIMEOUT = 10

# Scanner Settings
DEFAULT_TOP_SYMBOLS = 30
DEFAULT_REFRESH_INTERVAL = 60  # seconds
DEFAULT_MIN_CONFIDENCE = 60

# Technical Analysis Parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
SUPPORT_RESISTANCE_WINDOW = 20

# Two-Pole Oscillator Settings
TWO_POLE_LENGTH = 20
SMA_PERIOD_1 = 25
SMA_PERIOD_2 = 25
STDEV_PERIOD = 25

# Bot Strategy Thresholds
LONG_BOT_THRESHOLDS = {
    'rsi_max': 40,
    'oscillator_max': -0.3,
    'min_confidence': 60
}

SHORT_BOT_THRESHOLDS = {
    'rsi_min': 60,
    'oscillator_min': 0.3,
    'min_confidence': 60
}

RANGE_BOT_THRESHOLDS = {
    'oscillator_range': (-0.5, 0.5),
    'rsi_range': (30, 70),
    'min_range_size': 0.02,
    'max_range_size': 0.08,
    'min_confidence': 50
}

# Risk Management
DEFAULT_STOP_LOSS_PERCENT = 3.0  # 3%
MIN_EXPECTED_PROFIT = 0.5  # 0.5%

# UI Configuration
CHART_HEIGHT = 600
MAX_DISPLAYED_RESULTS = 15

# Timeframes for analysis
TIMEFRAMES = ['1h', '15m', '5m', '1m']

# Color scheme
COLORS = {
    'long': '#00ff88',
    'short': '#ff4444',
    'range': '#ffaa00',
    'neutral': '#888888'
}