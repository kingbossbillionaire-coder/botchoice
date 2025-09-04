# Binance Futures Scanner

A real-time Binance futures trading scanner that analyzes the top 30 assets and recommends profitable bot strategies using advanced technical analysis including the Two-Pole Oscillator indicator.

## Features

- 📊 Real-time analysis of top 30 Binance futures by volume
- 🤖 Bot strategy recommendations (Long, Short, Range)
- 📈 Multi-timeframe trend analysis (1h, 15m, 5m, 1m)
- 🎯 Support/resistance level detection
- 💰 Expected profit calculations with stop-loss levels
- 🌙 Dark/Light mode toggle
- ⚡ Auto-refresh every minute
- 📱 Responsive design

## Technical Indicators Used

- Two-Pole Oscillator (custom implementation from Pine Script)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Support/Resistance levels

## Quick Start

### Deploy on Streamlit Cloud (Recommended)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy from your forked repository
5. Set the main file path to `app.py`

### Deploy on Vercel

1. Install Vercel CLI: `npm i -g vercel`
2. Clone this repository
3. Run `vercel` in the project directory
4. Follow the deployment prompts

### Local Development

```bash
# Clone the repository
git clone <your-repo-url>
cd binance-futures-scanner

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Configuration

The app automatically fetches data from Binance's public API. No API keys required for basic functionality.

### Settings Available:
- Refresh interval (1-5 minutes)
- Minimum confidence threshold
- Bot type filters (Long/Short/Range)
- Chart display toggle

## Bot Strategy Logic

### Long Bot
- Triggered when RSI < 40, Two-Pole Oscillator < -0.3
- Price below Bollinger Band middle line
- Recent buy signals from oscillator

### Short Bot
- Triggered when RSI > 60, Two-Pole Oscillator > 0.3
- Price above Bollinger Band middle line
- Recent sell signals from oscillator

### Range Bot
- Triggered when oscillator is neutral (-0.5 to 0.5)
- Clear support/resistance levels identified
- RSI between 30-70 (neutral zone)

## Data Sources

- Binance Futures API for real-time price data
- 24h volume data for ranking
- Multiple timeframe analysis for trend confirmation

## Disclaimer

This tool is for educational and informational purposes only. Always do your own research and never invest more than you can afford to lose. Past performance does not guarantee future results.

## License

MIT License - Feel free to modify and distribute.