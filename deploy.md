# Deployment Guide

## Option 1: Streamlit Cloud (Recommended - Easiest)

1. **Create GitHub Repository:**
   - Create a new repository on GitHub
   - Upload all the files from this project

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub account
   - Select your repository
   - Set main file path to `app.py`
   - Click "Deploy"

3. **Configuration:**
   - The app will automatically install dependencies from `requirements.txt`
   - No additional configuration needed

## Option 2: Vercel (Alternative)

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Deploy:**
   ```bash
   git clone <your-repo-url>
   cd binance-futures-scanner
   vercel
   ```

3. **Follow prompts:**
   - Link to your Vercel account
   - Set up project settings
   - Deploy

## Option 3: Heroku (Free tier discontinued, but still available)

1. **Create Heroku app:**
   ```bash
   heroku create your-app-name
   ```

2. **Add buildpack:**
   ```bash
   heroku buildpacks:add heroku/python
   ```

3. **Deploy:**
   ```bash
   git push heroku main
   ```

## Option 4: Railway

1. **Go to [railway.app](https://railway.app)**
2. **Connect GitHub repository**
3. **Deploy automatically**

## Option 5: Local Development

```bash
# Clone repository
git clone <your-repo-url>
cd binance-futures-scanner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Troubleshooting

### Common Issues:

1. **TA-Lib Installation Error:**
   - On Windows: Download wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
   - On Linux/Mac: `sudo apt-get install ta-lib` or `brew install ta-lib`

2. **Memory Issues:**
   - Reduce the number of symbols analyzed
   - Increase refresh interval
   - Disable chart display

3. **API Rate Limits:**
   - The app uses public endpoints with generous limits
   - If issues persist, add small delays between requests

### Performance Optimization:

- Use caching for API calls
- Limit historical data fetching
- Optimize indicator calculations
- Use session state for data persistence

## Environment Variables (Optional)

Create a `.env` file for custom configuration:

```
REFRESH_INTERVAL=60
MIN_CONFIDENCE=60
MAX_SYMBOLS=30
ENABLE_ALERTS=true
```

## Monitoring

- Check app logs for any errors
- Monitor API response times
- Watch for rate limiting issues
- Verify data accuracy periodically