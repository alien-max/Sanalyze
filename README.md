# 📈 Market Predictor

A web application for forecasting financial market trends using multiple Machine Learning models.

## Features

- **10 ML models** — 5 regression (price prediction) + 5 classification (direction prediction)
- **Rich feature engineering** — 30+ technical indicators (RSI, MACD, Bollinger Bands, ATR, Stochastic, Williams %R, OBV, …) plus 20-day lag features
- **Any asset** — stocks (AAPL, NVDA, …), crypto (BTC-USD, ETH-USD), forex (EURUSD=X), indices (^SPX)
- **Ensemble consensus** — mean forecast + confidence band across all models
- **Interactive charts** — candlestick + forecast overlay, model comparison, feature importance
- **Disk caching** — yfinance data cached for 4 hours to avoid redundant API calls

## Models

| Task | Models |
|------|--------|
| Regression (price) | Decision Tree · Random Forest · AdaBoost · CatBoost · SVR |
| Classification (direction) | Decision Tree · Random Forest · AdaBoost · CatBoost · SVC |

## Technical indicators

| Category | Indicators |
|----------|-----------|
| Trend | EMA 9/21/50, MACD, MACD Signal, MACD Histogram |
| Momentum | RSI(14), Stochastic %K/%D, Williams %R, ROC |
| Volatility | ATR(14), Bollinger Bands (upper/mid/lower/width/%B) |
| Volume | OBV, Volume EMA, Volume Ratio |
| Price-derived | Daily return, Log return, 52-week price position |
| Lag features | Close lag 1–20, Return lag 1–20, Rolling mean/std (5/10/20d) |

## Project structure

```
market-predictor/
├── app.py                        # Streamlit entry point
├── data/
│   └── yfinance_loader.py        # Data download + disk cache
├── features/
│   └── engineering.py            # All feature engineering
├── models/
│   └── ml_models.py              # Train, evaluate, forecast
├── visualization/
│   └── charts.py                 # Plotly charts & tables
└── requirements.txt
```

## Quickstart

```bash
git clone https://github.com/alien-max/Sanalyze
cd Sanalyze
pip install -r requirements.txt
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501), enter a ticker symbol (e.g. `AAPL` or `BTC-USD`), choose a forecast horizon, and click **Run forecast**.

## Disclaimer

This project is for **educational purposes only**. ML predictions are not financial advice. Past performance does not guarantee future results.