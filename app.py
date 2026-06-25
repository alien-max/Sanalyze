import streamlit as st
import pandas as pd
import numpy as np
from data.yfinance_loader import load_ohlcv, get_symbol_info
from features.engineering import prepare_dataset, get_feature_columns
from models.ml_models import (
    train_all_models, evaluate_models,
    get_feature_importance, forecast_next_n_days, split_train_test,
)
from visualization.charts import (
    plot_price_with_forecast, plot_model_comparison,
    plot_feature_importance, build_summary_table, build_metrics_table,
)

POPULAR_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'INTC', 'PLTR', 'UBER', 'COIN', 'JPM', 'V', 'WMT', 'DIS', 'BA', 'NKE', 'PYPL', 'GC=F', 'SI=F', 'CL=F', 'BZ=F', 'NG=F', 'HG=F', 'ZC=F', 'ZS=F', 'ZW=F', 'KC=F', 'CT=F', 'CC=F', 'LE=F', 'HE=F', 'ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'DX=F', 'ZT=F', 'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'TRX-USD', 'AVAX-USD', 'DOT-USD', 'LINK-USD', 'MATIC-USD', 'LTC-USD', 'BCH-USD', 'ATOM-USD', 'UNI-USD', 'ETC-USD', 'XLM-USD', 'APT-USD', 'PEPE-USD']

st.set_page_config(
    page_title="Sanalyze",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("📈 Sanalyze")
    st.caption("ML-powered price & direction forecasting")
    st.divider()

    symbol = st.selectbox(
        "Ticker symbol",
        options=POPULAR_SYMBOLS,
        index=None,
        accept_new_options=True,
        placeholder="Type symbol... (AAPL, BTC-USD, TSLA)"
    )
    symbol = (symbol or "").upper().strip()
    horizon = st.slider("Forecast horizon (days)", min_value=1, max_value=30, value=7, step=1,)

    st.divider()
    run = st.button("▶ Run forecast", type="primary", use_container_width=True)

st.title(f"Market forecast — {symbol}")

if not run:
    st.info("Configure the symbol and horizon in the sidebar, then click **Run forecast**.")
    st.stop()


with st.spinner(f"Downloading data for {symbol}…"):
    try:
        raw_df = load_ohlcv(symbol, period='max')
        info = get_symbol_info(symbol)
    except ValueError as e:
        st.error(str(e))
        st.stop()

if len(raw_df) < 100:
    st.error(f"Not enough data for '{symbol}' ({len(raw_df)} rows). Try a different symbol.")
    st.stop()

with st.spinner("Engineering features…"):
    df = prepare_dataset(raw_df, horizon=1, lags=20)
    feature_cols = get_feature_columns(df)

with st.spinner("Training models…"):
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = split_train_test(df, feature_cols)
    trained = train_all_models(X_train, y_reg_train, y_cls_train)

with st.spinner("Evaluating…"):
    eval_results = evaluate_models(trained, X_test, y_reg_test, y_cls_test)
    importances = get_feature_importance(trained, feature_cols)

with st.spinner(f"Forecasting next {horizon} days…"):
    forecasts = forecast_next_n_days(trained, df, feature_cols, horizon)
    reg_forecasts = forecasts["regression"]
    cls_forecasts = forecasts["classification"]

last_price = float(raw_df["Close"].iloc[-1])
prev_price = float(raw_df["Close"].iloc[-2])
price_change = (last_price - prev_price) / prev_price * 100

future_dates = pd.bdate_range(start=raw_df.index[-1], periods=horizon + 1)[1:]

all_preds = np.array(list(reg_forecasts.values()))
mean_final = float(all_preds[:, -1].mean())
ensemble_change = (mean_final - last_price) / last_price * 100

up_votes = sum(1 for preds in cls_forecasts.values() if preds[-1] == 1)
consensus = "▲ Up" if up_votes > len(cls_forecasts) // 2 else "▼ Down"
consensus_cls = "up" if "Up" in consensus else "down"

col1, col2, col3, col4, col5 = st.columns(5)

def _kpi(col, label, value, sub="", cls=""):
    col.markdown(
        f'<div class="metric-card"><div class="val {cls}">{value}</div>'
        f'<div class="lbl">{label}</div>'
        f'{"<div class=\"lbl\">" + sub + "</div>" if sub else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )

_kpi(col1, "Last price", f"{last_price:,.4f}", info.get("currency", ""))
_kpi(col2, "1-day change", f"{price_change:+.2f}%", cls="up" if price_change >= 0 else "down")
_kpi(col3, f"Ensemble in {horizon}d", f"{mean_final:,.4f}", f"{ensemble_change:+.2f}%", cls="up" if ensemble_change >= 0 else "down")
_kpi(col4, "Consensus direction", consensus, f"{up_votes}/{len(cls_forecasts)} models", cls=consensus_cls)
_kpi(col5, "Training rows", f"{len(X_train):,}", f"{len(X_test):,} test rows")

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Forecast chart",
    "📋 Prediction table",
    "🏆 Model performance",
    "🔍 Feature importance",
])

with tab1:
    fig = plot_price_with_forecast(raw_df, reg_forecasts, horizon, symbol)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Direction consensus per day")
    cols = st.columns(min(horizon, 7))
    for i in range(min(horizon, len(future_dates))):
        up = sum(1 for preds in cls_forecasts.values() if preds[i] == 1)
        total = len(cls_forecasts)
        conf = max(up, total - up) / total * 100
        arrow = "▲" if up > total // 2 else "▼"
        color = "up" if up > total // 2 else "down"
        cols[i % 7].markdown(
            f'<div class="metric-card"><div class="val {color}">{arrow}</div>'
            f'<div class="lbl">{future_dates[i].strftime("%d %b")}</div>'
            f'<div class="lbl">{conf:.0f}% conf.</div></div>',
            unsafe_allow_html=True,
        )

with tab2:
    st.subheader("Detailed forecast table")
    summary_df = build_summary_table(reg_forecasts, cls_forecasts, last_price, future_dates)
    st.dataframe(summary_df, use_container_width=True)
    st.caption("▲ = predicted price increase vs previous day · ▼ = decrease · Ensemble Δ% relative to last close")

with tab3:
    st.subheader("Model performance on test set (last 20% of history)")
    reg_table, cls_table = build_metrics_table(eval_results)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Regression metrics**")
        st.dataframe(reg_table.style.highlight_min(subset=["RMSE", "MAE"], color="#1d4a35").highlight_max(subset=["R²"], color="#1d4a35"), use_container_width=True)
    with c2:
        st.markdown("**Classification metrics**")
        st.dataframe(cls_table.style.highlight_max(subset=["Accuracy", "F1"], color="#1d4a35"), use_container_width=True)

    st.plotly_chart(plot_model_comparison(eval_results), use_container_width=True)

with tab4:
    st.subheader("Feature importance (tree-based models)")

    if importances:
        fig_imp = plot_feature_importance(importances, top_n=20)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Based on impurity-based importance from regression models.")
    else:
        st.info("No feature importance available (SVR/SVC do not expose feature importances).")

st.divider()
st.caption(
    "⚠️ This tool is for educational purposes only. "
    "ML predictions are not financial advice. Past performance does not guarantee future results."
)