import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


MODEL_COLORS = {
    "Decision Tree": "#7C6FCD",
    "Random Forest": "#1D9E75",
    "AdaBoost": "#EF9F27",
    "CatBoost": "#D85A30",
    "SVR": "#D4537E",
    "SVC": "#D4537E",
}

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=13),
    margin=dict(l=16, r=16, t=40, b=16),
)

def plot_price_with_forecast(
    df: pd.DataFrame,
    reg_forecasts: dict,
    horizon: int,
    symbol: str,
    lookback_days: int = 120,
) -> go.Figure:
    recent = df.tail(lookback_days)
    last_date = recent.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=horizon + 1)[1:]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
    )

    fig.add_trace(
        go.Candlestick(
            x=recent.index,
            open=recent["Open"], high=recent["High"],
            low=recent["Low"], close=recent["Close"],
            name="Price",
            increasing_line_color="#1D9E75",
            decreasing_line_color="#D85A30",
            increasing_fillcolor="#1D9E75",
            decreasing_fillcolor="#D85A30",
            line=dict(width=1),
        ),
        row=1, col=1,
    )

    all_preds = np.array(list(reg_forecasts.values()))
    mean_pred = all_preds.mean(axis=0)
    std_pred = all_preds.std(axis=0)

    x_band = [last_date] + list(future_dates) + list(reversed(future_dates)) + [last_date]
    upper = [float(recent["Close"].iloc[-1])] + list(mean_pred + std_pred)
    lower = [float(recent["Close"].iloc[-1])] + list(mean_pred - std_pred)
    y_band = upper + list(reversed(lower)) + [float(recent["Close"].iloc[-1])]

    fig.add_trace(
        go.Scatter(
            x=x_band, y=y_band,
            fill="toself",
            fillcolor="rgba(120, 120, 200, 0.15)",
            line=dict(width=0),
            name="Confidence band",
            showlegend=True,
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    for name, preds in reg_forecasts.items():
        x_vals = [last_date] + list(future_dates)
        y_vals = [float(recent["Close"].iloc[-1])] + list(preds)
        color = MODEL_COLORS.get(name, "#888")
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=y_vals,
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2, dash="dot"),
                marker=dict(size=5),
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=[last_date] + list(future_dates),
            y=[float(recent["Close"].iloc[-1])] + list(mean_pred),
            mode="lines",
            name="Ensemble mean",
            line=dict(color="#ffffff", width=2.5),
        ),
        row=1, col=1,
    )

    colors = ["#1D9E75" if c >= o else "#D85A30" for c, o in zip(recent["Close"], recent["Open"])]
    fig.add_trace(
        go.Bar(x=recent.index, y=recent["Volume"], name="Volume", marker_color=colors, showlegend=False, opacity=0.6),
        row=2, col=1,
    )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=520,
        **CHART_THEME,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_xaxes(showgrid=False)

    return fig

def plot_model_comparison(eval_results: dict) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Regression metrics", "Classification metrics"),
        horizontal_spacing=0.12,
    )

    reg = eval_results["regression"]
    reg_names = list(reg.keys())
    metrics_reg = ["RMSE", "MAE"]
    for metric in metrics_reg:
        vals = [reg[n][metric] for n in reg_names]
        fig.add_trace(
            go.Bar(
                name=metric,
                x=reg_names,
                y=vals,
                marker_color=["#7C6FCD" if metric == "RMSE" else "#1D9E75"] * len(reg_names),
                text=[f"{v:.4f}" for v in vals],
                textposition="outside",
            ),
            row=1, col=1,
        )

    cls = eval_results["classification"]
    cls_names = list(cls.keys())
    metrics_cls = ["Accuracy", "F1"]
    for metric in metrics_cls:
        vals = [cls[n][metric] for n in cls_names]
        fig.add_trace(
            go.Bar(
                name=metric,
                x=cls_names,
                y=vals,
                marker_color=["#EF9F27" if metric == "Accuracy" else "#D85A30"] * len(cls_names),
                text=[f"{v:.3f}" for v in vals],
                textposition="outside",
            ),
            row=1, col=2,
        )

    fig.update_layout(
        barmode="group",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        **CHART_THEME,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=True, zerolinecolor="rgba(128,128,128,0.3)")

    return fig

def plot_feature_importance(importances: dict, top_n: int = 20) -> go.Figure:
    if not importances:
        return go.Figure()

    n_models = len(importances)
    fig = make_subplots(
        rows=1, cols=n_models,
        subplot_titles=list(importances.keys()),
        horizontal_spacing=0.1,
    )

    for i, (name, series) in enumerate(importances.items(), start=1):
        top = series.head(top_n)
        color = MODEL_COLORS.get(name, "#888")
        fig.add_trace(
            go.Bar(
                x=top.values,
                y=top.index,
                orientation="h",
                name=name,
                marker_color=color,
                showlegend=False,
            ),
            row=1, col=i,
        )

    fig.update_layout(
        height=max(380, top_n * 18),
        **CHART_THEME,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")

    return fig

def build_summary_table(
    reg_forecasts: dict,
    cls_forecasts: dict,
    last_price: float,
    future_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    rows = []
    n_days = len(future_dates)

    for i in range(n_days):
        row = {"Date": future_dates[i].strftime("%Y-%m-%d")}

        for name, preds in reg_forecasts.items():
            row[f"{name} price"] = round(preds[i], 4)

        for name, preds in cls_forecasts.items():
            row[f"{name} direction"] = "▲ Up" if preds[i] == 1 else "▼ Down"

        all_prices = [preds[i] for preds in reg_forecasts.values()]
        row["Ensemble price"] = round(float(np.mean(all_prices)), 4)
        row["Ensemble Δ%"] = round((row["Ensemble price"] - last_price) / last_price * 100, 2)

        up_votes = sum(1 for preds in cls_forecasts.values() if preds[i] == 1)
        total = len(cls_forecasts)
        row["Consensus"] = f"▲ Up ({up_votes}/{total})" if up_votes > total // 2 else f"▼ Down ({total-up_votes}/{total})"

        rows.append(row)

    return pd.DataFrame(rows).set_index("Date")

def build_metrics_table(eval_results: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    reg = eval_results["regression"]
    reg_rows = [
        {"Model": name, "RMSE": round(m["RMSE"], 4), "MAE": round(m["MAE"], 4), "R²": round(m["R²"], 4)}
        for name, m in reg.items()
    ]

    cls = eval_results["classification"]
    cls_rows = [
        {
            "Model": name,
            "Accuracy": round(m["Accuracy"], 4),
            "F1": round(m["F1"], 4),
            "Precision": round(m["Precision"], 4),
            "Recall": round(m["Recall"], 4),
        }
        for name, m in cls.items()
    ]

    return pd.DataFrame(reg_rows).set_index("Model"), pd.DataFrame(cls_rows).set_index("Model")