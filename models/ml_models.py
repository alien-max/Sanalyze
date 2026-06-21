import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score
)
from catboost import CatBoostRegressor, CatBoostClassifier

REGRESSION_MODELS = {
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "CatBoost": CatBoostRegressor(iterations=100, depth=6, learning_rate=0.05, random_seed=42, verbose=0),
    "SVR": Pipeline([("scaler", StandardScaler()), ("model", SVR(C=1.0, epsilon=0.1, kernel="rbf"))]),
}

CLASSIFICATION_MODELS = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "CatBoost": CatBoostClassifier(iterations=100, depth=6, learning_rate=0.05, random_seed=42, verbose=0),
    "SVC": Pipeline([("scaler", StandardScaler()), ("model", SVC(C=1.0, kernel="rbf", probability=True))]),
}

def train_all_models(
    X_train: pd.DataFrame,
    y_reg: pd.Series,
    y_cls: pd.Series,
) -> dict:
    trained = {"regression": {}, "classification": {}}

    for name, model in REGRESSION_MODELS.items():
        m = _clone_model(model)
        m.fit(X_train, y_reg)
        trained["regression"][name] = m

    for name, model in CLASSIFICATION_MODELS.items():
        m = _clone_model(model)
        m.fit(X_train, y_cls)
        trained["classification"][name] = m

    return trained

def _clone_model(model):
    from sklearn.base import clone as sk_clone
    try:
        return sk_clone(model)
    except Exception:
        import copy
        return copy.deepcopy(model)

def evaluate_models(
    trained: dict,
    X_test: pd.DataFrame,
    y_reg_test: pd.Series,
    y_cls_test: pd.Series,
) -> dict:
    results = {"regression": {}, "classification": {}}
    for name, model in trained["regression"].items():
        preds = model.predict(X_test)
        results["regression"][name] = {
            "MAE": mean_absolute_error(y_reg_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_reg_test, preds)),
            "R²": r2_score(y_reg_test, preds),
            "predictions": preds,
        }

    for name, model in trained["classification"].items():
        preds = model.predict(X_test)
        results["classification"][name] = {
            "Accuracy": accuracy_score(y_cls_test, preds),
            "F1": f1_score(y_cls_test, preds, zero_division=0),
            "Precision": precision_score(y_cls_test, preds, zero_division=0),
            "Recall": recall_score(y_cls_test, preds, zero_division=0),
            "predictions": preds,
        }

    return results

def get_feature_importance(trained: dict, feature_names: list) -> dict:
    importances = {}
    for name, model in trained["regression"].items():
        raw = model
        if hasattr(model, "named_steps"):
            raw = model.named_steps.get("model", model)
        if hasattr(raw, "feature_importances_"):
            importances[name] = pd.Series(
                raw.feature_importances_, index=feature_names
            ).sort_values(ascending=False)
    return importances

def forecast_next_n_days(
    trained: dict,
    df_full: pd.DataFrame,
    feature_cols: list,
    horizon: int,
) -> dict:
    last_row = df_full[feature_cols].iloc[-1:].copy()
    last_price = df_full["Close"].iloc[-1]

    reg_forecasts = {name: [] for name in trained["regression"]}
    cls_forecasts = {name: [] for name in trained["classification"]}

    current_row = last_row.copy()
    current_price = last_price

    for _ in range(horizon):
        for name, model in trained["regression"].items():
            pred = float(model.predict(current_row)[0])
            reg_forecasts[name].append(pred)

        for name, model in trained["classification"].items():
            pred = int(model.predict(current_row)[0])
            cls_forecasts[name].append(pred)

        next_row = current_row.copy()
        avg_reg_pred = np.mean([v[-1] for v in reg_forecasts.values()])
        next_return = (avg_reg_pred - current_price) / (current_price + 1e-10)

        lag_close_cols = sorted([c for c in feature_cols if c.startswith("close_lag_")], key=lambda x: int(x.split("_")[-1]))
        lag_return_cols = sorted([c for c in feature_cols if c.startswith("return_lag_")], key=lambda x: int(x.split("_")[-1]))

        for i in range(len(lag_close_cols) - 1, 0, -1):
            next_row[lag_close_cols[i]] = next_row[lag_close_cols[i - 1]]
        if lag_close_cols:
            next_row[lag_close_cols[0]] = current_price

        for i in range(len(lag_return_cols) - 1, 0, -1):
            next_row[lag_return_cols[i]] = next_row[lag_return_cols[i - 1]]
        if lag_return_cols:
            next_row[lag_return_cols[0]] = next_return

        current_price = avg_reg_pred
        current_row = next_row

    return {"regression": reg_forecasts, "classification": cls_forecasts}

def split_train_test(
    df: pd.DataFrame,
    feature_cols: list,
    test_ratio: float = 0.2,
) -> tuple:
    n = len(df)
    split = int(n * (1 - test_ratio))

    X = df[feature_cols]
    y_reg = df["target_reg"]
    y_cls = df["target_cls"]

    return (
        X.iloc[:split], X.iloc[split:],
        y_reg.iloc[:split], y_reg.iloc[split:],
        y_cls.iloc[:split], y_cls.iloc[split:],
    )