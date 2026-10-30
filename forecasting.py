"""Shared helpers used by the command-line predictor and the dashboard."""
from pathlib import Path
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "demand_models.joblib"
DATA_PATH = ROOT / "data" / "salesdaily.csv"
LAGS, ROLLS = (1, 2, 3, 7, 14, 21, 28, 35, 56, 365), (7, 14, 28, 56)

def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run python train_model.py first.")
    return joblib.load(MODEL_PATH)

def load_history(csv_path=DATA_PATH):
    raw = pd.read_csv(csv_path)
    date_col = next((c for c in raw.columns if c.lower() == "datum"), raw.columns[0])
    raw[date_col] = pd.to_datetime(raw[date_col], dayfirst=False, errors="coerce")
    raw = raw.dropna(subset=[date_col]).set_index(date_col).sort_index()
    medicines = [c for c in raw.columns if c.upper() in {"M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"}]
    if not medicines:
        raise ValueError("CSV has no recognized medicine group columns.")
    sales = raw[medicines].apply(pd.to_numeric, errors="coerce").clip(lower=0)
    idx = pd.date_range(sales.index.min(), sales.index.max(), freq="D")
    return sales.groupby(level=0).sum().reindex(idx).fillna(0)

def forecast_one(bundle, history, medicine, horizon):
    """Return forecast quantity, historical residual range and chosen estimator."""
    if medicine not in history.columns:
        raise ValueError(f"Medicine group {medicine} is not in the sales history.")
    y = history[medicine].astype(float).asfreq("D", fill_value=0)
    if len(y) < 365:
        raise ValueError("At least 365 daily records are needed for this model.")
    day = y.index[-1] + pd.Timedelta(days=1)
    doy = day.dayofyear
    features = {"weekday": day.dayofweek, "month": day.month, "day_of_year": doy,
                "week_of_year": int(day.isocalendar().week),
                "week_sin": np.sin(2 * np.pi * day.dayofweek / 7),
                "week_cos": np.cos(2 * np.pi * day.dayofweek / 7),
                "year_sin": np.sin(2 * np.pi * doy / 365.25),
                "year_cos": np.cos(2 * np.pi * doy / 365.25)}
    for lag in LAGS:
        features[f"lag_{lag}"] = float(y.iloc[-lag])
    for window in ROLLS:
        features[f"mean_{window}"] = float(y.tail(window).mean())
    for window in (7, 28):
        features[f"std_{window}"] = float(y.tail(window).std())
    saved = bundle["models"].get((medicine, horizon))
    if saved is None:
        raise ValueError(f"No trained {horizon}-day model for {medicine}.")
    if saved["name"] == "WeeklyProfileBaseline":
        recent = y.tail(56)
        weekday_average = recent.groupby(recent.index.dayofweek).mean()
        forecast_days = pd.date_range(day, periods=horizon, freq="D")
        raw = float(sum(weekday_average.get(d.dayofweek, recent.mean()) for d in forecast_days))
    else:
        row = pd.DataFrame([features])[bundle["features"]]
        raw = float(saved["model"].predict(row)[0])
    quantity = max(0, int(np.ceil(raw)))
    low = max(0, int(np.floor(raw + saved["residual_low"])))
    high = max(quantity, int(np.ceil(raw + saved["residual_high"])))
    return {"date": day, "quantity": quantity, "low": low, "high": high,
            "model": saved["name"], "daily_average": raw / horizon}
