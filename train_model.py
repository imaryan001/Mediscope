"""Compare models with leakage-safe rolling validation, then refit the winner."""
from pathlib import Path
import json
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")  # Avoid a slow, restricted Windows core-count probe.
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "salesdaily.csv"
MODEL_PATH = ROOT / "models" / "demand_models.joblib"
REPORT_PATH = ROOT / "reports" / "evaluation.json"
HORIZONS = (7, 30)
LAGS = (1, 2, 3, 7, 14, 21, 28, 35, 56, 365)
ROLLS = (7, 14, 28, 56)
FEATURES = ["weekday", "month", "day_of_year", "week_of_year", "week_sin", "week_cos", "year_sin", "year_cos"] + [f"lag_{n}" for n in LAGS] + [f"mean_{n}" for n in ROLLS] + [f"std_{n}" for n in (7, 28)]

def load_sales():
    """Read dates as month/day/year (as used in the source CSV), then make a daily panel."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run python download_data.py first.")
    raw = pd.read_csv(DATA_PATH)
    date_col = next((c for c in raw.columns if c.lower() == "datum"), raw.columns[0])
    raw[date_col] = pd.to_datetime(raw[date_col], dayfirst=False, errors="coerce")
    raw = raw.dropna(subset=[date_col]).set_index(date_col).sort_index()
    # Only the eight ATC columns are medicine series; Year/Month/Hour are metadata.
    known = [c for c in raw.columns if c.upper() in {"M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"}]
    sales = raw[known].apply(pd.to_numeric, errors="coerce").clip(lower=0)
    idx = pd.date_range(sales.index.min(), sales.index.max(), freq="D")
    return sales.groupby(level=0).sum().reindex(idx).fillna(0)

def make_features(y):
    """Every feature uses data from t-1 or earlier, never the day being forecast."""
    dates = y.index
    x = pd.DataFrame(index=dates)
    day_of_year = dates.dayofyear.to_numpy()
    x["weekday"] = dates.dayofweek
    x["month"] = dates.month
    x["day_of_year"] = day_of_year
    x["week_of_year"] = dates.isocalendar().week.to_numpy(dtype=int)
    x["week_sin"] = np.sin(2 * np.pi * dates.dayofweek / 7)
    x["week_cos"] = np.cos(2 * np.pi * dates.dayofweek / 7)
    x["year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    x["year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    past = y.shift(1)
    for lag in LAGS:
        x[f"lag_{lag}"] = y.shift(lag)
    for window in ROLLS:
        x[f"mean_{window}"] = past.rolling(window).mean()
    for window in (7, 28):
        x[f"std_{window}"] = past.rolling(window).std()
    return x.replace([np.inf, -np.inf], np.nan)

def make_target(y, horizon):
    """At date t, target is the sum of sales on t+1 ... t+horizon."""
    future = pd.concat([y.shift(-step) for step in range(1, horizon + 1)], axis=1)
    return future.sum(axis=1, min_count=horizon)

def model_options(seed=42):
    return {
        "ExtraTrees": ExtraTreesRegressor(n_estimators=120, min_samples_leaf=2, max_features=0.9, random_state=seed, n_jobs=1),
        "RandomForest": RandomForestRegressor(n_estimators=120, min_samples_leaf=2, max_features=0.9, random_state=seed, n_jobs=1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=160, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed),
    }

def score(actual, predicted):
    actual, predicted = np.asarray(actual), np.maximum(0, np.asarray(predicted))
    return {"mae": round(float(mean_absolute_error(actual, predicted)), 3),
            "wape_percent": round(float(np.abs(actual - predicted).sum() / max(np.abs(actual).sum(), 1e-9) * 100), 2)}

def weekly_profile_baseline(y, forecast_date, horizon):
    """Repeat the average recent demand for each weekday over the horizon."""
    history = y.loc[:forecast_date - pd.Timedelta(days=1)].tail(56)
    by_day = history.groupby(history.index.dayofweek).mean()
    days = pd.date_range(forecast_date, periods=horizon, freq="D")
    return float(sum(by_day.get(d.dayofweek, history.mean()) for d in days))

def run():
    sales = load_sales()
    if len(sales) < 700:
        raise ValueError("Need at least two years of daily data for rolling evaluation and yearly features.")
    # Three chronological validation blocks; train labels are purged before each block
    # so that none of their future target days overlap validation dates.
    fold_starts = [sales.index[-150], sales.index[-120]]
    models, report = {}, {"dataset": "Pharma Sales Data (Milan Zdravkovic; Kaggle/GitHub copy)",
                          "source_csv": DATA_PATH.name, "rows": len(sales),
                          "date_start": str(sales.index.min().date()), "date_end": str(sales.index.max().date()),
                          "validation": "3 rolling 30-day origin blocks; training labels purged by forecast horizon",
                          "medicine_groups": {}}
    for medicine in sales.columns:
        y = sales[medicine].astype(float)
        x = make_features(y)
        group_report = {}
        for horizon in HORIZONS:
            target = make_target(y, horizon)
            valid = x.notna().all(axis=1) & target.notna()
            candidates = list(model_options())
            oof = {name: {"actual": [], "prediction": []} for name in candidates}
            for fold_no, val_start in enumerate(fold_starts, 1):
                val_end = val_start + pd.Timedelta(days=30)
                val_idx = sales.index[(sales.index >= val_start) & (sales.index < val_end) & valid]
                train_idx = sales.index[(sales.index < val_start - pd.Timedelta(days=horizon)) & valid]
                if len(train_idx) < 365 or len(val_idx) < 10:
                    continue
                for name, estimator in model_options(seed=42 + fold_no).items():
                    estimator.fit(x.loc[train_idx, FEATURES], target.loc[train_idx])
                    pred = np.maximum(0, estimator.predict(x.loc[val_idx, FEATURES]))
                    oof[name]["actual"].extend(target.loc[val_idx].tolist())
                    oof[name]["prediction"].extend(pred.tolist())
            evaluated = {name: score(v["actual"], v["prediction"]) for name, v in oof.items() if v["actual"]}
            if not evaluated:
                continue
            baseline_actual, baseline_pred = [], []
            for fold_start in fold_starts:
                val_idx = sales.index[(sales.index >= fold_start) & (sales.index < fold_start + pd.Timedelta(days=30)) & valid]
                val_idx = val_idx[val_idx + pd.Timedelta(days=horizon) <= sales.index[-1]]
                for d in val_idx:
                    baseline_actual.append(float(target.loc[d]))
                    baseline_pred.append(weekly_profile_baseline(y, d, horizon))
            baseline_score = score(baseline_actual, baseline_pred)
            best_tree = min(evaluated, key=lambda name: evaluated[name]["wape_percent"])
            if baseline_score["wape_percent"] <= evaluated[best_tree]["wape_percent"]:
                winner = "WeeklyProfileBaseline"
                residuals = np.asarray(baseline_actual) - np.asarray(baseline_pred)
                final = None  # Keep the simpler seasonal baseline when it validates better.
                selected_score = baseline_score
            else:
                winner = best_tree
                final = model_options()[winner]
                train_idx = sales.index[valid]
                final.fit(x.loc[train_idx, FEATURES], target.loc[train_idx])
                residuals = np.asarray(oof[winner]["actual"]) - np.asarray(oof[winner]["prediction"])
                selected_score = evaluated[winner]
            models[(medicine, horizon)] = {"model": final, "name": winner,
                                           "residual_low": float(np.quantile(residuals, 0.10)),
                                           "residual_high": float(np.quantile(residuals, 0.90))}
            group_report[str(horizon)] = {"selected_model": winner,
                "rolling_validation": selected_score,
                "best_tree_model": best_tree,
                "best_tree_score": evaluated[best_tree],
                "weekly_profile_baseline": baseline_score,
                "candidate_scores": evaluated,
                "validation_forecasts": len(baseline_actual)}
        report["medicine_groups"][medicine] = group_report
    if not models:
        raise ValueError("No models were trained. Check source CSV format and date range.")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "columns": list(sales.columns), "features": FEATURES,
                 "last_sales": sales.tail(400), "labeled_dataset": report["dataset"]}, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Rows: {len(sales)} ({report['date_start']} to {report['date_end']})")
    print(f"Trained {len(models)} models; saved {MODEL_PATH}")
    print(f"Rolling evaluation saved: {REPORT_PATH}")
    for med, horizons in report["medicine_groups"].items():
        for horizon, result in horizons.items():
            print(f"{med:5s} | {horizon:>2s}d | {result['selected_model']:22s} WAPE={result['rolling_validation']['wape_percent']:6.2f}% | weekly baseline={result['weekly_profile_baseline']['wape_percent']:6.2f}%")

if __name__ == "__main__":
    run()
