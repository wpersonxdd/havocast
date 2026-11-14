"""Train next-day PM2.5 models for Tashkent and compare them with simple baselines on the latest year.

Split by date: train on everything before TEST_START, test on TEST_START onwards. Days are never shuffled.

Models that use tomorrow's weather are trained with the observed weather as a stand-in for a forecast,
then tested twice: with observed weather (a perfect forecast) and with the real day-ahead forecasts
that were issued at the time. The second number is the honest one.

The warning threshold for schools is chosen on the training years only (time-series cross-validation),
treating a missed smog day as MISS_COST times worse than a false alarm.

Writes outputs/results.json, outputs/test_predictions.csv and the demo model to outputs/model.joblib.
Usage: python src/train.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FORECAST_FEATURES, THRESHOLD, TODAY_FEATURES, build, real_forecast_features

OUT = Path(__file__).resolve().parent.parent / "outputs"
TEST_START = "2025-11-01"
WINTER = [11, 12, 1, 2]
MISS_COST = 3  # a smog day with children outside is treated as 3x worse than moving one PE class indoors


def warning_scores(y_true, y_pred, warn_at=THRESHOLD) -> dict:
    """How well a forecast warns about days whose real mean is above THRESHOLD."""
    real, warned = y_true > THRESHOLD, y_pred > warn_at
    caught = int((real & warned).sum())
    return {
        "bad_days": int(real.sum()),
        "caught": caught,
        "missed": int((real & ~warned).sum()),
        "false_alarms": int((~real & warned).sum()),
        "recall": round(caught / real.sum(), 3) if real.sum() else None,
        "precision": round(caught / warned.sum(), 3) if warned.sum() else None,
    }


def evaluate(name, y_true, y_pred, months, warn_at=THRESHOLD) -> dict:
    winter = np.isin(months, WINTER)
    return {
        "model": name,
        "mae": round(mean_absolute_error(y_true, y_pred), 2),
        "mae_winter": round(mean_absolute_error(y_true[winter], y_pred[winter]), 2),
        "mae_summer": round(mean_absolute_error(y_true[~winter], y_pred[~winter]), 2),
        "warn_at": warn_at,
        **warning_scores(y_true, y_pred, warn_at),
    }


def models():
    return {
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)),
        "boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                  min_samples_leaf=20, random_state=0),
    }


def choose_warning_threshold(model, X, y) -> float:
    """Pick the predicted level at which to warn, using only training years (no test data)."""
    oof_pred, oof_true = [], []
    for tr, va in TimeSeriesSplit(n_splits=5).split(X):
        m = clone(model).fit(X.iloc[tr], y[tr])
        oof_pred.append(m.predict(X.iloc[va]))
        oof_true.append(y[va])
    p, t = np.concatenate(oof_pred), np.concatenate(oof_true)
    best, best_cost = THRESHOLD, None
    for warn_at in np.arange(20, 40.5, 0.5):
        s = warning_scores(t, p, warn_at)
        cost = MISS_COST * s["missed"] + s["false_alarms"]
        if best_cost is None or cost < best_cost:
            best, best_cost = float(warn_at), cost
    return best


def main():
    OUT.mkdir(exist_ok=True)
    d = build().dropna(subset=["target", "pm_7day"])
    train, test = d[d.index < TEST_START], d[d.index >= TEST_START]
    y_tr, y_te = train["target"].to_numpy(), test["target"].to_numpy()
    months = (test.index + pd.Timedelta(days=1)).month.to_numpy()
    print(f"train {train.index.min().date()}..{train.index.max().date()} ({len(train)} days), "
          f"test {test.index.min().date()}..{test.index.max().date()} ({len(test)} days)")

    # Test rows with real day-ahead forecasts instead of observed tomorrow weather
    real_fc = real_forecast_features().reindex(test.index)
    test_real = test.copy()
    test_real[FORECAST_FEATURES] = real_fc[FORECAST_FEATURES].to_numpy()
    print("test days missing a real forecast:", int(real_fc.isna().any(axis=1).sum()))

    results, preds = [], {}
    preds["persistence"] = test["pm_today"].to_numpy()
    preds["monthly_average"] = train.groupby((train.index + pd.Timedelta(days=1)).month)["target"].mean() \
        .reindex(months).to_numpy()
    for name in ["persistence", "monthly_average"]:
        results.append(evaluate(name, y_te, preds[name], months))

    fitted = {}
    for feat_name, feats in [("today", TODAY_FEATURES), ("today+forecast", TODAY_FEATURES + FORECAST_FEATURES)]:
        for m_name, model in models().items():
            model.fit(train[feats], y_tr)
            name = f"{m_name} ({feat_name})"
            fitted[name] = (model, feats)
            if feat_name == "today":
                preds[name] = model.predict(test[feats])
                results.append(evaluate(name, y_te, preds[name], months))
            else:
                preds[name + " perfect weather"] = model.predict(test[feats])
                results.append(evaluate(name + " perfect weather", y_te, preds[name + " perfect weather"], months))
                preds[name + " real forecast"] = model.predict(test_real[feats])
                results.append(evaluate(name + " real forecast", y_te, preds[name + " real forecast"], months))

    # The demo model: boosting with tomorrow's forecast, plus a warning threshold tuned for schools
    demo_name = "boosting (today+forecast)"
    demo_model, demo_feats = fitted[demo_name]
    warn_at = choose_warning_threshold(demo_model, train[demo_feats], y_tr)
    results.append(evaluate(f"{demo_name} real forecast, school warning", y_te,
                            preds[demo_name + " real forecast"], months, warn_at))
    results.append(evaluate("persistence, school warning", y_te, preds["persistence"], months, warn_at))

    table = pd.DataFrame(results).set_index("model")
    with pd.option_context("display.width", 200):
        print(table.to_string())
    print(f"school warning threshold (chosen on training years): {warn_at} ug/m3")

    joblib.dump({"model": demo_model, "features": demo_feats, "name": demo_name, "warn_at": warn_at},
                OUT / "model.joblib")
    pd.DataFrame({"date": (test.index + pd.Timedelta(days=1)).date, "actual": y_te, **preds}) \
        .to_csv(OUT / "test_predictions.csv", index=False)
    (OUT / "results.json").write_text(json.dumps({
        "test_start": TEST_START, "threshold": THRESHOLD, "warn_at": warn_at, "miss_cost": MISS_COST,
        "results": results}, indent=1))


if __name__ == "__main__":
    main()
