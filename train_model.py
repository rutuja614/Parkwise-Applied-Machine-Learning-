"""Train, compare and save the parking-occupancy model.

    python train_model.py               # train (creates sample data if missing)
    python train_model.py --regenerate  # rebuild the sample dataset first

Real data: data/parking_data.csv and data/observed_log.csv use the columns
    date, hour, area, special_event, rain, occupied
Rows in observed_log.csv (collected in the app's Data Lab) are weighted higher.
"""
import argparse
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor

from features import (AREAS, BUNDLE_VERSION, DATA_COLUMNS, FEATURE_COLUMNS,
                      OPEN_HOURS, build_features)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "parking_data.csv"
OBSERVED_PATH = BASE_DIR / "data" / "observed_log.csv"
MODEL_PATH = BASE_DIR / "models" / "parking_model.pkl"
REAL_WEIGHT = 20  # one real observation counts as 20 synthetic rows

PROFILES = {
    "Main Gate Parking": dict(base=0.15, peaks=[(9.5, 1.5, 0.70), (16.5, 1.5, 0.35)], weekend=0.35),
    "Library Parking":   dict(base=0.15, peaks=[(11, 2.5, 0.40), (15, 2.5, 0.45)],   weekend=0.55),
    "Canteen Parking":   dict(base=0.10, peaks=[(13, 1.2, 0.75), (17, 1.0, 0.25)],   weekend=0.40),
    "Staff Parking":     dict(base=0.30, peaks=[(11, 3.5, 0.60)],                    weekend=0.20),
}

CANDIDATES = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=42),
}


def generate_dataset(days: int = 240, seed: int = 42) -> pd.DataFrame:
    """Synthetic sample data - NOT real measurements."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2026-01-01", periods=days):
        weekend = d.dayofweek >= 5
        event = int(rng.random() < (0.03 if weekend else 0.10))
        rain = int(rng.random() < 0.20)
        for area, cap in AREAS.items():
            p = PROFILES[area]
            for hour in OPEN_HOURS:
                rate = p["base"] + sum(h * np.exp(-0.5 * ((hour - c) / w) ** 2) for c, w, h in p["peaks"])
                if weekend:
                    rate *= p["weekend"]
                rate += 0.15 * event + 0.05 * rain + rng.normal(0, 0.04)
                rows.append((d.date().isoformat(), hour, area, event, rain,
                             int(round(np.clip(rate, 0.02, 1.0) * cap))))
    return pd.DataFrame(rows, columns=DATA_COLUMNS)


def _cv_mae(model, X, y, w, folds=3):
    scores = []
    for tr, va in KFold(folds, shuffle=True, random_state=42).split(X):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr])
        scores.append(mean_absolute_error(y.iloc[va], m.predict(X.iloc[va])))
    return float(np.mean(scores))


def train(regenerate: bool = False) -> dict:
    DATA_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    if regenerate or not DATA_PATH.exists():
        generate_dataset().to_csv(DATA_PATH, index=False)

    df = pd.read_csv(DATA_PATH)
    df["real"] = 0
    n_real = 0
    if OBSERVED_PATH.exists():
        obs = pd.read_csv(OBSERVED_PATH)
        if len(obs):
            obs["real"] = 1
            n_real = len(obs)
            df = pd.concat([df, obs[DATA_COLUMNS + ["real"]]], ignore_index=True)
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek

    X, y = build_features(df), df["occupied"]
    w = pd.Series(np.where(df["real"] == 1, REAL_WEIGHT, 1.0), index=df.index)
    tr_idx, te_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    X_tr, X_te, y_tr, y_te, w_tr = X.loc[tr_idx], X.loc[te_idx], y.loc[tr_idx], y.loc[te_idx], w.loc[tr_idx]

    comparison, fitted = [], {}
    for name, mdl in CANDIDATES.items():
        t0 = time.time()
        cv = _cv_mae(mdl, X_tr, y_tr, w_tr)
        m = clone(mdl).fit(X_tr, y_tr, sample_weight=w_tr)
        p = m.predict(X_te)
        fitted[name] = (m, p)
        comparison.append({"Model": name, "CV MAE": round(cv, 3),
                           "Test MAE": round(float(mean_absolute_error(y_te, p)), 3),
                           "Test R2": round(float(r2_score(y_te, p)), 4),
                           "Train time (s)": round(time.time() - t0, 1)})
    best = min(comparison, key=lambda r: r["CV MAE"])["Model"]
    best_model, best_pred = fitted[best]

    area_mean = y_tr.groupby(df.loc[tr_idx, "area"]).mean()
    baseline = df.loc[te_idx, "area"].map(area_mean)
    resid = np.abs(y_te.values - best_pred)
    area_mae = {a: round(float(resid[(df.loc[te_idx, "area"] == a).values].mean()), 2) for a in AREAS}

    sample = X_te.sample(min(1500, len(X_te)), random_state=1)
    perm = permutation_importance(best_model, sample, y_te.loc[sample.index], n_repeats=3,
                                  random_state=42, scoring="neg_mean_absolute_error")
    imp = np.clip(perm.importances_mean, 0, None)
    imp = imp / imp.sum() if imp.sum() > 0 else imp

    show = np.random.default_rng(3).choice(len(y_te), min(300, len(y_te)), replace=False)
    final = clone(CANDIDATES[best]).fit(X, y, sample_weight=w)  # deploy on all data

    bundle = {
        "version": BUNDLE_VERSION, "model": final, "best_name": best,
        "feature_columns": FEATURE_COLUMNS, "comparison": comparison,
        "metrics": {"mae": float(mean_absolute_error(y_te, best_pred)),
                    "r2": float(r2_score(y_te, best_pred)),
                    "baseline_mae": float(mean_absolute_error(y_te, baseline)),
                    "interval": int(np.ceil(np.percentile(resid, 90))),
                    "train_rows": int(len(X_tr)), "test_rows": int(len(X_te))},
        "importances": dict(zip(FEATURE_COLUMNS, map(float, imp))),
        "area_mae": area_mae,
        "test_sample": {"actual": y_te.values[show].tolist(), "pred": best_pred[show].round(1).tolist()},
        "real_rows": n_real,
        "data_source": ("Synthetic sample data" if not n_real else
                        f"Synthetic sample data + {n_real} real observations (weight x{REAL_WEIGHT})"),
        "trained_at": datetime.now().strftime("%d %b %Y, %H:%M"),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    write_report(bundle)
    return bundle


def write_report(b: dict) -> None:
    """Developer-only diagnostics. The web app never displays any of this."""
    m = b["metrics"]
    lines = [f"ParkWise training report - {b['trained_at']}", f"Data: {b['data_source']}",
             f"Rows: {m['train_rows']} train / {m['test_rows']} test", "",
             pd.DataFrame(b["comparison"]).to_string(index=False), "",
             f"Selected model : {b['best_name']}",
             f"Test MAE       : {m['mae']:.3f} spaces (baseline {m['baseline_mae']:.3f})",
             f"Test R2        : {m['r2']:.4f}",
             f"90% interval   : +/-{m['interval']} spaces", "",
             "MAE by zone    : " + ", ".join(f"{a}={v}" for a, v in b["area_mae"].items()), "",
             "Feature importance:"]
    lines += [f"  {k:<28}{v:.3f}" for k, v in sorted(b["importances"].items(), key=lambda kv: -kv[1])]
    (MODEL_PATH.parent / "training_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true")
    b = train(ap.parse_args().regenerate)
    print(pd.DataFrame(b["comparison"]).to_string(index=False))
    m = b["metrics"]
    print(f"\nBest model: {b['best_name']} | MAE {m['mae']:.2f} (baseline {m['baseline_mae']:.2f}) | "
          f"R2 {m['r2']:.3f} | 90% interval ±{m['interval']}")
    print(f"Saved to {MODEL_PATH}")
