"""Prediction engine: model loading, forecasting, recommendations, data helpers, reports."""
import io
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import ui
from features import (AREAS, BLUE, BUILDINGS, BUNDLE_VERSION, DATA_COLUMNS, DAYS, GREEN, AMBER, RED,
                      METERS_PER_UNIT, OPEN_HOURS, WALK_M_PER_MIN, ZONE_POS, build_features)
from train_model import DATA_PATH, MODEL_PATH, OBSERVED_PATH, train

HISTORY_PATH = DATA_PATH.parent / "prediction_history.csv"


# ------------------------------------------------------------------ model
def _try_load():
    try:
        with open(MODEL_PATH, "rb") as f:
            b = pickle.load(f)
        return b if isinstance(b, dict) and b.get("version") == BUNDLE_VERSION else None
    except Exception:
        return None


@st.cache_resource(show_spinner="Preparing model (first run only)...")
def load_bundle():
    return _try_load() or train()


def predict(bundle, hours, dow, area, event, rain):
    df = pd.DataFrame({"hour": list(hours), "day_of_week": dow, "area": area,
                       "special_event": event, "rain": rain})
    out = np.rint(bundle["model"].predict(build_features(df))).astype(int)
    return np.clip(out, 0, AREAS[area])


def status_of(available, capacity):
    if available <= 0:
        return "Parking Full", "🔴", "full"
    if available / capacity <= 0.20:
        return "Limited Availability", "🟠", "limited"
    return "Spaces Available", "🟢", "ok"


def snapshot(bundle, hour, dow, event, rain):
    hw, rows = bundle["metrics"]["interval"], []
    for area, cap in AREAS.items():
        occ = int(predict(bundle, [hour], dow, area, event, rain)[0])
        free = cap - occ
        rows.append({"Parking Area": area, "Capacity": cap, "Occupied": occ, "Available": free,
                     "Occupancy %": round(100 * occ / cap),
                     "Free Low": max(0, free - hw), "Free High": min(cap, free + hw),
                     "Status": status_of(free, cap)[0]})
    return pd.DataFrame(rows)


def recommend(bundle, destination, hour, dow, event, rain):
    """Rank zones by walking time + a penalty when a zone is nearly full."""
    snap = snapshot(bundle, hour, dow, event, rain)
    bx, by = BUILDINGS[destination]
    walk_m = [int(np.hypot(ZONE_POS[a][0] - bx, ZONE_POS[a][1] - by) * METERS_PER_UNIT)
              for a in snap["Parking Area"]]
    snap["Walk (m)"] = walk_m
    snap["Walk (min)"] = np.round(np.array(walk_m) / WALK_M_PER_MIN, 1)
    ratio = snap["Available"] / snap["Capacity"]
    snap["Score"] = (snap["Walk (min)"] + 10 * np.clip((0.30 - ratio) / 0.30, 0, None)).round(1)
    snap = snap[snap["Available"] > 0].sort_values("Score").reset_index(drop=True)
    snap.insert(0, "Rank", range(1, len(snap) + 1))
    return snap


def day_grid(bundle, dow, event, rain):
    """Predicted occupancy % : rows = zones, columns = hours."""
    return pd.DataFrame({h: [100 * predict(bundle, [h], dow, a, event, rain)[0] / c
                             for a, c in AREAS.items()] for h in OPEN_HOURS}, index=list(AREAS))


def week_grid(bundle, area, event, rain):
    """Predicted FREE spaces : rows = weekdays, columns = hours."""
    cap = AREAS[area]
    return pd.DataFrame({h: [cap - predict(bundle, [h], d, area, event, rain)[0] for d in range(7)]
                         for h in OPEN_HOURS}, index=DAYS)


def make_pdf(bundle, dow, hour, event, rain):
    snap = snapshot(bundle, hour, dow, event, rain)
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.05, 0.93, "ParkWise - Campus Parking Report", fontsize=20, fontweight="bold", color="#11213F")
        fig.text(0.05, 0.895, f"{DAYS[dow]}  |  {hour:02d}:00  |  Event: {'Yes' if event else 'No'}  |  "
                 f"Weather: {'Rainy' if rain else 'Clear'}  |  Generated {datetime.now():%d %b %Y %H:%M}",
                 fontsize=10, color="#64748B")
        ax = fig.add_axes([0.05, 0.52, 0.9, 0.32]); ax.axis("off")
        cols = ["Parking Area", "Capacity", "Occupied", "Available", "Occupancy %", "Status"]
        t = ax.table(cellText=snap[cols].values, colLabels=cols, loc="center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 2)
        for (r, _), cell in t.get_celld().items():
            if r == 0:
                cell.set_facecolor("#11213F"); cell.set_text_props(color="white", fontweight="bold")
        ax2 = fig.add_axes([0.09, 0.08, 0.84, 0.36])
        ax2.bar(snap["Parking Area"], snap["Occupied"], color=BLUE, label="Occupied")
        ax2.bar(snap["Parking Area"], snap["Available"], bottom=snap["Occupied"], color="#CBD5E1", label="Available")
        ax2.set_ylabel("Spaces"); ax2.legend(frameon=False); ui.style_ax(ax2)
        pdf.savefig(fig); plt.close(fig)
        fig = ui.draw_heat(day_grid(bundle, dow, event, rain), f"Predicted occupancy - {DAYS[dow]}",
                           figsize=(11.69, 4.5))
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig = ui.draw_map(snap); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return buf.getvalue()


# ------------------------------------------------------------------ history
def read_history():
    try:
        return pd.read_csv(HISTORY_PATH)
    except Exception:
        return pd.DataFrame()


def save_record(rec):
    new = pd.DataFrame([rec])
    out = pd.concat([read_history(), new], ignore_index=True) if HISTORY_PATH.exists() else new
    out.to_csv(HISTORY_PATH, index=False)


# ------------------------------------------------------------------ real observations
def read_observed():
    try:
        return pd.read_csv(OBSERVED_PATH)
    except Exception:
        return pd.DataFrame(columns=DATA_COLUMNS)


def append_observed(rows: pd.DataFrame):
    out = pd.concat([read_observed(), rows[DATA_COLUMNS]], ignore_index=True)
    out.to_csv(OBSERVED_PATH, index=False)
    return len(out)


def validate_upload(df: pd.DataFrame):
    """Returns (clean_rows, problem_message or None)."""
    missing = [c for c in DATA_COLUMNS if c not in df.columns]
    if missing:
        return None, f"Missing columns: {', '.join(missing)}"
    d = df[DATA_COLUMNS].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ("hour", "special_event", "rain", "occupied"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    d = d[d["area"].isin(AREAS) & d["hour"].between(0, 23) & d["special_event"].isin([0, 1])
          & d["rain"].isin([0, 1]) & (d["occupied"] >= 0)]
    d = d[d["occupied"] <= d["area"].map(AREAS)]
    d["date"] = d["date"].dt.date.astype(str)
    d[["hour", "special_event", "rain", "occupied"]] = d[["hour", "special_event", "rain", "occupied"]].astype(int)
    dropped = len(df) - len(d)
    return d, (f"{dropped} invalid row(s) were skipped." if dropped else None)
