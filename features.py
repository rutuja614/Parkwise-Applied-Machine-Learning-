"""Shared configuration and feature engineering (single source of truth)."""
import pandas as pd

BUNDLE_VERSION = 2

# ---- Campus configuration -- edit these to match YOUR campus --------------
AREAS = {
    "Main Gate Parking": 100,
    "Library Parking": 60,
    "Canteen Parking": 50,
    "Staff Parking": 40,
}
# Schematic map coordinates (0-100 grid). 1 unit = 10 m for walking distances.
ZONE_POS = {
    "Main Gate Parking": (50, 14),
    "Library Parking": (16, 56),
    "Canteen Parking": (84, 56),
    "Staff Parking": (76, 90),
}
BUILDINGS = {
    "Main Block": (50, 62),
    "Library": (18, 76),
    "Canteen": (82, 78),
    "Admin Office": (50, 88),
    "Labs": (22, 34),
    "Hostel": (88, 28),
}
WALK_M_PER_MIN = 80
METERS_PER_UNIT = 10

OPEN_HOURS = list(range(7, 21))  # 07:00 - 20:00
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
GREEN, AMBER, RED, BLUE, NAVY = "#10B981", "#F59E0B", "#EF4444", "#2563EB", "#11213F"

BASE_FEATURES = ["hour", "day_of_week", "is_weekend", "special_event", "rain", "capacity"]
FEATURE_COLUMNS = BASE_FEATURES + [f"area_{a}" for a in AREAS]
DATA_COLUMNS = ["date", "hour", "area", "special_event", "rain", "occupied"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df needs: hour, day_of_week (0=Mon), area, special_event (0/1), rain (0/1)."""
    X = pd.DataFrame(index=df.index)
    X["hour"] = df["hour"].astype(int)
    X["day_of_week"] = df["day_of_week"].astype(int)
    X["is_weekend"] = (X["day_of_week"] >= 5).astype(int)
    X["special_event"] = df["special_event"].astype(int)
    X["rain"] = df["rain"].astype(int)
    X["capacity"] = df["area"].map(AREAS).astype(int)
    for a in AREAS:
        X[f"area_{a}"] = (df["area"] == a).astype(int)
    return X[FEATURE_COLUMNS]
