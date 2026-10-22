"""Feature engineering. Only information available BEFORE the outcome is used."""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURES = ["vessel_arrivals", "queue_length", "port_capacity",
                 "weather_severity", "historical_delay_hours"]
ENGINEERED = ["load_ratio", "weather_x_load", "season_sin", "season_cos"]
FORBIDDEN = {"actual_delay_hours", "congested"}   # outcome columns must never be features


def port_columns(port_ids) -> list[str]:
    return [f"port_{p}" for p in port_ids]


def feature_names(port_ids) -> list[str]:
    return BASE_FEATURES + ENGINEERED + port_columns(port_ids)


def feature_groups(port_ids) -> dict[str, list[str]]:
    """Columns that are summed into one human-readable factor in explanations."""
    return {"seasonality": ["season_sin", "season_cos"], "port_profile": port_columns(port_ids)}


def assert_no_leakage(names) -> None:
    bad = FORBIDDEN & set(names)
    if bad:
        raise ValueError(f"Leakage: outcome columns used as features: {bad}")


def build_features(df: pd.DataFrame, port_ids) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in BASE_FEATURES:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    cap = out["port_capacity"].where(out["port_capacity"] > 0)
    out["load_ratio"] = (out["vessel_arrivals"] + out["queue_length"]) / cap      # demand vs capacity
    out["weather_x_load"] = out["weather_severity"] * out["load_ratio"]           # interaction
    ang = 2 * np.pi * pd.to_datetime(df["timestamp"]).dt.dayofyear.to_numpy() / 365.25
    out["season_sin"], out["season_cos"] = np.sin(ang), np.cos(ang)               # seasonality
    for p in port_ids:
        out[f"port_{p}"] = (df["port_id"] == p).astype(float)                     # port variability
    names = feature_names(port_ids)
    assert_no_leakage(names)
    return out[names]
