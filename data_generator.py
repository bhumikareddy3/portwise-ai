"""Synthetic port-congestion data generator (fully reproducible).

HOW THE DATA IS SIMULATED  (see SIMULATION_NOTES, also shown in the app)
-----------------------------------------------------------------------
Each row is a *snapshot* of one port taken every few days. Per snapshot we
simulate a hidden "true" state, then expose only noisy/partial observations
as features. Ground truth is the simulated delay, not a function of the
observed columns, so the model cannot trivially reverse-engineer the label.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config

# Port archetypes. Capacity / arrivals are "vessels per ~48h planning window".
PORT_PROFILES = {
    "INNSA": dict(name="Nhava Sheva (JNPT)", capacity=26, arrivals=22, weather_sens=1.0,
                  labor_risk=0.06, base_delay=5.0, peak_doy=200),
    "INMAA": dict(name="Chennai",            capacity=18, arrivals=14, weather_sens=1.2,
                  labor_risk=0.05, base_delay=4.5, peak_doy=300),
    "SGSIN": dict(name="Singapore",          capacity=40, arrivals=34, weather_sens=0.5,
                  labor_risk=0.02, base_delay=3.0, peak_doy=330),
    "NLRTM": dict(name="Rotterdam",          capacity=34, arrivals=27, weather_sens=0.8,
                  labor_risk=0.08, base_delay=4.0, peak_doy=15),
    "USLAX": dict(name="Los Angeles",        capacity=30, arrivals=26, weather_sens=0.6,
                  labor_risk=0.07, base_delay=5.5, peak_doy=45),
}

# Simulator constants (kept explicit so the report can quote them)
CARRYOVER = 0.5          # share of last snapshot's unserved backlog still queued now
WEATHER_CAP_LOSS = 0.04  # capacity lost per weather-severity point (x port sensitivity)
LABOR_CAP_FACTOR = 0.80  # capacity multiplier during a (hidden) labour disruption
LABOR_EXTRA_DELAY = 2.0  # extra hours during a labour disruption
LOAD_KNEE = 1.15         # delay starts rising once demand/capacity exceeds this
DELAY_SLOPE = 20.0       # hours of delay per unit of load above the knee
WEATHER_DELAY = 0.8      # direct delay (pilotage/berthing suspension) above severity 6
NOISE_SIGMA = 0.30       # multiplicative log-normal noise on delay
WEATHER_SENSOR_SD = 0.6  # measurement noise on observed weather severity

SIMULATION_NOTES = f"""
**Congestion label:** `congested = actual_delay_hours > threshold` (default {config.CONGESTION_THRESHOLD_HOURS:g} h, configurable).

**Per snapshot (port, time) the simulator draws:**
1. *Weather (hidden true severity, 0-10)* = seasonal baseline (port-specific storm season) + AR(1) persistence + rare storms (6%/snapshot).
   The observed `weather_severity` adds sensor noise (SD {WEATHER_SENSOR_SD}) and ~{config.MISSING_RATE_WEATHER:.0%} of values are missing.
2. *Vessel arrivals* ~ Poisson(port base rate x seasonal x weekday uplift), with 8% chance of a 35% demand surge.
3. *Port capacity* = port base x U(0.90, 1.05), with a 7% chance of a maintenance cut (-25%).
4. *Hidden labour disruption* (never given to the model): persistent Markov state that cuts capacity by 20% and adds {LABOR_EXTRA_DELAY} h.
5. *Queue length* = {CARRYOVER:.0%} of the previous snapshot's unserved backlog + Poisson(1) (only PAST information).
   ~{config.MISSING_RATE_QUEUE:.0%} of values are missing.
6. *Effective capacity* = capacity x (1 - {WEATHER_CAP_LOSS} x weather x port sensitivity) x labour factor.
7. *Load* = (arrivals + queue) / effective capacity.
8. *Delay (h)* = port base delay + {DELAY_SLOPE:g} x max(0, load - {LOAD_KNEE}) + weather term + labour term,
   then multiplied by log-normal noise (sigma={NOISE_SIGMA}) plus N(0,1) additive noise.

**Leakage control:** `historical_delay_hours` = mean of the previous {config.HISTORY_WINDOW} snapshots' *actual* delays for the same port
(shifted by one step, so it never includes the current row). The first snapshot of a port uses a constant prior ({config.PRIOR_DELAY_HOURS:g} h).
"""


def make_label(df: pd.DataFrame, threshold_hours: float = config.CONGESTION_THRESHOLD_HOURS) -> pd.Series:
    """Ground-truth congestion label derived from the realised delay."""
    return (df["actual_delay_hours"] > threshold_hours).astype(int)


def _simulate_port(port_idx: int, prof: dict, rng: np.random.Generator) -> pd.DataFrame:
    n = config.N_OBS_PER_PORT
    start = pd.Timestamp(config.START_DATE) + pd.Timedelta(hours=6 + 2 * port_idx)
    ts = pd.DatetimeIndex([start + pd.Timedelta(days=config.SNAPSHOT_STEP_DAYS * i) for i in range(n)])
    doy = ts.dayofyear.values
    season = 0.5 * (1 + np.cos(2 * np.pi * (doy - prof["peak_doy"]) / 365))  # 0..1

    ar, labor = 0.0, False
    q_prev = arr_prev = eff_prev = None
    rows = []
    for i in range(n):
        ar = 0.5 * ar + rng.normal(0, 1.0)
        storm = rng.random() < 0.06
        w_true = float(np.clip(1.5 + 3.0 * season[i] + ar + (4.0 if storm else 0.0), 0, 10))

        labor = (rng.random() < 0.8) if labor else (rng.random() < prof["labor_risk"])

        cap = prof["capacity"] * rng.uniform(0.90, 1.05)
        if rng.random() < 0.07:
            cap *= 0.75
        cap = max(1, int(round(cap)))

        lam = prof["arrivals"] * (1 + 0.10 * season[i]) * (1.08 if ts[i].dayofweek in (0, 1) else 1.0)
        if rng.random() < 0.08:
            lam *= 1.35
        arrivals = int(rng.poisson(lam))

        if i == 0:
            queue = int(rng.poisson(3))
        else:
            queue = int(round(CARRYOVER * max(0.0, q_prev + arr_prev - eff_prev) + rng.poisson(1)))

        eff = cap * (1 - WEATHER_CAP_LOSS * prof["weather_sens"] * w_true) * (LABOR_CAP_FACTOR if labor else 1.0)
        load = (arrivals + queue) / eff

        delay = (prof["base_delay"]
                 + DELAY_SLOPE * max(0.0, load - LOAD_KNEE)
                 + WEATHER_DELAY * prof["weather_sens"] * max(0.0, w_true - 6.0)
                 + (LABOR_EXTRA_DELAY if labor else 0.0))
        delay = max(0.5, delay * rng.lognormal(0, NOISE_SIGMA) + rng.normal(0, 1.0))

        w_obs = float(np.clip(w_true + rng.normal(0, WEATHER_SENSOR_SD), 0, 10))
        rows.append(dict(port_id=None, timestamp=ts[i], vessel_arrivals=arrivals, queue_length=float(queue),
                         port_capacity=cap, weather_severity=round(w_obs, 1),
                         actual_delay_hours=round(delay, 1)))
        q_prev, arr_prev, eff_prev = queue, arrivals, eff
    return pd.DataFrame(rows)


def generate_dataset(seed: int = config.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for idx, (pid, prof) in enumerate(PORT_PROFILES.items()):
        f = _simulate_port(idx, prof, rng)
        f["port_id"] = pid
        frames.append(f)
    df = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "port_id"]).reset_index(drop=True)

    # Historical delay: PAST snapshots only (shift(1) before rolling) -> no leakage.
    df["historical_delay_hours"] = (
        df.groupby("port_id")["actual_delay_hours"]
          .transform(lambda s: s.shift(1).rolling(config.HISTORY_WINDOW, min_periods=1).mean())
          .fillna(config.PRIOR_DELAY_HOURS).round(2)
    )

    # Incomplete data: random missingness in operational columns.
    df.loc[rng.random(len(df)) < config.MISSING_RATE_WEATHER, "weather_severity"] = np.nan
    df.loc[rng.random(len(df)) < config.MISSING_RATE_QUEUE, "queue_length"] = np.nan

    df["congested"] = make_label(df)
    cols = ["port_id", "timestamp", "vessel_arrivals", "queue_length", "port_capacity",
            "weather_severity", "historical_delay_hours", "actual_delay_hours", "congested"]
    return df[cols]


def load_or_generate(path=config.DATA_PATH, force: bool = False) -> pd.DataFrame:
    """Load the CSV if present (so real data can be dropped in), else generate it."""
    if path.exists() and not force:
        df = pd.read_csv(path, parse_dates=["timestamp"])
    else:
        df = generate_dataset()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the synthetic PortWise dataset.")
    ap.add_argument("--force", action="store_true", help="regenerate even if the CSV exists")
    d = load_or_generate(force=ap.parse_args().force)
    print(f"{len(d)} rows -> {config.DATA_PATH}; congestion rate = {d['congested'].mean():.1%}")
    print(d.groupby("port_id")["congested"].mean().round(3).to_string())
