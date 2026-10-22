"""Explainable AI: SHAP (Random Forest) / exact linear contributions (Logistic Regression),
plain-English sentences and operational recommendations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_generator import PORT_PROFILES
from features import build_features, feature_groups
from model import ModelBundle, predict_frame

LABELS = {
    "vessel_arrivals": "Inbound vessel arrivals", "queue_length": "Vessels already queued",
    "port_capacity": "Port capacity", "weather_severity": "Weather severity",
    "historical_delay_hours": "Recent delay history", "load_ratio": "Demand vs capacity",
    "weather_x_load": "Weather x demand interaction", "seasonality": "Seasonality",
    "port_profile": "Port characteristics",
}

ACTIONS = {
    "vessel_arrivals": "Smooth the arrival peak: slot/appointment booking, ask carriers to adjust ETAs, or divert surplus calls to an alternate terminal.",
    "load_ratio": "Demand is close to or above capacity: stagger arrivals and prioritise vessels that can berth and clear fastest.",
    "queue_length": "Clear the backlog first: add berth/pilot/tug shifts and expedite vessels that are ready to sail.",
    "port_capacity": "Restore or supplement capacity: defer planned maintenance, open reserve berths or yard equipment.",
    "weather_severity": "Add weather buffer to ETAs and berth windows, brief shippers early, and plan pilotage around the forecast.",
    "weather_x_load": "Weather is compounding heavy demand: keep contingency shifts on standby and re-plan berth windows after each forecast update.",
    "historical_delay_hours": "Persistent recent delays point to a structural bottleneck: review gate/yard processes and warn customers proactively.",
    "seasonality": "Seasonal pressure: apply the seasonal peak playbook (extra shifts, earlier pre-advice).",
    "port_profile": "Port-specific structural factors dominate: review this terminal's standing constraints.",
}
CATEGORY_ACTIONS = {
    "High": "Issue a congestion alert to shipping lines, agents and consignees and add buffer to planned berth windows.",
    "Medium": "Pre-alert planning teams; monitor queue and arrivals each shift and keep contingency shifts on standby.",
    "Low": "No intervention needed; continue normal berth planning and keep monitoring.",
}


@dataclass
class Explanation:
    probability: float
    risk: str
    confidence: float
    contributions: pd.Series      # grouped factors, positive = pushes towards congestion
    drivers_up: list
    drivers_down: list
    sentence: str
    actions: list
    method: str


def _join(items: list) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _weather_word(v: float) -> str:
    return "mild" if v < 4 else ("moderate" if v < 7 else "severe")


def _describe(group: str, direction: str, vals: dict, ref, port_name: str) -> str:
    v = vals.get(group)
    if group in vals and pd.isna(v):
        return f"missing {LABELS[group].lower()} data (imputed)"
    up = direction == "up"
    if group == "vessel_arrivals":
        return (f"{v:.0f} inbound vessels (typical here: {ref['vessel_arrivals']:.0f})" if up
                else f"light inbound traffic ({v:.0f} vessels)")
    if group == "queue_length":
        return f"a queue of {v:.0f} vessels already waiting" if up else f"a short queue ({v:.0f} vessels)"
    if group == "port_capacity":
        if not up:
            return f"ample capacity ({v:.0f} vessels)"
        typ = ref["port_capacity"]
        return (f"reduced capacity ({v:.0f} vs typical {typ:.0f})" if v < 0.97 * typ
                else f"limited capacity headroom ({v:.0f} vessels vs the current load)")
    if group == "weather_severity":
        return (f"{_weather_word(v)} weather (severity {v:.1f}/10)" if up
                else f"calm weather (severity {v:.1f}/10)")
    if group == "historical_delay_hours":
        return f"recent delays averaging {v:.1f} h" if up else f"low recent delays ({v:.1f} h)"
    if group == "load_ratio":
        return f"demand at {v * 100:.0f}% of capacity" if up else f"demand only {v * 100:.0f}% of capacity"
    if group == "weather_x_load":
        return "weather compounding heavy demand" if up else "weather not compounding demand"
    if group == "seasonality":
        return "the seasonal traffic pattern" if up else "favourable seasonal timing"
    return f"structural characteristics of {port_name}" if up else f"{port_name}'s generally lower congestion profile"


class PortWiseExplainer:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self._shap = None
        try:
            import shap  # optional dependency -> graceful fallback
            self._shap = shap.TreeExplainer(bundle.models["rf"].estimator_.named_steps["clf"])
        except Exception:
            self._shap = None

    def method_name(self, model: str = "rf") -> str:
        if model == "lr":
            return "Logistic Regression: coefficient x standardised value (exact, log-odds)"
        return "SHAP TreeExplainer (probability units)" if self._shap is not None \
            else "Occlusion fallback: change in probability when a feature is set to its training median"

    # ---- raw per-feature contributions (rows x features)
    def contributions(self, X: pd.DataFrame, model: str = "rf") -> pd.DataFrame:
        pipe, names = self.bundle.models[model].estimator_, self.bundle.feature_names
        Xt = pipe[:-1].transform(X)
        if model == "lr":
            return pd.DataFrame(Xt * pipe.named_steps["clf"].coef_[0], index=X.index, columns=names)
        if self._shap is not None:
            sv = self._shap.shap_values(Xt, check_additivity=False)
            sv = sv[1] if isinstance(sv, list) else np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            return pd.DataFrame(sv, index=X.index, columns=names)
        med = self.bundle.reference.loc["__global__"]
        p0 = pipe.predict_proba(X)[:, 1]
        out = {}
        for c in names:
            Xo = X.copy()
            Xo[c] = med[c] if c in med.index else X[c].median()
            out[c] = p0 - pipe.predict_proba(Xo)[:, 1]
        return pd.DataFrame(out, index=X.index)

    def group(self, contrib: pd.DataFrame) -> pd.DataFrame:
        groups = feature_groups(self.bundle.port_ids)
        grouped = {c for cols in groups.values() for c in cols}
        out = {c: contrib[c] for c in contrib.columns if c not in grouped}
        for g, cols in groups.items():
            out[g] = contrib[cols].sum(axis=1)
        return pd.DataFrame(out, index=contrib.index)

    def global_importance(self, X: pd.DataFrame, model: str = "rf") -> pd.Series:
        """Mean |contribution| per factor over the rows in X (SHAP-based for the RF)."""
        return self.group(self.contributions(X, model)).abs().mean().sort_values(ascending=False)

    # ---- local explanation for ONE raw row
    def explain_prediction(self, raw_row: pd.DataFrame, model: str = "rf") -> Explanation:
        b = self.bundle
        pr = predict_frame(b, raw_row, model).iloc[0]
        X = build_features(raw_row, b.port_ids)
        contrib = self.group(self.contributions(X, model)).iloc[0]
        contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index)

        row, x = raw_row.iloc[0], X.iloc[0]
        port = row["port_id"]
        port_name = PORT_PROFILES.get(port, {}).get("name", port)
        ref = b.reference.loc[port] if port in b.reference.index else b.reference.loc["__global__"]
        vals = {k: row[k] for k in ["vessel_arrivals", "queue_length", "port_capacity",
                                    "weather_severity", "historical_delay_hours"]}
        vals["load_ratio"] = x["load_ratio"]

        floor = 0.05 * contrib.abs().sum()
        up = [g for g, v in contrib.items() if v > floor][:3]
        down = [g for g, v in contrib.items() if v < -floor][:2]
        ups = [_describe(g, "up", vals, ref, port_name) for g in up]
        downs = [_describe(g, "down", vals, ref, port_name) for g in down]

        head = f"{pr.risk_category.upper()} risk ({pr.probability:.0%} congestion probability, {pr.confidence:.0%} confidence) at {port_name}"
        if pr.risk_category == "Low":
            sentence = head + (f": conditions look manageable, supported by {_join(downs)}." if downs
                               else ": no factor stands out as a congestion driver.")
            if ups:
                sentence += f" Watch {_join(ups)}."
        else:
            sentence = head + (f": driven mainly by {_join(ups)}." if ups
                               else ": no single dominant driver; the risk comes from several small factors.")
            if downs:
                sentence += f" Partly offset by {_join(downs)}."

        actions = [CATEGORY_ACTIONS[pr.risk_category]]
        if pr.risk_category != "Low":
            actions += [ACTIONS[g] for g in up[:2] if g in ACTIONS]
        return Explanation(float(pr.probability), pr.risk_category, float(pr.confidence), contrib,
                           up, down, sentence, actions, self.method_name(model))
