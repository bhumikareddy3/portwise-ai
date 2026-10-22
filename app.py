"""PortWise AI - Streamlit operator dashboard.   Run:  streamlit run app.py"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config
from data_generator import PORT_PROFILES, SIMULATION_NOTES, load_or_generate, make_label
from explainer import LABELS, PortWiseExplainer
from features import build_features
from model import load_or_train, metrics_table, predict_frame

st.set_page_config(page_title="PortWise AI", page_icon="🚢", layout="wide")

RISK_COLORS = {"Low": "#2e7d32", "Medium": "#ef6c00", "High": "#c62828"}
MODEL_LABELS = {"rf": "Random Forest (calibrated)", "lr": "Logistic Regression (calibrated)"}


# ----------------------------------------------------------------------------- cached resources
@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_or_generate()


@st.cache_resource(show_spinner="Training models...")
def get_bundle(threshold: float):
    return load_or_train(get_data(), threshold)


@st.cache_resource(show_spinner=False)
def get_explainer(threshold: float):
    return PortWiseExplainer(get_bundle(threshold))


def port_label(p: str) -> str:
    return f"{p} - {PORT_PROFILES[p]['name']}" if p in PORT_PROFILES else p


def risk_badge(cat: str) -> str:
    return f"<span style='background:{RISK_COLORS[cat]};color:white;padding:4px 14px;border-radius:12px;font-weight:700'>{cat.upper()}</span>"


# ----------------------------------------------------------------------------- plots
def plot_confusion(cm, title):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    ax.set_xticks([0, 1], ["Not congested", "Congested"], fontsize=8)
    ax.set_yticks([0, 1], ["Not congested", "Congested"], fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); return fig


def plot_roc(metrics):
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    for k, lab in [("lr", "Logistic Reg."), ("rf", "Random Forest"), ("baseline", "Naive rule")]:
        m = metrics[k]
        ax.plot(m["roc_curve"]["fpr"], m["roc_curve"]["tpr"], label=f"{lab} (AUC {m['roc_auc']:.2f})",
                ls="--" if k == "baseline" else "-")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=8)
    ax.set_title("ROC curve (hold-out test set)", fontsize=10); fig.tight_layout(); return fig


def plot_calibration(metrics):
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    for k, lab in [("lr", "Logistic Reg."), ("rf", "Random Forest")]:
        c = metrics[k]["calibration"]; ax.plot(c["prob_pred"], c["prob_true"], marker="o", label=lab)
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed congestion rate"); ax.legend(fontsize=8)
    ax.set_title("Reliability (test set, quantile bins)", fontsize=10); fig.tight_layout(); return fig


def plot_bars(series: pd.Series, title: str, signed: bool):
    s = series.iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 0.45 * len(s) + 1))
    colors = ["#c62828" if v > 0 else "#2e7d32" for v in s] if signed else "#1f77b4"
    ax.barh([LABELS.get(i, i) for i in s.index], s.values, color=colors)
    ax.axvline(0, color="black", lw=0.6); ax.set_title(title, fontsize=10); ax.tick_params(labelsize=8)
    fig.tight_layout(); return fig


def show(fig):
    st.pyplot(fig); plt.close(fig)


# ----------------------------------------------------------------------------- shared widgets
def record_picker(df, bundle, key):
    port = st.selectbox("Port", sorted(df["port_id"].unique()), format_func=port_label, key=f"{key}_port")
    sub = df[df["port_id"] == port].sort_values("timestamp")
    test_set = set(bundle.test_index)
    rid = st.selectbox("Observation (default: most recent)", sub.index.tolist(), index=len(sub) - 1,
                       format_func=lambda i: f"{df.loc[i, 'timestamp']:%Y-%m-%d %H:%M}  [{'TEST' if i in test_set else 'train'}]",
                       key=f"{key}_rec_{port}")
    if rid not in test_set:
        st.caption("This record was part of the training period, so its prediction is in-sample.")
    return df.loc[[rid]]


def fill_missing(row: pd.DataFrame, bundle) -> pd.DataFrame:
    row = row.copy()
    ref = bundle.reference.loc[row["port_id"].iloc[0]]
    for c in ["queue_length", "weather_severity", "vessel_arrivals", "port_capacity", "historical_delay_hours"]:
        if pd.isna(row[c].iloc[0]):
            row[c] = ref[c]
    return row


# ----------------------------------------------------------------------------- pages
def page_overview(df, bundle, explainer, mk):
    st.title("🚢 PortWise AI - Port Congestion Decision Support")
    st.caption(f"Congestion = actual delay > {bundle.threshold_hours:g} h. Data is **synthetic** (see below); "
               "metrics describe the simulator, not any real port.")
    c = st.columns(5)
    c[0].metric("Records", len(df)); c[1].metric("Ports", df["port_id"].nunique())
    c[2].metric("Period", f"{df.timestamp.min():%b %Y} - {df.timestamp.max():%b %Y}")
    c[3].metric("Congestion rate", f"{df['congested'].mean():.1%}")
    c[4].metric("Train / Test", f"{bundle.n_train} / {bundle.n_test}")

    with st.expander("How the synthetic data and ground truth were simulated"):
        st.markdown(SIMULATION_NOTES)
    st.dataframe(df.head(12))

    a, b = st.columns(2)
    with a:
        st.subheader("Congestion rate by port")
        st.bar_chart(df.groupby("port_id")["congested"].mean())
    with b:
        st.subheader("Delay distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(df["actual_delay_hours"], bins=30, color="#1f77b4")
        ax.axvline(bundle.threshold_hours, color="red", ls="--", label="congestion threshold")
        ax.set_xlabel("actual delay (h)"); ax.legend(fontsize=8); fig.tight_layout(); show(fig)

    st.header("Model evaluation (chronological hold-out)")
    st.caption(f"Trained on the first {bundle.n_train} records; tested on the last {bundle.n_test} "
               f"(from {bundle.cutoff:%Y-%m-%d}). Decision threshold {bundle.metrics['decision_threshold']:.2f}. "
               "All numbers are computed live from the models - none are hard-coded.")
    st.dataframe(metrics_table(bundle))
    st.caption(f"With only {bundle.n_test} test rows the metrics are noisy - see the bootstrap confidence interval on ROC-AUC.")

    c1, c2, c3, c4 = st.columns(4)
    with c1: show(plot_confusion(bundle.metrics["lr"]["confusion_matrix"], "Logistic Regression"))
    with c2: show(plot_confusion(bundle.metrics["rf"]["confusion_matrix"], "Random Forest"))
    with c3: show(plot_roc(bundle.metrics))
    with c4: show(plot_calibration(bundle.metrics))

    st.header("Global feature importance")
    Xte = build_features(df.loc[bundle.test_index], bundle.port_ids)
    imp = explainer.global_importance(Xte, mk)
    show(plot_bars(imp, f"Mean |contribution| on test set - {MODEL_LABELS[mk]}", signed=False))
    st.caption(f"Method: {explainer.method_name(mk)}")


def page_prediction(df, bundle, explainer, mk):
    st.title("Prediction & Explanation")
    row = record_picker(df, bundle, "pe")
    r = row.iloc[0]
    exp = explainer.explain_prediction(row, mk)
    other = "lr" if mk == "rf" else "rf"
    p_other = predict_frame(bundle, row, other)["probability"].iloc[0]

    c = st.columns(4)
    c[0].metric("Congestion probability", f"{exp.probability:.0%}")
    c[1].markdown("**Risk category**<br>" + risk_badge(exp.risk), unsafe_allow_html=True)
    c[2].metric("Confidence", f"{exp.confidence:.0%}", help="|2p-1|: 0% = model is on the fence (p=50%), 100% = certain.")
    c[3].metric(f"{MODEL_LABELS[other].split(' (')[0]} says", f"{p_other:.0%}",
                help="Agreement between two different model families is a useful sanity check.")
    st.info(exp.sentence)

    left, right = st.columns([3, 2])
    with left:
        show(plot_bars(exp.contributions, "Top contributing factors (red = raises risk, green = lowers risk)", signed=True))
        st.caption(f"Method: {exp.method}")
    with right:
        st.subheader("Suggested actions")
        for a in exp.actions:
            st.markdown(f"- {a}")
        st.caption("Actions are rule-based heuristics mapped from the top drivers; they are not causal estimates.")
        st.subheader("Observation")
        st.dataframe(row[["vessel_arrivals", "queue_length", "port_capacity", "weather_severity",
                          "historical_delay_hours"]].T.rename(columns={row.index[0]: "value"}))
    with st.expander("Reveal ground truth (evaluation only - not used as a feature)"):
        lab = make_label(row, bundle.threshold_hours).iloc[0]
        st.write(f"Actual delay: **{r['actual_delay_hours']:.1f} h** -> {'congested' if lab else 'not congested'}")


def page_whatif(df, bundle, explainer, mk):
    st.title("What-If Simulator")
    st.warning("Scenario outputs are **model-based projections**, not guaranteed real-world outcomes. "
               "The model learned correlations from (synthetic) history and cannot promise a given intervention will work.")
    row = fill_missing(record_picker(df, bundle, "wi"), bundle)
    base = row.iloc[0]
    rid, ver = row.index[0], st.session_state.get("wi_ver", 0)
    k = lambda n: f"wi_{n}_{rid}_{ver}"

    st.subheader("Scenario inputs")
    s = st.columns(4)
    arrivals = s[0].slider("Vessel arrivals", 0, max(60, int(df.vessel_arrivals.max() * 1.5)), int(base.vessel_arrivals), key=k("arr"))
    capacity = s[1].slider("Port capacity", 1, max(60, int(df.port_capacity.max() * 1.5)), int(base.port_capacity), key=k("cap"))
    queue = s[2].slider("Queue length", 0, max(40, int(df.queue_length.max() * 1.5)), int(base.queue_length), key=k("q"))
    weather = s[3].slider("Weather severity (0-10)", 0.0, 10.0, float(base.weather_severity), 0.1, key=k("w"))
    if st.button("Reset to baseline"):
        st.session_state["wi_ver"] = ver + 1
        st.rerun()

    scen = row.copy()
    scen["vessel_arrivals"], scen["port_capacity"], scen["queue_length"], scen["weather_severity"] = arrivals, capacity, queue, weather

    eb, es = explainer.explain_prediction(row, mk), explainer.explain_prediction(scen, mk)
    cb, cs = st.columns(2)
    for col, title, e in [(cb, "Baseline (selected record)", eb), (cs, "Scenario", es)]:
        with col:
            st.subheader(title)
            m = st.columns(3)
            m[0].metric("Probability", f"{e.probability:.0%}",
                        delta=None if e is eb else f"{(e.probability - eb.probability) * 100:+.0f} pp", delta_color="inverse")
            m[1].markdown("**Risk**<br>" + risk_badge(e.risk), unsafe_allow_html=True)
            m[2].metric("Confidence", f"{e.confidence:.0%}")
            st.info(e.sentence)

    st.subheader("Sensitivity: how risk changes along one lever (other scenario inputs held fixed)")
    var = st.selectbox("Lever", ["vessel_arrivals", "port_capacity", "queue_length", "weather_severity"],
                       format_func=lambda v: LABELS[v])
    hi = {"vessel_arrivals": max(60, int(df.vessel_arrivals.max() * 1.5)), "port_capacity": max(60, int(df.port_capacity.max() * 1.5)),
          "queue_length": max(40, int(df.queue_length.max() * 1.5)), "weather_severity": 10}[var]
    lo = 1 if var == "port_capacity" else 0
    grid = np.linspace(lo, hi, 40)
    sweep = pd.concat([scen] * len(grid), ignore_index=True)
    sweep[var] = grid
    probs = predict_frame(bundle, sweep, mk)["probability"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(grid, probs, color="#1f77b4")
    ax.axhspan(0, config.RISK_LOW, color="#2e7d32", alpha=0.08); ax.axhspan(config.RISK_HIGH, 1, color="#c62828", alpha=0.08)
    ax.axvline(float(base[var]), color="grey", ls=":", label="baseline value")
    ax.axvline(float(scen[var].iloc[0]), color="red", ls="--", label="scenario value")
    ax.set_xlabel(LABELS[var]); ax.set_ylabel("P(congestion)"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.tight_layout(); show(fig)
    st.caption("Green band = Low risk, red band = High risk. Projections only - treat steep regions as 'sensitive', not as promises.")


# ----------------------------------------------------------------------------- main
def main():
    st.sidebar.title("PortWise AI")
    page = st.sidebar.radio("Navigate", ["Overview", "Prediction & Explanation", "What-If Simulator"])
    thr = st.sidebar.number_input("Congestion threshold (hours of delay)", 4.0, 48.0, config.CONGESTION_THRESHOLD_HOURS, 1.0,
                                  help="Changing this redefines the label and retrains the models.")
    mk = st.sidebar.selectbox("Active model", list(MODEL_LABELS), format_func=MODEL_LABELS.get)
    st.sidebar.markdown(f"**Risk bands:** Low < {config.RISK_LOW:.0%} <= Medium < {config.RISK_HIGH:.0%} <= High")

    df = get_data().copy()
    df["congested"] = make_label(df, thr)
    bundle, explainer = get_bundle(float(thr)), get_explainer(float(thr))
    {"Overview": page_overview, "Prediction & Explanation": page_prediction,
     "What-If Simulator": page_whatif}[page](df, bundle, explainer, mk)


main()
