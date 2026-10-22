# PortWise AI — Port Congestion Prediction & Decision Support

Take-home project for the Aiqon AI internship. Given a port snapshot (inbound vessels, queue, capacity, weather,
recent delay history), PortWise AI estimates the **probability of congestion**, gives a **confidence score** and
**risk category**, **explains the drivers in plain English**, and lets an operator test **what-if scenarios**.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
python data_generator.py          # creates data/port_data.csv (also auto-created on first app run)
python train.py                   # trains, evaluates, writes reports/metrics.json + reports/results.md
pytest -q                         # leakage / reproducibility / split / output sanity tests
streamlit run app.py              # operator dashboard
```

## Problem definition

* **Congestion** = `actual_delay_hours > 12` (configurable: `config.CONGESTION_THRESHOLD_HOURS` or the sidebar in the app).
* **Task:** binary classification — P(congested | information available at snapshot time).
* **Why a classifier, not a regressor:** operators act on "is this port about to jam?" and a probability maps directly to alert levels.

## Dataset (synthetic, reproducible — seed 42)

450 records = 5 ports (INNSA, INMAA, SGSIN, NLRTM, USLAX) × 90 snapshots, one every 4 days (~1 year).
Columns: `port_id, timestamp, vessel_arrivals, queue_length, port_capacity, weather_severity, historical_delay_hours, actual_delay_hours`
(+ `congested`, the derived label). Full simulator description: `data_generator.SIMULATION_NOTES` (also shown in the app).

Key design choices so the problem is not trivial:

| Real-world complexity | How it appears in the data |
|---|---|
| Port variability | Different capacity, demand, weather sensitivity, labour risk and base delay per port |
| Temporal dynamics / seasonality | Port-specific storm season, weekday demand uplift, AR(1) weather persistence, backlog carry-over |
| Hidden causes | A persistent **labour disruption** state affects delay but is **never given to the model** |
| Noisy / incomplete data | Weather sensor noise (SD 0.6); ~3% missing weather, ~2% missing queue (median-imputed inside the pipeline) |
| Interactions | Delay is driven by (arrivals + queue) / weather-and-labour-adjusted capacity, so weather × load matters |

**Ground truth** is the simulated delay (a noisy function of the *true* load, including hidden factors), not a formula over the observed columns.

### Leakage prevention
* `historical_delay_hours` = mean of the previous 3 snapshots' actual delays for the same port, **shifted by one step**; the first snapshot of a port uses a constant prior (8 h). Unit-tested.
* `queue_length` is the backlog already present at snapshot time (built from earlier snapshots only).
* `actual_delay_hours` / `congested` are blocked from the feature set (`features.assert_no_leakage`, tested).
* **Chronological split:** first 80% of time = train, last 20% = test. Imputers/scalers are fit on train only (inside sklearn Pipelines).
* **Calibration is time-aware:** Platt scaling fitted on forward-chaining (`TimeSeriesSplit`) out-of-fold predictions — no future information.

## Features
Raw: arrivals, queue, capacity, weather, recent delay. Engineered: `load_ratio = (arrivals + queue) / capacity`,
`weather × load` interaction, seasonality (sin/cos of day-of-year), port one-hot.

## Models and outputs
* **Logistic Regression** (baseline, standardised) and **Random Forest** (300 trees, depth ≤ 6, min leaf 5).
* **Probability:** calibrated P(congestion). **Risk:** Low < 35% ≤ Medium < 65% ≤ High (`config.py`).
* **Confidence:** `|2p − 1|` — 0% when the model is on the fence, 100% when certain. This is a *decisiveness* score, not a formal credible interval.
* A **naive persistence rule** ("delay > threshold because recent delay was high") is reported as a sanity baseline.

## Results (chronological hold-out, seed 42 — computed by `train.py`, not hand-written)

Chronological split: 360 train / 90 test (test starts 2025-10-16).

| Model                                |   Precision |   Recall |    F1 |   ROC-AUC | ROC-AUC 95% CI   |   Brier |
|:-------------------------------------|------------:|---------:|------:|----------:|:-----------------|--------:|
| Logistic Regression                  |       0.659 |    0.964 | 0.783 |     0.946 | 0.90-0.98        |   0.142 |
| Random Forest                        |       0.587 |    0.964 | 0.73  |     0.943 | 0.89-0.98        |   0.139 |
| Naive rule: recent delay > threshold |       0.429 |    0.429 | 0.429 |     0.674 | 0.56-0.78        | nan     |

See `reports/results.md` / `reports/metrics.json` (includes confusion matrices, ROC and calibration data). Metrics use a 0.5 decision threshold.

**How to read these honestly**
* The test set has only **90 rows** (~28 congested) — hence the bootstrap CI on ROC-AUC. Differences between LR and RF are within noise.
* Models **rank** risk well (AUC ≈ 0.94 vs 0.67 for the naive rule) but **over-estimate** absolute probability in the test window
  (mean predicted ≈ 0.5 vs 31% actual): late-year traffic/queues are heavier than anything in training, a genuine temporal distribution shift.
  Operationally this means a higher alert threshold (e.g. 0.6–0.7) trades some recall for much better precision; choose it from the cost of a false alarm vs a missed jam.
* Because the data is simulated, **these numbers say the pipeline works, not how accurate it would be on real ports.**

## Explainability
* **Random Forest → SHAP TreeExplainer** (local + global, additive; verified to sum to the model output). Falls back to an occlusion method if `shap` is unavailable.
* **Logistic Regression → exact coefficient × standardised value.**
* Port one-hots and seasonality pairs are grouped into single human-readable factors.
* `explainer.explain_prediction` produces sentences like *"HIGH risk (93% probability, 86% confidence) at Chennai: driven mainly by demand at 206% of capacity, weather compounding heavy demand…"*
  plus **suggested actions** (stagger arrivals, add shifts, buffer ETAs…). Actions are rule-based heuristics mapped from the top drivers, not causal estimates.

## Dashboard (`app.py`)
1. **Overview** — dataset summary, simulator notes, metrics table, confusion matrices, ROC, reliability curve, global importance.
2. **Prediction & Explanation** — pick port + observation → probability, risk, confidence, second-model cross-check, SHAP factors, plain-English reason, actions, optional ground-truth reveal.
3. **What-If Simulator** — baseline vs scenario side-by-side (sliders for arrivals, capacity, queue, weather), probability delta, and a sensitivity curve along any lever. Marked clearly as model-based projections.

## Repository layout
```
app.py              Streamlit dashboard (3 pages)
data_generator.py   synthetic simulator + label definition + SIMULATION_NOTES
features.py         leakage-safe feature engineering
model.py            LR + RF pipelines, time-aware calibration, evaluation, risk bands
explainer.py        SHAP / linear contributions, plain-English text, recommendations
train.py            CLI: train + evaluate + write reports
config.py           all tunables (threshold, seed, risk bands, sizes)
tests/              pytest suite
data/ artifacts/ reports/
```

## Using real data (e.g. AIS)
Replace `data/port_data.csv` with a file using the same 8 columns; nothing else changes. Suggested derivation from AIS:
`vessel_arrivals` = vessels with an ETA/entry into the port geofence in the next window; `queue_length` = vessels with speed ≈ 0 inside the anchorage polygon;
`actual_delay_hours` = berth time − arrival at anchorage; `port_capacity` = historical max throughput per window; weather from a reanalysis/forecast API.
Note MarineCadastre AIS covers US waters only and has no delay labels — you must derive them.

## Limitations / future work
Synthetic data; 5 ports; single-step (not multi-day-ahead) prediction; confidence is not a statistical interval; recommendations are heuristic;
no cost model for choosing the alert threshold; with real data add rolling arrival-density features and per-port models or hierarchical effects.

## Assumptions
Stated in code (`data_generator.py`, `config.py`) and above: 12 h threshold, 4-day snapshots, 50% backlog carry-over, 80/20 chronological split, 0.5 decision threshold.
