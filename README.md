PortWise AI — Port Congestion Prediction & Decision Support

Take-home project for the Aiqon AI internship. Given a port snapshot (inbound vessels, queue, capacity, weather, recent delay history), PortWise AI estimates the probability of congestion, gives a confidence score and risk category, explains the drivers in plain English, and lets an operator test what-if scenarios.

📄 Technical report: reports/PortWise_AI_Technical_Report.pdf (2 pages: approach, assumptions, results)

Quick start

Requires Python 3.10+ (developed and tested with Python 3.12). Run all commands from the project folder.

bash
# 1. create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Mac/Linux
.venv\Scripts\activate             # Windows (Command Prompt / VS Code terminal)

# 2. install dependencies
pip install -r requirements.txt

# 3. (optional) regenerate the synthetic dataset, then train and evaluate
python data_generator.py --force   # writes data/port_data.csv (also auto-created on first app run)
python train.py                    # trains, evaluates, writes reports/metrics.json + reports/results.md

# 4. run the tests (leakage, reproducibility, chronological split, output sanity)
pytest -q

# 5. launch the operator dashboard
streamlit run app.py               # opens http://localhost:8501

The first dashboard load trains the models once (a few seconds) and caches them. If a running dashboard does not pick up a change to the CSV or code, stop it (Ctrl+C) and start it again.

'''Problem definition'''
Congestion = actual_delay_hours > 12 (configurable: config.CONGESTION_THRESHOLD_HOURS or the sidebar in the app).
Task: binary classification — P(congested | information available at snapshot time).
Why a classifier, not a regressor: operators act on "is this port at risk of congestion?" and a probability maps directly to alert levels.
Dataset (synthetic, reproducible — seed 42)

450 records = 5 ports (INNSA, INMAA, SGSIN, NLRTM, USLAX) × 90 snapshots, collected at exact four-day intervals over a 356-day span (1 Jan – 23 Dec 2025). Columns: port_id, timestamp, vessel_arrivals, queue_length, port_capacity, weather_severity, historical_delay_hours, actual_delay_hours (+ congested, the derived label). Full simulator description: data_generator.SIMULATION_NOTES (also shown in the app). Overall congestion rate is 30.2%, varying by port (Singapore 13% to Los Angeles 38%).

Key design choices so the problem is not trivial:

Real-world complexity	How it appears in the data
Port variability	Different capacity, demand, weather sensitivity, labour risk and base delay per port
Temporal dynamics / seasonality	Port-specific storm season, weekday demand uplift, AR(1) weather persistence, backlog carry-over
Hidden causes	A persistent labour disruption state affects delay but is never given to the model
Noisy / incomplete data	Weather sensor noise (SD 0.6); ~3% missing weather, ~2% missing queue (median-imputed inside the pipeline)
Interactions	Delay is driven by (arrivals + queue) / weather-and-labour-adjusted capacity, so weather × load matters

Ground truth is the simulated delay (a noisy function of the true load, including hidden factors), not a formula over the observed columns.

'''Leakage prevention'''
historical_delay_hours = mean of the actual delays from the previous three snapshots for the same port, shifted by one step; the first snapshot of a port uses a constant prior (8 h). Unit-tested.
queue_length is the backlog already present at snapshot time (built from earlier snapshots only).
actual_delay_hours / congested are blocked from the feature set (features.assert_no_leakage, tested).
Chronological split: first 80% of time = train (360 rows), last 20% = test (90 rows, from 2025-10-16). Imputers/scalers are fit on train only (inside sklearn Pipelines).
Calibration is time-aware: Platt scaling is fitted on forward-chaining (TimeSeriesSplit) out-of-fold predictions and labels from the training period only; the test period is used exclusively for evaluation.
Features

Raw: arrivals, queue, capacity, weather, recent delay. Engineered: load_ratio = (arrivals + queue) / capacity, weather × load interaction, seasonality (sin/cos of day-of-year), port one-hot.

'''Models and outputs'''
Logistic Regression (baseline, standardised) and Random Forest (300 trees, depth ≤ 6, min leaf 5). The dashboard defaults to the Random Forest (nonlinear effects, SHAP explanations); LR performs comparably here, so either is defensible at this sample size.
Probability: calibrated P(congestion). Risk bands: Low if p < 0.35, Medium if 0.35 ≤ p < 0.65, High if p ≥ 0.65 (config.py).
Confidence score: |2p − 1| — a prediction-decisiveness measure: 0% at p = 50%, 100% at p = 0% or 100%. It is not a probability of being correct or a statistical confidence interval.
A naive persistence rule ("congested if recent delay > threshold") is reported as a sanity baseline.
Results (chronological hold-out, seed 42 — computed by train.py, not hand-written)

Split: 360 train / 90 test (test starts 2025-10-16); decision threshold 0.5.

Model	Precision	Recall	F1	ROC-AUC	95% CI	Brier	TN	FP	FN	TP
Logistic Regression	0.659	0.964	0.783	0.946	0.90-0.98	0.142	48	14	1	27
Random Forest	0.587	0.964	0.730	0.943	0.89-0.98	0.139	43	19	1	27
Naive rule	0.429	0.429	0.429	0.674	0.56-0.78	n/a	46	16	16	12

Naive rule = congested if recent delay > threshold; its scores are not probabilities, so it has no Brier score. TN/FP/FN/TP are test-set confusion counts (positive = congested).

Full output (including ROC and calibration data) is in reports/results.md and reports/metrics.json. If you change the data, seed or threshold, re-run python train.py — these numbers will change.

'''How to read these honestly'''

The test set has only 90 rows (28 congested) — hence the bootstrap CI on ROC-AUC. Differences between LR and RF are within noise.
Models rank risk well (AUC ≈ 0.94 vs 0.67 for the naive rule) but over-estimate absolute probability on the test set (mean predicted ≈ 0.50 vs 31% observed). The test window (Oct–Dec) has heavier traffic and queues than the training period, consistent with a temporal distribution shift, so probabilities should be treated cautiously.
The alert threshold is an operational choice. A higher threshold (e.g. 0.6–0.7) trades some recall for better precision; the trade-off table in the report is exploratory (chosen after seeing test results). In practice select the threshold on validation data using the operational cost of a false alarm versus a missed congestion event.
Because the data is simulated, these numbers show the pipeline works, not how accurate it would be on real ports.
Explainability
Random Forest → SHAP TreeExplainer (local + global). Its raw output is the forest's class-1 probability (not log-odds): base value (~0.30) + a record's SHAP contributions reproduces the forest's raw probability exactly. This is before Platt scaling, so SHAP magnitudes are in raw-forest units and are not additive on the displayed calibrated probability; the direction and ranking of drivers carry over. Falls back to an occlusion method if shap is unavailable.
Logistic Regression → exact coefficient × standardised value.
Port one-hots and seasonality pairs are grouped into single human-readable factors.
explainer.explain_prediction produces sentences like "HIGH risk (99% congestion probability, 98% confidence) at Los Angeles: driven mainly by demand at 173% of capacity, weather compounding heavy demand and recent delays averaging 22.0 h." (the final record in the dataset, a test-period record), plus suggested actions (stagger arrivals, add shifts, buffer ETAs…). Actions are rule-based heuristics mapped from the top drivers, not causal estimates.
Dashboard (app.py)
Overview — dataset summary, simulator notes, metrics table, confusion matrices, ROC, reliability curve, global importance.
Prediction & Explanation — pick port + observation → probability, risk, confidence, second-model cross-check, SHAP factors, plain-English reason, actions, optional ground-truth reveal.
What-If Simulator — baseline vs scenario side-by-side (sliders for arrivals, capacity, queue, weather), probability delta, and a sensitivity curve along any lever. Marked clearly as model-based projections.

The sidebar also lets you change the congestion threshold (which redefines the label and retrains) and switch the active model.

'''Repository layout'''
app.py              Streamlit dashboard (3 pages)
data_generator.py   synthetic simulator + label definition + SIMULATION_NOTES
features.py         leakage-safe feature engineering
model.py            LR + RF pipelines, time-aware calibration, evaluation, risk bands
explainer.py        SHAP / linear contributions, plain-English text, recommendations
train.py            CLI: train + evaluate + write reports
config.py           all tunables (threshold, seed, risk bands, sizes)
conftest.py         lets pytest import the flat modules
tests/              pytest suite
data/               port_data.csv (synthetic dataset)
artifacts/          cached trained models (git-ignored)
reports/            technical report PDF, metrics.json, results.md
Using real data (e.g. AIS)

Replace data/port_data.csv with a file using the same 8 columns (sorted by time); nothing else changes, and new port IDs appear automatically. Suggested derivation from AIS: vessel_arrivals = vessels with an ETA/entry into the port geofence in the next window; queue_length = vessels with speed ≈ 0 inside the anchorage polygon; actual_delay_hours = berth time − arrival at anchorage; port_capacity = historical max throughput per window; weather from a reanalysis/forecast API. Ensure historical_delay_hours uses only past delays. Note MarineCadastre AIS covers US waters only and has no delay labels — you must derive them.

''''Limitations / future work'''

Synthetic data; 5 ports; single-step (not multi-day-ahead) prediction; four-day snapshots, so time-of-day effects are not modelled; the training period does not cover a complete annual seasonal cycle; confidence is not a statistical interval; recommendations are heuristic; no cost model for choosing the alert threshold; with real data add rolling arrival-density features and per-port or hierarchical models.

'''Assumptions'''

Stated in code (data_generator.py, config.py) and above: 12 h threshold, 4-day snapshots, 50% backlog carry-over, 80/20 chronological split, 0.5 decision threshold, risk bands 35% / 65%.

'''Tested with'''

Python 3.12, scikit-learn 1.8.0, pandas 3.0.2, NumPy 2.4.4, SHAP 0.54.1, Streamlit 1.66.0, matplotlib 3.10.8.