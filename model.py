"""ML pipeline: Logistic Regression baseline + Random Forest, time-aware calibration,
chronological evaluation, risk categories."""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from data_generator import make_label
from features import BASE_FEATURES, build_features, feature_names


# ----------------------------------------------------------------------------- calibration
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class TimeAwareCalibratedClassifier:
    """Platt scaling fitted on *out-of-fold, forward-chaining* predictions.

    Standard random-fold calibration would leak future information. Here each fold's
    model only sees the past (TimeSeriesSplit); the sigmoid is fitted on the resulting
    out-of-fold scores; the final estimator is then refit on all training data.
    """

    def __init__(self, estimator, n_splits: int = 4):
        self.estimator = estimator
        self.n_splits = n_splits

    def fit(self, X: pd.DataFrame, y):
        X = X.reset_index(drop=True)
        y = pd.Series(np.asarray(y))
        scores, truth = [], []
        for tr, va in TimeSeriesSplit(n_splits=self.n_splits).split(X):
            if y.iloc[tr].nunique() < 2:
                continue
            m = clone(self.estimator).fit(X.iloc[tr], y.iloc[tr])
            scores.append(m.predict_proba(X.iloc[va])[:, 1])
            truth.append(y.iloc[va].to_numpy())
        self.calibrator_ = None
        if scores:
            s, t = np.concatenate(scores), np.concatenate(truth)
            if len(np.unique(t)) == 2:
                self.calibrator_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(s).reshape(-1, 1), t)
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        s = self.estimator_.predict_proba(X)[:, 1]
        p = s if self.calibrator_ is None else self.calibrator_.predict_proba(_logit(s).reshape(-1, 1))[:, 1]
        return np.column_stack([1 - p, p])


# ----------------------------------------------------------------------------- bundle
@dataclass
class ModelBundle:
    threshold_hours: float
    port_ids: list
    feature_names: list
    models: dict            # "lr" / "rf" -> TimeAwareCalibratedClassifier
    metrics: dict
    reference: pd.DataFrame  # per-port training medians (+ "__global__") for plain-English comparisons
    test_index: list
    cutoff: pd.Timestamp
    n_train: int
    n_test: int


def risk_category(p: float, low: float = config.RISK_LOW, high: float = config.RISK_HIGH) -> str:
    return "Low" if p < low else ("High" if p >= high else "Medium")


def predict_frame(bundle: ModelBundle, raw: pd.DataFrame, model: str = "rf") -> pd.DataFrame:
    """Probability, risk category and confidence for raw observation rows.

    confidence = |2p - 1|: 0 when the model is on the fence (p=0.5), 1 when certain.
    """
    X = build_features(raw, bundle.port_ids)
    p = bundle.models[model].predict_proba(X)[:, 1]
    return pd.DataFrame({"probability": p,
                         "risk_category": [risk_category(v) for v in p],
                         "confidence": np.abs(2 * p - 1)}, index=raw.index)


# ----------------------------------------------------------------------------- evaluation
def _evaluate(y, score, pred, rng, probabilistic: bool) -> dict:
    y, score, pred = np.asarray(y), np.asarray(score, dtype=float), np.asarray(pred)
    out = {"precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0)),
           "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, score))
        fpr, tpr, _ = roc_curve(y, score)
        out["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        aucs = []
        for _ in range(500):                      # bootstrap CI: the test set is small
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) == 2:
                aucs.append(roc_auc_score(y[idx], score[idx]))
        out["roc_auc_ci"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
        if probabilistic:
            out["brier"] = float(brier_score_loss(y, score))
            out["avg_precision"] = float(average_precision_score(y, score))
            pt, pp = calibration_curve(y, score, n_bins=5, strategy="quantile")
            out["calibration"] = {"prob_true": pt.tolist(), "prob_pred": pp.tolist()}
    else:
        out["roc_auc"], out["roc_auc_ci"] = float("nan"), [float("nan")] * 2
    return out


def chronological_split(df: pd.DataFrame, test_fraction: float = config.TEST_FRACTION):
    df = df.sort_values(["timestamp", "port_id"], kind="stable")
    n_test = int(round(len(df) * test_fraction))
    train, test = df.iloc[:-n_test], df.iloc[-n_test:]
    assert train["timestamp"].max() <= test["timestamp"].min(), "split is not chronological"
    return train, test


def _estimators(seed: int) -> dict:
    lr = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=2000, random_state=seed))])
    rf = Pipeline([("impute", SimpleImputer(strategy="median")),
                   ("clf", RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                                  random_state=seed, n_jobs=-1))])
    return {"lr": lr, "rf": rf}


def train_and_evaluate(df: pd.DataFrame, threshold_hours: float = config.CONGESTION_THRESHOLD_HOURS,
                       seed: int = config.SEED) -> ModelBundle:
    train, test = chronological_split(df)
    port_ids = sorted(df["port_id"].unique())
    Xtr, Xte = build_features(train, port_ids), build_features(test, port_ids)
    ytr, yte = make_label(train, threshold_hours), make_label(test, threshold_hours)

    models = {k: TimeAwareCalibratedClassifier(est).fit(Xtr, ytr) for k, est in _estimators(seed).items()}

    rng = np.random.default_rng(seed)
    metrics = {"threshold_hours": threshold_hours, "decision_threshold": config.DECISION_THRESHOLD,
               "base_rate_train": float(ytr.mean()), "base_rate_test": float(yte.mean())}
    for k, m in models.items():
        p = m.predict_proba(Xte)[:, 1]
        metrics[k] = _evaluate(yte, p, p >= config.DECISION_THRESHOLD, rng, probabilistic=True)
    hist = test["historical_delay_hours"].to_numpy()          # naive rule: "recent delays were long"
    metrics["baseline"] = _evaluate(yte, hist, hist > threshold_hours, rng, probabilistic=False)

    src = Xtr.assign(port_id=train["port_id"].to_numpy())
    cols = BASE_FEATURES + ["load_ratio"]
    reference = pd.concat([src.groupby("port_id")[cols].median(),
                           Xtr[cols].median().to_frame("__global__").T])

    return ModelBundle(threshold_hours, port_ids, feature_names(port_ids), models, metrics, reference,
                       test.index.tolist(), pd.Timestamp(test["timestamp"].min()), len(train), len(test))


def metrics_table(bundle: ModelBundle) -> pd.DataFrame:
    rows = []
    for key, label in [("lr", "Logistic Regression"), ("rf", "Random Forest"),
                       ("baseline", "Naive rule: recent delay > threshold")]:
        m = bundle.metrics[key]
        rows.append({"Model": label, "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"],
                     "ROC-AUC": m["roc_auc"],
                     "ROC-AUC 95% CI": f"{m['roc_auc_ci'][0]:.2f}-{m['roc_auc_ci'][1]:.2f}",
                     "Brier": m.get("brier", np.nan)})
    return pd.DataFrame(rows).round(3)


# ----------------------------------------------------------------------------- persistence
def load_or_train(df: pd.DataFrame, threshold_hours: float = config.CONGESTION_THRESHOLD_HOURS,
                  seed: int = config.SEED, force: bool = False) -> ModelBundle:
    fp = int(pd.util.hash_pandas_object(df, index=True).sum() % 2**31)
    path = config.ARTIFACT_DIR / f"bundle_t{threshold_hours:g}_s{seed}_{fp}.joblib"
    if path.exists() and not force:
        try:
            return joblib.load(path)
        except Exception:
            pass
    bundle = train_and_evaluate(df, threshold_hours, seed)
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for old in config.ARTIFACT_DIR.glob(f"bundle_t{threshold_hours:g}_s{seed}_*.joblib"):
        old.unlink()
    joblib.dump(bundle, path)
    return bundle
