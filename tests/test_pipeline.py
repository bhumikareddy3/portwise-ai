import numpy as np
import pytest

import config
from data_generator import generate_dataset, make_label
from features import FORBIDDEN, build_features, feature_names
from model import chronological_split, predict_frame, train_and_evaluate


@pytest.fixture(scope="module")
def df():
    return generate_dataset(seed=config.SEED)


def test_reproducible(df):
    assert df.equals(generate_dataset(seed=config.SEED))


def test_size_and_columns(df):
    assert 300 <= len(df) <= 500
    for c in ["port_id", "timestamp", "vessel_arrivals", "queue_length", "port_capacity",
              "weather_severity", "historical_delay_hours", "actual_delay_hours"]:
        assert c in df.columns


def test_features_exclude_outcomes(df):
    ports = sorted(df.port_id.unique())
    assert not FORBIDDEN & set(feature_names(ports))
    assert not FORBIDDEN & set(build_features(df, ports).columns)


def test_history_uses_only_past(df):
    for _, g in df.sort_values("timestamp").groupby("port_id"):
        a, h = g["actual_delay_hours"].to_numpy(), g["historical_delay_hours"].to_numpy()
        for i in range(1, len(g)):
            expected = a[max(0, i - config.HISTORY_WINDOW):i].mean()
            assert abs(h[i] - expected) < 0.011


def test_chronological_split(df):
    tr, te = chronological_split(df)
    assert tr["timestamp"].max() <= te["timestamp"].min()


def test_model_outputs_valid(df):
    b = train_and_evaluate(df)
    out = predict_frame(b, df.loc[b.test_index], "rf")
    assert out["probability"].between(0, 1).all()
    assert set(out["risk_category"]) <= {"Low", "Medium", "High"}
    assert 0 <= b.metrics["rf"]["roc_auc"] <= 1
    assert np.isclose(sum(sum(r) for r in b.metrics["rf"]["confusion_matrix"]), b.n_test)
    assert make_label(df).isin([0, 1]).all()
