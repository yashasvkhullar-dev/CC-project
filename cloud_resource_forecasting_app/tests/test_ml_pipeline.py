import numpy as np
import pandas as pd
import pytest

from ml_pipeline import clean_dataset, build_supervised_dataset, cost_summary


def sample_df(n=60):
    times = pd.date_range("2026-01-01", periods=n, freq="5min")
    return pd.DataFrame({
        "timestamp": list(times) * 2,
        "server_id": ["server-1"] * n + ["S002"] * n,
        "cpu_utilization": [float((i * 7) % 100) for i in range(n)] +
                           [float((i * 11 + 5) % 100) for i in range(n)],
        "memory_utilization": [50.0] * (2 * n),
        "workload_level": ["med"] * (2 * n),
    })


def test_clean_dataset_normalizes_and_aggregates():
    raw = sample_df()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    cleaned, notes = clean_dataset(raw)
    assert cleaned["server_id"].str.match(r"^S\d{3}$").all()
    assert not cleaned.duplicated(["server_id", "timestamp"]).any()
    assert "medium" in cleaned["workload_level"].unique()
    assert notes


def test_missing_required_columns_fails():
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_dataset(pd.DataFrame({"timestamp": ["2026-01-01"]}))


def test_target_is_five_minutes_ahead_for_same_server():
    cleaned, _ = clean_dataset(sample_df())
    base, numeric, categorical = build_supervised_dataset(cleaned)
    row = base[(base["server_id"] == "S001")].sort_values("timestamp").iloc[0]
    matching = cleaned[(cleaned["server_id"] == row["server_id"]) &
                       (cleaned["timestamp"] == row["timestamp"] + pd.Timedelta(minutes=5))]
    assert len(matching) == 1
    assert row["target_cpu_5min"] == matching["cpu_utilization"].iloc[0]
    assert "target_cpu_5min" not in numeric + categorical
    assert "timestamp" not in numeric + categorical


def test_cost_summary_math():
    summary = cost_summary([10, 50, 90], rates={"Small": 0.02, "Medium": 0.04, "Large": 0.08})
    assert summary["intervals"] == 3
    assert summary["fixed_cost"] == pytest.approx(0.08 * 0.25)
    assert summary["scaled_cost"] == pytest.approx((0.02 + 0.04 + 0.08) * (5 / 60))
    assert summary["tier_counts"] == {"Small": 1, "Medium": 1, "Large": 1}


def test_cost_thresholds_must_be_ordered():
    with pytest.raises(ValueError):
        cost_summary([50], small_max=70, medium_max=35)
