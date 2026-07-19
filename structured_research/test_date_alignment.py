from pathlib import Path
"""
test_date_alignment.py
======================
Offline regression checks for date-window slicing and metric alignment.
Run:  python structured_research/test_date_alignment.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    calculate_metrics,
    query_entsoe_generation,
    query_ecmwf_weather_nuts3,
    slice_time_index,
)


def test_metrics_inner_align():
    act = pd.Series(
        np.arange(8760, dtype=float),
        index=pd.date_range("2023-01-01", periods=8760, freq="h"),
    )
    sim = act.loc["2023-01-01":"2023-01-31"] * 0.8
    m = calculate_metrics(sim, act)
    assert m["n"] == 744, m["n"]
    assert abs(m["ratio"] - 80.0) < 1e-6, m["ratio"]
    assert abs(m["act_sum"] - act.loc["2023-01-01":"2023-01-31"].sum()) < 1e-6
    print("OK  calculate_metrics inner-aligns before sums")


def test_entsoe_january_slice():
    date_range = pd.date_range("2023-01-01", "2023-01-31 23:00", freq="h")
    zones = ["DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNET"]
    wind, solar = query_entsoe_generation(
        None, "2023-01-01", "2023-01-31 23:00", zones, date_range
    )
    assert len(wind) == 744, len(wind)
    assert len(solar) == 744, len(solar)
    # Full-year DE_50HZ wind is ~38126 GWh; January must be much smaller
    jan_gwh = wind["DE_50HZ"].sum() / 1e3
    assert 3000 < jan_gwh < 8000, jan_gwh
    print(f"OK  ENTSO-E January slice: DE_50HZ wind = {jan_gwh:.1f} GWh (n={len(wind)})")


def test_weather_january_slice():
    start, end = "2023-01-01", "2023-01-31 23:00"
    df = query_ecmwf_weather_nuts3(None, start, end, nuts_prefix="DE")
    assert "time" in df.columns
    tmin, tmax = df["time"].min(), df["time"].max()
    assert tmin >= pd.Timestamp("2023-01-01")
    assert tmax <= pd.Timestamp("2023-01-31 23:00")
    print(f"OK  weather NUTS3 January slice: {tmin} → {tmax} ({len(df)} rows)")


def test_slice_helper():
    s = pd.Series(1.0, index=pd.date_range("2023-01-01", periods=100, freq="h"))
    out = slice_time_index(s, start_date="2023-01-02", end_date="2023-01-02 23:00")
    assert len(out) == 24
    print("OK  slice_time_index label slice")


if __name__ == "__main__":
    test_metrics_inner_align()
    test_slice_helper()
    test_entsoe_january_slice()
    test_weather_january_slice()
    print("\nAll date-alignment checks passed.")
