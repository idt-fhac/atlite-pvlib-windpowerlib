"""
Investigate why matched-ERA5 Windpowerlib >> Atlite wind.

Controlled national experiments on germany_2023.nc:
  A) Atlite V112 (baseline from matched study / recompute)
  B) WPL V112/3000, hub=80 m, u from ERA5 100 m log→80 m (Atlite-like shear)
  C) WPL V112 + artificial cut-out at 25 m/s (Atlite add_cutout_windspeed)
  D) WPL V112 using raw wnd100m at hub (no down-extrapolation) — upper bias check
  E) Existing class-based WPL from matched_era5_timeseries.csv

Also dumps power-curve and wind-speed distribution diagnostics.

Outputs → structured_research/results/:
  matched_wind_v112_investigation.csv
  matched_wind_v112_investigation.md
  matched_wind_powercurve_compare.csv
"""

from __future__ import annotations

from pathlib import Path

import sys
import logging

import atlite
import numpy as np
import pandas as pd
from tqdm import tqdm
from windpowerlib import ModelChain, WindTurbine
from atlite.resource import get_windturbineconfig
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    resolve_engine,
    offline_mode,
    load_plz_nuts,
    map_row_to_tso,
    classify_wind_turbines,
    query_mastr_wind,
    query_entsoe_generation,
    calculate_metrics,
    ensure_results_dir,
    cutout_path,
    result_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("v112_invest")

ZONES = ["DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNET"]


def metrics_row(sim, act, label: str) -> dict:
    m = calculate_metrics(sim, act)
    return {
        "case": label,
        "Ratio_vs_ENTSOE_%": round(m["ratio"], 2),
        "Corr_ENTSOE": round(m["corr"], 4),
        "Sim_GWh": round(m["sim_sum"] / 1e3, 1),
        "ENTSOE_GWh": round(m["act_sum"] / 1e3, 1),
    }


def vs_atlite(sim, atl, label: str) -> dict:
    m = calculate_metrics(sim, atl)
    return {
        "case": label,
        "Ratio_vs_Atlite_%": round(m["ratio"], 2),
        "Corr_Atlite": round(m["corr"], 4),
        "WPL_GWh": round(m["sim_sum"] / 1e3, 1),
        "Atlite_GWh": round(m["act_sum"] / 1e3, 1),
    }


def log_down_to_hub(u100: np.ndarray, z0: np.ndarray, hub: float = 80.0) -> np.ndarray:
    z0 = np.maximum(z0.astype(float), 1e-4)
    return u100 * np.log(hub / z0) / np.log(100.0 / z0)


def build_hub_winds(wind_df, nuts3_coords, ds, date_range):
    """Per-NUTS3: capacity_kw, u80 (log←100m), u100."""
    cap = wind_df.groupby("nuts3")["maxPower"].sum()  # kW
    out = {}
    for nuts, capacity_kw in tqdm(cap.items(), desc="Extract hub winds"):
        if nuts not in nuts3_coords.index:
            continue
        lat = float(nuts3_coords.loc[nuts, "latitude"])
        lon = float(nuts3_coords.loc[nuts, "longitude"])
        yi = int(np.abs(ds.y.values - lat).argmin())
        xi = int(np.abs(ds.x.values - lon).argmin())
        u100 = ds["wnd100m"].isel(y=yi, x=xi).to_pandas()
        z0 = ds["roughness"].isel(y=yi, x=xi).to_pandas()
        if getattr(u100.index, "tz", None) is not None:
            u100.index = u100.index.tz_localize(None)
            z0.index = z0.index.tz_localize(None)
        u100 = u100.reindex(date_range, method="nearest").astype(float)
        z0 = z0.reindex(date_range, method="nearest").astype(float)
        u80 = log_down_to_hub(u100.values, z0.values, 80.0)
        out[nuts] = {
            "capacity_kw": float(capacity_kw),
            "u100": u100.values,
            "u80": u80,
        }
    return out


def run_wpl_v112_from_hubs(hubs, date_range, *, mode: str):
    """
    mode:
      'atlite_like' — u80 via log from wnd100m + roughness; V112 curve as-is
      'with_cutout' — same + force P=0 for u>=25
      'raw100' — feed wnd100m as if at hub 80 (wrong physically; bias check)
    """
    wt = WindTurbine(hub_height=80.0, turbine_type="V112/3000")
    sim = pd.Series(0.0, index=date_range)
    for nuts, h in tqdm(hubs.items(), desc=f"WPL V112 {mode}"):
        u_hub = h["u100"] if mode == "raw100" else h["u80"]
        ww = pd.DataFrame(
            np.asarray([np.full(len(date_range), 280.0), u_hub]).T,
            index=date_range,
            columns=[["temperature", "wind_speed"], [2, 80]],
        )
        mc = ModelChain(wt, wind_speed_model="interpolation_extrapolation").run_model(ww)
        power_mw = mc.power_output / wt.nominal_power * (h["capacity_kw"] * 1e3) / 1e6
        if mode == "with_cutout":
            power_mw = power_mw.where(pd.Series(u_hub, index=date_range) < 25.0, 0.0)
        sim = sim + power_mw
    return sim


def apply_curve_mw(u: np.ndarray, V: np.ndarray, POW: np.ndarray) -> np.ndarray:
    """Linear interpolate power curve; clamp outside range to edge values."""
    return np.interp(u, V, POW, left=POW[0], right=POW[-1])


def run_curve_on_u80(hubs, date_range, V, POW, *, rated_mw: float, cutout_at: float | None = None):
    """Capacity-factor power curve on same u80 winds (no windpowerlib). POW in MW."""
    sim = pd.Series(0.0, index=date_range)
    for h in hubs.values():
        p_mw = apply_curve_mw(h["u80"], V, POW)
        if cutout_at is not None:
            p_mw = np.where(h["u80"] >= cutout_at, 0.0, p_mw)
        cf = p_mw / rated_mw
        sim = sim + pd.Series(cf * (h["capacity_kw"] * 1e3) / 1e6, index=date_range)
    return sim


def powercurve_table():
    cfg = get_windturbineconfig("Vestas_V112_3MW", add_cutout_windspeed=True)
    wt = WindTurbine(hub_height=80.0, turbine_type="V112/3000")
    vs = np.arange(0, 30, 1.0)
    atl = np.interp(vs, cfg["V"], cfg["POW"])  # MW
    wpl = np.interp(vs, wt.power_curve.wind_speed.values, wt.power_curve.value.values) / 1e6
    return pd.DataFrame({
        "V_ms": vs,
        "atlite_V112_MW": atl,
        "wpl_V112_MW": wpl,
        "ratio_wpl_atlite": np.where(atl > 1e-6, wpl / atl, np.nan),
    })


def main():
    ensure_results_dir()
    date_range = pd.date_range("2023-01-01", "2023-12-31 23:00:00", freq="h")
    engine = resolve_engine()
    logger.info("Mode: %s", "OFFLINE" if offline_mode() or engine is None else "DB")

    cutout = atlite.Cutout(cutout_path("germany_2023.nc"))
    ds = cutout.data

    wind_df = query_mastr_wind(engine, "2023-01-01", "2023-12-31")
    plz = load_plz_nuts(engine)
    wind_df["nuts3"] = wind_df["plzCode"].map(plz["nuts3"])
    wind_df = wind_df.dropna(subset=["nuts3"])
    wind_df["tso"] = wind_df.apply(map_row_to_tso, axis=1)
    wind_df = wind_df[wind_df["tso"] != "UNKNOWN"]
    wind_df = classify_wind_turbines(wind_df)
    nuts3_coords = plz.groupby("nuts3")[["latitude", "longitude"]].mean()

    entsoe_w, _ = query_entsoe_generation(engine, "2023-01-01", "2023-12-31", ZONES, date_range)
    act = sum((entsoe_w[z].reindex(date_range, fill_value=0.0) for z in ZONES),
              start=pd.Series(0.0, index=date_range))

    # Atlite national (same as matched study)
    logger.info("Recomputing Atlite V112 national...")
    atl_parts = []
    for tso in ZONES:
        t_wind = wind_df[wind_df["tso"] == tso].copy()
        t_wind["x"] = t_wind["lon"]
        t_wind["y"] = t_wind["lat"]
        t_wind["capacity_mw"] = t_wind["maxPower"] / 1e3
        layout = cutout.layout_from_capacity_list(t_wind, col="capacity_mw")
        w_ds = cutout.wind(
            turbine="Vestas_V112_3MW",
            layout=layout,
            add_cutout_windspeed=True,
        )
        atl_parts.append(pd.Series(w_ds.to_series().values, index=date_range))
    atl = sum(atl_parts, start=pd.Series(0.0, index=date_range))

    # Prior class-based WPL if available
    ts_path = result_path("matched_era5_timeseries.csv")
    class_wpl = None
    if ts_path.exists():
        prev = pd.read_csv(ts_path, index_col=0, parse_dates=True)
        class_wpl = prev["wpl_wind"].reindex(date_range)

    logger.info("Extracting NUTS3 hub winds once...")
    hubs = build_hub_winds(wind_df, nuts3_coords, ds, date_range)

    logger.info("WPL V112 atlite-like shear...")
    wpl_like = run_wpl_v112_from_hubs(hubs, date_range, mode="atlite_like")
    logger.info("WPL V112 + cut-out mask...")
    wpl_cut = run_wpl_v112_from_hubs(hubs, date_range, mode="with_cutout")
    logger.info("WPL V112 raw 100m as hub (bias check)...")
    wpl_raw = run_wpl_v112_from_hubs(hubs, date_range, mode="raw100")

    # Same u80, pure curve interp (isolates oedb vs atlite curve on identical winds)
    cfg = get_windturbineconfig("Vestas_V112_3MW", add_cutout_windspeed=True)
    wt_pc = WindTurbine(hub_height=80.0, turbine_type="V112/3000")
    rated_atl = float(cfg["P"])
    rated_wpl = float(wt_pc.nominal_power) / 1e6
    curve_atl = run_curve_on_u80(
        hubs, date_range, np.asarray(cfg["V"]), np.asarray(cfg["POW"]),
        rated_mw=rated_atl, cutout_at=None,
    )
    # Atlite config already has cut-out padded to 0
    curve_wpl = run_curve_on_u80(
        hubs, date_range,
        wt_pc.power_curve.wind_speed.values,
        wt_pc.power_curve.value.values / 1e6,
        rated_mw=rated_wpl, cutout_at=None,
    )
    curve_wpl_cut = run_curve_on_u80(
        hubs, date_range,
        wt_pc.power_curve.wind_speed.values,
        wt_pc.power_curve.value.values / 1e6,
        rated_mw=rated_wpl, cutout_at=25.0,
    )

    rows_e = [
        metrics_row(atl, act, "Atlite V112 (layout)"),
        metrics_row(wpl_like, act, "WPL V112 hub80 log←100m"),
        metrics_row(wpl_cut, act, "WPL V112 + cutout@25"),
        metrics_row(wpl_raw, act, "WPL V112 using wnd100m raw"),
        metrics_row(curve_atl, act, "Curve-Atlite on NUTS3 u80"),
        metrics_row(curve_wpl, act, "Curve-WPL on NUTS3 u80"),
        metrics_row(curve_wpl_cut, act, "Curve-WPL+cutout on NUTS3 u80"),
    ]
    rows_a = [
        vs_atlite(wpl_like, atl, "WPL V112 hub80 log←100m"),
        vs_atlite(wpl_cut, atl, "WPL V112 + cutout@25"),
        vs_atlite(wpl_raw, atl, "WPL V112 using wnd100m raw"),
        vs_atlite(curve_atl, atl, "Curve-Atlite on NUTS3 u80"),
        vs_atlite(curve_wpl, atl, "Curve-WPL on NUTS3 u80"),
        vs_atlite(curve_wpl_cut, atl, "Curve-WPL+cutout on NUTS3 u80"),
    ]
    if class_wpl is not None:
        rows_e.append(metrics_row(class_wpl, act, "WPL MaStR fleet (matched study)"))
        rows_a.append(vs_atlite(class_wpl, atl, "WPL MaStR fleet (matched study)"))

    # Curve-only ratio on identical winds
    curve_delta = vs_atlite(curve_wpl, curve_atl, "Curve-WPL / Curve-Atlite (same u80)")
    rows_a.append({
        "case": curve_delta["case"],
        "Ratio_vs_Atlite_%": curve_delta["Ratio_vs_Atlite_%"],
        "Corr_Atlite": curve_delta["Corr_Atlite"],
        "WPL_GWh": curve_delta["WPL_GWh"],
        "Atlite_GWh": curve_delta["Atlite_GWh"],
    })

    entsoe_df = pd.DataFrame(rows_e)
    atlite_df = pd.DataFrame(rows_a)
    entsoe_df.to_csv(result_path("matched_wind_v112_vs_entsoe.csv"), index=False)
    atlite_df.to_csv(result_path("matched_wind_v112_vs_atlite.csv"), index=False)

    pc = powercurve_table()
    pc.to_csv(result_path("matched_wind_powercurve_compare.csv"), index=False)

    # Wind speed / capacity diagnostics
    total_gw = wind_df["maxPower"].sum() / 1e6
    by_class = wind_df.groupby("class")["maxPower"].sum() / 1e6

    # Hours where |u| high and curves diverge
    # sample one cell mean u100
    u100_mean = float(ds["wnd100m"].mean())
    hours_ge_25 = int((ds["wnd100m"] >= 25).sum())  # cell-hours, rough

    lines = [
        "# Why matched ERA5 Windpowerlib ≫ Atlite",
        "",
        "## Code differences found",
        "1. **Turbine fleet:** matched study used WPL **4 SP classes** (hub 80–120 m). "
        "Atlite uses a single **Vestas V112** at **hub_height=80 m** for all capacity "
        "(PyPSA-Eur onshore default).",
        "2. **Hub wind:** Atlite **down-extrapolates** ERA5 `wnd100m` → 80 m with logarithmic + `roughness`. "
        "Class WPL **up-extrapolated** to 105–120 m for many MW → systematically higher hub winds.",
        "3. **Power curve cut-out:** Atlite `add_cutout_windspeed=True` forces **P=0 at max V (~25 m/s)**. "
        f"WPL `V112/3000` stays at **rated power through 25 m/s** (no cut-out in oedb curve). "
        f"Atlite V112 hub={cfg['hub_height']} m, P={cfg['P']} MW.",
        "4. **Low-wind curve:** at 4–5 m/s WPL V112 power ≫ Atlite V112 (see powercurve CSV).",
        "5. **Spatial aggregation:** Atlite uses plant-level `layout_from_capacity_list`; "
        "WPL uses NUTS3-centroid weather × county capacity (small extra error, high correlation).",
        "",
        f"## Capacity: {total_gw:.2f} GW total",
        by_class.to_string(),
        "",
        f"Cutout mean wnd100m ≈ {u100_mean:.2f} m/s; gridcell-hours with wnd100m≥25: {hours_ge_25}",
        "",
        "## Results vs ENTSO-E",
        entsoe_df.to_string(index=False),
        "",
        "## Results vs Atlite (same ERA5 cutout)",
        atlite_df.to_string(index=False),
        "",
        "## Reading",
        "- If **WPL V112 log←100m** ≈ Atlite, the 142% gap was mostly **class turbines + taller hubs**.",
        "- If still high, residual is **power curve** (esp. cut-out / low-V) and centroid vs layout.",
        "- Adding cut-out@25 shows how much storm hours inflate WPL.",
        "",
    ]
    path = result_path("matched_wind_v112_investigation.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Wrote %s", path)
    print(entsoe_df.to_string(index=False))
    print(atlite_df.to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
