"""
Investigate why OEDS/Windpowerlib under-predicts wind at night vs day.

Analyses (offline-capable):
  1) Instant diagnostics from saved national/TSO timeseries
  2) ECMWF 10 m day/night wind stats vs log-extrapolated hub height
  3) ERA5 100 m (Atlite cutout) vs ECMWF→100 m log at capacity-weighted NUTS3
  4) z0 sweep with day/night metrics split
  5) Dual-z0 (day fixed 0.2, night elevated)
  6) Hellman/power-law alpha sweep (hub winds fed at hub height)

Outputs → structured_research/results/:
  night_wind_diagnostics.csv
  night_wind_weather_compare.csv
  night_wind_z0_daynight.csv
  night_wind_dual_z0.csv
  night_wind_alpha_daynight.csv
  night_wind_sensitivity_summary.md
  night_wind_z0_daynight.png
"""

from __future__ import annotations

from pathlib import Path

import sys
import time
import logging

import atlite
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from windpowerlib import ModelChain, WindTurbine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    resolve_engine,
    offline_mode,
    load_plz_nuts,
    map_row_to_tso,
    classify_wind_turbines,
    query_mastr_wind,
    query_entsoe_generation,
    query_ecmwf_weather_nuts3,
    calculate_metrics,
    ensure_results_dir,
    result_path,
    cutout_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("night_wind")

ZONES = ["DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNET"]
DAY_HOURS = list(range(6, 18))
NIGHT_HOURS = list(range(18, 24)) + list(range(0, 6))

TURBINE_MODELS = {
    "class_low": WindTurbine(hub_height=120.0, turbine_type="V112/3000"),
    "class_med_low": WindTurbine(hub_height=105.0, turbine_type="V90/2000"),
    "class_med": WindTurbine(hub_height=100.0, turbine_type="E-82/2300"),
    "class_high": WindTurbine(hub_height=80.0, turbine_type="E-70/2000"),
}

Z0_SWEEP = [0.03, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20]
NIGHT_Z0_SWEEP = [0.20, 0.40, 0.80, 1.20, 2.00]
ALPHA_SWEEP = [0.10, 0.14, 0.20, 0.28, 0.35, 0.45]
HUB_REF = 100.0  # m, for weather compare


def day_night_mask(index: pd.DatetimeIndex, which: str) -> np.ndarray:
    h = index.hour
    if which == "Day":
        return np.isin(h, DAY_HOURS)
    if which == "Night":
        return np.isin(h, NIGHT_HOURS)
    raise ValueError(which)


def metrics_day_night(sim: pd.Series, act: pd.Series) -> dict:
    out = {}
    for which in ("Day", "Night", "Full"):
        if which == "Full":
            m = calculate_metrics(sim, act)
        else:
            mask = day_night_mask(act.index, which)
            m = calculate_metrics(sim[mask], act[mask])
        out[f"{which}_ratio"] = m["ratio"]
        out[f"{which}_corr"] = m["corr"]
        out[f"{which}_act_GWh"] = m["act_sum"] / 1e3
        out[f"{which}_sim_GWh"] = m["sim_sum"] / 1e3
    out["night_day_ratio_gap_pp"] = out["Night_ratio"] - out["Day_ratio"]
    return out


def log_extrapolate(u10: np.ndarray, z0: float, hub: float = HUB_REF) -> np.ndarray:
    z0 = max(float(z0), 1e-6)
    return u10 * np.log(hub / z0) / np.log(10.0 / z0)


def hellman(u10: np.ndarray, alpha: float, hub: float = HUB_REF) -> np.ndarray:
    return u10 * (hub / 10.0) ** alpha


# ---------------------------------------------------------------------------
# Instant diagnostics from saved timeseries
# ---------------------------------------------------------------------------
def diagnostics_from_timeseries() -> pd.DataFrame:
    rows = []
    nat = pd.read_csv(result_path("annual_seasonal_timeseries.csv"), index_col=0, parse_dates=True)
    for label, sim_col in [("OEDS", "oeds_wind"), ("Atlite", "atlite_wind")]:
        m = metrics_day_night(nat[sim_col], nat["entsoe_wind"])
        rows.append({"Scale": "Germany national", "Model": label, **m})

    # Mean power by hour-of-day (normalized)
    for label, col in [("ENTSO-E", "entsoe_wind"), ("OEDS", "oeds_wind"), ("Atlite", "atlite_wind")]:
        by_hour = nat[col].groupby(nat.index.hour).mean()
        day_m = by_hour.loc[DAY_HOURS].mean()
        night_m = by_hour.loc[NIGHT_HOURS].mean()
        rows.append({
            "Scale": "Germany national (mean MW)",
            "Model": label,
            "Day_ratio": day_m,
            "Night_ratio": night_m,
            "Full_ratio": nat[col].mean(),
            "Day_corr": np.nan,
            "Night_corr": np.nan,
            "Full_corr": np.nan,
            "Day_act_GWh": np.nan,
            "Night_act_GWh": np.nan,
            "Full_act_GWh": np.nan,
            "Day_sim_GWh": np.nan,
            "Night_sim_GWh": np.nan,
            "Full_sim_GWh": np.nan,
            "night_day_ratio_gap_pp": night_m - day_m,
        })

    tso_path = result_path("annual_seasonal_tso_timeseries.parquet")
    if tso_path.exists():
        tso = pd.read_parquet(tso_path)
        if not isinstance(tso.index, pd.DatetimeIndex):
            tso.index = pd.to_datetime(tso.index)
        for zone in ZONES:
            for label, prefix in [("OEDS", "oeds"), ("Atlite", "atlite")]:
                m = metrics_day_night(tso[f"{prefix}_wind_{zone}"], tso[f"entsoe_wind_{zone}"])
                rows.append({"Scale": zone, "Model": label, **m})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Weather: ECMWF 10m / log hub vs ERA5 100m
# ---------------------------------------------------------------------------
def weather_day_night_compare(wind_groups, weather_dict, nuts3_coords, date_range) -> pd.DataFrame:
    # Capacity-weighted top NUTS3 covering ~80% of capacity
    cap = wind_groups.sum(axis=1).sort_values(ascending=False)
    cum = cap.cumsum() / cap.sum()
    keep = cap.index[cum <= 0.80]
    if len(keep) < 20:
        keep = cap.head(40).index

    cutout_file = cutout_path("germany_2023.nc")
    cutout = atlite.Cutout(cutout_file) if cutout_file.exists() else None

    rows = []
    ecmwf_10_day, ecmwf_10_night = [], []
    ecmwf_100_day, ecmwf_100_night = [], []
    era5_100_day, era5_100_night = [], []
    weights = []

    for nuts in tqdm(keep, desc="Weather day/night NUTS3"):
        if nuts not in weather_dict:
            continue
        w = weather_dict[nuts].reindex(date_range, method="nearest")
        u10 = w["wind_speed"].astype(float).values
        u100_log = log_extrapolate(u10, 0.2, HUB_REF)
        day = day_night_mask(date_range, "Day")
        night = day_night_mask(date_range, "Night")
        wt = float(cap.loc[nuts])
        weights.append(wt)
        ecmwf_10_day.append(np.nanmean(u10[day]))
        ecmwf_10_night.append(np.nanmean(u10[night]))
        ecmwf_100_day.append(np.nanmean(u100_log[day]))
        ecmwf_100_night.append(np.nanmean(u100_log[night]))

        era_d = era_n = np.nan
        if cutout is not None and nuts in nuts3_coords.index:
            lat = float(nuts3_coords.loc[nuts, "latitude"])
            lon = float(nuts3_coords.loc[nuts, "longitude"])
            try:
                # nearest cell wind at 100 m if available
                ds = cutout.data
                # atlite stores wnd100m typically
                if "wnd100m" in ds:
                    # pick nearest lat/lon
                    lat_idx = int(np.abs(ds.y.values - lat).argmin())
                    lon_idx = int(np.abs(ds.x.values - lon).argmin())
                    series = ds["wnd100m"].isel(y=lat_idx, x=lon_idx).to_pandas()
                    series = series.reindex(date_range, method="nearest")
                    era_d = float(series[day].mean())
                    era_n = float(series[night].mean())
            except Exception as exc:  # noqa: BLE001
                logger.debug("ERA5 sample failed for %s: %s", nuts, exc)
        era5_100_day.append(era_d)
        era5_100_night.append(era_n)

        rows.append({
            "nuts3": nuts,
            "capacity_kW": wt,
            "ecmwf_10_day": ecmwf_10_day[-1],
            "ecmwf_10_night": ecmwf_10_night[-1],
            "ecmwf_100log_z0.2_day": ecmwf_100_day[-1],
            "ecmwf_100log_z0.2_night": ecmwf_100_night[-1],
            "era5_100_day": era_d,
            "era5_100_night": era_n,
            "ecmwf_10_night_day": ecmwf_10_night[-1] / ecmwf_10_day[-1] if ecmwf_10_day[-1] else np.nan,
            "era5_100_night_day": era_n / era_d if era_d and not np.isnan(era_d) else np.nan,
        })

    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    summary = {
        "ecmwf_10_day_wmean": np.nansum(np.asarray(ecmwf_10_day) * w),
        "ecmwf_10_night_wmean": np.nansum(np.asarray(ecmwf_10_night) * w),
        "ecmwf_100log_day_wmean": np.nansum(np.asarray(ecmwf_100_day) * w),
        "ecmwf_100log_night_wmean": np.nansum(np.asarray(ecmwf_100_night) * w),
        "era5_100_day_wmean": np.nansum(np.asarray(era5_100_day) * w),
        "era5_100_night_wmean": np.nansum(np.asarray(era5_100_night) * w),
    }
    summary["ecmwf_10_night_over_day"] = summary["ecmwf_10_night_wmean"] / summary["ecmwf_10_day_wmean"]
    summary["ecmwf_100log_night_over_day"] = summary["ecmwf_100log_day_wmean"] and (
        summary["ecmwf_100log_night_wmean"] / summary["ecmwf_100log_day_wmean"]
    )
    if summary["era5_100_day_wmean"] and not np.isnan(summary["era5_100_day_wmean"]):
        summary["era5_100_night_over_day"] = summary["era5_100_night_wmean"] / summary["era5_100_day_wmean"]
        summary["era5_over_ecmwf100log_day"] = summary["era5_100_day_wmean"] / summary["ecmwf_100log_day_wmean"]
        summary["era5_over_ecmwf100log_night"] = summary["era5_100_night_wmean"] / summary["ecmwf_100log_night_wmean"]

    df = pd.DataFrame(rows)
    df.attrs["summary"] = summary
    return df, summary


# ---------------------------------------------------------------------------
# Windpowerlib sims
# ---------------------------------------------------------------------------
def run_wind_sim(
    wind_groups,
    weather_dict,
    nuts3_to_tso,
    date_range,
    *,
    z0: float | None = None,
    z0_by_hour: np.ndarray | None = None,
    alpha: float | None = None,
):
    """If alpha is set, Hellman-extrapolate to each turbine hub and feed hub winds."""
    sim_w_tso = {tso: pd.Series(0.0, index=date_range) for tso in ZONES}
    for nuts in wind_groups.index:
        if nuts not in weather_dict or nuts not in nuts3_to_tso:
            continue
        tso = nuts3_to_tso[nuts]
        weather_df = weather_dict[nuts].reindex(date_range, method="nearest")
        u10 = weather_df["wind_speed"].values.astype(float)
        temp = weather_df["temp_air"].values
        row = wind_groups.loc[nuts]

        if alpha is not None:
            # per-class hub height Hellman
            for cls_name, capacity_kw in row.items():
                if capacity_kw <= 0:
                    continue
                wt = TURBINE_MODELS[cls_name]
                u_hub = hellman(u10, alpha, wt.hub_height)
                ww = pd.DataFrame(
                    np.asarray([temp, u_hub]).T,
                    index=date_range,
                    columns=[["temperature", "wind_speed"], [2, wt.hub_height]],
                )
                mc = ModelChain(wt, wind_speed_model="interpolation_extrapolation").run_model(ww)
                cls_power_mw = mc.power_output / wt.nominal_power * (capacity_kw * 1e3) / 1e6
                sim_w_tso[tso] += cls_power_mw
            continue

        if z0_by_hour is not None:
            z0_series = z0_by_hour
        else:
            z0_series = float(z0) * np.ones(len(date_range))

        ww = pd.DataFrame(
            np.asarray([z0_series, temp, u10]).T,
            index=date_range,
            columns=[["roughness_length", "temperature", "wind_speed"], [0, 2, 10]],
        )
        for cls_name, capacity_kw in row.items():
            if capacity_kw <= 0:
                continue
            wt = TURBINE_MODELS[cls_name]
            mc = ModelChain(wt).run_model(ww)
            cls_power_mw = mc.power_output / wt.nominal_power * (capacity_kw * 1e3) / 1e6
            sim_w_tso[tso] += cls_power_mw

    return sim_w_tso


def evaluate_sim(sim_w_tso, act_w_tso, date_range, tag: dict) -> list[dict]:
    rows = []
    sim_nat = pd.Series(0.0, index=date_range)
    act_nat = pd.Series(0.0, index=date_range)
    for t in ZONES:
        sim_nat = sim_nat + sim_w_tso[t].reindex(date_range, fill_value=0.0)
        act_nat = act_nat + act_w_tso[t].reindex(date_range, fill_value=0.0)
    m = metrics_day_night(sim_nat, act_nat)
    rows.append({**tag, "Scale": "Germany national", **m})
    for tso in ZONES:
        m = metrics_day_night(sim_w_tso[tso], act_w_tso[tso].reindex(date_range, fill_value=0.0))
        rows.append({**tag, "Scale": tso, **m})
    return rows


def plot_z0_daynight(df_nat: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df_nat["z0"], df_nat["Day_ratio"], "o-", label="Day ratio %", color="#1f77b4")
    ax.plot(df_nat["z0"], df_nat["Night_ratio"], "s-", label="Night ratio %", color="#d62728")
    ax.plot(df_nat["z0"], df_nat["Full_ratio"], "^-", label="Full-year ratio %", color="#333333", alpha=0.7)
    ax.axhline(100, color="gray", ls="--", lw=1)
    ax.axvline(0.2, color="gray", ls=":", lw=1, label="Default z0=0.2")
    ax.set_xlabel("Roughness length z0 (m)")
    ax.set_ylabel("Sim / ENTSO-E yield (%)")
    ax.set_title("OEDS wind: day vs night yield vs z0 (Germany 2023)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(diag, weather_summary, z0_df, dual_df, alpha_df, path: str):
    nat_oeds = diag[(diag.Scale == "Germany national") & (diag.Model == "OEDS")].iloc[0]
    nat_atl = diag[(diag.Scale == "Germany national") & (diag.Model == "Atlite")].iloc[0]
    z0_nat = z0_df[z0_df.Scale == "Germany national"].sort_values("z0")
    best_night = z0_nat.iloc[(z0_nat["Night_ratio"] - 100).abs().argmin()]
    best_bal = z0_nat.iloc[(z0_nat["night_day_ratio_gap_pp"]).abs().argmin()]

    dual_nat = dual_df[dual_df.Scale == "Germany national"].sort_values("night_z0")
    alpha_nat = alpha_df[alpha_df.Scale == "Germany national"].sort_values("alpha")

    lines = [
        "# Night wind collapse — sensitivity summary",
        "",
        "## Baseline (default OEDS z0=0.2 vs Atlite)",
        f"- OEDS national: day **{nat_oeds.Day_ratio:.1f}%**, night **{nat_oeds.Night_ratio:.1f}%** (gap {nat_oeds.night_day_ratio_gap_pp:.1f} pp)",
        f"- Atlite national: day **{nat_atl.Day_ratio:.1f}%**, night **{nat_atl.Night_ratio:.1f}%** (gap {nat_atl.night_day_ratio_gap_pp:.1f} pp)",
        "",
        "## Weather (capacity-weighted ~80% of MaStR wind)",
    ]
    for k, v in weather_summary.items():
        if isinstance(v, float):
            lines.append(f"- `{k}`: {v:.3f}")
        else:
            lines.append(f"- `{k}`: {v}")
    lines += [
        "",
        "## Interpretation keys",
        "- If ECMWF 10 m night/day ≈ ERA5 100 m night/day but hub-log night/day is flatter, "
        "neutral log understates nocturnal shear.",
        "- If ERA5 100 m ≫ ECMWF log-100 m especially at night, the collapse is largely "
        "weather-height / product, not power-curve.",
        "- If raising z0 lifts night ratio toward 100% faster than day, night needs stronger shear.",
        "",
        "## z0 sweep (uniform, all hours)",
        z0_nat[["z0", "Day_ratio", "Night_ratio", "Full_ratio", "night_day_ratio_gap_pp"]].to_string(index=False),
        "",
        f"- Closest night ratio to 100%: z0={best_night.z0} → night {best_night.Night_ratio:.1f}%, day {best_night.Day_ratio:.1f}%",
        f"- Smallest |night−day| gap: z0={best_bal.z0} → gap {best_bal.night_day_ratio_gap_pp:.1f} pp",
        "",
        "## Dual-z0 (day z0=0.2, night varied)",
        dual_nat[["night_z0", "Day_ratio", "Night_ratio", "Full_ratio", "night_day_ratio_gap_pp"]].to_string(index=False),
        "",
        "## Hellman alpha (hub winds at hub height)",
        alpha_nat[["alpha", "Day_ratio", "Night_ratio", "Full_ratio", "night_day_ratio_gap_pp"]].to_string(index=False),
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Wrote %s", path)


def load_inputs(date_range):
    engine = resolve_engine()
    logger.info("Mode: %s", "OFFLINE" if offline_mode() or engine is None else "DB-allowed")
    wind_df = query_mastr_wind(engine, "2023-01-01", "2023-12-31")
    plz_nuts = load_plz_nuts(engine)
    wind_df["nuts3"] = wind_df["plzCode"].map(plz_nuts["nuts3"])
    wind_df = wind_df.dropna(subset=["nuts3"])
    wind_df["tso"] = wind_df.apply(map_row_to_tso, axis=1)
    wind_df = wind_df[wind_df["tso"] != "UNKNOWN"]
    wind_df = classify_wind_turbines(wind_df)
    wind_groups = wind_df.groupby(["nuts3", "class"])["maxPower"].sum().unstack(fill_value=0.0)
    nuts3_to_tso = {nuts3: grp["tso"].iloc[0] for nuts3, grp in wind_df.groupby("nuts3")}
    nuts3_coords = plz_nuts.groupby("nuts3")[["latitude", "longitude"]].mean()

    weather_raw = query_ecmwf_weather_nuts3(
        engine, "2023-01-01", "2023-12-31", nuts_prefix="DE", date_range=date_range
    )
    weather_dict = {nuts: grp.set_index("time") for nuts, grp in weather_raw.groupby("nuts_id")}
    entsoe_wind, _ = query_entsoe_generation(engine, "2023-01-01", "2023-12-31", ZONES, date_range)
    act_w_tso = {tso: entsoe_wind[tso] for tso in ZONES}
    return wind_groups, weather_dict, nuts3_to_tso, nuts3_coords, act_w_tso


def main():
    ensure_results_dir()
    date_range = pd.date_range("2023-01-01 00:00:00", "2023-12-31 23:00:00", freq="h")

    logger.info("=== 1) Timeseries diagnostics ===")
    diag = diagnostics_from_timeseries()
    diag.to_csv(result_path("night_wind_diagnostics.csv"), index=False)
    print(diag[diag.Scale.isin(["Germany national", "Germany national (mean MW)"])][
        ["Scale", "Model", "Day_ratio", "Night_ratio", "Full_ratio", "night_day_ratio_gap_pp"]
    ].to_string(index=False))

    logger.info("=== 2) Load registry/weather ===")
    wind_groups, weather_dict, nuts3_to_tso, nuts3_coords, act_w_tso = load_inputs(date_range)

    logger.info("=== 3) Weather day/night compare ===")
    weather_df, weather_summary = weather_day_night_compare(
        wind_groups, weather_dict, nuts3_coords, date_range
    )
    weather_df.to_csv(result_path("night_wind_weather_compare.csv"), index=False)
    pd.Series(weather_summary).to_csv(result_path("night_wind_weather_summary.csv"))
    print("Weather summary:", weather_summary)

    logger.info("=== 4) Uniform z0 sweep with day/night metrics ===")
    z0_rows = []
    for z0 in Z0_SWEEP:
        t0 = time.time()
        sim = run_wind_sim(wind_groups, weather_dict, nuts3_to_tso, date_range, z0=z0)
        z0_rows.extend(evaluate_sim(sim, act_w_tso, date_range, {"z0": z0, "mode": "uniform_z0"}))
        logger.info("z0=%.2f done in %.1fs", z0, time.time() - t0)
    z0_df = pd.DataFrame(z0_rows)
    z0_df.to_csv(result_path("night_wind_z0_daynight.csv"), index=False)
    z0_nat = z0_df[z0_df.Scale == "Germany national"]
    plot_z0_daynight(z0_nat, result_path("night_wind_z0_daynight.png"))

    logger.info("=== 5) Dual-z0: day=0.2, night varied ===")
    dual_rows = []
    for night_z0 in NIGHT_Z0_SWEEP:
        z0_by_hour = np.where(day_night_mask(date_range, "Night"), night_z0, 0.2)
        t0 = time.time()
        sim = run_wind_sim(
            wind_groups, weather_dict, nuts3_to_tso, date_range, z0_by_hour=z0_by_hour
        )
        dual_rows.extend(
            evaluate_sim(sim, act_w_tso, date_range, {"night_z0": night_z0, "day_z0": 0.2, "mode": "dual_z0"})
        )
        logger.info("night_z0=%.2f done in %.1fs", night_z0, time.time() - t0)
    dual_df = pd.DataFrame(dual_rows)
    dual_df.to_csv(result_path("night_wind_dual_z0.csv"), index=False)

    logger.info("=== 6) Hellman alpha sweep ===")
    alpha_rows = []
    for alpha in ALPHA_SWEEP:
        t0 = time.time()
        sim = run_wind_sim(wind_groups, weather_dict, nuts3_to_tso, date_range, alpha=alpha)
        alpha_rows.extend(evaluate_sim(sim, act_w_tso, date_range, {"alpha": alpha, "mode": "hellman"}))
        logger.info("alpha=%.2f done in %.1fs", alpha, time.time() - t0)
    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df.to_csv(result_path("night_wind_alpha_daynight.csv"), index=False)

    write_summary(
        diag,
        weather_summary,
        z0_df,
        dual_df,
        alpha_df,
        result_path("night_wind_sensitivity_summary.md"),
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
