"""
Matched-ERA5 ablation: same Atlite cutout weather → Windpowerlib + PVLib vs Atlite.

Windpowerlib uses MaStR plant mapping (diameter/power → oedb catalogue + MaStR hubs).
Atlite wind uses a single Vestas V112 @ 80 m (PyPSA-Eur default) for all capacity.

Removes the ECMWF vs ERA5 confound so remaining deltas are conversion-library /
capacity-layout / orientation choices.

Outputs (structured_research/results/):
  matched_era5_national.csv
  matched_era5_tso.csv
  matched_era5_library_delta.csv   — WPL/PVLib vs Atlite (same weather)
  matched_era5_vs_entsoe.csv       — all three stacks vs ENTSO-E
  matched_era5_daynight_wind.csv
  matched_era5_timeseries.csv      — national hourly
  matched_era5_tso_timeseries.parquet — per-TSO hourly (paper figures)
  matched_era5_summary.md
  matched_era5_library_delta.png
"""

from __future__ import annotations

from pathlib import Path

import sys
import logging
import time

import atlite
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.irradiance import erbs
from pvlib.temperature import sapm_cell

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import (
    ATLITE_WIND_TURBINE,
    SAPM_A,
    SAPM_B,
    SAPM_DT,
    TEMP_COEFF,
    ZONES,
    capacity_factor,
)
from utils import (
    resolve_engine,
    offline_mode,
    load_plz_nuts,
    map_row_to_tso,
    map_mastr_to_wpl_turbine,
    classify_wind_turbines,
    parse_solar_orientation,
    query_mastr_wind,
    query_mastr_solar,
    query_entsoe_generation,
    calculate_metrics,
    ensure_results_dir,
    cutout_path,
    result_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("matched_era5")

DAY_HOURS = list(range(6, 18))
HUB_BIN_M = 5.0  # round hubs for aggregation only (runtime); still MaStR-based

LABEL_ATLITE = "Atlite (ERA5 cutout)"
LABEL_WPL = "Windpowerlib MaStR fleet (ERA5 cutout)"
LABEL_PVLIB_MASTR = "PVLib MaStR orient (ERA5 cutout)"
LABEL_PVLIB_UNIFORM = "PVLib 30/180 (ERA5 cutout)"


def day_night_metrics(sim: pd.Series, act: pd.Series) -> dict:
    out = {}
    for name, hours in [("Day", DAY_HOURS), ("Night", list(range(18, 24)) + list(range(0, 6))), ("Full", None)]:
        if hours is None:
            m = calculate_metrics(sim, act)
        else:
            mask = sim.index.hour.isin(hours)
            m = calculate_metrics(sim[mask], act[mask])
        out[f"{name}_ratio"] = m["ratio"]
        out[f"{name}_corr"] = m["corr"]
        out[f"{name}_sim_GWh"] = m["sim_sum"] / 1e3
        out[f"{name}_act_GWh"] = m["act_sum"] / 1e3
    return out


def nearest_cell(ds, lat: float, lon: float):
    yi = int(np.abs(ds.y.values - lat).argmin())
    xi = int(np.abs(ds.x.values - lon).argmin())
    return yi, xi


def extract_cell_weather(ds, yi: int, xi: int, date_range: pd.DatetimeIndex) -> pd.DataFrame:
    def series(name):
        s = ds[name].isel(y=yi, x=xi).to_pandas()
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        return s.reindex(date_range, method="nearest")

    wnd100 = series("wnd100m").astype(float)
    alpha = series("wnd_shear_exp").astype(float).clip(0.05, 0.55)
    temp = series("temperature").astype(float)
    direct = series("influx_direct").astype(float).clip(lower=0.0)
    diffuse = series("influx_diffuse").astype(float).clip(lower=0.0)
    # Atlite influx is W/m²; GHI = direct + diffuse on horizontal
    ghi = (direct + diffuse).clip(lower=0.0)
    # Approximate 10 m wind from 100 m + shear for SAPM
    u10 = wnd100 / (10.0 ** alpha)
    return pd.DataFrame({
        "wnd100m": wnd100,
        "alpha": alpha,
        "temp_k": temp,
        "ghi": ghi,
        "dni_proxy": direct,   # not true DNI; used only if Erbs preferred
        "dhi": diffuse,
        "u10": u10,
    }, index=date_range)


def run_wpl_from_cutout(wind_groups, weather_by_nuts, nuts3_to_tso, date_range):
    """
    Windpowerlib on MaStR-mapped catalogue types + MaStR hub heights.

    wind_groups columns: nuts3, turbine_type, hub_m, capacity_kw
    """
    sim_tso = {t: pd.Series(0.0, index=date_range) for t in ZONES}
    for nuts, g in tqdm(wind_groups.groupby("nuts3"), desc="WPL MaStR fleet"):
        if nuts not in weather_by_nuts or nuts not in nuts3_to_tso:
            continue
        tso = nuts3_to_tso[nuts]
        w = weather_by_nuts[nuts]
        u100 = w["wnd100m"].to_numpy(dtype=float)
        alpha = w["alpha"].to_numpy(dtype=float)
        for _, r in g.iterrows():
            hub = float(r["hub_m"])
            u_hub = u100 * (hub / 100.0) ** alpha
            cf = capacity_factor(str(r["turbine_type"]), u_hub)
            sim_tso[tso] += cf * (float(r["capacity_kw"]) * 1e3) / 1e6
    return sim_tso


def run_pvlib_from_cutout(
    solar_groups,
    nuts3_coords,
    weather_by_nuts,
    nuts3_to_tso,
    date_range,
    *,
    force_uniform_30_180: bool = False,
):
    sim_tso = {t: pd.Series(0.0, index=date_range) for t in ZONES}
    groups = solar_groups
    if force_uniform_30_180:
        # Collapse all orientations to Atlite-like 30°/180°
        g = solar_groups.copy()
        g = g.groupby("nuts3", as_index=False)["maxPower"].sum()
        g["azimuth"] = 180.0
        g["tilt"] = 30.0
        groups = g

    for nuts, grp in tqdm(groups.groupby("nuts3"), desc=f"PVLib ERA5 ({'uniform' if force_uniform_30_180 else 'MaStR'})"):
        if nuts not in weather_by_nuts or nuts not in nuts3_to_tso:
            continue
        tso = nuts3_to_tso[nuts]
        w = weather_by_nuts[nuts]
        lat = float(nuts3_coords.loc[nuts, "latitude"]) if nuts in nuts3_coords.index else 50.0
        lon = float(nuts3_coords.loc[nuts, "longitude"]) if nuts in nuts3_coords.index else 10.0
        location = Location(lat, lon, tz="UTC")
        sun_pos = location.get_solarposition(date_range)
        ghi = w["ghi"]
        # Prefer Erbs on GHI for comparability with OEDS path; DHI from cutout as cross-check unused
        erbs_out = erbs(ghi, sun_pos["zenith"], date_range)
        temp_c = w["temp_k"] - 273.15
        u10 = w["u10"]

        for _, s_row in grp.iterrows():
            azimuth = float(s_row["azimuth"])
            tilt = float(s_row["tilt"])
            capacity_kw = float(s_row["maxPower"])
            system = PVSystem(
                surface_tilt=tilt,
                surface_azimuth=azimuth,
                module_parameters={"pdc0": capacity_kw},
            )
            irr = system.get_irradiance(
                solar_zenith=sun_pos["zenith"],
                solar_azimuth=sun_pos["azimuth"],
                dni=erbs_out["dni"],
                ghi=ghi,
                dhi=erbs_out["dhi"],
            )
            poa = irr["poa_global"]
            cell_temp = sapm_cell(poa, temp_c, u10, SAPM_A, SAPM_B, SAPM_DT)
            temp_factor = 1 + TEMP_COEFF * (cell_temp - 25.0)
            sim_tso[tso] += (poa * temp_factor * capacity_kw) / 1e6
    return sim_tso


def run_atlite(cutout, wind_df, solar_df, date_range):
    atlite_wind_tso = {}
    atlite_solar_tso = {}
    for tso in ZONES:
        logger.info("Atlite wind %s", tso)
        t_wind = wind_df[wind_df["tso"] == tso].copy()
        t_wind["x"] = t_wind["lon"]
        t_wind["y"] = t_wind["lat"]
        t_wind["capacity_mw"] = t_wind["maxPower"] / 1e3
        layout_w = cutout.layout_from_capacity_list(t_wind, col="capacity_mw")
        w_ds = cutout.wind(
            turbine=ATLITE_WIND_TURBINE,
            layout=layout_w,
            add_cutout_windspeed=True,
        )
        atlite_wind_tso[tso] = pd.Series(w_ds.to_series().values, index=date_range)

        logger.info("Atlite solar %s", tso)
        t_solar = solar_df[solar_df["tso"] == tso].copy()
        t_solar["x"] = t_solar["lon"]
        t_solar["y"] = t_solar["lat"]
        t_solar["capacity_mw"] = t_solar["maxPower"] / 1e3
        layout_s = cutout.layout_from_capacity_list(t_solar, col="capacity_mw")
        s_ds = cutout.pv(
            panel="CSi",
            orientation={"slope": 30.0, "azimuth": 180.0},
            layout=layout_s,
        )
        atlite_solar_tso[tso] = pd.Series(s_ds.to_series().values, index=date_range)

    return atlite_wind_tso, atlite_solar_tso


def compare_pair(sim_a, sim_b, label_a, label_b, scale: str, tech: str) -> dict:
    m = calculate_metrics(sim_a, sim_b)  # ratio = A/B
    return {
        "Scale": scale,
        "Technology": tech,
        "Numerator": label_a,
        "Denominator": label_b,
        "Ratio_%": round(m["ratio"], 2),
        "Correlation": round(m["corr"], 4),
        "MAE_MW": round(m["mae"], 1),
        "RMSE_MW": round(m["rmse"], 1),
        "Num_GWh": round(m["sim_sum"] / 1e3, 1),
        "Den_GWh": round(m["act_sum"] / 1e3, 1),
    }


def vs_entsoe(sim, act, model: str, scale: str, tech: str) -> dict:
    m = calculate_metrics(sim, act)
    return {
        "Scale": scale,
        "Technology": tech,
        "Model": model,
        "Ratio_%": round(m["ratio"], 2),
        "Correlation": round(m["corr"], 4),
        "MAE_MW": round(m["mae"], 1),
        "RMSE_MW": round(m["rmse"], 1),
        "Sim_GWh": round(m["sim_sum"] / 1e3, 1),
        "Actual_GWh": round(m["act_sum"] / 1e3, 1),
    }


def plot_delta(lib_df: pd.DataFrame, path: str):
    sub = lib_df[lib_df.Scale == "Germany national"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, tech in zip(axes, ["Wind", "Solar"]):
        s = sub[sub.Technology == tech]
        labels = [f"{r.Numerator.split()[0]}\n/ Atlite" for _, r in s.iterrows()]
        ax.bar(range(len(s)), s["Ratio_%"], color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(s)])
        ax.axhline(100, color="gray", ls="--")
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Energy ratio vs Atlite (%)")
        ax.set_title(f"{tech}: library delta on same ERA5 cutout")
        ax.grid(True, axis="y", alpha=0.3)
        for i, r in enumerate(s.itertuples()):
            ax.text(i, r._5 + 1, f"r={r.Correlation:.3f}", ha="center", fontsize=8)
    fig.suptitle("Matched ERA5: PVLib/Windpowerlib vs Atlite (weather held fixed)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(lib_df, entsoe_df, daynight_df, path: str):
    nat_lib = lib_df[lib_df.Scale == "Germany national"]
    nat_e = entsoe_df[entsoe_df.Scale == "Germany national"]
    lines = [
        "# Matched ERA5 cutout: Atlite vs Windpowerlib / PVLib",
        "",
        "Weather held fixed: `germany_2023.nc` (ERA5 via Atlite cutout).",
        "Remaining deltas are conversion library, turbine/panel model, and orientation/layout choices.",
        "",
        "## National library deltas (numerator / Atlite)",
        nat_lib.to_string(index=False),
        "",
        "## National vs ENTSO-E (same cutout weather for all sims)",
        nat_e.to_string(index=False),
        "",
        "## Wind day/night (national)",
        daynight_df.to_string(index=False),
        "",
        "## Interpretation",
        "- **Wind:** Windpowerlib uses MaStR diameter/power → oedb catalogue types + MaStR hub",
        "  heights (5 m bins for aggregation only). Atlite uses a single V112@80 m "
        "(PyPSA-Eur default) for all capacity.",
        "  The large energy gap vs Atlite is therefore mainly **fleet representation**, not weather.",
        "  Night collapse from ECMWF 10 m log disappears when both use ERA5 hub wind.",
        "- **Solar:** PVLib 30°/180° is the fair library match to Atlite orientation;",
        "  PVLib MaStR orientations show layout/diversity effect on top of library.",
        "- vs ENTSO-E still includes feed-in≠generation (solar) and omitted wakes/availability (wind).",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Wrote %s", path)


def main():
    ensure_results_dir()
    date_range = pd.date_range("2023-01-01", "2023-12-31 23:00:00", freq="h")
    engine = resolve_engine()
    logger.info("Mode: %s", "OFFLINE" if offline_mode() or engine is None else "DB-allowed")

    cutout_file = cutout_path("germany_2023.nc")
    if not cutout_file.exists():
        raise FileNotFoundError(f"Need {cutout_file}")

    logger.info("Loading cutout %s", cutout_file)
    cutout = atlite.Cutout(cutout_file)
    ds = cutout.data

    logger.info("Loading MaStR + ENTSO-E...")
    wind_df = query_mastr_wind(engine, "2023-01-01", "2023-12-31")
    solar_df = query_mastr_solar(engine, "2023-01-01", "2023-12-31")
    plz = load_plz_nuts(engine)
    wind_df["nuts3"] = wind_df["plzCode"].map(plz["nuts3"])
    solar_df["nuts3"] = solar_df["plzCode"].map(plz["nuts3"])
    wind_df = wind_df.dropna(subset=["nuts3"])
    solar_df = solar_df.dropna(subset=["nuts3"])
    wind_df["tso"] = wind_df.apply(map_row_to_tso, axis=1)
    solar_df["tso"] = solar_df.apply(map_row_to_tso, axis=1)
    wind_df = wind_df[wind_df["tso"] != "UNKNOWN"]
    solar_df = solar_df[solar_df["tso"] != "UNKNOWN"]
    # MaStR → WPL catalogue by diameter/power + MaStR hub heights (no 4-class SP aggregation)
    if "hub_m" not in wind_df.columns:
        raise RuntimeError(
            "mastr_wind.parquet lacks hub_m — refresh with extract_research_data.py (online)"
        )
    wind_df = map_mastr_to_wpl_turbine(classify_wind_turbines(wind_df))
    wind_df["hub_bin"] = (
        np.round(pd.to_numeric(wind_df["hub_m_used"], errors="coerce") / HUB_BIN_M) * HUB_BIN_M
    ).clip(30.0, 200.0)
    solar_df = parse_solar_orientation(solar_df)

    wind_groups = (
        wind_df.groupby(["nuts3", "wpl_type", "hub_bin"], as_index=False)["maxPower"]
        .sum()
        .rename(columns={"wpl_type": "turbine_type", "hub_bin": "hub_m", "maxPower": "capacity_kw"})
    )
    logger.info(
        "WPL MaStR fleet groups: %d (types=%d, GW=%.2f)",
        len(wind_groups),
        wind_groups["turbine_type"].nunique(),
        wind_groups["capacity_kw"].sum() / 1e6,
    )
    solar_groups = solar_df.groupby(["nuts3", "azimuth", "tilt"])["maxPower"].sum().reset_index()
    nuts3_coords = plz.groupby("nuts3")[["latitude", "longitude"]].mean()
    nuts3_to_tso = {}
    for nuts3, grp in wind_df.groupby("nuts3"):
        nuts3_to_tso[nuts3] = grp["tso"].iloc[0]
    for nuts3, grp in solar_df.groupby("nuts3"):
        nuts3_to_tso.setdefault(nuts3, grp["tso"].iloc[0])

    entsoe_w, entsoe_s = query_entsoe_generation(
        engine, "2023-01-01", "2023-12-31", ZONES, date_range
    )
    act_w_tso = {z: entsoe_w[z] for z in ZONES}
    act_s_tso = {z: entsoe_s[z] for z in ZONES}
    act_w_nat = sum((act_w_tso[z].reindex(date_range, fill_value=0.0) for z in ZONES),
                    start=pd.Series(0.0, index=date_range))
    act_s_nat = sum((act_s_tso[z].reindex(date_range, fill_value=0.0) for z in ZONES),
                    start=pd.Series(0.0, index=date_range))

    # Pre-extract cutout weather at NUTS3 centroids
    logger.info("Extracting cutout weather at NUTS3 centroids...")
    nuts_all = sorted(set(wind_groups["nuts3"]) | set(solar_groups["nuts3"]))
    weather_by_nuts = {}
    for nuts in tqdm(nuts_all, desc="Cutout cells"):
        if nuts not in nuts3_coords.index:
            continue
        lat = float(nuts3_coords.loc[nuts, "latitude"])
        lon = float(nuts3_coords.loc[nuts, "longitude"])
        yi, xi = nearest_cell(ds, lat, lon)
        weather_by_nuts[nuts] = extract_cell_weather(ds, yi, xi, date_range)

    t0 = time.time()
    logger.info("Running Windpowerlib MaStR fleet on ERA5 cutout...")
    wpl_tso = run_wpl_from_cutout(wind_groups, weather_by_nuts, nuts3_to_tso, date_range)
    logger.info("WPL done in %.1fs", time.time() - t0)

    t0 = time.time()
    logger.info("Running PVLib MaStR orientations on ERA5 cutout...")
    pv_mastr_tso = run_pvlib_from_cutout(
        solar_groups, nuts3_coords, weather_by_nuts, nuts3_to_tso, date_range, force_uniform_30_180=False
    )
    logger.info("PVLib MaStR done in %.1fs", time.time() - t0)

    t0 = time.time()
    logger.info("Running PVLib uniform 30/180 on ERA5 cutout...")
    pv_uni_tso = run_pvlib_from_cutout(
        solar_groups, nuts3_coords, weather_by_nuts, nuts3_to_tso, date_range, force_uniform_30_180=True
    )
    logger.info("PVLib uniform done in %.1fs", time.time() - t0)

    t0 = time.time()
    logger.info("Running Atlite on same cutout...")
    atl_w_tso, atl_s_tso = run_atlite(cutout, wind_df, solar_df, date_range)
    logger.info("Atlite done in %.1fs", time.time() - t0)

    def nat(d):
        return sum((d[z].reindex(date_range, fill_value=0.0) for z in ZONES),
                   start=pd.Series(0.0, index=date_range))

    wpl_nat = nat(wpl_tso)
    pv_mastr_nat = nat(pv_mastr_tso)
    pv_uni_nat = nat(pv_uni_tso)
    atl_w_nat = nat(atl_w_tso)
    atl_s_nat = nat(atl_s_tso)

    # Library deltas (same weather)
    lib_rows = []
    lib_rows.append(compare_pair(wpl_nat, atl_w_nat, LABEL_WPL, LABEL_ATLITE, "Germany national", "Wind"))
    lib_rows.append(compare_pair(pv_uni_nat, atl_s_nat, LABEL_PVLIB_UNIFORM, LABEL_ATLITE, "Germany national", "Solar"))
    lib_rows.append(compare_pair(pv_mastr_nat, atl_s_nat, LABEL_PVLIB_MASTR, LABEL_ATLITE, "Germany national", "Solar"))
    for z in ZONES:
        lib_rows.append(compare_pair(wpl_tso[z], atl_w_tso[z], LABEL_WPL, LABEL_ATLITE, z, "Wind"))
        lib_rows.append(compare_pair(pv_uni_tso[z], atl_s_tso[z], LABEL_PVLIB_UNIFORM, LABEL_ATLITE, z, "Solar"))
        lib_rows.append(compare_pair(pv_mastr_tso[z], atl_s_tso[z], LABEL_PVLIB_MASTR, LABEL_ATLITE, z, "Solar"))
    lib_df = pd.DataFrame(lib_rows)
    lib_df.to_csv(result_path("matched_era5_library_delta.csv"), index=False)

    # vs ENTSO-E
    e_rows = []
    for scale, wsim, ssim_u, ssim_m, aw, as_ in [
        ("Germany national", wpl_nat, pv_uni_nat, pv_mastr_nat, atl_w_nat, atl_s_nat),
    ]:
        e_rows.append(vs_entsoe(wsim, act_w_nat, LABEL_WPL, scale, "Wind"))
        e_rows.append(vs_entsoe(aw, act_w_nat, LABEL_ATLITE, scale, "Wind"))
        e_rows.append(vs_entsoe(ssim_u, act_s_nat, LABEL_PVLIB_UNIFORM, scale, "Solar"))
        e_rows.append(vs_entsoe(ssim_m, act_s_nat, LABEL_PVLIB_MASTR, scale, "Solar"))
        e_rows.append(vs_entsoe(as_, act_s_nat, LABEL_ATLITE, scale, "Solar"))
    for z in ZONES:
        e_rows.append(vs_entsoe(wpl_tso[z], act_w_tso[z], LABEL_WPL, z, "Wind"))
        e_rows.append(vs_entsoe(atl_w_tso[z], act_w_tso[z], LABEL_ATLITE, z, "Wind"))
        e_rows.append(vs_entsoe(pv_uni_tso[z], act_s_tso[z], LABEL_PVLIB_UNIFORM, z, "Solar"))
        e_rows.append(vs_entsoe(pv_mastr_tso[z], act_s_tso[z], LABEL_PVLIB_MASTR, z, "Solar"))
        e_rows.append(vs_entsoe(atl_s_tso[z], act_s_tso[z], LABEL_ATLITE, z, "Solar"))
    entsoe_df = pd.DataFrame(e_rows)
    entsoe_df.to_csv(result_path("matched_era5_vs_entsoe.csv"), index=False)

    # Compact national / TSO tables
    nat_rows = entsoe_df[entsoe_df.Scale == "Germany national"]
    nat_rows.to_csv(result_path("matched_era5_national.csv"), index=False)
    tso_rows = entsoe_df[entsoe_df.Scale != "Germany national"]
    tso_rows.to_csv(result_path("matched_era5_tso.csv"), index=False)

    # Day/night wind
    dn_rows = []
    for label, series in [(LABEL_WPL, wpl_nat), (LABEL_ATLITE, atl_w_nat)]:
        dn_rows.append({"Model": label, **day_night_metrics(series, act_w_nat)})
    daynight_df = pd.DataFrame(dn_rows)
    daynight_df.to_csv(result_path("matched_era5_daynight_wind.csv"), index=False)

    ts = pd.DataFrame({
        "entsoe_wind": act_w_nat,
        "entsoe_solar": act_s_nat,
        "atlite_wind": atl_w_nat,
        "atlite_solar": atl_s_nat,
        "wpl_wind": wpl_nat,
        "pvlib_uniform_solar": pv_uni_nat,
        "pvlib_mastr_solar": pv_mastr_nat,
    })
    ts.to_csv(result_path("matched_era5_timeseries.csv"))

    # Per-TSO hourly (paper figures; replaces annual_seasonal_tso_timeseries dependency)
    tso_ts = {}
    for z in ZONES:
        tso_ts[f"entsoe_wind_{z}"] = act_w_tso[z].reindex(date_range, fill_value=0.0)
        tso_ts[f"entsoe_solar_{z}"] = act_s_tso[z].reindex(date_range, fill_value=0.0)
        tso_ts[f"wpl_wind_{z}"] = wpl_tso[z].reindex(date_range, fill_value=0.0)
        tso_ts[f"atlite_wind_{z}"] = atl_w_tso[z].reindex(date_range, fill_value=0.0)
        tso_ts[f"pvlib_mastr_solar_{z}"] = pv_mastr_tso[z].reindex(date_range, fill_value=0.0)
        tso_ts[f"atlite_solar_{z}"] = atl_s_tso[z].reindex(date_range, fill_value=0.0)
    tso_hourly = pd.DataFrame(tso_ts, index=date_range)
    tso_path = result_path("matched_era5_tso_timeseries.parquet")
    tso_hourly.to_parquet(tso_path)
    logger.info("Wrote %s", tso_path)

    plot_delta(lib_df, result_path("matched_era5_library_delta.png"))
    write_summary(lib_df, entsoe_df, daynight_df, result_path("matched_era5_summary.md"))

    print("\n=== LIBRARY DELTA (national, same ERA5) ===")
    print(lib_df[lib_df.Scale == "Germany national"].to_string(index=False))
    print("\n=== VS ENTSO-E (national) ===")
    print(nat_rows.to_string(index=False))
    print("\n=== WIND DAY/NIGHT ===")
    print(daynight_df.to_string(index=False))
    logger.info("Done.")


if __name__ == "__main__":
    main()
