"""
Why Atlite overestimates Jülich PV vs measured AC and vs PVLib.

Compares three generation series + weather, and runs controls:
  A) Existing: Actual / PVLib(ECMWF) / Atlite(ERA5)
  B) PVLib on ERA5 GHI from same cutout cell (isolates library)
  C) Atlite CSi with inverter_efficiency forced to 1.0
  D) ERA5 vs ECMWF GHI energy and clear/cloudy ratios

Outputs → structured_research/results/:
  juelich_atlite_overest.md
  juelich_atlite_overest_hourly.csv (subset diagnostics)
  juelich_atlite_overest_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import sys

import atlite
import numpy as np
import pandas as pd
from atlite.resource import get_solarpanelconfig
from pvlib.irradiance import erbs
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.temperature import sapm_cell

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    ensure_results_dir,
    result_path,
    cutout_path,
    resolve_engine,
    query_ecmwf_weather,
    calculate_metrics,
)

LAT, LON = 50.92, 6.36
DB_LAT, DB_LON = 50.845454545454544, 6.454545454545454
CAP_KW = 216.960
TILT, AZIM = 25.0, 165.0
SAPM_A, SAPM_B, SAPM_DT = -3.47, -0.0594, 3.0
TEMP_COEFF = -0.004


def ratio_sum(sim, act):
    return 100.0 * sim.sum() / act.sum()


def main():
    ensure_results_dir()
    date_range = pd.date_range("2023-01-01", "2023-12-31 23:00:00", freq="h")

    gen = pd.read_csv(
        result_path("juelich_eview_comparison_complete.csv"),
        index_col=0,
        parse_dates=True,
    ).reindex(date_range).fillna(0.0)
    act = gen["actual_generation"].astype(float)
    pv_ecmwf = gen["oeds_temp_corrected"].astype(float)
    atl = gen["atlite_sim"].astype(float)

    # --- ERA5 from cutout ---
    cut = atlite.Cutout(cutout_path("juelich_2023.nc"))
    ds = cut.data
    yi = int(np.abs(ds.y.values - LAT).argmin())
    xi = int(np.abs(ds.x.values - LON).argmin())
    cell_y, cell_x = float(ds.y.values[yi]), float(ds.x.values[xi])

    def cell_series(name):
        s = ds[name].isel(y=yi, x=xi).to_pandas()
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        return s.reindex(date_range, method="nearest").astype(float)

    era5_direct = cell_series("influx_direct").clip(lower=0)
    era5_diffuse = cell_series("influx_diffuse").clip(lower=0)
    era5_ghi = (era5_direct + era5_diffuse).clip(lower=0)
    era5_temp_k = cell_series("temperature")
    era5_temp_c = era5_temp_k - 273.15

    # Atlite POA (total tilted) time series at cell — use irradiation convert
    poa_da = cut.irradiation(
        orientation={"slope": TILT, "azimuth": AZIM},
        irradiation="total",
        aggregate_time=None,
    )
    era5_poa = poa_da.isel(y=yi, x=xi).to_pandas()
    if getattr(era5_poa.index, "tz", None) is not None:
        era5_poa.index = era5_poa.index.tz_localize(None)
    era5_poa = era5_poa.reindex(date_range, method="nearest").astype(float).clip(lower=0)

    # CF and inverter-off control
    cf = cut.pv(
        panel="CSi",
        orientation={"slope": TILT, "azimuth": AZIM},
        aggregate_time=None,
    ).isel(y=yi, x=xi).to_pandas()
    if getattr(cf.index, "tz", None) is not None:
        cf.index = cf.index.tz_localize(None)
    cf = cf.reindex(date_range, method="nearest").astype(float)
    atl_from_cf = cf * CAP_KW  # kW

    pc = get_solarpanelconfig("CSi")
    pc_noinv = dict(pc)
    pc_noinv["inverter_efficiency"] = 1.0
    cf_noinv = cut.pv(
        panel=pc_noinv,
        orientation={"slope": TILT, "azimuth": AZIM},
        aggregate_time=None,
    ).isel(y=yi, x=xi).to_pandas()
    if getattr(cf_noinv.index, "tz", None) is not None:
        cf_noinv.index = cf_noinv.index.tz_localize(None)
    cf_noinv = cf_noinv.reindex(date_range, method="nearest").astype(float)
    atl_noinv = cf_noinv * CAP_KW

    # --- ECMWF weather (same as validate_juelich) ---
    engine = resolve_engine()
    weather = query_ecmwf_weather(
        engine,
        "2023-01-01 00:00:00",
        "2023-12-31 23:00:00",
        lat=DB_LAT,
        lon=DB_LON,
        date_range=date_range,
    ).reindex(date_range, method="nearest")
    ecmwf_ghi = (weather["ghi"] / 3600.0).astype(float)  # W/m2
    ecmwf_temp_c = weather["temp_air"].astype(float) - 273.15
    ecmwf_ws = weather["wind_speed"].astype(float)

    # PVLib on ERA5 GHI (Erbs + same geometry + SAPM)
    location = Location(LAT, LON, tz="UTC")
    sun = location.get_solarposition(date_range)
    erbs_e = erbs(era5_ghi, sun["zenith"], date_range)
    system = PVSystem(
        surface_tilt=TILT,
        surface_azimuth=AZIM,
        module_parameters={"pdc0": CAP_KW},
    )
    irr_e = system.get_irradiance(
        solar_zenith=sun["zenith"],
        solar_azimuth=sun["azimuth"],
        dni=erbs_e["dni"],
        ghi=era5_ghi,
        dhi=erbs_e["dhi"],
    )
    poa_pvlib_era5 = irr_e["poa_global"].astype(float)
    # approximate 10 m wind from nowhere — use 1 m/s placeholder or cutout roughness unused
    # use ECMWF wind as proxy for SAPM on ERA5 run
    cell_t = sapm_cell(poa_pvlib_era5, era5_temp_c, ecmwf_ws.clip(lower=0.5), SAPM_A, SAPM_B, SAPM_DT)
    pv_era5 = poa_pvlib_era5 * (1 + TEMP_COEFF * (cell_t - 25.0)) * CAP_KW / 1000.0

    # Also PVLib POA from ECMWF for irradiance compare
    erbs_c = erbs(ecmwf_ghi, sun["zenith"], date_range)
    irr_c = system.get_irradiance(
        solar_zenith=sun["zenith"],
        solar_azimuth=sun["azimuth"],
        dni=erbs_c["dni"],
        ghi=ecmwf_ghi,
        dhi=erbs_c["dhi"],
    )
    poa_pvlib_ecmwf = irr_c["poa_global"].astype(float)

    # --- Summaries ---
    rows = []
    for name, sim in [
        ("PVLib ECMWF+SAPM (stored)", pv_ecmwf),
        ("Atlite ERA5 CSi η_inv=0.9 (stored)", atl),
        ("Atlite from CF×kWp (recomputed)", atl_from_cf),
        ("Atlite CSi η_inv=1.0", atl_noinv),
        ("PVLib on ERA5 GHI+SAPM", pv_era5),
    ]:
        m = calculate_metrics(sim, act)
        rows.append({
            "case": name,
            "vs_actual_%": round(m["ratio"], 2),
            "corr": round(m["corr"], 4),
            "sim_MWh": round(m["sim_sum"] / 1e3, 2),
            "act_MWh": round(m["act_sum"] / 1e3, 2),
            "vs_atlite_%": round(100 * sim.sum() / atl.sum(), 2),
            "vs_pvlib_ecmwf_%": round(100 * sim.sum() / pv_ecmwf.sum(), 2),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(result_path("juelich_atlite_overest_summary.csv"), index=False)

    # Irradiance energy
    day = era5_ghi > 20
    irr = pd.DataFrame({
        "metric": [
            "GHI sum (kWh/m2)",
            "POA Atlite tilted sum",
            "POA PVLib-ERA5 sum",
            "POA PVLib-ECMWF sum",
            "GHI ERA5/ECMWF %",
            "POA Atlite/PVLib-ECMWF %",
            "POA PVLib-ERA5/PVLib-ECMWF %",
            "daytime GHI ERA5/ECMWF %",
        ],
        "value": [
            round(era5_ghi.sum() / 1000, 1),
            round(era5_poa.sum() / 1000, 1),
            round(poa_pvlib_era5.sum() / 1000, 1),
            round(poa_pvlib_ecmwf.sum() / 1000, 1),
            round(100 * era5_ghi.sum() / ecmwf_ghi.sum(), 2),
            round(100 * era5_poa.sum() / poa_pvlib_ecmwf.sum(), 2),
            round(100 * poa_pvlib_era5.sum() / poa_pvlib_ecmwf.sum(), 2),
            round(100 * era5_ghi[day].sum() / ecmwf_ghi[day].sum(), 2),
        ],
    })

    # Hourly diagnostic sample + full ratios by hour/month
    hourly = []
    for h in range(24):
        mask = date_range.hour == h
        if act[mask].sum() < 1:
            continue
        hourly.append({
            "hour": h,
            "actual_mean": round(act[mask].mean(), 2),
            "pvlib_ecmwf_mean": round(pv_ecmwf[mask].mean(), 2),
            "atlite_mean": round(atl[mask].mean(), 2),
            "pvlib_era5_mean": round(pv_era5[mask].mean(), 2),
            "atl/act_%": round(ratio_sum(atl[mask], act[mask]), 1),
            "pv_ecmwf/act_%": round(ratio_sum(pv_ecmwf[mask], act[mask]), 1),
            "pv_era5/act_%": round(ratio_sum(pv_era5[mask], act[mask]), 1),
            "atl/pv_ecmwf_%": round(ratio_sum(atl[mask], pv_ecmwf[mask]), 1) if pv_ecmwf[mask].sum() > 1 else np.nan,
            "era5_ghi/ecmwf_ghi_%": round(100 * era5_ghi[mask].sum() / max(ecmwf_ghi[mask].sum(), 1e-6), 1),
        })
    hourly_df = pd.DataFrame(hourly)
    hourly_df.to_csv(result_path("juelich_atlite_overest_hourly.csv"), index=False)

    monthly = []
    for m in range(1, 13):
        mask = date_range.month == m
        monthly.append({
            "month": m,
            "atl/act_%": round(ratio_sum(atl[mask], act[mask]), 1),
            "pv_ecmwf/act_%": round(ratio_sum(pv_ecmwf[mask], act[mask]), 1),
            "pv_era5/act_%": round(ratio_sum(pv_era5[mask], act[mask]), 1),
            "atl/pv_ecmwf_%": round(ratio_sum(atl[mask], pv_ecmwf[mask]), 1),
            "GHI_ERA5/ECMWF_%": round(100 * era5_ghi[mask].sum() / ecmwf_ghi[mask].sum(), 1),
            "POA_atl/POA_pvlibECMWF_%": round(100 * era5_poa[mask].sum() / poa_pvlib_ecmwf[mask].sum(), 1),
        })
    monthly_df = pd.DataFrame(monthly)

    # Huld at STC check
    # Decomposition of Atlite/PVLib gap
    gap_atl_pv = atl.sum() / pv_ecmwf.sum()
    gap_weather = pv_era5.sum() / pv_ecmwf.sum()  # same library, different weather
    gap_lib_on_era5 = atl.sum() / pv_era5.sum()   # same weather family, different library
    # note: atl uses Atlite POA path; pv_era5 uses Erbs on ERA5 GHI — still not identical POA

    lines = [
        "# Why Atlite overestimates Jülich (~117% vs meter)",
        "",
        f"Site: ({LAT}, {LON}), capacity {CAP_KW} kWp, tilt {TILT}°, azimuth {AZIM}°.",
        f"ERA5 cutout cell used: y={cell_y}, x={cell_x}.",
        f"ECMWF weather queried at DB grid ({DB_LAT:.3f}, {DB_LON:.3f}).",
        "",
        "## Code path (Atlite)",
        "- `cutout.pv(panel='CSi', orientation={slope:25, azimuth:165}, layout=...)`",
        "- CSi = **Huld 2010** model: `CF = (G/1000) * η_huld(G,T) * inverter_efficiency`",
        "- Default **`inverter_efficiency = 0.9`** in `atlite/resources/solarpanel/CSi.yaml`",
        "- Temperature in cutout is **Kelvin** (Huld refs `r_tamb=293`, `r_tmod=298`) — consistent.",
        "- PVLib path is ideal DC `POA/1000 * kWp` × SAPM temp — **no** inverter derate.",
        "- Therefore if POA were equal, Atlite should be *lower* than PVLib by ~10%. "
        "Observed Atlite > PVLib ⇒ **ERA5/Atlite irradiance path more than offsets the inverter.**",
        "",
        "## Generation vs actual",
        summary.to_string(index=False),
        "",
        "## Irradiance comparison",
        irr.to_string(index=False),
        "",
        "## Gap decomposition (energy ratios)",
        f"- Atlite / PVLib(ECMWF) = **{100*gap_atl_pv:.1f}%** (what we see in the three series)",
        f"- PVLib(ERA5) / PVLib(ECMWF) = **{100*gap_weather:.1f}%** → weather-product effect with same library",
        f"- Atlite / PVLib(ERA5) = **{100*gap_lib_on_era5:.1f}%** → library/POA-path effect on ERA5-class irradiance",
        f"- Atlite η_inv=1.0 / Atlite η_inv=0.9 = **{100*atl_noinv.sum()/atl.sum():.1f}%** (exactly 1/0.9 if model linear)",
        "",
        "## Monthly",
        monthly_df.to_string(index=False),
        "",
        "## Reading",
        "1. If PVLib(ERA5) ≈ Atlite: overestimate is mostly **ERA5 GHI hot vs ECMWF / meter**.",
        "2. If PVLib(ERA5) ≪ Atlite: Atlite **POA/Huld** adds extra (clearsky diffuse, transposition).",
        "3. Turning off inverter makes Atlite *worse* vs meter — inverter is not the cause of overestimate.",
        "4. Three power series are highly correlated (atl vs pvlib ~0.99); bias is mostly scale + shoulders.",
        "",
    ]
    path = result_path("juelich_atlite_overest.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # save aligned compare for plotting later
    out = pd.DataFrame({
        "actual": act,
        "pvlib_ecmwf": pv_ecmwf,
        "atlite": atl,
        "pvlib_era5": pv_era5,
        "atlite_noinv": atl_noinv,
        "era5_ghi": era5_ghi,
        "ecmwf_ghi": ecmwf_ghi,
        "era5_poa_atlite": era5_poa,
        "poa_pvlib_era5": poa_pvlib_era5,
        "poa_pvlib_ecmwf": poa_pvlib_ecmwf,
    }, index=date_range)
    out.to_csv(result_path("juelich_atlite_overest_timeseries.csv"))

    print(summary.to_string(index=False))
    print()
    print(irr.to_string(index=False))
    print()
    print(f"gap atl/pv_ecmwf={100*gap_atl_pv:.1f}%  pv_era5/pv_ecmwf={100*gap_weather:.1f}%  atl/pv_era5={100*gap_lib_on_era5:.1f}%")
    print("Wrote", path)


if __name__ == "__main__":
    main()
