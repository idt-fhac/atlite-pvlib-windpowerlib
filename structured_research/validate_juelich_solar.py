"""
Jülich single-plant PV validation: PVLib vs Atlite on the SAME ERA5 cutout.

Weather: juelich_2023.nc (ssrd = influx_direct + influx_diffuse, 2 m temperature).
Both libraries use identical site geometry (25° / 165°) and nameplate (216.96 kWp).

Derates applied after conversion:
  - Module aging from commission date (both libraries), 0.5 %/y linear.
  - Inverter efficiency 0.90 on PVLib only (Atlite CSi already includes η_inv=0.9).

Outputs → structured_research/results/ (and copies key PNGs to text/data/):
  juelich_eview_comparison_complete.csv
  juelich_eview_comparison_stats.csv
  juelich_seasonal_comparison.csv
  juelich_library_delta.csv
  juelich_factor_ablation.csv
  juelich_january_2weeks.png / juelich_june_2weeks.png
  juelich_daily_max_duration.png / juelich_hourly_duration.png
"""

from __future__ import annotations

from pathlib import Path

import sys
from datetime import date, datetime

import atlite
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, DayLocator
from pvlib.irradiance import erbs
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.temperature import sapm_cell

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import (
    AGING_RATE_PER_YEAR,
    AGING_REF_DATE,
    INVERTER_EFFICIENCY,
    SAPM_A,
    SAPM_B,
    SAPM_DT,
    TEMP_COEFF,
    copy_to_manuscript,
)
from utils import (
    resolve_engine,
    offline_mode,
    query_juelich_actual,
    calculate_metrics,
    metrics_by_season,
    plot_duration_curves,
    plot_scatter_comparison,
    plot_timeseries_comparison,
    ensure_results_dir,
    cutout_path,
    result_path,
    load_local_data,
)

LAT, LON = 50.92, 6.36
CAPACITY_KW = 216.960
TILT, AZIMUTH = 25.0, 165.0

# Aging: Campus Jülich plant prior. Exact MaStR unit at 216.96 kWp not uniquely
# matched; ~12-year plant in 2023 → mid-2011 commission.
COMMISSION_DATE = date(2011, 7, 1)


def aging_factor(
    commission: date = COMMISSION_DATE,
    ref: date = AGING_REF_DATE,
    rate: float = AGING_RATE_PER_YEAR,
) -> float:
    age_y = max(0.0, (ref - commission).days / 365.25)
    return float(np.clip(1.0 - rate * age_y, 0.70, 1.0)), age_y


def cutout_weather(cutout: atlite.Cutout, date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """Nearest-cell ERA5 fields from the Jülich cutout."""
    ds = cutout.data
    yi = int(np.abs(ds.y.values - LAT).argmin())
    xi = int(np.abs(ds.x.values - LON).argmin())
    print(
        f"  Cutout cell: y={float(ds.y.values[yi])}, x={float(ds.x.values[xi])} "
        f"(site {LAT}, {LON})"
    )

    def series(name: str) -> pd.Series:
        s = ds[name].isel(y=yi, x=xi).to_pandas()
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        return s.reindex(date_range, method="nearest").astype(float)

    ghi = series("influx_direct").clip(lower=0) + series("influx_diffuse").clip(lower=0)
    temp_k = series("temperature")
    u100 = series("wnd100m").clip(lower=0)
    u10 = u100 * (10.0 / 100.0) ** 0.14
    return pd.DataFrame(
        {"ghi": ghi, "temp_c": temp_k - 273.15, "wind_10m": u10, "wnd100m": u100},
        index=date_range,
    )


def run_pvlib(weather: pd.DataFrame, date_range: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    location = Location(LAT, LON, tz="UTC")
    sun_pos = location.get_solarposition(date_range)
    erbs_out = erbs(weather["ghi"], sun_pos["zenith"], date_range)
    system = PVSystem(
        surface_tilt=TILT,
        surface_azimuth=AZIMUTH,
        module_parameters={"pdc0": CAPACITY_KW},
    )
    irr = system.get_irradiance(
        solar_zenith=sun_pos["zenith"],
        solar_azimuth=sun_pos["azimuth"],
        dni=erbs_out["dni"],
        ghi=weather["ghi"],
        dhi=erbs_out["dhi"],
    )
    poa = irr["poa_global"]
    standard = (poa * CAPACITY_KW) / 1000.0
    standard.name = "pvlib_standard"
    cell_temp = sapm_cell(
        poa_global=poa,
        temp_air=weather["temp_c"],
        wind_speed=weather["wind_10m"],
        a=SAPM_A,
        b=SAPM_B,
        deltaT=SAPM_DT,
    )
    temp_corr = standard * (1 + TEMP_COEFF * (cell_temp - 25.0))
    temp_corr.name = "pvlib_temp_corrected"
    return standard, temp_corr


def run_atlite(cutout: atlite.Cutout, date_range: pd.DatetimeIndex) -> pd.Series:
    layout = cutout.layout_from_capacity_list(
        pd.DataFrame({"x": [LON], "y": [LAT], "maxPower": [CAPACITY_KW / 1000.0]}),
        col="maxPower",
    )
    gen = (
        cutout.pv(
            panel="CSi",
            orientation={"slope": TILT, "azimuth": AZIMUTH},
            layout=layout,
        ).to_series()
        * 1000.0
    )
    gen.index = date_range
    gen.name = "atlite_sim"
    return gen


def _actual_overlap_series() -> pd.Series:
    """Meter series with gaps as NaN (not zero-filled)."""
    raw = load_local_data("juelich_actuals.parquet")
    if not isinstance(raw.index, pd.DatetimeIndex):
        if "time" in raw.columns:
            raw = raw.set_index("time")
        raw.index = pd.to_datetime(raw.index)
    if getattr(raw.index, "tz", None) is not None:
        raw.index = raw.index.tz_convert("UTC").tz_localize(None)
    return raw["generation"].sort_index()


def write_factor_ablation(
    act: pd.Series,
    pv_lin: pd.Series,
    pv_sapm: pd.Series,
    atl: pd.Series,
    eta_age: float,
) -> pd.DataFrame:
    """
    Factor ablation on meter-overlap hours.

    Constant multipliers (η_inv, aging) cannot raise Pearson r (scale-invariant);
    they reduce energy bias and MAE/RMSE. SAPM can change r slightly via shape.
    """
    ov = act.dropna()
    idx = ov.index
    configs = [
        ("PVLib linear (raw)", pv_lin.reindex(idx)),
        ("PVLib + SAPM (temp only)", pv_sapm.reindex(idx)),
        ("PVLib linear × η_inv", (pv_lin * INVERTER_EFFICIENCY).reindex(idx)),
        ("PVLib linear × aging", (pv_lin * eta_age).reindex(idx)),
        ("PVLib SAPM × η_inv", (pv_sapm * INVERTER_EFFICIENCY).reindex(idx)),
        ("PVLib SAPM × aging", (pv_sapm * eta_age).reindex(idx)),
        (
            "PVLib SAPM × η_inv × aging",
            (pv_sapm * INVERTER_EFFICIENCY * eta_age).reindex(idx),
        ),
        ("Atlite CSi (η_inv in panel)", atl.reindex(idx)),
        ("Atlite × aging", (atl * eta_age).reindex(idx)),
    ]
    rows = []
    base_mae = base_rmse = None
    for name, sim in configs:
        m = calculate_metrics(sim, ov)
        day = (ov.index.hour >= 7) & (ov.index.hour <= 17) & (ov > 1)
        mday = calculate_metrics(sim[day], ov[day]) if int(day.sum()) > 10 else m
        if base_mae is None:
            base_mae, base_rmse = m["mae"], m["rmse"]
        rows.append(
            {
                "Configuration": name,
                "Ratio_%": round(m["ratio"], 2),
                "Corr_full": round(m["corr"], 4),
                "Corr_daytime": round(mday["corr"], 4),
                "MAE_kW": round(m["mae"], 3),
                "RMSE_kW": round(m["rmse"], 3),
                "MAE_reduction_%_vs_raw": round(
                    100.0 * (base_mae - m["mae"]) / base_mae, 2
                ),
                "RMSE_reduction_%_vs_raw": round(
                    100.0 * (base_rmse - m["rmse"]) / base_rmse, 2
                ),
                "n": int(m["n"]),
            }
        )
    abl = pd.DataFrame(rows)
    abl.to_csv(result_path("juelich_factor_ablation.csv"), index=False)
    print("\n=== Factor ablation (overlap hours; scalars do not raise r) ===")
    print(abl.to_string(index=False))
    return abl


def plot_hourly_duration(comp: pd.DataFrame, outfile: str) -> None:
    """Classic duration curve of all hourly power values."""
    act_raw = _actual_overlap_series()
    # Use overlap hours only for actual; sims on same index
    idx = act_raw.dropna().index.intersection(comp.index)
    act = act_raw.loc[idx].to_numpy()
    pv = comp.loc[idx, "oeds_temp_corrected"].to_numpy()
    atl = comp.loc[idx, "atlite_sim"].to_numpy()
    n = len(act)
    x = np.arange(1, n + 1) / n * 100.0
    act_s = np.sort(act)[::-1]
    pv_s = np.sort(pv)[::-1]
    atl_s = np.sort(atl)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, act_s, color="black", lw=2.0, label="Actual")
    ax.plot(
        x,
        pv_s,
        color="C0",
        lw=1.6,
        ls="--",
        label="PVLib SAPM ×η_inv×aging",
    )
    ax.plot(
        x,
        atl_s,
        color="C2",
        lw=1.6,
        ls="-.",
        label="Atlite CSi ×aging",
    )
    ax.set_xlabel("Percentage of time (%)")
    ax.set_ylabel("Power (kW)")
    ax.set_title(
        "Jülich PV 2023 — hourly generation duration curve\n"
        f"(n={n} overlap hours; sims include inverter/aging derates)"
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved {outfile}")


def plot_daily_max_duration(comp: pd.DataFrame, outfile: str) -> None:
    """Duration curve of per-day maximum power for actual / PVLib / Atlite."""
    act_raw = _actual_overlap_series()
    daily_act = act_raw.groupby(act_raw.index.normalize()).max()
    daily_pv = comp["oeds_temp_corrected"].groupby(comp.index.normalize()).max()
    daily_atl = comp["atlite_sim"].groupby(comp.index.normalize()).max()
    days = daily_act.index.intersection(daily_pv.index).intersection(daily_atl.index)
    df = pd.DataFrame(
        {
            "actual": daily_act.loc[days],
            "pvlib": daily_pv.loc[days],
            "atlite": daily_atl.loc[days],
        }
    ).dropna()
    df = df[df["actual"] > 0]
    n = len(df)
    x = np.arange(1, n + 1) / n * 100.0
    act_s = np.sort(df["actual"].to_numpy())[::-1]
    pv_s = np.sort(df["pvlib"].to_numpy())[::-1]
    atl_s = np.sort(df["atlite"].to_numpy())[::-1]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, act_s, color="black", lw=2.0, label="Actual (daily max)")
    ax.plot(
        x,
        pv_s,
        color="C0",
        lw=1.6,
        ls="--",
        label="PVLib SAPM ×η_inv×aging (daily max)",
    )
    ax.plot(
        x,
        atl_s,
        color="C2",
        lw=1.6,
        ls="-.",
        label="Atlite CSi ×aging (daily max)",
    )
    ax.set_xlabel("Percentage of days (%)")
    ax.set_ylabel("Daily maximum power (kW)")
    ax.set_title(
        "Jülich PV 2023 — duration curve of daily maximum generation\n"
        f"(n={n} days with metered peak > 0; sims include inverter/aging derates)"
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved {outfile}")


def plot_two_week_window(
    comp: pd.DataFrame,
    start: str,
    end: str,
    title: str,
    outfile: str,
    snow_span: tuple[str, str] | None = None,
) -> None:
    act_raw = _actual_overlap_series()
    w = comp.loc[start:end].copy()
    w["actual_overlap"] = act_raw.reindex(w.index)

    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.plot(
        w.index,
        w["actual_overlap"],
        color="black",
        lw=1.8,
        label="Actual (metered; gaps left blank)",
    )
    ax.plot(
        w.index,
        w["oeds_temp_corrected"],
        color="C0",
        lw=1.3,
        ls="--",
        label="PVLib (ERA5 + SAPM × η_inv × aging)",
    )
    ax.plot(
        w.index,
        w["atlite_sim"],
        color="C2",
        lw=1.3,
        ls="-.",
        label="Atlite (ERA5 CSi × aging)",
    )
    if snow_span is not None:
        ax.axvspan(
            pd.Timestamp(snow_span[0]),
            pd.Timestamp(snow_span[1]),
            color="0.85",
            alpha=0.7,
            zorder=0,
            label="Snow on modules",
        )
    ax.set_xlim(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23))
    ax.set_ylabel("Power (kW)")
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%d %b"))
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved {outfile}")


def main():
    eta_age, age_y = aging_factor()
    print(f"Mode: {'OFFLINE (cache-only)' if offline_mode() else 'DB allowed (prefer cache)'}")
    print("Weather source: ERA5 cutout juelich_2023.nc (ssrd) for BOTH PVLib and Atlite")
    print(
        f"Derates: aging η={eta_age:.4f} "
        f"(commission {COMMISSION_DATE}, age={age_y:.2f} y @ {100*DEGRADATION_RATE_PER_YEAR:.1f}%/y); "
        f"PVLib inverter η={INVERTER_EFFICIENCY:.2f} (Atlite CSi already includes 0.90)"
    )

    timescale_engine = resolve_engine("timescale")
    date_range = pd.date_range(datetime(2023, 1, 1), datetime(2023, 12, 31, 23), freq="h")

    print("Fetching measured AC generation...")
    actual_ov = query_juelich_actual(
        timescale_engine,
        "2023-01-01 00:00:00",
        "2023-12-31 23:59:59",
        date_range=date_range,
    )["generation"]
    # Zero-fill for plots/metrics that expect a full 8760 index (nights / gaps → 0)
    actual = actual_ov.reindex(date_range, fill_value=0.0)
    actual.name = "actual_generation"

    print("Loading cutout weather...")
    cutout = atlite.Cutout(cutout_path("juelich_2023.nc"))
    weather = cutout_weather(cutout, date_range)

    print("Running PVLib on cutout ssrd...")
    pv_std_raw, pv_temp_raw = run_pvlib(weather, date_range)
    # PVLib: inverter (parity with Atlite) × module aging
    pv_factor = INVERTER_EFFICIENCY * eta_age
    pv_std = pv_std_raw * pv_factor
    pv_temp = pv_temp_raw * pv_factor
    pv_std.name = "oeds_standard"
    pv_temp.name = "oeds_temp_corrected"

    print("Running Atlite on same cutout...")
    atl_raw = run_atlite(cutout, date_range)
    # Atlite: aging only (η_inv already in CSi panel config)
    atl = atl_raw * eta_age
    atl.name = "atlite_sim"

    ensure_results_dir()
    write_factor_ablation(actual_ov, pv_std_raw, pv_temp_raw, atl_raw, eta_age)

    comp = pd.DataFrame(
        {
            "actual_generation": actual,
            "oeds_standard": pv_std,
            "oeds_temp_corrected": pv_temp,
            "atlite_sim": atl,
            "pvlib_linear_raw": pv_std_raw,
            "pvlib_sapm_raw": pv_temp_raw,
            "atlite_raw": atl_raw,
        }
    ).fillna(0.0)
    comp.to_csv(result_path("juelich_eview_comparison_complete.csv"))

    m_std = calculate_metrics(comp["oeds_standard"], actual)
    m_temp = calculate_metrics(comp["oeds_temp_corrected"], actual)
    m_atl = calculate_metrics(comp["atlite_sim"], actual)
    m_lib = calculate_metrics(comp["oeds_temp_corrected"], comp["atlite_sim"])

    print("\n=== JÜLICH 2023 — matched ERA5 cutout + aging + PVLib inverter ===")
    print(f"Actual:              {m_std['act_sum']:.1f} kWh")
    print(
        f"PVLib (no temp)×η:   {m_std['sim_sum']:.1f} kWh | {m_std['ratio']:.2f}% | r={m_std['corr']:.4f}"
    )
    print(
        f"PVLib + SAPM×η:      {m_temp['sim_sum']:.1f} kWh | {m_temp['ratio']:.2f}% | r={m_temp['corr']:.4f}"
    )
    print(
        f"Atlite CSi×aging:    {m_atl['sim_sum']:.1f} kWh | {m_atl['ratio']:.2f}% | r={m_atl['corr']:.4f}"
    )
    print(f"PVLib+SAPM / Atlite: {m_lib['ratio']:.2f}% | r={m_lib['corr']:.4f}")
    print(
        f"Factors: PVLib total η={pv_factor:.4f} "
        f"(inv={INVERTER_EFFICIENCY} × age={eta_age:.4f}); Atlite age={eta_age:.4f}"
    )

    stats = pd.DataFrame(
        {
            "Metric": [
                "Actual Total (kWh)",
                "PVLib standard×η_inv×aging Total (kWh)",
                "PVLib SAPM×η_inv×aging Total (kWh)",
                "Atlite×aging Total (kWh)",
                "PVLib standard×η Correlation",
                "PVLib SAPM×η Correlation",
                "Atlite×aging Correlation",
                "PVLib standard×η MAE (kW)",
                "PVLib SAPM×η MAE (kW)",
                "Atlite×aging MAE (kW)",
                "PVLib standard×η RMSE (kW)",
                "PVLib SAPM×η RMSE (kW)",
                "Atlite×aging RMSE (kW)",
                "PVLib SAPM×η / Atlite×aging Ratio_%",
                "PVLib SAPM×η / Atlite×aging Corr",
                "Inverter_efficiency_PVLib",
                "Aging_factor",
                "Age_years_mid2023",
                "Commission_date",
                "Degradation_rate_per_year",
                "PVLib_total_factor",
            ],
            "Value": [
                m_std["act_sum"],
                m_std["sim_sum"],
                m_temp["sim_sum"],
                m_atl["sim_sum"],
                m_std["corr"],
                m_temp["corr"],
                m_atl["corr"],
                m_std["mae"],
                m_temp["mae"],
                m_atl["mae"],
                m_std["rmse"],
                m_temp["rmse"],
                m_atl["rmse"],
                m_lib["ratio"],
                m_lib["corr"],
                INVERTER_EFFICIENCY,
                eta_age,
                age_y,
                str(COMMISSION_DATE),
                DEGRADATION_RATE_PER_YEAR,
                pv_factor,
            ],
        }
    )
    stats.to_csv(result_path("juelich_eview_comparison_stats.csv"), index=False)

    seasonal_rows = []
    for model_name, series in [
        ("PVLib (ERA5, standard×η_inv×aging)", comp["oeds_standard"]),
        ("PVLib (ERA5, SAPM×η_inv×aging)", comp["oeds_temp_corrected"]),
        ("Atlite (ERA5, CSi×aging)", comp["atlite_sim"]),
    ]:
        sdf = metrics_by_season(series, actual)
        sdf.insert(0, "Scale", "Single")
        sdf.insert(1, "Site", "Juelich_PV")
        sdf.insert(2, "Technology", "Solar")
        sdf.insert(3, "Model", model_name)
        seasonal_rows.append(sdf)
    seasonal = pd.concat(seasonal_rows, ignore_index=True)
    seasonal.to_csv(result_path("juelich_seasonal_comparison.csv"), index=False)

    lib_rows = []
    for code, months in [
        ("DJF", [12, 1, 2]),
        ("MAM", [3, 4, 5]),
        ("JJA", [6, 7, 8]),
        ("SON", [9, 10, 11]),
        ("FULL", list(range(1, 13))),
    ]:
        mask = date_range.month.isin(months)
        m = calculate_metrics(
            comp.loc[mask, "oeds_temp_corrected"], comp.loc[mask, "atlite_sim"]
        )
        lib_rows.append(
            {
                "Season": code,
                "PVLib_SAPM_kWh": round(m["sim_sum"], 1),
                "Atlite_kWh": round(m["act_sum"], 1),
                "Ratio_PVLib_Atlite_%": round(m["ratio"], 2),
                "Corr": round(m["corr"], 4),
                "PVLib_vs_Actual_%": round(
                    calculate_metrics(
                        comp.loc[mask, "oeds_temp_corrected"], actual[mask]
                    )["ratio"],
                    2,
                ),
                "Atlite_vs_Actual_%": round(
                    calculate_metrics(comp.loc[mask, "atlite_sim"], actual[mask])[
                        "ratio"
                    ],
                    2,
                ),
            }
        )
    lib_df = pd.DataFrame(lib_rows)
    lib_df.to_csv(result_path("juelich_library_delta.csv"), index=False)
    print("\nSeasonal PVLib×η / Atlite×aging (same cutout):")
    print(lib_df.to_string(index=False))

    print("Generating plots...")
    plot_timeseries_comparison(
        comp,
        {
            "actual_generation": "Actual Measured (eview)",
            "oeds_standard": "PVLib ×η_inv×aging (no temp)",
            "oeds_temp_corrected": "PVLib SAPM ×η_inv×aging",
            "atlite_sim": "Atlite CSi ×aging",
        },
        "Jülich PV — ERA5 cutout + inverter/aging (1–15 Jun 2023)",
        "Generation (kW)",
        result_path("juelich_eview_comparison.png"),
        sample_range=slice("2023-06-01", "2023-06-15"),
        colors=["black", "tab:red", "tab:blue", "tab:green"],
        linestyles=["-", "--", "-.", ":"],
    )
    plot_duration_curves(
        {
            "Actual Measured": comp["actual_generation"],
            "PVLib SAPM ×η_inv×aging": comp["oeds_temp_corrected"],
            "Atlite CSi ×aging": comp["atlite_sim"],
        },
        "Jülich PV duration curve 2023 — ERA5 + inverter/aging",
        "Power Output (kW)",
        result_path("juelich_duration_comparison.png"),
        colors=["black", "tab:blue", "tab:green"],
    )
    plot_scatter_comparison(
        comp["actual_generation"],
        {
            "PVLib SAPM ×η_inv×aging": comp["oeds_temp_corrected"],
            "Atlite CSi ×aging": comp["atlite_sim"],
        },
        "Jülich PV — simulated vs actual (ERA5 + inverter/aging)",
        "Simulated Power (kW)",
        result_path("juelich_scatter_comparison.png"),
        colors=["tab:blue", "tab:green"],
    )

    plot_two_week_window(
        comp,
        "2023-01-08",
        "2023-01-21",
        "Campus Jülich PV — 8–21 January 2023 "
        f"(η_inv={INVERTER_EFFICIENCY:.2f} on PVLib, aging η={eta_age:.3f} both)",
        result_path("juelich_january_2weeks.png"),
        snow_span=("2023-01-19", "2023-01-22"),
    )
    plot_two_week_window(
        comp,
        "2023-06-05",
        "2023-06-18",
        "Campus Jülich PV — 5–18 June 2023 "
        f"(η_inv={INVERTER_EFFICIENCY:.2f} on PVLib, aging η={eta_age:.3f} both)",
        result_path("juelich_june_2weeks.png"),
    )
    plot_daily_max_duration(comp, result_path("juelich_daily_max_duration.png"))
    plot_hourly_duration(comp, result_path("juelich_hourly_duration.png"))

    for name in [
        "juelich_january_2weeks.png",
        "juelich_june_2weeks.png",
        "juelich_daily_max_duration.png",
        "juelich_hourly_duration.png",
        "juelich_eview_comparison.png",
        "juelich_duration_comparison.png",
        "juelich_scatter_comparison.png",
    ]:
        src = result_path(name)
        if src.is_file():
            copy_to_manuscript(src)

    print("Done →", ensure_results_dir())


if __name__ == "__main__":
    main()
