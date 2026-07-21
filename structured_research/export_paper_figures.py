#!/usr/bin/env python3
"""
export_paper_figures.py
=======================
Rebuild manuscript figures from cached results:

  - Vector (SVG → PDF): fortnights, duration curves, seasonal bars
  - Raster (PNG only): scatters

Paper includes PDFs for vector figures and PNGs for scatters.

    cd paper
    OFFLINE_MODE=1 .venv/bin/python structured_research/export_paper_figures.py

Copies land in latex/data/ (and structured_research/results/).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, DayLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.constants import FEED_IN_SCALE, FLEET_AGING_FALLBACK, INVERTER_EFFICIENCY
from lib.figures import (
    PAPER_RASTER_STEMS,
    PAPER_VECTOR_STEMS,
    save_raster_figure,
    save_vector_figure,
)
from lib.manuscript import copy_to_manuscript
from utils import ensure_results_dir, result_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("export_paper_figures")


def _load_juelich_comp() -> pd.DataFrame:
    path = result_path("juelich_eview_comparison_complete.csv")
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing {path}; run validate_juelich_solar.py first."
        )
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _actual_overlap_from_comp(comp: pd.DataFrame) -> pd.Series:
    """Prefer non-zero meter hours as overlap; leave true zeros at night as 0."""
    # Gaps in the paper plots are NaN (blank). Heuristic: where sims produce
    # daytime power but actual is exactly 0 across a contiguous gap is hard
    # offline — use stored actual and mask only explicit NaNs if present.
    act = comp["actual_generation"].astype(float).copy()
    return act


def plot_juelich_fortnight(
    comp: pd.DataFrame,
    start: str,
    end: str,
    title: str,
    stem: str,
    snow_span: tuple[str, str] | None = None,
) -> None:
    act = _actual_overlap_from_comp(comp)
    w = comp.loc[start:end].copy()
    # Blank meter gaps: if actual is 0 and both sims > 5 kW, treat as gap
    a = act.reindex(w.index)
    pv = w["oeds_temp_corrected"]
    atl = w["atlite_sim"]
    gap = (a == 0) & ((pv > 5) | (atl > 5))
    a = a.mask(gap)

    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.plot(w.index, a, color="black", lw=1.8, label="Actual (metered; gaps left blank)")
    ax.plot(
        w.index,
        pv,
        color="C0",
        lw=1.3,
        ls="--",
        label="PVLib (ERA5 + SAPM × η_inv × aging)",
    )
    ax.plot(
        w.index,
        atl,
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
    save_vector_figure(fig, stem)
    plt.close(fig)


def plot_juelich_hourly_duration(comp: pd.DataFrame) -> None:
    act = _actual_overlap_from_comp(comp)
    idx = act.index.intersection(comp.index)
    # Overlap-ish: drop hours where actual is NaN
    mask = act.loc[idx].notna()
    act_v = act.loc[idx][mask].to_numpy()
    pv = comp.loc[idx, "oeds_temp_corrected"][mask].to_numpy()
    atl = comp.loc[idx, "atlite_sim"][mask].to_numpy()
    n = len(act_v)
    x = np.arange(1, n + 1) / n * 100.0

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, np.sort(act_v)[::-1], color="black", lw=2.0, label="Actual")
    ax.plot(x, np.sort(pv)[::-1], color="C0", lw=1.6, ls="--", label="PVLib SAPM ×η_inv×aging")
    ax.plot(x, np.sort(atl)[::-1], color="C2", lw=1.6, ls="-.", label="Atlite CSi ×aging")
    ax.set_xlabel("Percentage of time (%)")
    ax.set_ylabel("Power (kW)")
    ax.set_title(
        "Jülich PV 2023 — hourly generation duration curve\n"
        f"(n={n} hours; sims include inverter/aging derates)"
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    save_vector_figure(fig, "juelich_hourly_duration")
    plt.close(fig)


def plot_juelich_daily_max_duration(comp: pd.DataFrame) -> None:
    act = _actual_overlap_from_comp(comp)
    daily_act = act.groupby(act.index.normalize()).max()
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

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, np.sort(df["actual"].to_numpy())[::-1], color="black", lw=2.0, label="Actual (daily max)")
    ax.plot(
        x,
        np.sort(df["pvlib"].to_numpy())[::-1],
        color="C0",
        lw=1.6,
        ls="--",
        label="PVLib SAPM ×η_inv×aging (daily max)",
    )
    ax.plot(
        x,
        np.sort(df["atlite"].to_numpy())[::-1],
        color="C2",
        lw=1.6,
        ls="-.",
        label="Atlite CSi ×aging (daily max)",
    )
    ax.set_xlabel("Percentage of days (%)")
    ax.set_ylabel("Daily maximum power (kW)")
    ax.set_title(
        "Jülich PV 2023 — duration curve of daily maximum generation\n"
        f"(n={n} days with measured max > 0; sims include inverter/aging)"
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    save_vector_figure(fig, "juelich_daily_max_duration")
    plt.close(fig)


def plot_juelich_scatter(comp: pd.DataFrame) -> None:
    """Raster scatter (kept as PNG)."""
    act = _actual_overlap_from_comp(comp)
    idx = act.dropna().index.intersection(comp.index)
    a = act.loc[idx]
    # Prefer daytime-ish points for readability
    pv = comp.loc[idx, "oeds_temp_corrected"]
    atl = comp.loc[idx, "atlite_sim"]
    mask = a.notna()
    a, pv, atl = a[mask], pv[mask], atl[mask]

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.scatter(a, pv, s=8, alpha=0.15, color="C0", label="PVLib SAPM ×η_inv×aging", rasterized=True)
    ax.scatter(a, atl, s=8, alpha=0.15, color="C2", label="Atlite CSi ×aging", rasterized=True)
    lim = float(max(a.max(), pv.max(), atl.max()) * 1.02)
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Measured AC (kW)")
    ax.set_ylabel("Simulated AC (kW)")
    ax.set_title("Jülich PV 2023 — hourly scatter after inverter/aging")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    save_raster_figure(fig, "juelich_scatter_comparison", dpi=150)
    plt.close(fig)


def plot_seasonal_summary() -> None:
    ts = pd.read_csv(result_path("matched_era5_timeseries.csv"), index_col=0, parse_dates=True)
    seasons = [
        ("Winter", [12, 1, 2]),
        ("Spring", [3, 4, 5]),
        ("Summer", [6, 7, 8]),
        ("Autumn", [9, 10, 11]),
        ("Full year", None),
    ]

    def ratio(sim, act, months):
        if months:
            m = sim.index.month.isin(months)
            sim, act = sim[m], act[m]
        return 100.0 * sim.sum() / act.sum()

    labels = [s[0] for s in seasons]
    w_wpl, w_atl, s_pv, s_atl = [], [], [], []
    for _, months in seasons:
        w_wpl.append(ratio(ts.wpl_wind, ts.entsoe_wind, months))
        w_atl.append(ratio(ts.atlite_wind, ts.entsoe_wind, months))
        s_pv.append(ratio(ts.pvlib_mastr_solar, ts.entsoe_solar, months))
        s_atl.append(ratio(ts.atlite_solar, ts.entsoe_solar, months))

    x = np.arange(len(labels))
    w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    ax = axes[0]
    ax.bar(x - 0.5 * w, w_wpl, w, label="WPL MaStR fleet", color="C0")
    ax.bar(x + 0.5 * w, w_atl, w, label="Atlite V112@80 m", color="C2")
    ax.axhline(100, color="0.3", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Yield ratio vs ENTSO-E (%)")
    ax.set_title("Onshore wind\n(fleet vs single-turbine proxy)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, max(max(w_wpl), max(w_atl)) * 1.12)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x - 0.5 * w, s_pv, w, label="PVLib MaStR", color="C0")
    ax.bar(x + 0.5 * w, s_atl, w, label="Atlite CSi", color="C2")
    ax.axhline(100, color="0.3", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Yield ratio vs ENTSO-E (%)")
    ax.set_title("Solar PV\n(vs public-grid feed-in)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, max(max(s_pv), max(s_atl)) * 1.08)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Germany 2023 — matched ERA5 cutout vs ENTSO-E", fontsize=11, y=1.02)
    fig.tight_layout()
    save_vector_figure(fig, "annual_seasonal_summary")
    plt.close(fig)


TSO_ORDER = ("DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNET")
TSO_LABELS = {
    "DE_50HZ": "50Hertz",
    "DE_AMPRION": "Amprion",
    "DE_TENNET": "TenneT",
    "DE_TRANSNET": "TransnetBW",
}


def plot_tso_yield_ratios() -> None:
    """Full-year TSO yield ratios vs ENTSO-E (matched ERA5) — paper fig for all four zones."""
    path = Path(result_path("matched_era5_tso.csv"))
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run validate_matched_era5.py first.")
    df = pd.read_csv(path)

    def pick(tech: str, model_substr: str) -> list[float]:
        out = []
        for z in TSO_ORDER:
            row = df[
                (df["Scale"] == z)
                & (df["Technology"] == tech)
                & (df["Model"].str.contains(model_substr, regex=False))
            ]
            if row.empty:
                raise ValueError(f"No row for {z} / {tech} / {model_substr}")
            # Prefer MaStR orientation for solar when both PVLib rows exist
            if tech == "Solar" and model_substr == "PVLib":
                mastr = row[row["Model"].str.contains("MaStR", regex=False)]
                row = mastr if not mastr.empty else row
            out.append(float(row.iloc[0]["Ratio_%"]))
        return out

    w_wpl = pick("Wind", "Windpowerlib")
    w_atl = pick("Wind", "Atlite")
    s_pv = pick("Solar", "PVLib")
    s_atl = pick("Solar", "Atlite")

    # Tidy table for manuscript / reuse
    tidy = pd.DataFrame(
        {
            "TSO": [TSO_LABELS[z] for z in TSO_ORDER],
            "Scale": list(TSO_ORDER),
            "Wind_WPL_MaStR_%": w_wpl,
            "Wind_Atlite_V112_%": w_atl,
            "Solar_PVLib_MaStR_%": s_pv,
            "Solar_Atlite_%": s_atl,
        }
    )
    tidy_path = Path(result_path("tso_yield_ratios.csv"))
    tidy.to_csv(tidy_path, index=False)
    copy_to_manuscript(tidy_path)

    labels = [TSO_LABELS[z] for z in TSO_ORDER]
    x = np.arange(len(labels))
    w = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=False)

    ax = axes[0]
    ax.bar(x - 1.5 * w, w_wpl, w, label="WPL MaStR fleet", color="C0")
    ax.bar(x - 0.5 * w, w_atl, w, label="Atlite V112@80 m", color="C2")
    ax.axhline(100, color="0.3", ls="--", lw=1, label="ENTSO-E = 100%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Yield ratio vs ENTSO-E (%)")
    ax.set_title("Onshore wind (full year 2023)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, max(max(w_wpl), max(w_atl), 100) * 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x - 1.5 * w, s_pv, w, label="PVLib MaStR", color="C0")
    ax.bar(x - 0.5 * w, s_atl, w, label="Atlite CSi", color="C2")
    ax.axhline(100, color="0.3", ls="--", lw=1, label="ENTSO-E = 100%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Yield ratio vs ENTSO-E (%)")
    ax.set_title("Solar PV vs public-grid feed-in")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, max(max(s_pv), max(s_atl), 100) * 1.08)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "German TSO zones 2023 — matched ERA5 cutout vs ENTSO-E",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    save_vector_figure(fig, "tso_yield_ratios")
    plt.close(fig)
    logger.info("Wrote tso_yield_ratios (+ tso_yield_ratios.csv)")


def plot_wind_library_test() -> None:
    """Bar chart: Atlite V112 vs WPL V112 vs WPL MaStR fleet (national)."""
    path = Path(result_path("matched_era5_wind_library_test.csv"))
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; ship paper-path library-test summary CSV."
        )
    df = pd.read_csv(path)
    short = {
        "Atlite V112 @ 80 m": "Atlite\nV112@80 m",
        "Windpowerlib V112 @ 80 m (matched proxy)": "WPL V112@80 m\n(matched)",
        "Windpowerlib MaStR fleet (types + hubs)": "WPL MaStR\nfleet",
    }
    labels = [short.get(c, c) for c in df["Configuration"]]
    ratios = df["Ratio_%"].astype(float).to_numpy()
    colors = ["C2", "C0", "C1"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(np.arange(len(labels)), ratios, color=colors, width=0.65)
    ax.axhline(100, color="0.3", ls="--", lw=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Yield ratio vs ENTSO-E (%)")
    ax.set_title(
        "National onshore wind 2023 — library test vs fleet\n"
        "(matched ERA5 cutout; free-stream, no wakes)"
    )
    ax.set_ylim(0, max(ratios) * 1.18)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, r in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            r + 2,
            f"{r:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    save_vector_figure(fig, "wind_library_test")
    plt.close(fig)


def plot_solar_residual_budget() -> None:
    """Waterfall-style national solar residual after physics derates + literature BTM."""
    path = Path(result_path("national_solar_derates_vs_entsoe.csv"))
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run investigate_national_solar_derates.py.")
    df = pd.read_csv(path)
    full = df[df["Window"] == "Full_8760"]

    def gwh(config_substr: str) -> float:
        row = full[full["Configuration"].str.contains(config_substr, regex=False)]
        if row.empty:
            raise ValueError(f"No derate row matching {config_substr}")
        return float(row.iloc[0]["Sim_GWh"])

    entsoe = float(full.iloc[0]["ENTSOE_GWh"])
    pv = gwh("PVLib MaStR × η_inv × aging")
    atl = gwh("Atlite × aging")
    btm_twh = 8.20
    btm_gwh = btm_twh * 1000.0

    # Persist tidy budget used by the figure / paper table cross-check
    budget = pd.DataFrame(
        [
            {
                "Quantity": "Simulated generation (after physics)",
                "PVLib_TWh": round(pv / 1000.0, 1),
                "Atlite_TWh": round(atl / 1000.0, 1),
            },
            {
                "Quantity": "ENTSO-E solar feed-in",
                "PVLib_TWh": round(entsoe / 1000.0, 1),
                "Atlite_TWh": round(entsoe / 1000.0, 1),
            },
            {
                "Quantity": "Gap to feed-in",
                "PVLib_TWh": round((pv - entsoe) / 1000.0, 1),
                "Atlite_TWh": round((atl - entsoe) / 1000.0, 1),
            },
            {
                "Quantity": "Literature BTM self-consumption",
                "PVLib_TWh": btm_twh,
                "Atlite_TWh": btm_twh,
            },
            {
                "Quantity": "Gap after BTM (unattributed)",
                "PVLib_TWh": round((pv - entsoe) / 1000.0 - btm_twh, 1),
                "Atlite_TWh": round((atl - entsoe) / 1000.0 - btm_twh, 1),
            },
        ]
    )
    budget_path = Path(result_path("solar_residual_budget.csv"))
    budget.to_csv(budget_path, index=False)
    copy_to_manuscript(budget_path)

    categories = ["Simulated\n(after physics)", "ENTSO-E\nfeed-in", "BTM\n(known)", "Unattributed\nresidual"]
    # Stacked composition of simulated energy for each library
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, sim_gwh, title, color in (
        (axes[0], pv, "PVLib MaStR ×η_inv×aging", "C0"),
        (axes[1], atl, "Atlite CSi ×aging", "C2"),
    ):
        gap = sim_gwh - entsoe
        unattr = gap - btm_gwh
        vals = [sim_gwh / 1000.0, entsoe / 1000.0, btm_twh, unattr / 1000.0]
        bar_colors = [color, "0.35", "C1", "0.7"]
        bars = ax.bar(np.arange(len(categories)), vals, color=bar_colors, width=0.7)
        ax.set_xticks(np.arange(len(categories)))
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylabel("Energy (TWh)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.8,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_ylim(0, max(sim_gwh / 1000.0, entsoe / 1000.0) * 1.2)

    fig.suptitle(
        "National solar 2023 residual vs ENTSO-E feed-in (matched ERA5)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    save_vector_figure(fig, "solar_residual_budget")
    plt.close(fig)


def plot_national_solar_duration_and_scatters() -> None:
    ts = pd.read_csv(result_path("matched_era5_timeseries.csv"), index_col=0, parse_dates=True)
    eta_age = FLEET_AGING_FALLBACK
    ent = ts["entsoe_solar"].astype(float)
    pv = ts["pvlib_mastr_solar"].astype(float) * INVERTER_EFFICIENCY * eta_age
    atl = ts["atlite_solar"].astype(float) * eta_age

    # Daily-max duration (vector)
    daily = pd.DataFrame(
        {
            "entsoe": ent.groupby(ent.index.normalize()).max(),
            "pvlib": pv.groupby(pv.index.normalize()).max(),
            "atlite": atl.groupby(atl.index.normalize()).max(),
        }
    ).dropna()
    n = len(daily)
    x = np.arange(1, n + 1) / n * 100.0
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, np.sort(daily["entsoe"].to_numpy())[::-1], color="black", lw=2.0, label="ENTSO-E feed-in (daily max)")
    ax.plot(
        x,
        np.sort(daily["pvlib"].to_numpy())[::-1],
        color="C0",
        lw=1.6,
        ls="--",
        label="PVLib MaStR ×η_inv×aging (daily max)",
    )
    ax.plot(
        x,
        np.sort(daily["atlite"].to_numpy())[::-1],
        color="C2",
        lw=1.6,
        ls="-.",
        label="Atlite CSi ×aging (daily max)",
    )
    # Illustrative scalar overlays (dashed lighter)
    ax.plot(
        x,
        np.sort((daily["pvlib"] * FEED_IN_SCALE).to_numpy())[::-1],
        color="C0",
        lw=1.0,
        ls=":",
        alpha=0.8,
        label=f"PVLib ×{FEED_IN_SCALE:g} (illustrative)",
    )
    ax.plot(
        x,
        np.sort((daily["atlite"] * FEED_IN_SCALE).to_numpy())[::-1],
        color="C2",
        lw=1.0,
        ls=":",
        alpha=0.8,
        label=f"Atlite ×{FEED_IN_SCALE:g} (illustrative)",
    )
    ax.set_xlabel("Percentage of days (%)")
    ax.set_ylabel("Daily maximum power (MW)")
    ax.set_title(
        "Germany national solar 2023 — duration curve of daily maximum\n"
        f"(n={n} days; sims = generation after inverter/aging; ENTSO-E = public feed-in)"
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8)
    fig.tight_layout()
    save_vector_figure(fig, "national_daily_max_duration")
    plt.close(fig)

    # Scatters (PNG)
    for scale, stem in (
        (None, "national_solar_scatter"),
        (FEED_IN_SCALE, "national_solar_scatter_scaled"),
    ):
        idx = ent.index.intersection(pv.index).intersection(atl.index)
        e = ent.loc[idx].to_numpy(dtype=float)
        p = pv.loc[idx].to_numpy(dtype=float)
        a = atl.loc[idx].to_numpy(dtype=float)
        note = " (physics derates only)"
        if scale is not None:
            p = p * scale
            a = a * scale
            note = f" ×{scale:g} (illustrative)"
        fig, ax = plt.subplots(figsize=(7.5, 7.0))
        ax.scatter(e, p, s=6, alpha=0.12, color="C0", label="PVLib MaStR ×η_inv×aging" + note, rasterized=True)
        ax.scatter(e, a, s=6, alpha=0.12, color="C2", label="Atlite CSi ×aging" + note, rasterized=True)
        lim = float(np.nanmax([e.max(), p.max(), a.max()]) * 1.02)
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("ENTSO-E solar feed-in (MW)")
        ax.set_ylabel("Simulated generation (MW)")
        ax.set_title("Germany national solar 2023 — hourly scatter vs feed-in")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", framealpha=0.95, fontsize=8)
        fig.tight_layout()
        save_raster_figure(fig, stem, dpi=150)
        plt.close(fig)


def plot_tso_matched_weeks_from_cache() -> None:
    """Rebuild 4-panel TSO week figures from cached sample weeks (offline-friendly)."""
    sample_path = Path(result_path("matched_era5_tso_sample_weeks.csv"))
    parquet_path = Path(result_path("matched_era5_tso_timeseries.parquet"))
    if sample_path.exists():
        tso = pd.read_csv(sample_path, index_col=0, parse_dates=True)
        logger.info("TSO week panels from %s", sample_path.name)
    elif parquet_path.exists():
        tso = pd.read_parquet(parquet_path)
        if not isinstance(tso.index, pd.DatetimeIndex):
            tso.index = pd.to_datetime(tso.index)
        logger.info("TSO week panels from %s", parquet_path.name)
    else:
        logger.warning(
            "Skipping tso_matched_week_* — missing sample weeks CSV and parquet "
            "(run plot_weeks after validate_matched_era5)."
        )
        return

    # Import locally to avoid circular imports with plot_week_and_diurnal helpers
    from plot_week_and_diurnal import (
        WEEK_SOLAR,
        WEEK_WIND,
        plot_tso_four_panel,
    )

    plot_tso_four_panel(
        tso,
        tech="wind",
        start=WEEK_WIND[0],
        end=WEEK_WIND[1],
        stem="tso_matched_week_wind",
        ylabel="Power (MW)",
        title=(
            f"German TSO onshore wind — matched ERA5 sample week "
            f"{WEEK_WIND[0]} to {WEEK_WIND[1]}"
        ),
    )
    plot_tso_four_panel(
        tso,
        tech="solar",
        start=WEEK_SOLAR[0],
        end=WEEK_SOLAR[1],
        stem="tso_matched_week_solar",
        ylabel="Power (MW)",
        title=(
            f"German TSO solar — matched ERA5 sample week "
            f"{WEEK_SOLAR[0]} to {WEEK_SOLAR[1]}"
        ),
    )


def main() -> None:
    ensure_results_dir()
    logger.info("Exporting paper figures (SVG→PDF for vector; PNG for scatters)")

    comp = _load_juelich_comp()
    plot_juelich_fortnight(
        comp,
        "2023-01-08",
        "2023-01-21",
        "Campus Jülich PV, 8–21 January 2023 (matched ERA5; inverter/aging derates)",
        "juelich_january_2weeks",
        snow_span=("2023-01-19", "2023-01-22"),
    )
    plot_juelich_fortnight(
        comp,
        "2023-06-05",
        "2023-06-18",
        "Campus Jülich PV, 5–18 June 2023 (matched ERA5; inverter/aging derates)",
        "juelich_june_2weeks",
    )
    plot_juelich_hourly_duration(comp)
    plot_juelich_daily_max_duration(comp)
    plot_juelich_scatter(comp)

    plot_seasonal_summary()
    plot_tso_yield_ratios()
    plot_wind_library_test()
    plot_solar_residual_budget()
    plot_national_solar_duration_and_scatters()
    plot_tso_matched_weeks_from_cache()

    # Ensure any pre-existing scatter PNGs from pipeline are still in latex/data
    for stem in PAPER_RASTER_STEMS:
        png = Path(result_path(f"{stem}.png"))
        if png.exists():
            copy_to_manuscript(png)

    logger.info("Vector stems: %s", ", ".join(PAPER_VECTOR_STEMS))
    logger.info("Raster stems: %s", ", ".join(PAPER_RASTER_STEMS))
    logger.info("Done — paper should \\includegraphics PDF for vector, PNG for scatter.")


if __name__ == "__main__":
    main()
