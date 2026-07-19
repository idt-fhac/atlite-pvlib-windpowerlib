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

Copies land in text/data/ (and structured_research/results/).
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
    plot_national_solar_duration_and_scatters()

    # Ensure any pre-existing scatter PNGs from pipeline are still in text/data
    for stem in PAPER_RASTER_STEMS:
        png = Path(result_path(f"{stem}.png"))
        if png.exists():
            copy_to_manuscript(png)

    logger.info("Vector stems: %s", ", ".join(PAPER_VECTOR_STEMS))
    logger.info("Raster stems: %s", ", ".join(PAPER_RASTER_STEMS))
    logger.info("Done — paper should \\includegraphics PDF for vector, PNG for scatter.")


if __name__ == "__main__":
    main()
