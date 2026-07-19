"""
Apply Jülich-style inverter + fleet aging to matched-ERA5 national solar series.

Requires prior run of validate_matched_era5.py (matched_era5_timeseries.csv).
Queries MaStR for capacity-weighted η_age. Compares to ENTSO-E feed-in.

Outputs → results/:
  national_solar_derates_vs_entsoe.csv
  national_solar_derates_seasonal.csv
  juelich_full_vs_daytime_metrics.csv
  national_solar_derates_investigation.md
  national_hourly_duration.png
  national_daily_max_duration.png
  national_solar_scatter.png          (physics derates vs ENTSO-E)
  national_solar_scatter_scaled.png   (illustrative ×0.62 vs ENTSO-E)
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import (
    AGING_RATE_PER_YEAR,
    AGING_REF_DATE,
    FEED_IN_SCALE,
    FLEET_AGING_FALLBACK,
    INVERTER_EFFICIENCY,
    copy_to_manuscript,
)
from utils import (
    resolve_engine,
    calculate_metrics,
    ensure_results_dir,
    result_path,
    load_local_data,
)


DAY = list(range(7, 18))
INV = INVERTER_EFFICIENCY
RATE = AGING_RATE_PER_YEAR
REF = AGING_REF_DATE
FEED = FEED_IN_SCALE



def plot_hourly_duration(
    ent: pd.Series, pv: pd.Series, atl: pd.Series, outfile: str
) -> None:
    """National hourly power duration curve (generation sims vs ENTSO-E feed-in)."""
    idx = ent.index.intersection(pv.index).intersection(atl.index)
    e = ent.loc[idx].to_numpy(dtype=float)
    p = pv.loc[idx].to_numpy(dtype=float)
    a = atl.loc[idx].to_numpy(dtype=float)
    n = len(e)
    x = np.arange(1, n + 1) / n * 100.0

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, np.sort(e)[::-1], color="black", lw=2.0, label="ENTSO-E feed-in")
    ax.plot(
        x,
        np.sort(p)[::-1],
        color="C0",
        lw=1.6,
        ls="--",
        label="PVLib MaStR ×η_inv×aging (gen.)",
    )
    ax.plot(
        x,
        np.sort(a)[::-1],
        color="C2",
        lw=1.6,
        ls="-.",
        label="Atlite CSi ×aging (gen.)",
    )
    ax.set_xlabel("Percentage of time (%)")
    ax.set_ylabel("Power (MW)")
    ax.set_title(
        "Germany national solar 2023 — hourly duration curve\n"
        f"(n={n} h; sims = generation after inverter/aging; ENTSO-E = public feed-in)"
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved {outfile}")


def plot_daily_max_duration(
    ent: pd.Series, pv: pd.Series, atl: pd.Series, outfile: str
) -> None:
    """Duration curve of per-day maximum national power."""
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
    ax.plot(
        x,
        np.sort(daily["entsoe"].to_numpy())[::-1],
        color="black",
        lw=2.0,
        label="ENTSO-E feed-in (daily max)",
    )
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
    ax.set_xlabel("Percentage of days (%)")
    ax.set_ylabel("Daily maximum power (MW)")
    ax.set_title(
        "Germany national solar 2023 — duration curve of daily maximum\n"
        f"(n={n} days; sims = generation after inverter/aging; ENTSO-E = public feed-in)"
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved {outfile}")


def plot_national_scatter(
    ent: pd.Series,
    pv: pd.Series,
    atl: pd.Series,
    outfile: str,
    *,
    scale: float | None = None,
    title: str | None = None,
) -> None:
    """Hourly scatter: simulated national solar vs ENTSO-E feed-in."""
    idx = ent.index.intersection(pv.index).intersection(atl.index)
    e = ent.loc[idx].to_numpy(dtype=float)
    p = pv.loc[idx].to_numpy(dtype=float)
    a = atl.loc[idx].to_numpy(dtype=float)
    if scale is not None:
        p = p * scale
        a = a * scale
        scale_note = f" ×{scale:g} (illustrative)"
    else:
        scale_note = " (physics derates only)"

    max_val = float(np.nanmax([e.max(), p.max(), a.max()]))
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(e, p, s=6, alpha=0.12, color="C0", label="PVLib MaStR ×η_inv×aging" + scale_note, rasterized=True)
    ax.scatter(e, a, s=6, alpha=0.12, color="C2", label="Atlite CSi ×aging" + scale_note, rasterized=True)
    ax.plot([0, max_val], [0, max_val], color="black", ls="--", lw=1.8, label="y = x")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ENTSO-E solar feed-in (MW)")
    ax.set_ylabel("Simulated power (MW)")
    ax.set_title(
        title
        or (
            "Germany national solar 2023 — simulated vs ENTSO-E feed-in\n"
            + (
                f"Illustrative energy-matching scalar ×{scale:g}"
                if scale is not None
                else "After inverter / fleet-aging (generation)"
            )
        )
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved {outfile}")


def fleet_aging_factor(engine) -> tuple[float, float, float]:
    with engine.connect() as c:
        df = pd.read_sql(
            text(
                """
            SELECT s."Inbetriebnahmedatum" AS commission, s."Bruttoleistung" AS kw
            FROM mastr.solar_extended s
            WHERE s."EinheitBetriebsstatus" = 'In Betrieb'
              AND s."Inbetriebnahmedatum" IS NOT NULL
              AND s."Inbetriebnahmedatum" <= '2023-12-31'
              AND (s."DatumEndgueltigeStilllegung" IS NULL
                   OR s."DatumEndgueltigeStilllegung" > '2023-01-01')
            """
            ),
            c,
        )
    df["commission"] = pd.to_datetime(df["commission"], errors="coerce")
    df = df.dropna(subset=["commission"])
    df["age_y"] = (
        (pd.Timestamp(REF) - df["commission"]).dt.days / 365.25
    ).clip(lower=0, upper=40)
    eta = float(
        ((1 - RATE * df["age_y"]).clip(0.70, 1.0) * df["kw"]).sum() / df["kw"].sum()
    )
    mean_age = float((df["age_y"] * df["kw"]).sum() / df["kw"].sum())
    gw = float(df["kw"].sum() / 1e6)
    return eta, mean_age, gw


def main():
    ensure_results_dir()
    engine = resolve_engine("oeds")
    if engine is None:
        # Offline fallback: use last known fleet factor from investigation
        eta_age, mean_age, gw = 0.9656, 6.88, 83.04
        print("OFFLINE: using cached fleet η_age=0.9656")
    else:
        eta_age, mean_age, gw = fleet_aging_factor(engine)
    print(f"Fleet GW={gw:.2f} mean_age={mean_age:.2f}y η_age={eta_age:.4f}")

    ts = pd.read_csv(
        result_path("matched_era5_timeseries.csv"), index_col=0, parse_dates=True
    )
    ent = ts["entsoe_solar"].astype(float)
    pv_raw = ts["pvlib_mastr_solar"].astype(float)
    pv_uni = ts["pvlib_uniform_solar"].astype(float)
    atl_raw = ts["atlite_solar"].astype(float)

    configs = {
        "PVLib MaStR raw (ERA5+SAPM)": pv_raw,
        "PVLib MaStR × η_inv": pv_raw * INV,
        "PVLib MaStR × aging": pv_raw * eta_age,
        "PVLib MaStR × η_inv × aging": pv_raw * INV * eta_age,
        "PVLib 30/180 × η_inv × aging": pv_uni * INV * eta_age,
        "Atlite CSi raw (ERA5)": atl_raw,
        "Atlite × aging": atl_raw * eta_age,
    }

    rows = []
    for name, sim in configs.items():
        for window, mask in [
            ("Full_8760", np.ones(len(sim), dtype=bool)),
            ("Daytime_07-17", sim.index.hour.isin(DAY)),
        ]:
            m = calculate_metrics(sim[mask], ent[mask])
            rows.append(
                {
                    "Configuration": name,
                    "Window": window,
                    "Ratio_%": round(m["ratio"], 2),
                    "Corr": round(m["corr"], 4),
                    "MAE_MW": round(m["mae"], 1),
                    "RMSE_MW": round(m["rmse"], 1),
                    "Sim_GWh": round(m["sim_sum"] / 1e3, 1),
                    "ENTSOE_GWh": round(m["act_sum"] / 1e3, 1),
                    "n": int(m["n"]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(result_path("national_solar_derates_vs_entsoe.csv"), index=False)
    print(out.to_string(index=False))

    seasonal = []
    for code, months in [
        ("DJF", [12, 1, 2]),
        ("MAM", [3, 4, 5]),
        ("JJA", [6, 7, 8]),
        ("SON", [9, 10, 11]),
        ("FULL", list(range(1, 13))),
    ]:
        mask = ts.index.month.isin(months)
        for name, sim in [
            ("PVLib MaStR × η_inv × aging", pv_raw * INV * eta_age),
            ("Atlite × aging", atl_raw * eta_age),
            ("PVLib MaStR raw", pv_raw),
            ("Atlite raw", atl_raw),
        ]:
            m = calculate_metrics(sim[mask], ent[mask])
            seasonal.append(
                {
                    "Season": code,
                    "Model": name,
                    "Ratio_%": round(m["ratio"], 1),
                    "Corr": round(m["corr"], 4),
                }
            )
    pd.DataFrame(seasonal).to_csv(
        result_path("national_solar_derates_seasonal.csv"), index=False
    )

    # Jülich window table (if plant results present)
    try:
        comp = pd.read_csv(
            result_path("juelich_eview_comparison_complete.csv"),
            index_col=0,
            parse_dates=True,
        )
        act = load_local_data("juelich_actuals.parquet")
        if not isinstance(act.index, pd.DatetimeIndex):
            act = act.set_index("time")
        act.index = pd.to_datetime(act.index)
        act = act["generation"]
        jrows = []
        for label, col in [
            ("PVLib SAPM × η_inv × aging", "oeds_temp_corrected"),
            ("Atlite × aging", "atlite_sim"),
        ]:
            sim = comp[col]
            for wname, maker in [
                (
                    "Full_8760_gaps0",
                    lambda: (comp["actual_generation"], sim),
                ),
                ("Overlap_hours", lambda: (act, sim.reindex(act.index))),
                ("Daytime_overlap_07-17_act>1", None),
            ]:
                if wname.startswith("Daytime"):
                    df = pd.DataFrame({"a": act, "s": sim.reindex(act.index)}).dropna()
                    df = df[(df.index.hour.isin(DAY)) & (df.a > 1)]
                    m = calculate_metrics(df["s"], df["a"])
                else:
                    a, s = maker()
                    df = pd.DataFrame({"a": a, "s": s}).dropna()
                    m = calculate_metrics(df["s"], df["a"])
                jrows.append(
                    {
                        "Model": label,
                        "Window": wname,
                        "Ratio_%": round(m["ratio"], 2),
                        "Corr": round(m["corr"], 4),
                        "MAE_kW": round(m["mae"], 3),
                        "RMSE_kW": round(m["rmse"], 3),
                        "n": int(m["n"]),
                    }
                )
        pd.DataFrame(jrows).to_csv(
            result_path("juelich_full_vs_daytime_metrics.csv"), index=False
        )
    except FileNotFoundError:
        print("Skipping Jülich window table (plant results missing)")

    btm_gwh = 8200.0
    feed = ent.sum() / 1e3
    print(f"\nENTSO-E {feed:.1f} GWh; +BTM → {feed+btm_gwh:.1f} GWh")
    pv = pv_raw * INV * eta_age
    atl = atl_raw * eta_age
    for name, sim in [
        ("PVLib ×η_inv×age", pv),
        ("Atlite ×age", atl),
    ]:
        print(
            f"  {name}: {100*sim.sum()/ent.sum():.1f}% feed-in; "
            f"{100*sim.sum()/(ent.sum()+btm_gwh*1e3):.1f}% feed-in+BTM"
        )

    hourly_path = result_path("national_hourly_duration.png")
    daily_path = result_path("national_daily_max_duration.png")
    scatter_path = result_path("national_solar_scatter.png")
    scatter_scaled_path = result_path("national_solar_scatter_scaled.png")
    plot_hourly_duration(ent, pv, atl, hourly_path)
    plot_daily_max_duration(ent, pv, atl, daily_path)
    plot_national_scatter(ent, pv, atl, scatter_path, scale=None)
    plot_national_scatter(ent, pv, atl, scatter_scaled_path, scale=FEED)
    for src in (hourly_path, daily_path, scatter_path, scatter_scaled_path):
        copy_to_manuscript(src)

    # Note: plot_tso_solar_feedin_scale.py may overwrite national_*_duration.png
    # with an illustrative ×0.62 overlay for the appendix figure.
    md = result_path("national_solar_derates_investigation.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(
            "# National matched-ERA5 solar + aging/inverter\n\n"
            f"Fleet capacity-weighted aging (MaStR mid-2023): η_age=**{eta_age:.4f}** "
            f"(mean age {mean_age:.2f} y at 0.5%/y).\n"
            f"PVLib total factor η_inv×η_age=**{INV * eta_age:.3f}**. "
            "Atlite: aging only (η_inv already in CSi).\n\n"
            "## vs ENTSO-E feed-in (full year)\n\n"
            f"| Configuration | Ratio |\n|---|---:|\n"
            f"| PVLib × η_inv × aging | **{100*pv.sum()/ent.sum():.1f}%** |\n"
            f"| Atlite × aging | **{100*atl.sum()/ent.sum():.1f}%** |\n\n"
            f"vs feed-in+BTM (8.2 TWh): PVLib **{100*pv.sum()/(ent.sum()+btm_gwh*1e3):.1f}%**, "
            f"Atlite **{100*atl.sum()/(ent.sum()+btm_gwh*1e3):.1f}%**.\n\n"
            "## Takeaway\n\n"
            "- Physics derates target **generation**, not ENTSO-E feed-in.\n"
            "- Literature BTM closes only part of the feed-in gap; a constant energy-matching "
            "scalar (~0.62) aligns annual sums but does **not** recover the feed-in duration-curve "
            "or scatter shape — residual left for future work.\n"
            "- Outputs: `national_solar_derates_vs_entsoe.csv`, "
            "`national_hourly_duration.png`, `national_daily_max_duration.png`, "
            "`national_solar_scatter.png`, `national_solar_scatter_scaled.png` "
            "(final appendix daily-max with illustrative ×0.62 from "
            "`plot_tso_solar_feedin_scale.py`).\n"
        )
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
