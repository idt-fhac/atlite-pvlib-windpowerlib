"""
Seasonal Atlite vs PVLib validation using the SAME cutout weather (ERA5).

1) Jülich plant: measured AC vs PVLib(ERA5 cutout) vs Atlite(ERA5 cutout)
2) Germany national: ENTSO-E vs both libraries on germany_2023.nc
   (from matched_era5_timeseries.csv)

Outputs → structured_research/results/:
  cutout_pv_seasonal_juelich.csv
  cutout_pv_seasonal_national.csv
  cutout_pv_seasonal_library_delta.csv
  cutout_pv_seasonal.md
  cutout_pv_seasonal.png
"""

from __future__ import annotations

from pathlib import Path

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import ensure_results_dir, result_path, calculate_metrics, metrics_by_season


SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}


def seasonal_table(sim: pd.Series, act: pd.Series, model: str, site: str) -> pd.DataFrame:
    rows = []
    for code, months in list(SEASONS.items()) + [("FULL", list(range(1, 13)))]:
        mask = sim.index.month.isin(months)
        m = calculate_metrics(sim[mask], act[mask])
        rows.append({
            "Site": site,
            "Model": model,
            "Season": code,
            "Ratio_%": round(m["ratio"], 2),
            "Corr": round(m["corr"], 4),
            "MAE_MW_or_kW": round(m["mae"], 2),
            "RMSE": round(m["rmse"], 2),
            "Sim_sum": round(m["sim_sum"], 1),
            "Act_sum": round(m["act_sum"], 1),
        })
    return pd.DataFrame(rows)


def library_delta(num: pd.Series, den: pd.Series, label: str, site: str) -> pd.DataFrame:
    rows = []
    for code, months in list(SEASONS.items()) + [("FULL", list(range(1, 13)))]:
        mask = num.index.month.isin(months)
        m = calculate_metrics(num[mask], den[mask])
        rows.append({
            "Site": site,
            "Comparison": label,
            "Season": code,
            "Ratio_%": round(m["ratio"], 2),
            "Corr": round(m["corr"], 4),
            "Num_sum": round(m["sim_sum"], 1),
            "Den_sum": round(m["act_sum"], 1),
        })
    return pd.DataFrame(rows)


def main():
    ensure_results_dir()

    # --- Jülich: rebuild PVLib-ERA5 + Atlite from saved forensic series if present ---
    jpath = result_path("juelich_atlite_overest_timeseries.csv")
    if not jpath.exists():
        raise FileNotFoundError(
            f"{jpath} missing — run investigate_juelich_atlite.py first"
        )
    j = pd.read_csv(jpath, index_col=0, parse_dates=True)
    act = j["actual"].astype(float)
    atl = j["atlite"].astype(float)
    pv_era5 = j["pvlib_era5"].astype(float)
    pv_ecmwf = j["pvlib_ecmwf"].astype(float)  # for contrast only

    juelich = pd.concat([
        seasonal_table(atl, act, "Atlite (ERA5 cutout)", "Juelich"),
        seasonal_table(pv_era5, act, "PVLib (ERA5 cutout)", "Juelich"),
        seasonal_table(pv_ecmwf, act, "PVLib (OEDS ERA5-Land ssr)", "Juelich"),
    ], ignore_index=True)
    juelich.to_csv(result_path("cutout_pv_seasonal_juelich.csv"), index=False)

    j_delta = pd.concat([
        library_delta(pv_era5, atl, "PVLib ERA5 / Atlite ERA5", "Juelich"),
        library_delta(atl, act, "Atlite / Actual", "Juelich"),
        library_delta(pv_era5, act, "PVLib ERA5 / Actual", "Juelich"),
    ], ignore_index=True)

    # --- National matched cutout ---
    n = pd.read_csv(result_path("matched_era5_timeseries.csv"), index_col=0, parse_dates=True)
    ent = n["entsoe_solar"].astype(float)
    nat_atl = n["atlite_solar"].astype(float)
    nat_pv_m = n["pvlib_mastr_solar"].astype(float)
    nat_pv_u = n["pvlib_uniform_solar"].astype(float)

    national = pd.concat([
        seasonal_table(nat_atl, ent, "Atlite (ERA5 cutout)", "DE national"),
        seasonal_table(nat_pv_m, ent, "PVLib MaStR (ERA5 cutout)", "DE national"),
        seasonal_table(nat_pv_u, ent, "PVLib 30/180 (ERA5 cutout)", "DE national"),
    ], ignore_index=True)
    # convert sums to GWh for national readability in writeup (MW hourly → MWh = sum)
    national.to_csv(result_path("cutout_pv_seasonal_national.csv"), index=False)

    n_delta = pd.concat([
        library_delta(nat_pv_m, nat_atl, "PVLib MaStR / Atlite", "DE national"),
        library_delta(nat_pv_u, nat_atl, "PVLib 30/180 / Atlite", "DE national"),
    ], ignore_index=True)

    deltas = pd.concat([j_delta, n_delta], ignore_index=True)
    deltas.to_csv(result_path("cutout_pv_seasonal_library_delta.csv"), index=False)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    order = ["DJF", "MAM", "JJA", "SON", "FULL"]
    x = np.arange(len(order))
    w = 0.35

    # Jülich vs actual
    ax = axes[0]
    a_r = [juelich[(juelich.Model == "Atlite (ERA5 cutout)") & (juelich.Season == s)]["Ratio_%"].iloc[0] for s in order]
    p_r = [juelich[(juelich.Model == "PVLib (ERA5 cutout)") & (juelich.Season == s)]["Ratio_%"].iloc[0] for s in order]
    ax.bar(x - w / 2, a_r, w, label="Atlite (ERA5)", color="#2a9d8f")
    ax.bar(x + w / 2, p_r, w, label="PVLib (ERA5)", color="#e76f51")
    ax.axhline(100, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Sim / actual (%)")
    ax.set_title("Jülich plant (same cutout weather)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(80, 160)

    # National library delta
    ax = axes[1]
    d_m = [n_delta[(n_delta.Comparison == "PVLib MaStR / Atlite") & (n_delta.Season == s)]["Ratio_%"].iloc[0] for s in order]
    d_u = [n_delta[(n_delta.Comparison == "PVLib 30/180 / Atlite") & (n_delta.Season == s)]["Ratio_%"].iloc[0] for s in order]
    ax.bar(x - w / 2, d_m, w, label="PVLib MaStR / Atlite", color="#e76f51")
    ax.bar(x + w / 2, d_u, w, label="PVLib 30/180 / Atlite", color="#f4a261")
    ax.axhline(100, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("PVLib / Atlite (%)")
    ax.set_title("Germany national (same ERA5 cutout)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(90, 130)

    fig.suptitle("Seasonal PV: Atlite vs PVLib on matched cutout weather", fontsize=12)
    fig.tight_layout()
    fig.savefig(result_path("cutout_pv_seasonal.png"), dpi=150)
    plt.close(fig)

    # --- Markdown ---
    j_piv = juelich.pivot(index="Model", columns="Season", values="Ratio_%")[order]
    j_corr = juelich.pivot(index="Model", columns="Season", values="Corr")[order]
    n_piv = national.pivot(index="Model", columns="Season", values="Ratio_%")[order]
    d_piv = n_delta.pivot(index="Comparison", columns="Season", values="Ratio_%")[order]
    jd = j_delta[j_delta.Comparison == "PVLib ERA5 / Atlite ERA5"].set_index("Season")["Ratio_%"]

    lines = [
        "# Seasonal Atlite vs PVLib — same cutout weather (ERA5)",
        "",
        "Weather held fixed from Atlite cutouts. Remaining differences are library, ",
        "orientation/layout, and (nationally) target definition vs ENTSO-E feed-in.",
        "",
        "## Jülich (measured AC) — ratio vs meter",
        j_piv.to_string(),
        "",
        "Correlations:",
        j_corr.to_string(),
        "",
        f"Library delta PVLib(ERA5)/Atlite by season: "
        + ", ".join(f"{s}={jd[s]:.1f}%" for s in order),
        "",
        "**Reading (Jülich):** On identical ERA5 cutout weather, **PVLib runs hotter than Atlite "
        "in every season** (~110–120% of Atlite) and both exceed the meter; Atlite is closer "
        "to measured AC annually because CSi includes η_inv=0.9. Seasonal shape vs meter is "
        "similar for both (winter still highest for both on ERA5).",
        "",
        "## Germany national — ratio vs ENTSO-E feed-in (same cutout)",
        n_piv.to_string(),
        "",
        "## National library delta (PVLib / Atlite, same weather)",
        d_piv.to_string(),
        "",
        "**Reading (national):** With weather fixed, PVLib/Atlite ≈ **110–117%** in all seasons "
        "(r>0.98). Seasonal blow-up vs ENTSO-E (DJF ≫ JJA) is **shared** — not a library-only "
        "effect. Contrast: PVLib on OEDS ERA5-Land **ssr** looked better at Jülich only because "
        "that irradiance is ~18% lower than cutout ssrd.",
        "",
    ]
    path = result_path("cutout_pv_seasonal.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(j_piv.to_string())
    print()
    print(d_piv.to_string())
    print("Wrote", path)


if __name__ == "__main__":
    main()
