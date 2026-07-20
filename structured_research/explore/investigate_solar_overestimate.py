"""
Investigate national solar overestimation vs ENTSO-E feed-in.

Uses existing 2023 timeseries (no full re-sim). Levers tested:
  1) system_losses / inverter derate (PVLib currently has none nationally)
  2) year-end vs mid-year capacity timing (MaStR growth during 2023)
  3) literature BTM self-consumption added to ENTSO-E target
  4) empirical Jülich plant derate prior
  5) midday peak ratio (curtailment / inverter hint)

Outputs → structured_research/results/:
  solar_overest_sensitivity.csv
  solar_overest_budget.csv
  solar_overest_investigation.md
"""

from __future__ import annotations

from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import ensure_results_dir, result_path, calculate_metrics

# Fraunhofer ISE literature BTM (TWh) — same as paper.tex
BTM_TWH = 8.2

# MaStR capacity trajectory 2023 (GW) from mastr.solar_extended query
GW_START = 67.608242
GW_ADDED_MONTHLY = [
    1.021001, 0.927214, 1.265984, 1.191448, 1.361171, 1.393043,
    1.509611, 1.397448, 1.231916, 1.531533, 1.434913, 1.162094,
]

# Jülich empirical derates (temp-corrected / actual)
JUELICH_DERATE = 0.9483  # system losses temp-corrected ≈ 0.948


def capacity_scale_by_month(index: pd.DatetimeIndex) -> pd.Series:
    """Scale factor = mid-month capacity / year-end capacity."""
    caps = [GW_START]
    for a in GW_ADDED_MONTHLY:
        caps.append(caps[-1] + a)
    gw_end = caps[-1]
    mid = [(caps[i] + caps[i + 1]) / 2.0 for i in range(12)]
    month_scale = {m + 1: mid[m] / gw_end for m in range(12)}
    return pd.Series([month_scale[ts.month] for ts in index], index=index)


def metrics_vs(sim: pd.Series, act: pd.Series, label: str) -> dict:
    m = calculate_metrics(sim, act)
    return {
        "case": label,
        "Ratio_%": round(m["ratio"], 2),
        "Corr": round(m["corr"], 4),
        "Sim_TWh": round(m["sim_sum"] / 1e6, 2),
        "Target_TWh": round(m["act_sum"] / 1e6, 2),
        "Gap_TWh": round((m["sim_sum"] - m["act_sum"]) / 1e6, 2),
    }


def main():
    ensure_results_dir()
    ts = pd.read_csv(
        result_path("annual_seasonal_timeseries.csv"),
        index_col=0,
        parse_dates=True,
    )
    matched = pd.read_csv(
        result_path("matched_era5_timeseries.csv"),
        index_col=0,
        parse_dates=True,
    )

    entsoe = ts["entsoe_solar"].astype(float)
    oeds = ts["oeds_solar"].astype(float)       # ECMWF + PVLib, no national losses
    atlite = ts["atlite_solar"].astype(float)   # ERA5 + CSi (inverter_eff=0.9 built-in)
    # matched ERA5 PVLib (also no losses)
    pv_era5 = matched["pvlib_mastr_solar"].astype(float).reindex(ts.index)
    atl_era5 = matched["atlite_solar"].astype(float).reindex(ts.index)

    # ENTSO-E + flat BTM (spread BTM energy proportional to feed-in daytime shape)
    # Simple: add constant energy via scaling feed-in up by (55.2+8.2)/55.2
    feed_twh = entsoe.sum() / 1e6
    scale_btm = (feed_twh + BTM_TWH) / feed_twh
    entsoe_btm = entsoe * scale_btm

    cap_scale = capacity_scale_by_month(ts.index)
    avg_cap_ratio = float(cap_scale.mean())

    rows = []
    targets = {
        "ENTSO-E feed-in": entsoe,
        "ENTSO-E + BTM 8.2 TWh": entsoe_btm,
    }

    # Baseline + lever combinations for OEDS (ECMWF)
    variants = {
        "OEDS raw (ECMWF, no losses)": oeds,
        f"OEDS × Jülich derate ({JUELICH_DERATE:.3f})": oeds * JUELICH_DERATE,
        "OEDS × inverter 0.96": oeds * 0.96,
        "OEDS × PVWatts-like 0.86": oeds * 0.86,
        "OEDS × capacity timing": oeds * cap_scale,
        "OEDS × timing × 0.86": oeds * cap_scale * 0.86,
        "OEDS × timing × Jülich": oeds * cap_scale * JUELICH_DERATE,
        "OEDS × timing × 0.90": oeds * cap_scale * 0.90,
        "Atlite raw (ERA5, η_inv=0.9)": atlite,
        f"Atlite × Jülich extra ({JUELICH_DERATE:.3f})": atlite * JUELICH_DERATE,
        "Atlite × capacity timing": atlite * cap_scale,
        "Atlite × timing × 0.95": atlite * cap_scale * 0.95,
        "Atlite × timing × 0.86/0.9": atlite * cap_scale * (0.86 / 0.9),  # extra beyond built-in inv
        "PVLib MaStR on ERA5 raw": pv_era5,
        "PVLib ERA5 × timing × 0.86": pv_era5 * cap_scale * 0.86,
    }

    for tname, target in targets.items():
        for vname, sim in variants.items():
            r = metrics_vs(sim, target, f"{vname} | vs {tname}")
            r["model"] = vname
            r["target"] = tname
            rows.append(r)

    sens = pd.DataFrame(rows)
    sens.to_csv(result_path("solar_overest_sensitivity.csv"), index=False)

    # Compact budget for paper-relevant cases
    budget_cases = [
        ("OEDS raw", oeds, entsoe),
        ("OEDS after BTM target", oeds, entsoe_btm),
        ("OEDS × timing", oeds * cap_scale, entsoe),
        ("OEDS × timing × 0.86", oeds * cap_scale * 0.86, entsoe),
        ("OEDS × timing × 0.86 | +BTM", oeds * cap_scale * 0.86, entsoe_btm),
        ("Atlite raw", atlite, entsoe),
        ("Atlite × timing × (0.86/0.9)", atlite * cap_scale * (0.86 / 0.9), entsoe_btm),
    ]
    budget = pd.DataFrame([metrics_vs(s, t, n) for n, s, t in budget_cases])
    budget.to_csv(result_path("solar_overest_budget.csv"), index=False)

    # Midday peak diagnostic (Jun–Aug, hour 11–13 UTC)
    summer = ts.index.month.isin([6, 7, 8]) & ts.index.hour.isin([11, 12, 13])
    # Find derate that hits 100% vs feed-in and vs feed-in+BTM
    def needed_derate(sim, target):
        return float(target.sum() / sim.sum())

    need = pd.DataFrame([
        {"stack": "OEDS", "vs": "feed-in", "derate_to_100%": round(needed_derate(oeds, entsoe), 3)},
        {"stack": "OEDS", "vs": "feed-in+BTM", "derate_to_100%": round(needed_derate(oeds, entsoe_btm), 3)},
        {"stack": "OEDS×timing", "vs": "feed-in+BTM",
         "derate_to_100%": round(needed_derate(oeds * cap_scale, entsoe_btm), 3)},
        {"stack": "Atlite", "vs": "feed-in", "derate_to_100%": round(needed_derate(atlite, entsoe), 3)},
        {"stack": "Atlite", "vs": "feed-in+BTM", "derate_to_100%": round(needed_derate(atlite, entsoe_btm), 3)},
        {"stack": "Atlite×timing", "vs": "feed-in+BTM",
         "derate_to_100%": round(needed_derate(atlite * cap_scale, entsoe_btm), 3)},
    ])

    # Best OEDS cases for summary table
    focus = sens[
        sens["model"].isin([
            "OEDS raw (ECMWF, no losses)",
            "OEDS × PVWatts-like 0.86",
            "OEDS × capacity timing",
            "OEDS × timing × 0.86",
            "OEDS × timing × 0.90",
            "Atlite raw (ERA5, η_inv=0.9)",
            "Atlite × capacity timing",
            "Atlite × timing × 0.86/0.9",
        ])
    ]

    lines = [
        "# Why national solar is overestimated",
        "",
        "## Code facts",
        "- **National PVLib/OEDS** uses ideal DC: `P = POA × capacity` with **no** inverter, "
        "soiling, mismatch, or availability losses (`validate_germany_national.py` / TSO scripts).",
        "- **Atlite CSi** already applies `inverter_efficiency: 0.9` (see `CSi.yaml`); still high vs feed-in.",
        "- **Jülich plant** (measured AC): temp-corrected PVLib needs only ~**5%** empirical derate "
        f"({JUELICH_DERATE:.3f}) → physical model is roughly OK when geometry and AC meter match.",
        "- **Capacity filter** uses year-end MaStR (`Inbetriebnahmedatum ≤ 2023-12-31`) for every hour. "
        f"DB growth 2023: {GW_START:.1f} → {GW_START + sum(GW_ADDED_MONTHLY):.1f} GW; "
        f"time-average / year-end ≈ **{avg_cap_ratio:.3f}** (~{100*(1-avg_cap_ratio):.0f}% energy inflate if ignored).",
        "- MaStR end-of-year capacity ≈ ENTSO-E nameplate (paper ~97%) → **not** plant over-counting.",
        "",
        "## Gap decomposition (OEDS ECMWF 81.8 TWh → ENTSO-E 55.2 TWh)",
        f"1. **BTM self-consumption (literature):** +{BTM_TWH} TWh to target → target ≈ {feed_twh + BTM_TWH:.1f} TWh.",
        f"2. **Capacity timing:** ×{avg_cap_ratio:.3f} on sim (~{(1-avg_cap_ratio)*oeds.sum()/1e6:.1f} TWh).",
        "3. **System losses:** national PVLib missing ~10–14% typical (inverter+soiling+mismatch+availability). "
        "PVWatts-like **0.86** is a standard prior; Jülich alone only supports ~0.95.",
        "4. **Residual:** weather-product bias, curtailment, snow, orientation/aggregation — after (1–3) "
        "often within ~0–10% depending on derate choice.",
        "",
        "## Derate needed for exact 100% energy match",
        need.to_string(index=False),
        "",
        "## Key sensitivity results",
        focus.to_string(index=False),
        "",
        "## Compact budget",
        budget.to_string(index=False),
        "",
        f"## Summer noon (Jun–Aug h11–13 UTC) mean MW ratio",
        f"OEDS/ENTSO-E = {float(oeds[summer].mean()/entsoe[summer].mean()):.2f}; "
        f"Atlite/ENTSO-E = {float(atlite[summer].mean()/entsoe[summer].mean()):.2f}",
        "(If midday ratio ≫ annual ratio, look to inverter clipping / grid curtailment; "
        "if similar, bias is mostly scalar capacity/losses/BTM.)",
        "",
        "## Recommended parameter to improve prediction",
        "1. **Apply `system_losses≈0.86–0.90`** on national PVLib (or inverter_efficiency × residual losses). "
        "Atlite already has 0.9 inverter — add ~0.95–0.97 extra for soiling/availability if calibrating.",
        "2. **Time-vary capacity** (or scale by monthly avg/end ≈ 0.90) instead of year-end MaStR for all hours.",
        "3. **Do not treat ENTSO-E solar as generation** — compare to feed-in+BTM or report a calibrated "
        "energy factor. Plant-level Jülich shows the conversion physics is not the main national error.",
        "",
        "## What this does *not* fix",
        "- Library vs weather confounding (matched ERA5 PVLib is even higher ~102 TWh).",
        "- Hour-resolved BTM / curtailment (scalar BTM is a literature total only).",
        "",
    ]
    path = result_path("solar_overest_investigation.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(need.to_string(index=False))
    print()
    print(budget.to_string(index=False))
    print()
    print(f"avg_cap_ratio={avg_cap_ratio:.4f}")
    print("Wrote", path)


if __name__ == "__main__":
    main()
