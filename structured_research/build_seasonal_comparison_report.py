"""
build_seasonal_comparison_report.py
===================================
Paper seasonal / multi-scale matrix from plant CSVs + matched ERA5.

Reads: juelich / kelmarsh seasonal CSVs + matched_era5_vs_entsoe.csv
       (+ matched_era5_daynight_wind.csv for national day/night note)

Writes:
  results/main_seasonal_comparison_matrix.csv
  results/main_seasonal_comparison_report.md
"""

from __future__ import annotations

from pathlib import Path

import sys
import logging
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import ZONE_LABEL
from utils import ensure_results_dir, result_path, SEASON_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("seasonal_report")


def _load_csv(name: str) -> pd.DataFrame | None:
    path = result_path(name)
    if not path.exists():
        logger.warning("Missing artifact: %s", path)
        return None
    return pd.read_csv(path)


def _explain_solar(ratio_pvlib: float, ratio_atlite: float, corr_pvlib: float, corr_atlite: float, season: str) -> str:
    parts = []
    if ratio_pvlib > 120 or ratio_atlite > 120:
        parts.append("High vs ENTSO-E feed-in (generation ≠ feed-in; BTM/timing residual).")
    if abs(ratio_pvlib - ratio_atlite) < 5:
        parts.append("Libraries agree closely on energy.")
    elif abs(ratio_pvlib - 100) < abs(ratio_atlite - 100):
        parts.append("PVLib closer on energy.")
    else:
        parts.append("Atlite closer on energy.")
    if corr_pvlib >= corr_atlite:
        parts.append("PVLib higher or equal r.")
    else:
        parts.append("Atlite higher r.")
    return " ".join(parts)


def _explain_wind(ratio_wpl: float, ratio_atlite: float, corr_wpl: float, corr_atlite: float, season: str, site: str) -> str:
    parts = []
    if "Kelmarsh" in site:
        parts.append("Kelmarsh is a UK SCADA check; not a German roughness calibration.")
    if abs(ratio_wpl - 100) < abs(ratio_atlite - 100):
        parts.append("Windpowerlib closer on energy.")
    else:
        parts.append("Atlite closer on energy (fleet proxy vs MaStR fleet — not library-only).")
    if corr_wpl >= corr_atlite:
        parts.append("WPL higher or equal r.")
    else:
        parts.append("Atlite higher r.")
    return " ".join(parts)


def build_matrix() -> pd.DataFrame:
    rows = []

    juelich = _load_csv("juelich_seasonal_comparison.csv")
    if juelich is not None:
        for season in list(SEASON_LABELS.values()) + ["Full Year"]:
            sub = juelich[juelich["Season"] == season]
            if sub.empty:
                continue
            pv = sub[sub["Model"].str.contains("temp-corrected|SAPM|PVLib", case=False, regex=True)]
            at = sub[sub["Model"].str.contains("Atlite", case=False)]
            if pv.empty or at.empty:
                continue
            pv, at = pv.iloc[0], at.iloc[0]
            rows.append({
                "Scale": "Single",
                "Site": "Juelich_PV",
                "Technology": "Solar",
                "Season": season,
                "Actual_ref": "Measured AC",
                "PVLib_or_WPL_Ratio_%": pv["Ratio_%"],
                "PVLib_or_WPL_Corr": pv["Correlation"],
                "Atlite_Ratio_%": at["Ratio_%"],
                "Atlite_Corr": at["Correlation"],
                "Closer_yield": "PVLib" if abs(pv["Ratio_%"] - 100) <= abs(at["Ratio_%"] - 100) else "Atlite",
                "Higher_corr": "PVLib" if pv["Correlation"] >= at["Correlation"] else "Atlite",
                "Explanation": _explain_solar(pv["Ratio_%"], at["Ratio_%"], pv["Correlation"], at["Correlation"], season),
            })

    kel = _load_csv("kelmarsh_seasonal_comparison.csv")
    if kel is not None:
        for season in list(SEASON_LABELS.values()) + ["Full Year"]:
            sub = kel[kel["Season"] == season]
            if sub.empty:
                continue
            wpl = sub[sub["Model"].str.contains("Windpowerlib", case=False)]
            at = sub[sub["Model"].str.contains("Atlite", case=False)]
            if wpl.empty or at.empty:
                continue
            wpl, at = wpl.iloc[0], at.iloc[0]
            rows.append({
                "Scale": "Single",
                "Site": "Kelmarsh_Wind",
                "Technology": "Wind",
                "Season": season,
                "Actual_ref": "SCADA",
                "PVLib_or_WPL_Ratio_%": wpl["Ratio_%"],
                "PVLib_or_WPL_Corr": wpl["Correlation"],
                "Atlite_Ratio_%": at["Ratio_%"],
                "Atlite_Corr": at["Correlation"],
                "Closer_yield": "Windpowerlib" if abs(wpl["Ratio_%"] - 100) <= abs(at["Ratio_%"] - 100) else "Atlite",
                "Higher_corr": "Windpowerlib" if wpl["Correlation"] >= at["Correlation"] else "Atlite",
                "Explanation": _explain_wind(wpl["Ratio_%"], at["Ratio_%"], wpl["Correlation"], at["Correlation"], season, "Kelmarsh"),
            })

    # National + TSO from matched ERA5 (full-year metrics in vs_entsoe)
    matched = _load_csv("matched_era5_vs_entsoe.csv")
    if matched is not None:
        for _, r in matched.iterrows():
            tech = r["Technology"]
            scale = r["Scale"]
            model = str(r["Model"])
            is_atlite = model.lower().startswith("atlite")
            is_wpl = "windpowerlib" in model.lower() or "wpl" in model.lower()
            is_pvlib = "pvlib" in model.lower() and "mastr" in model.lower()
            if tech == "Wind" and not (is_atlite or is_wpl):
                continue
            if tech == "Solar" and not (is_atlite or is_pvlib):
                continue
            site = "Germany" if scale == "Germany national" else ZONE_LABEL.get(scale, scale)
            rows.append({
                "Scale": "National" if scale == "Germany national" else "TSO",
                "Site": site,
                "Technology": tech,
                "Season": "Full Year",
                "Actual_ref": "ENTSO-E feed-in",
                "PVLib_or_WPL_Ratio_%": r["Ratio_%"] if not is_atlite else None,
                "PVLib_or_WPL_Corr": r["Correlation"] if not is_atlite else None,
                "Atlite_Ratio_%": r["Ratio_%"] if is_atlite else None,
                "Atlite_Corr": r["Correlation"] if is_atlite else None,
                "Closer_yield": "",
                "Higher_corr": "",
                "Explanation": "Matched ERA5 cutout; wind fleet not equivalent (MaStR vs V112@80 m).",
                "_model": model,
            })

        # Pivot wind/solar pairs into single rows per site/tech
        df_raw = pd.DataFrame(rows)
        # Keep plant rows as-is; collapse matched rows
        plant = df_raw[df_raw["Scale"].isin(["Single"])].copy()
        matched_rows = df_raw[df_raw["Scale"].isin(["National", "TSO"])].copy()
        merged = []
        if not matched_rows.empty:
            for (scale, site, tech), g in matched_rows.groupby(["Scale", "Site", "Technology"]):
                lib = g[g["_model"].str.contains("Windpowerlib|PVLib MaStR", case=False, regex=True, na=False)]
                atl = g[g["_model"].str.startswith("Atlite", na=False)]
                if lib.empty or atl.empty:
                    continue
                lib, atl = lib.iloc[0], atl.iloc[0]
                rr_l, rr_a = float(lib["PVLib_or_WPL_Ratio_%"]), float(atl["Atlite_Ratio_%"])
                cr_l, cr_a = float(lib["PVLib_or_WPL_Corr"]), float(atl["Atlite_Corr"])
                merged.append({
                    "Scale": scale,
                    "Site": site,
                    "Technology": tech,
                    "Season": "Full Year",
                    "Actual_ref": "ENTSO-E feed-in",
                    "PVLib_or_WPL_Ratio_%": rr_l,
                    "PVLib_or_WPL_Corr": cr_l,
                    "Atlite_Ratio_%": rr_a,
                    "Atlite_Corr": cr_a,
                    "Closer_yield": ("Windpowerlib" if tech == "Wind" else "PVLib")
                    if abs(rr_l - 100) <= abs(rr_a - 100)
                    else "Atlite",
                    "Higher_corr": ("Windpowerlib" if tech == "Wind" else "PVLib")
                    if cr_l >= cr_a
                    else "Atlite",
                    "Explanation": (
                        _explain_wind(rr_l, rr_a, cr_l, cr_a, "Full Year", site)
                        if tech == "Wind"
                        else _explain_solar(rr_l, rr_a, cr_l, cr_a, "Full Year")
                    ),
                })
        out = pd.concat([plant.drop(columns=["_model"], errors="ignore"), pd.DataFrame(merged)], ignore_index=True)
        return out

    return pd.DataFrame(rows)


def write_markdown(df: pd.DataFrame, path: str) -> None:
    lines = [
        "# Main seasonal / multi-scale comparison (paper path)",
        "",
        "Sources: Jülich / Kelmarsh seasonal CSVs + `matched_era5_vs_entsoe.csv`.",
        "National/TSO rows are **full-year** on matched ERA5 (see PAPER_CONTRACT.md).",
        "",
        df.to_string(index=False),
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Wrote %s", path)


def main():
    ensure_results_dir()
    df = build_matrix()
    df.to_csv(result_path("main_seasonal_comparison_matrix.csv"), index=False)
    write_markdown(df, result_path("main_seasonal_comparison_report.md"))
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
