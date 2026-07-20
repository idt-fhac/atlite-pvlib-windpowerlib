"""Seasonal structure of solar overestimation — not explainable by flat losses alone."""

from __future__ import annotations

from pathlib import Path

import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import ensure_results_dir, result_path

GW_START = 67.608242
ADDS = [
    1.021001, 0.927214, 1.265984, 1.191448, 1.361171, 1.393043,
    1.509611, 1.397448, 1.231916, 1.531533, 1.434913, 1.162094,
]
BTM_MWH = 8.2e6


def capacity_scale(index: pd.DatetimeIndex) -> pd.Series:
    caps = [GW_START]
    for a in ADDS:
        caps.append(caps[-1] + a)
    mid = [(caps[i] + caps[i + 1]) / 2.0 for i in range(12)]
    return pd.Series([mid[t.month - 1] / caps[-1] for t in index], index=index), mid, caps[-1]


def main():
    ensure_results_dir()
    ts = pd.read_csv(result_path("annual_seasonal_timeseries.csv"), index_col=0, parse_dates=True)
    e = ts["entsoe_solar"].astype(float)
    o = ts["oeds_solar"].astype(float)
    a = ts["atlite_solar"].astype(float)
    scale, mid, gw_end = capacity_scale(ts.index)

    o_t = o * scale
    o_tl = o * scale * 0.86
    a_t = a * scale
    a_tl = a * scale * (0.86 / 0.9)
    e_btm = e * ((e.sum() + BTM_MWH) / e.sum())

    seasons = {
        "DJF": ts.index.month.isin([12, 1, 2]),
        "MAM": ts.index.month.isin([3, 4, 5]),
        "JJA": ts.index.month.isin([6, 7, 8]),
        "SON": ts.index.month.isin([9, 10, 11]),
    }

    rows = []
    sims = {
        "OEDS raw": o,
        "OEDS x timing": o_t,
        "OEDS x timing x 0.86": o_tl,
        "Atlite raw": a,
        "Atlite x timing": a_t,
        "Atlite x timing x 0.86/0.9": a_tl,
    }
    for sname, mask in seasons.items():
        for mname, sim in sims.items():
            for tname, tgt in [("feed-in", e), ("feed-in+BTM", e_btm)]:
                r = 100.0 * sim[mask].sum() / tgt[mask].sum()
                rows.append({
                    "season": sname,
                    "model": mname,
                    "target": tname,
                    "ratio_%": round(r, 1),
                    "sim_TWh": round(sim[mask].sum() / 1e6, 2),
                    "tgt_TWh": round(tgt[mask].sum() / 1e6, 2),
                })
    seas = pd.DataFrame(rows)
    seas.to_csv(result_path("solar_overest_seasonal.csv"), index=False)

    monthly = []
    for m in range(1, 13):
        mask = ts.index.month == m
        monthly.append({
            "month": m,
            "cap_scale": round(mid[m - 1] / gw_end, 3),
            "ENTSOE_TWh": round(e[mask].sum() / 1e6, 2),
            "OEDS_raw_%": round(100 * o[mask].sum() / e[mask].sum(), 1),
            "OEDS_timing_%": round(100 * o_t[mask].sum() / e[mask].sum(), 1),
            "OEDS_t_x086_%": round(100 * o_tl[mask].sum() / e[mask].sum(), 1),
            "OEDS_t_x086_vsBTM_%": round(100 * o_tl[mask].sum() / e_btm[mask].sum(), 1),
            "Atlite_raw_%": round(100 * a[mask].sum() / e[mask].sum(), 1),
            "Atlite_t_x086_%": round(100 * a_tl[mask].sum() / e_btm[mask].sum(), 1),
        })
    mon = pd.DataFrame(monthly)
    mon.to_csv(result_path("solar_overest_monthly.csv"), index=False)

    # Same absolute feed-in band: winter vs summer (isolates seasonal physics)
    band = (e >= 5000) & (e <= 15000)
    band_rows = []
    for label, months in [("DJF", [12, 1, 2]), ("JJA", [6, 7, 8])]:
        mask = band & ts.index.month.isin(months)
        band_rows.append({
            "season": label,
            "n_hours": int(mask.sum()),
            "mean_ENTSOE_MW": round(float(e[mask].mean()), 0),
            "OEDS_raw_%": round(100 * o[mask].sum() / e[mask].sum(), 1),
            "OEDS_t_x086_%": round(100 * o_tl[mask].sum() / e[mask].sum(), 1),
            "Atlite_raw_%": round(100 * a[mask].sum() / e[mask].sum(), 1),
        })
    band_df = pd.DataFrame(band_rows)

    # Daytime terciles
    day = e > 500
    q = e[day].quantile([0.33, 0.66])
    terc = []
    for label, mask in [
        ("low", day & (e <= q.iloc[0])),
        ("mid", day & (e > q.iloc[0]) & (e <= q.iloc[1])),
        ("high", day & (e > q.iloc[1])),
    ]:
        terc.append({
            "tercile": label,
            "OEDS_raw_%": round(100 * o[mask].sum() / e[mask].sum(), 1),
            "OEDS_t_x086_%": round(100 * o_tl[mask].sum() / e[mask].sum(), 1),
            "Atlite_raw_%": round(100 * a[mask].sum() / e[mask].sum(), 1),
        })
    terc_df = pd.DataFrame(terc)

    # Amplitude after corrections
    def amp(model, target="feed-in"):
        sub = seas[(seas.model == model) & (seas.target == target)].set_index("season")["ratio_%"]
        return sub["DJF"], sub["JJA"], sub["DJF"] - sub["JJA"]

    lines = [
        "# Solar overestimate is seasonal — not only scalar losses/capacity",
        "",
        "## Point",
        "Flat `system_losses` and year-end capacity inflate **annual** energy, but they do **not**",
        "remove the winter–summer skill gap. After `timing × 0.86`, OEDS is near ~100% annually vs",
        "feed-in+BTM, yet winter remains systematically high vs summer.",
        "",
        "## Seasonal ratios vs ENTSO-E feed-in",
    ]
    piv = seas[seas.target == "feed-in"].pivot(index="model", columns="season", values="ratio_%")
    piv = piv[["DJF", "MAM", "JJA", "SON"]]
    piv["DJF_minus_JJA_pp"] = piv["DJF"] - piv["JJA"]
    lines += [piv.to_string(), ""]

    lines += ["## Seasonal ratios vs feed-in + BTM (∝ feed-in)", ""]
    piv2 = seas[seas.target == "feed-in+BTM"].pivot(index="model", columns="season", values="ratio_%")
    piv2 = piv2[["DJF", "MAM", "JJA", "SON"]]
    piv2["DJF_minus_JJA_pp"] = piv2["DJF"] - piv2["JJA"]
    lines += [piv2.to_string(), ""]

    d0, j0, a0 = amp("OEDS raw")
    d1, j1, a1 = amp("OEDS x timing")
    d2, j2, a2 = amp("OEDS x timing x 0.86")
    d3, j3, a3 = amp("OEDS x timing x 0.86", "feed-in+BTM")
    lines += [
        "## Does timing/loss close the seasonal gap?",
        f"- OEDS raw:              DJF={d0:.1f}%  JJA={j0:.1f}%  Δ={a0:+.1f} pp",
        f"- OEDS × timing:         DJF={d1:.1f}%  JJA={j1:.1f}%  Δ={a1:+.1f} pp  "
        f"(timing *increases* winter relative bias: more capacity missing in early year)",
        f"- OEDS × timing × 0.86:  DJF={d2:.1f}%  JJA={j2:.1f}%  Δ={a2:+.1f} pp  (flat loss does not shrink Δ)",
        f"- same vs feed-in+BTM:   DJF={d3:.1f}%  JJA={j3:.1f}%  Δ={a3:+.1f} pp",
        "",
        "## Same absolute feed-in band (5–15 GW) — winter vs summer",
        "Controls for 'winter has less sun so relative errors look bigger'.",
        band_df.to_string(index=False),
        "",
        "## Daytime feed-in terciles (all year)",
        terc_df.to_string(index=False),
        "",
        "## Monthly OEDS ratios",
        mon.to_string(index=False),
        "",
        "## Interpretation — remaining seasonal drivers",
        "1. **Low-sun / diffuse physics:** Erbs split + POA model errors grow at high zenith (winter); "
        "Atlite CSi shows the same DJF≫JJA pattern → not PVLib-only.",
        "2. **Snow / soiling seasonality:** winter underperformance of real fleet vs clear-module sim "
        "(snow cover, dirt) — a flat annual loss under-derates winter.",
        "3. **BTM seasonality:** self-consumption share likely differs by season (midday load overlap vs heating). "
        "A scalar +8.2 TWh proportional to feed-in does not capture that. Needs seasonal BTM, not annual total.",
        "4. **Irradiance product bias** may itself be seasonal (ECMWF/ERA5 winter GHI errors).",
        "5. Capacity timing moves annual energy but **widens** DJF−JJA slightly (more correction in H1).",
        "",
        "## What to try next (seasonal levers, not more flat derate)",
        "- Winter-only or albedo/snow derate; or month-varying `system_losses`.",
        "- Compare sim GHI vs ground pyranometer network (DWD) by season — isolate weather vs conversion.",
        "- Residual after matching irradiance: library/orientation/snow.",
        "- Do **not** claim 0.86 losses 'solve' solar — they calibrate annual energy only.",
        "",
    ]
    path = result_path("solar_overest_seasonal.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(piv.to_string())
    print()
    print(piv2.to_string())
    print()
    print(band_df.to_string(index=False))
    print()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
