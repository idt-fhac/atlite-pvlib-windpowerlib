# National matched-ERA5 solar + aging/inverter

Fleet capacity-weighted aging (MaStR mid-2023): η_age=**0.9656** (mean age 6.88 y at 0.5%/y).
PVLib total factor η_inv×η_age=**0.869**. Atlite: aging only (η_inv already in CSi).

## vs ENTSO-E feed-in (full year)

| Configuration | Ratio |
|---|---:|
| PVLib × η_inv × aging | **160.5%** |
| Atlite × aging | **156.9%** |

vs feed-in+BTM (8.2 TWh): PVLib **139.8%**, Atlite **136.6%**.

## Takeaway

- Physics derates target **generation**, not ENTSO-E feed-in.
- Literature BTM closes only part of the feed-in gap; a constant energy-matching scalar (~0.62) aligns annual sums but does **not** recover the feed-in duration-curve or scatter shape — residual left for future work.
- Outputs: `national_solar_derates_vs_entsoe.csv`, `national_hourly_duration.png`, `national_daily_max_duration.png`, `national_solar_scatter.png`, `national_solar_scatter_scaled.png` (final appendix daily-max with illustrative ×0.62 from `plot_tso_solar_feedin_scale.py`).
