# Jülich PV investigation close-out (matched ERA5 cutout)

## Protocol (code: `validate_juelich_solar.py`)
- Weather: `juelich_2023.nc` for **both** PVLib and Atlite (ssrd as GHI).
- PVLib: Erbs + SAPM; then `η_inv = 0.90` × `η_age = 0.940`.
- Atlite: CSi (already `inverter_efficiency=0.90`); then `η_age` only.
- Aging prior: commission 2011-07-01, 0.5 %/y, ref mid-2023 (12.0 y).

## Main results (8760 with gaps→0)
| Config | vs meter | r |
|---|---:|---:|
| PVLib SAPM × η_inv × aging | 111.9% | 0.918 |
| Atlite × aging | 109.8% | 0.921 |
| PVLib / Atlite | 102.0% | 0.996 |

## Factor ablation (overlap hours) — correlation vs error
Constant multipliers **do not raise Pearson r** (scale-invariant). They cut bias and MAE/RMSE.

| Step | Ratio | r | MAE ↓ vs raw |
|---|---:|---:|---:|
| PVLib linear raw | 132.6% | 0.8936 | — |
| + SAPM | 129.0% | 0.8941 | 8.5% |
| + η_inv | 116.1% | 0.8941 | 26.1% |
| + aging | 109.2% | 0.8941 | 33.4% |
| Atlite × aging | 107.3% | 0.8993 | 37.5% |

## Qualitative residuals
- Snow from ~19 Jan for several days (models keep producing).
- Meter gaps filled as 0 inflate plots; ~2–3 pp annual only.
- Clear summer daily peaks still a bit high (duration-curve upper tail).
- Cloudy hours structurally over-predicted.

## Outputs for paper
- `juelich_factor_ablation.csv`
- `juelich_january_2weeks.png`, `juelich_june_2weeks.png`
- `juelich_hourly_duration.png`, `juelich_daily_max_duration.png`
- Copied to `text/data/` for manuscript appendix.
