# TSO solar capacity + generation breakdown (2023)

## Capacity (MaStR vs ENTSO-E)

ENTSO-E TSO solar capacity is published only for **50Hertz** and **Amprion** at 2024-01-01; TenneT/Transnet end-2023 values below are **imputed** from DE_LU residual using start-2023 shares. Start-2023 TSO values from local `entsoe_capacity.parquet` (sum 63.5 GW ≈ DE_LU 63.4 GW).

| TSO | ENTSO-E start GW | ENTSO-E end GW | MaStR start | MaStR mid | MaStR end | MaStR/ENTSOE start | MaStR/ENTSOE end |
|---|---:|---:|---:|---:|---:|---:|---:|
| DE_50HZ | 16.5 | 20.7 | 18.0 | 19.4 | 21.0 | 109% | 101% |
| DE_AMPRION | 14.5 | 17.6 | 12.7 | 14.5 | 16.6 | **88%** | **95%** |
| DE_TENNET | 24.4 | 29.1† | 28.5 | 31.6 | 35.1 | **117%** | **121%**† |
| DE_TRANSNET | 8.1 | 9.7† | 8.3 | 9.2 | 10.3 | 102% | 107%† |
| DE national | 63.5 | 77.0 | 67.6 | 74.8 | 83.0 | 106% | 108% |

† Imputed. National MaStR end / DE_LU end = **107.8%** — inventory is slightly high but in the acceptable ballpark (not a 40%+ error).

**Note:** Earlier `tso_capacity_comparison.csv` mixed MaStR ~end-year (~83 GW) with ENTSO-E **start**-year (~63 GW) → inflated TSO ratios (esp. TenneT 144%). Aligned dates fix most of that.

## Generation vs feed-in after η_inv × aging (matched ERA5)

| TSO | PVLib energy ratio | Atlite energy ratio | Cap ratio end | FLH ratio (sim mid-cap / feed-in mean-cap) |
|---|---:|---:|---:|---:|
| DE_50HZ | 145% | 140% | ~101% | ~135–139% |
| DE_AMPRION | 138% | 139% | ~95% | ~152–154% |
| DE_TENNET | **187%** | **181%** | ~121%† | ~153–158% |
| DE_TRANSNET | 156% | 154% | ~107%† | ~148–150% |
| DE national | 161% | 157% | 108% | ~148–151% |

Correlations stay high everywhere (~0.91–0.94).

## Reading

1. **Capacity is mostly OK nationally (~106–108%)**; Amprion is slightly low in MaStR, TenneT somewhat high.
2. **Energy overestimate vs feed-in is everywhere** (~140–160%), not a single-TSO bug.
3. **TenneT is the outlier on energy (~180–187%)**, partly capacity (~+20%) and partly the same FLH gap as others (~+50–55% after capacity normalization).
4. Capacity-normalized FLH ratios remain ~135–160% in all zones → leftover is **feed-in ≠ generation** (BTM, timing, curtailment), not a broken TSO capacity map alone.

Outputs: `tso_solar_capacity_check.csv`, `tso_solar_derates_breakdown.csv`.
