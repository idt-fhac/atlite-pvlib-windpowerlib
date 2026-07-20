# Night wind collapse — sensitivity summary

## Root cause (short)
ENTSO-E and Atlite (ERA5 **100 m**) both have **higher** mean wind power at night than by day.
OEDS starts from ECMWF **10 m**, which is **weaker** at night (night/day ≈ 0.88). Neutral log
shear preserves that ratio at hub height, so hub winds stay night-weak while reality (ERA5 100 m)
is night-strong (night/day ≈ 1.05). ERA5 100 m exceeds ECMWF→100 m log by ~11% by day but
**~33% at night**. Uniform z0 cannot fix the diurnal shape (raising z0 overshoots day before
night recovers). Dual-z0 (day 0.2 / night ≈ 0.8) or true hub-height weather restores balance.

## Baseline (default OEDS z0=0.2 vs Atlite)
- OEDS national: day **100.4%**, night **66.7%** (gap -33.7 pp)
- Atlite national: day **95.7%**, night **92.5%** (gap -3.2 pp)
- Mean MW: ENTSO-E day 12.98 / night 13.94; OEDS 13.03 / 9.30; Atlite 12.42 / 12.89

## Weather (capacity-weighted ~80% of MaStR wind)
- `ecmwf_10_day_wmean`: 3.604
- `ecmwf_10_night_wmean`: 3.167
- `ecmwf_100log_day_wmean`: 5.726
- `ecmwf_100log_night_wmean`: 5.031
- `era5_100_day_wmean`: 6.373
- `era5_100_night_wmean`: 6.688
- `ecmwf_10_night_over_day`: 0.879
- `ecmwf_100log_night_over_day`: 0.879
- `era5_100_night_over_day`: 1.049
- `era5_over_ecmwf100log_day`: 1.113
- `era5_over_ecmwf100log_night`: 1.329

## Interpretation keys
- If ECMWF 10 m night/day ≈ ERA5 100 m night/day but hub-log night/day is flatter, neutral log understates nocturnal shear.
- If ERA5 100 m ≫ ECMWF log-100 m especially at night, the collapse is largely weather-height / product, not power-curve.
- If raising z0 lifts night ratio toward 100% faster than day, night needs stronger shear.

## z0 sweep (uniform, all hours)
  z0  Day_ratio  Night_ratio  Full_ratio  night_day_ratio_gap_pp
0.03  71.843792    46.930395   58.942455              -24.913397
0.05  77.333466    50.690946   63.536708              -26.642520
0.10  86.979088    57.343440   71.632346              -29.635648
0.20 100.408682    66.697190   82.951275              -33.711492
0.40 119.992850    80.529794   99.557011              -39.463055
0.80 150.005895   102.448682  125.378519              -47.557213
1.20 175.290509   121.859802  147.621558              -53.430707

- Closest night ratio to 100%: z0=0.8 → night 102.4%, day 150.0%
- Smallest |night−day| gap: z0=0.03 → gap -24.9 pp

## Dual-z0 (day z0=0.2, night varied)
 night_z0  Day_ratio  Night_ratio  Full_ratio  night_day_ratio_gap_pp
      0.2 100.408682    66.697190   82.951275              -33.711492
      0.4 100.408682    80.529794   90.114453              -19.878888
      0.8 100.408682   102.448682  101.465091                2.040000
      1.2 100.408682   121.859802  111.517088               21.451119
      2.0 100.408682   158.222671  130.347503               57.813988

## Hellman alpha (hub winds at hub height)
 alpha  Day_ratio  Night_ratio  Full_ratio  night_day_ratio_gap_pp
  0.10  53.442864    34.426068   43.595067              -19.016796
  0.14  69.872295    45.572724   57.288826              -24.299571
  0.20 100.531456    66.772155   83.049291              -33.759301
  0.28 150.509853   102.835245  125.821683              -47.674607
  0.35 197.753844   140.160635  167.929353              -57.593208
  0.45 256.773845   197.209965  225.928847              -59.563879

