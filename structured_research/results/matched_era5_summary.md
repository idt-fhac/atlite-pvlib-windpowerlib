# Matched ERA5 cutout: Atlite vs Windpowerlib / PVLib

Weather held fixed: `germany_2023.nc` (ERA5 via Atlite cutout).
Remaining deltas are conversion library, turbine/panel model, and orientation/layout choices.

## National library deltas (numerator / Atlite)
           Scale Technology                              Numerator          Denominator  Ratio_%  Correlation  MAE_MW  RMSE_MW  Num_GWh  Den_GWh
Germany national       Wind Windpowerlib MaStR fleet (ERA5 cutout) Atlite (ERA5 cutout)   110.47       0.9974  1813.2   2132.0 166302.6 150547.4
Germany national      Solar             PVLib 30/180 (ERA5 cutout) Atlite (ERA5 cutout)   116.81       0.9970  1846.9   3240.8 104799.6  89720.8
Germany national      Solar       PVLib MaStR orient (ERA5 cutout) Atlite (ERA5 cutout)   113.68       0.9966  1522.0   2553.7 101999.0  89720.8

## National vs ENTSO-E (same cutout weather for all sims)
           Scale Technology                                  Model  Ratio_%  Correlation  MAE_MW  RMSE_MW  Sim_GWh  Actual_GWh
Germany national       Wind Windpowerlib MaStR fleet (ERA5 cutout)   141.02       0.9763  5611.6   7845.4 166302.6    117931.3
Germany national       Wind                   Atlite (ERA5 cutout)   127.66       0.9738  4110.9   6537.9 150547.4    117931.3
Germany national      Solar             PVLib 30/180 (ERA5 cutout)   189.81       0.9366  5988.6  10809.1 104799.6     55212.0
Germany national      Solar       PVLib MaStR orient (ERA5 cutout)   184.74       0.9346  5678.6  10086.5 101999.0     55212.0
Germany national      Solar                   Atlite (ERA5 cutout)   162.50       0.9396  4557.0   7986.9  89720.8     55212.0

## Wind day/night (national)
                                 Model  Day_ratio  Day_corr  Day_sim_GWh  Day_act_GWh  Night_ratio  Night_corr  Night_sim_GWh  Night_act_GWh  Full_ratio  Full_corr  Full_sim_GWh  Full_act_GWh
Windpowerlib MaStR fleet (ERA5 cutout) 140.346449  0.972916 79802.240468   56860.8905   141.640311    0.980247   86500.324608      61070.414  141.016472   0.976321 166302.565076   117931.3045
                  Atlite (ERA5 cutout) 129.118527  0.972168 73417.943974   56860.8905   126.295919    0.976065   77129.440350      61070.414  127.656846   0.973817 150547.384324   117931.3045

## Interpretation
- **Wind:** Windpowerlib uses MaStR diameter/power → oedb catalogue types + MaStR hub
  heights (5 m bins for aggregation only). Atlite still uses a single V112@80 m for all capacity.
  The large energy gap vs Atlite is therefore mainly **fleet representation**, not weather.
  Night collapse from ECMWF 10 m log disappears when both use ERA5 hub wind.
- **Solar:** PVLib 30°/180° is the fair library match to Atlite orientation;
  PVLib MaStR orientations show layout/diversity effect on top of library.
- vs ENTSO-E still includes feed-in≠generation (solar) and omitted wakes/availability (wind).
