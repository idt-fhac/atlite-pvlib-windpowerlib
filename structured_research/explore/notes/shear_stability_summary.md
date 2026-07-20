# Shear / stability inspection (10 m → 100 m)

## Does temperature enter the current OEDS wind stack?
- **Shear: no.** `windpowerlib.logarithmic_profile` is neutral: only `u`, heights, `z0`.
- **Density: not in our runs.** `ModelChain(..., density_correction=False)` (default) with `power_curve` → temperature is carried in the weather frame but **does not change power**.
- So the night collapse cannot be blamed on missing temperature *in the power curve*; it is missing **stability-dependent shear** (and/or hub-height weather).

## Implied α (ERA5 100 m / ECMWF 10 m)
- Day mean implied α ≈ **0.239**
- Night mean implied α ≈ **0.329**
- Neutral-log equivalent α for z0=0.2 at 100 m: **0.161** (constant — no day/night)

## Bin diagnostics (log bias vs ERA5)
                   bin      n  alpha_implied_mean  alpha_era5_mean  u100_log_mean  u100_era5_mean  log_bias_ms  log_ratio  u10_mean  temp_mean
                   all 312868            0.284397         0.230202       5.473461        6.590055    -1.116594   0.830564  3.445480 284.056525
                   day 156146            0.239478         0.186254       5.825688        6.436319    -0.610631   0.905127  3.667202 285.317720
                 night 156722            0.329152         0.273988       5.122528        6.743227    -1.620698   0.759655  3.224572 282.799965
            hour_00-02  39208            0.332858         0.278425       5.090568        6.749976    -1.659408   0.754161  3.204453 282.165748
            hour_03-05  39135            0.336353         0.270965       5.063569        6.736948    -1.673378   0.751612  3.187457 281.747746
            hour_06-08  38923            0.290177         0.214808       5.358862        6.508660    -1.149798   0.823343  3.373341 283.042545
            hour_09-11  39049            0.202827         0.161791       6.061847        6.257723    -0.195875   0.968699  3.815862 285.457591
            hour_12-14  39065            0.205360         0.159897       6.233966        6.477561    -0.243595   0.962394  3.924208 286.705694
            hour_15-17  39109            0.259694         0.208588       5.646679        6.501451    -0.854772   0.868526  3.554518 286.056008
            hour_18-20  39198            0.318660         0.267566       5.202631        6.715646    -1.513016   0.774703  3.274995 284.320899
            hour_21-23  39181            0.328747         0.278992       5.133264        6.770337    -1.637073   0.758199  3.231330 282.964007
            night_cold  51718            0.334167         0.271255       5.093222        6.747173    -1.653951   0.754868  3.206124 275.391757
            night_mild  53287            0.325080         0.266959       5.810772        7.540100    -1.729327   0.770649  3.657813 282.702669
            night_warm  51717            0.328333         0.283963       4.442698        5.918216    -1.475517   0.750682  2.796627 290.308566
 night_cooling_dT<-0.3  63839            0.331889         0.288745       4.480806        6.032081    -1.551275   0.742829  2.820615 284.583056
night_neutral_|dT|<0.3  73556            0.327571         0.265605       5.669062        7.365023    -1.695961   0.769728  3.568608 280.883307
  night_warming_dT>0.3  19291            0.326212         0.257221       5.149775        6.712157    -1.562382   0.767231  3.241723 284.198601
           night_u10<3  85381            0.352763         0.300363       3.275468        4.769241    -1.493773   0.686790  2.061869 283.252625
          night_u10>=5  22034            0.262315         0.213567      10.135130       11.636544    -1.501414   0.870974  6.379946 281.568360
          day_ghi>=400  27452            0.152411         0.125147       5.540878        5.071458     0.469420   1.092561  3.487918 292.580171
           day_ghi<150  78049            0.294295         0.227487       6.063740        7.398663    -1.334923   0.819572  3.817053 280.958954

## Extrapolator skill vs ERA5 100 m (same ECMWF u10)
                   model subset     rmse      mae      bias  ratio_mean     corr  alpha_day_used  alpha_night_used
        neutral_log_z0.2    all 1.649844 1.375640 -1.116594    0.830564 0.917756        0.239478          0.329152
        neutral_log_z0.2    day 1.372326 1.065379 -0.610631    0.905127 0.924661        0.239478          0.329152
        neutral_log_z0.2  night 1.886162 1.684761 -1.620698    0.759655 0.942655        0.239478          0.329152
       hellman_alpha_1/7    all 2.204688 1.878033 -1.802577    0.726470 0.917756        0.239478          0.329152
       hellman_alpha_1/7    day 1.870804 1.469315 -1.340758    0.791689 0.924661        0.239478          0.329152
       hellman_alpha_1/7  night 2.493268 2.285249 -2.262699    0.664449 0.942655        0.239478          0.329152
hellman_alpha_era5_field    all 1.226791 0.991831 -0.795938    0.879221 0.955371        0.239478          0.329152
hellman_alpha_era5_field    day 1.186251 0.936734 -0.769317    0.880473 0.963331        0.239478          0.329152
hellman_alpha_era5_field  night 1.265892 1.046725 -0.822460    0.878032 0.945153        0.239478          0.329152
    hellman_dual_learned    all 1.254750 0.957842  0.033304    1.005054 0.927917        0.239478          0.329152
    hellman_dual_learned    day 1.242084 0.996418 -0.071111    0.988952 0.924661        0.239478          0.329152
    hellman_dual_learned  night 1.267244 0.919408  0.137336    1.020366 0.942655        0.239478          0.329152
 hellman_solar_rad_class    all 1.352654 1.054680 -0.717787    0.891080 0.931188        0.239478          0.329152
 hellman_solar_rad_class    day 1.494793 1.155086 -0.946517    0.852941 0.939241        0.239478          0.329152
 hellman_solar_rad_class  night 1.194331 0.954643 -0.489897    0.927350 0.930262        0.239478          0.329152
   hellman_temp_tendency    all 1.499936 1.159278 -0.767186    0.883584 0.907336        0.239478          0.329152
   hellman_temp_tendency    day 1.753029 1.368381 -1.210518    0.811924 0.927733        0.239478          0.329152
   hellman_temp_tendency  night 1.195631 0.950943 -0.325483    0.951732 0.916690        0.239478          0.329152

## National power ratios when feeding Hellman hub winds into Windpowerlib
                  model  Day_ratio  Day_corr  Night_ratio  Night_corr  Full_ratio  Full_corr
   hellman_dual_learned 124.195397  0.928334   128.655664    0.966626  126.505134   0.944581
hellman_solar_rad_class  85.768021  0.938339   105.024921    0.963345   95.740156   0.943556
  hellman_temp_tendency  75.785769  0.923544   112.066524    0.939974   94.573662   0.906497
     hellman_fixed_0.20 100.531456  0.926330    66.772155    0.930026   83.049291   0.907653

## Practical improvement path
1. Best physics with current fields: use a **diurnal / stability-dependent α** (solar-radiation class or dual day/night α fitted to ERA5).
2. Better: ingest **hub-height wind** (ERA5 100 m or ECMWF 100/120 m if available in DB).
3. Temperature alone (density) will not fix night yield; use T / GHI as **stability proxies for α**.
