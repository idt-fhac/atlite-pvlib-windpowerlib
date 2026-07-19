"""Shared constants for the paper pipeline (single source of truth)."""

from datetime import date

# Atlite onshore wind — PyPSA-Eur default turbine YAML (hub 80 m)
ATLITE_WIND_TURBINE = "Vestas_V112_3MW"
WPL_MATCHED_PROXY_TYPE = "V112/3000"  # oedb sibling for matched-proxy runs
ATLITE_WIND_HUB_M = 80.0

# PVLib / Atlite solar physics
SAPM_A = -3.47
SAPM_B = -0.0594
SAPM_DT = 3.0
TEMP_COEFF = -0.004  # 1/°C
INVERTER_EFFICIENCY = 0.90
AGING_RATE_PER_YEAR = 0.005
AGING_REF_DATE = date(2023, 7, 1)

# Illustrative national feed-in energy match only — not a validated model
FEED_IN_SCALE = 0.62

# Default fleet aging when MaStR offline (capacity-weighted 2023 DE prior)
FLEET_AGING_FALLBACK = 0.9656

ZONES = ["DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNET"]
ZONE_LABEL = {
    "DE_50HZ": "50Hertz",
    "DE_AMPRION": "Amprion",
    "DE_TENNET": "TenneT",
    "DE_TRANSNET": "TransnetBW",
}
