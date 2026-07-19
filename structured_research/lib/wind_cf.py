"""Windpowerlib capacity-factor interpolation (shared by paper + explore)."""

from __future__ import annotations

import numpy as np
from windpowerlib import WindTurbine

_CF_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def capacity_factor(turbine_type: str, u_hub: np.ndarray) -> np.ndarray:
    """Interpolate oedb power curve → capacity factor (avoids ModelChain overhead)."""
    if turbine_type not in _CF_CACHE:
        wt = WindTurbine(hub_height=100.0, turbine_type=turbine_type)
        pc = wt.power_curve
        u = np.asarray(pc["wind_speed"], dtype=float)
        cf = np.clip(np.asarray(pc["value"], dtype=float) / float(wt.nominal_power), 0.0, 1.0)
        _CF_CACHE[turbine_type] = (u, cf)
    u_grid, cf_grid = _CF_CACHE[turbine_type]
    return np.interp(u_hub, u_grid, cf_grid, left=0.0, right=cf_grid[-1])
