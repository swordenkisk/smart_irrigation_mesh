"""
smart_irrigation_mesh — Soil Model + ET₀ Calculator
=====================================================
Implements:
  1. Bucket soil moisture model  (dθ/dt = P + I - ET₀·Kc - D)
  2. Penman-Monteith ET₀ equation (FAO-56 standard)

These are the physical foundation of all irrigation decisions.
Every valve open/close is derived from these equations.

Author: swordenkisk | github.com/swordenkisk/smart_irrigation_mesh
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional


# ─── Crop Coefficients (FAO-56 Table 12) ─────────────────────────

CROP_KC: Dict[str, Dict[str, float]] = {
    "wheat"     : {"initial": 0.30, "mid": 1.15, "late": 0.25, "root_depth_m": 1.0},
    "tomato"    : {"initial": 0.60, "mid": 1.15, "late": 0.70, "root_depth_m": 0.7},
    "potato"    : {"initial": 0.50, "mid": 1.15, "late": 0.75, "root_depth_m": 0.4},
    "maize"     : {"initial": 0.30, "mid": 1.20, "late": 0.35, "root_depth_m": 1.2},
    "date_palm" : {"initial": 0.90, "mid": 0.95, "late": 0.95, "root_depth_m": 1.5},
    "olive"     : {"initial": 0.65, "mid": 0.70, "late": 0.70, "root_depth_m": 1.2},
    "cotton"    : {"initial": 0.35, "mid": 1.15, "late": 0.50, "root_depth_m": 1.0},
    "barley"    : {"initial": 0.30, "mid": 1.15, "late": 0.25, "root_depth_m": 1.0},
}

# Soil texture properties [θ_wilting, θ_field_capacity, θ_saturation]
SOIL_TEXTURE: Dict[str, Dict[str, float]] = {
    "sandy"       : {"wp": 0.05, "fc": 0.14, "sat": 0.43},
    "sandy_loam"  : {"wp": 0.08, "fc": 0.20, "sat": 0.46},
    "loam"        : {"wp": 0.12, "fc": 0.27, "sat": 0.47},
    "clay_loam"   : {"wp": 0.18, "fc": 0.34, "sat": 0.48},
    "clay"        : {"wp": 0.25, "fc": 0.40, "sat": 0.50},
    "silt_loam"   : {"wp": 0.10, "fc": 0.33, "sat": 0.46},  # common in Algeria
}


# ─── Weather Observation ──────────────────────────────────────────

@dataclass
class WeatherObs:
    """Hourly weather observation."""
    hour          : int
    temp_c        : float     # Air temperature °C
    rh_pct        : float     # Relative humidity %
    wind_m_s      : float     # Wind speed m/s at 2m height
    solar_rad_mj  : float     # Solar radiation MJ/m²/hr
    precip_mm     : float     # Precipitation mm/hr


# ─── Penman-Monteith ET₀ ─────────────────────────────────────────

class ET0Calculator:
    """
    FAO-56 Penman-Monteith reference evapotranspiration.

    ET₀ [mm/hr] = [0.408·Δ·(Rn-G) + γ·(37/(T+273))·u₂·(eₛ-eₐ)]
                  / [Δ + γ·(1 + 0.34·u₂)]

    This is the international standard (FAO Irrigation Paper 56).
    Used by every serious irrigation system in the world.
    """

    def compute(self, obs: WeatherObs, elevation_m: float = 200.0) -> float:
        """Compute hourly ET₀ in mm/hr."""
        T   = obs.temp_c
        RH  = obs.rh_pct / 100.0
        u2  = obs.wind_m_s
        Rn  = obs.solar_rad_mj * 0.77  # net radiation ≈ 0.77 × incoming
        G   = 0.1 * Rn                 # soil heat flux ≈ 0.1 × Rn (daytime)

        # Saturation vapour pressure [kPa]
        es = 0.6108 * math.exp(17.27 * T / (T + 237.3))
        ea = es * RH

        # Slope of vapour pressure curve Δ [kPa/°C]
        delta = 4098 * es / (T + 237.3) ** 2

        # Atmospheric pressure [kPa]
        P = 101.3 * ((293 - 0.0065 * elevation_m) / 293) ** 5.26

        # Psychrometric constant γ [kPa/°C]
        gamma = 0.000665 * P

        # Penman-Monteith
        numerator   = 0.408 * delta * (Rn - G) + gamma * (37 / (T + 273)) * u2 * (es - ea)
        denominator = delta + gamma * (1 + 0.34 * u2)

        et0 = max(numerator / denominator, 0.0)
        return et0   # mm/hr


# ─── Soil Bucket Model ────────────────────────────────────────────

@dataclass
class SoilState:
    """Current state of one soil zone."""
    zone_id      : str
    moisture     : float     # θ volumetric [m³/m³]
    crop         : str
    soil_texture : str
    growth_stage : str = "mid"     # initial | mid | late
    depth_m      : float = 0.3     # sensor depth

    @property
    def soil_props(self) -> Dict:
        return SOIL_TEXTURE.get(self.soil_texture, SOIL_TEXTURE["loam"])

    @property
    def crop_props(self) -> Dict:
        return CROP_KC.get(self.crop, CROP_KC["wheat"])

    @property
    def kc(self) -> float:
        return self.crop_props[self.growth_stage]

    @property
    def field_capacity(self) -> float:
        return self.soil_props["fc"]

    @property
    def wilting_point(self) -> float:
        return self.soil_props["wp"]

    @property
    def root_depth(self) -> float:
        return self.crop_props["root_depth_m"]

    @property
    def needs_irrigation(self) -> bool:
        """MAD (Management Allowed Depletion) threshold = 50% of plant-available water."""
        paw       = self.field_capacity - self.wilting_point
        threshold = self.field_capacity - 0.50 * paw
        return self.moisture < threshold

    @property
    def depletion_mm(self) -> float:
        """How much water (mm) needed to reach field capacity."""
        deficit = max(self.field_capacity - self.moisture, 0.0)
        return deficit * self.root_depth * 1000   # convert to mm


class SoilBucketModel:
    """
    FAO bucket model for soil moisture dynamics.

    dθ/dt = [P(t) + I(t) - ET₀(t)·Kc - D(t)] / (root_depth × 1000)

    D(t) = drainage when θ > field_capacity (excess drains away)
    """

    def __init__(self, dt_hours: float = 1.0):
        self.dt = dt_hours
        self.et0 = ET0Calculator()

    def step(
        self,
        state   : SoilState,
        weather : WeatherObs,
        irrig_mm: float = 0.0,
    ) -> SoilState:
        """
        Advance soil moisture by one time step dt.
        Returns updated SoilState.
        """
        props  = state.soil_props
        et0_hr = self.et0.compute(weather)
        etc_hr = et0_hr * state.kc     # crop evapotranspiration [mm/hr]

        # Water balance [mm]
        inflow    = weather.precip_mm + irrig_mm
        outflow   = etc_hr * self.dt
        # Convert mm → m³/m³
        dz        = state.root_depth   # m
        delta_theta = (inflow - outflow) / (dz * 1000)

        new_moisture = state.moisture + delta_theta

        # Drainage: excess above saturation is lost
        if new_moisture > props["sat"]:
            new_moisture = props["sat"]
        # Cannot go below absolute dry
        if new_moisture < 0:
            new_moisture = 0.0

        return SoilState(
            zone_id      = state.zone_id,
            moisture     = new_moisture,
            crop         = state.crop,
            soil_texture = state.soil_texture,
            growth_stage = state.growth_stage,
            depth_m      = state.depth_m,
        )

    def irrigation_volume(self, state: SoilState, zone_area_m2: float) -> float:
        """
        Compute irrigation volume [m³] to bring zone to 90% field capacity.
        I* = min((θ_fc×0.9 - θ_current) × root_depth, max_depth=0.030m) × area
        Capped at 30mm depth per event — agronomic best practice.
        """
        target    = state.field_capacity * 0.90
        deficit   = max(target - state.moisture, 0.0)
        depth_m   = min(deficit * state.root_depth, 0.030)   # max 30mm/event
        return depth_m * zone_area_m2
