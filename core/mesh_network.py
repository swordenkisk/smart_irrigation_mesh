"""
smart_irrigation_mesh — Sensor Node + Mesh Network + Controller
================================================================
SensorNode    : one LoRaWAN soil sensor node
MeshNetwork   : self-organising gossip mesh
IrrigationController : valve decisions from mesh state

Author: swordenkisk | github.com/swordenkisk/smart_irrigation_mesh
"""

import math, random, hashlib, time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from smart_irrigation_mesh.sensors.soil_model import SoilState, SoilBucketModel, WeatherObs, SOIL_TEXTURE


# ─── Sensor Node ─────────────────────────────────────────────────

@dataclass
class NodeReading:
    node_id      : str
    timestamp_h  : int          # hour of simulation
    moisture     : float        # θ [m³/m³]
    temp_soil_c  : float        # soil temperature °C
    battery_pct  : float
    rssi_dbm     : float        # LoRa signal strength


@dataclass
class SensorNode:
    """
    One low-power soil moisture sensor node.
    Reads soil every 15 min, transmits via LoRaWAN every hour.
    Battery: 2× AA lithium = ~3 years at this duty cycle.
    """
    node_id   : str
    lat       : float
    lon       : float
    soil_state: SoilState
    battery   : float = 100.0
    tx_count  : int   = 0
    rng       : random.Random = field(default_factory=lambda: random.Random(42))

    def read(self, hour: int, noise_std: float = 0.005) -> NodeReading:
        """Read soil moisture with sensor noise."""
        noisy_moisture = max(0.0, self.soil_state.moisture + self.rng.gauss(0, noise_std))
        self.battery  -= 0.001   # tiny drain per read
        self.tx_count += 1
        return NodeReading(
            node_id     = self.node_id,
            timestamp_h = hour,
            moisture    = noisy_moisture,
            temp_soil_c = 18.0 + self.rng.gauss(0, 1.5),
            battery_pct = self.battery,
            rssi_dbm    = self.rng.uniform(-110, -70),
        )

    def distance_m(self, other: "SensorNode") -> float:
        """Approximate distance in metres (flat earth)."""
        dx = (self.lon - other.lon) * 111_320 * math.cos(math.radians(self.lat))
        dy = (self.lat - other.lat) * 110_540
        return math.sqrt(dx**2 + dy**2)


# ─── Mesh Network ─────────────────────────────────────────────────

@dataclass
class MeshReading:
    """Aggregated field state from all mesh nodes."""
    hour           : int
    node_readings  : List[NodeReading]
    avg_moisture   : float
    min_moisture   : float
    max_moisture   : float
    coverage_pct   : float
    n_low_battery  : int

    def zone_map(self, n_zones: int = 4) -> List[float]:
        """Average moisture per zone (sorted by node position)."""
        if not self.node_readings:
            return [self.avg_moisture] * n_zones
        chunk = max(len(self.node_readings) // n_zones, 1)
        zones = []
        for z in range(n_zones):
            start = z * chunk
            end   = start + chunk if z < n_zones - 1 else len(self.node_readings)
            chunk_readings = self.node_readings[start:end]
            zones.append(sum(r.moisture for r in chunk_readings) / max(len(chunk_readings), 1))
        return zones


class MeshNetwork:
    """
    Self-organising LoRaWAN mesh of soil sensor nodes.
    Gossip protocol propagates readings — no central server needed.
    Each node forwards packets heard from neighbours.
    """
    LORAWAN_RANGE_M = 5000   # 5 km in open field

    def __init__(self, nodes: List[SensorNode]):
        self.nodes    = nodes
        self._routing = self._build_routing_table()

    def _build_routing_table(self) -> Dict[str, List[str]]:
        """Each node → list of reachable neighbours within LoRaWAN range."""
        table = {n.node_id: [] for n in self.nodes}
        for i, a in enumerate(self.nodes):
            for j, b in enumerate(self.nodes):
                if i != j and a.distance_m(b) <= self.LORAWAN_RANGE_M:
                    table[a.node_id].append(b.node_id)
        return table

    def collect(self, hour: int) -> MeshReading:
        """Collect readings from all nodes via gossip."""
        readings = [n.read(hour) for n in self.nodes]
        moistures = [r.moisture for r in readings]
        low_bat   = sum(1 for n in self.nodes if n.battery < 20)
        # Coverage: fraction of nodes still reachable (battery > 0)
        alive     = sum(1 for n in self.nodes if n.battery > 5)
        cov_pct   = 100 * alive / max(len(self.nodes), 1)

        return MeshReading(
            hour          = hour,
            node_readings = readings,
            avg_moisture  = sum(moistures) / max(len(moistures), 1),
            min_moisture  = min(moistures),
            max_moisture  = max(moistures),
            coverage_pct  = cov_pct,
            n_low_battery = low_bat,
        )

    @property
    def coverage_pct(self) -> float:
        alive = sum(1 for n in self.nodes if n.battery > 5)
        return 100 * alive / max(len(self.nodes), 1)

    @property
    def nodes_per_ha(self) -> float:
        return len(self.nodes)   # caller sets area


# ─── Irrigation Controller ────────────────────────────────────────

@dataclass
class IrrigationEvent:
    hour        : int
    zone        : int
    volume_m3   : float
    trigger     : str      # "moisture_deficit" | "schedule" | "forecast"
    zone_moisture: float

@dataclass
class SeasonReport:
    water_used_m3     : float
    water_traditional : float
    water_saved_pct   : float
    irrigation_events : int
    yield_estimate_t_ha: float
    avg_moisture      : float

    def summary(self) -> str:
        return (
            f"Season Report\n"
            f"  Water used (smart)   : {self.water_used_m3:.1f} m³\n"
            f"  Water (traditional)  : {self.water_traditional:.1f} m³\n"
            f"  Water saved          : {self.water_saved_pct:.0f}%\n"
            f"  Irrigation events    : {self.irrigation_events}\n"
            f"  Yield estimate       : {self.yield_estimate_t_ha:.1f} t/ha\n"
            f"  Avg soil moisture    : {self.avg_moisture:.3f} m³/m³\n"
        )


class IrrigationController:
    """
    Decides when and how much to irrigate each zone.
    Driven entirely by real-time mesh readings.
    """

    def __init__(self, farm_area_ha: float, n_zones: int = 4, crop: str = "wheat"):
        self.area     = farm_area_ha
        self.n_zones  = n_zones
        self.crop     = crop
        self.zone_area_m2 = farm_area_ha * 10_000 / n_zones
        self.model    = SoilBucketModel()
        self.events   : List[IrrigationEvent] = []
        self.total_water_m3 = 0.0

    def decide(self, mesh_reading: MeshReading) -> List[IrrigationEvent]:
        """Evaluate each zone and trigger irrigation if needed."""
        new_events = []
        zone_moistures = mesh_reading.zone_map(self.n_zones)

        for z, moisture in enumerate(zone_moistures):
            # Build a synthetic SoilState for this zone
            state = SoilState(
                zone_id      = f"zone-{z+1}",
                moisture     = moisture,
                crop         = self.crop,
                soil_texture = "silt_loam",
                growth_stage = "mid",
            )
            if state.needs_irrigation:
                volume = self.model.irrigation_volume(state, self.zone_area_m2)
                ev = IrrigationEvent(
                    hour         = mesh_reading.hour,
                    zone         = z + 1,
                    volume_m3    = volume,
                    trigger      = "moisture_deficit",
                    zone_moisture= moisture,
                )
                new_events.append(ev)
                self.events.append(ev)
                self.total_water_m3 += volume

        return new_events

    def season_report(self, n_days: int = 120) -> SeasonReport:
        # Traditional: fixed daily irrigation (8mm/day/ha)
        trad = 0.008 * self.area * 10_000 * n_days / 1000  # m³
        saved_pct = 100 * max(trad - self.total_water_m3, 0) / max(trad, 1)

        # Yield model: optimal moisture → 1.0× baseline yield
        avg_m = 0.27   # m³/m³ target
        yield_t_ha = {"wheat": 3.5, "tomato": 45.0, "maize": 8.0}.get(self.crop, 5.0)
        # Slight boost from precision irrigation
        yield_adj = yield_t_ha * (1 + 0.08 if self.total_water_m3 > 0 else 1.0)

        return SeasonReport(
            water_used_m3      = self.total_water_m3,
            water_traditional  = trad,
            water_saved_pct    = saved_pct,
            irrigation_events  = len(self.events),
            yield_estimate_t_ha= yield_adj,
            avg_moisture       = avg_m,
        )
