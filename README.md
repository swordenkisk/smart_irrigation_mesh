# 💧 smart_irrigation_mesh
### Smart Irrigation Mesh Network — Federated ML Water Optimisation
#### *Low-power IoT sensor mesh — soil moisture + weather + crop model — zero water waste*

<div align="center">

![Score](https://img.shields.io/badge/idea%20score-89.0%2F100-brightgreen)
![Domain](https://img.shields.io/badge/domain-AgriTech-green)
![Novelty](https://img.shields.io/badge/novelty-97%2F100-blue)
![Feasibility](https://img.shields.io/badge/feasibility-79%2F100-blue)
![Impact](https://img.shields.io/badge/impact-97%2F100-orange)
![Author](https://img.shields.io/badge/author-swordenkisk-black)
![License](https://img.shields.io/badge/license-MIT-purple)
![Protocol](https://img.shields.io/badge/protocol-LoRaWAN-teal)

</div>

---

## 🌍 The Problem

**Agriculture consumes 70% of all global fresh water — and most is wasted.**

- Traditional irrigation: fixed schedule, uniform zones, no feedback
- A farmer irrigates at 6am every day — regardless of last night's rain
- Result: 40–60% of irrigation water evaporates, runs off, or over-saturates roots
- In Algeria alone: **7 billion m³/year** wasted on inefficient irrigation
- The Sahara aquifer is being depleted at 10× its recharge rate

**The root cause:** Farmers have no real-time data on what the soil actually needs.

---

## ✅ The Solution

A **self-organising mesh network** of low-cost soil sensors that:

1. **Senses** — each node reads soil moisture, temperature, salinity, pH every 15 min
2. **Communicates** — LoRaWAN gossip protocol propagates readings across the mesh (10km range)
3. **Predicts** — on-device ML forecasts soil moisture 24h ahead using weather API
4. **Decides** — irrigation controller opens/closes valves per zone, per need
5. **Learns** — federated learning: each farm improves the shared model without sharing raw data

```
Soil sensor mesh (LoRaWAN)
        ↓ readings every 15 min
Mesh gateway (edge aggregation)
        ↓ fused field map
ML forecaster (soil moisture 24h ahead)
        ↓ irrigation schedule
Valve controller (zone-by-zone)
        ↓ feedback
Federated model update (privacy-preserving)
```

---

## 🧮 Core Algorithms

### Soil Moisture Model (Bucket Model)
```
dθ/dt = P(t) + I(t) - ET₀(t)·Kc - D(t)

  θ    = volumetric soil moisture [m³/m³]
  P(t) = precipitation rate
  I(t) = irrigation rate (what we control)
  ET₀  = reference evapotranspiration (Penman-Monteith)
  Kc   = crop coefficient (wheat=1.15, tomato=1.05, date palm=0.9)
  D(t) = deep drainage (when θ > field capacity)
```

### Penman-Monteith Evapotranspiration
```
ET₀ = [0.408·Δ·(Rn-G) + γ·(900/(T+273))·u₂·(eₛ-eₐ)] / [Δ + γ·(1+0.34·u₂)]

  Rn = net radiation, G = soil heat flux
  T  = air temperature, u₂ = wind speed at 2m
  eₛ = saturation vapour pressure
  eₐ = actual vapour pressure
  Δ  = slope of vapour pressure curve
  γ  = psychrometric constant
```

### Irrigation Decision Rule
```
Trigger:  θ(t) < θ_threshold(crop, growth_stage)
Volume:   I* = (θ_fc - θ(t)) × D_root × A_zone
Cutoff:   stop when θ(t+Δt_predicted) ≥ θ_fc × 0.9
```

### Federated Learning (FedAvg)
```
Round r:
  Each farm k computes: Δwₖ = ∇L(wᵣ, Dₖ)   (local gradient)
  Server aggregates:    wᵣ₊₁ = Σₖ (nₖ/n) · (wᵣ - η·Δwₖ)
  Privacy: only Δwₖ is shared — raw data Dₖ never leaves the farm
```

---

## 🏗️ Architecture

```
smart_irrigation_mesh/
├── core/
│   ├── sensor_node.py         # SensorNode — soil readings + LoRaWAN tx
│   ├── mesh_network.py        # Mesh topology + gossip routing
│   └── irrigation_controller.py # Valve control + decision logic
├── sensors/
│   ├── soil_model.py          # Bucket model — soil moisture dynamics
│   └── et0_calculator.py      # Penman-Monteith ET₀ computation
├── ml/
│   ├── moisture_forecaster.py # 24h soil moisture forecast (LSTM-lite)
│   └── federated_learner.py   # FedAvg — privacy-preserving farm learning
├── mesh/
│   ├── lorawan_sim.py         # LoRaWAN packet simulation
│   └── gossip_protocol.py     # Epidemic mesh propagation
└── examples/
    └── algeria_farm_demo.py   # Full Algeria farm simulation
```

---

## ⚡ Quick Start

```python
from smart_irrigation_mesh import IrrigationMesh

mesh = IrrigationMesh(
    farm_name   = "Ferme Mitidja — Algeria",
    area_ha     = 50.0,
    crop        = "wheat",
    n_nodes     = 20,           # sensor nodes
    lat, lon    = 36.5, 3.1,    # GPS coordinates
)

# Deploy mesh — nodes self-organise
mesh.deploy()
print(f"Mesh coverage: {mesh.coverage_pct:.0f}%")
print(f"Node density : {mesh.nodes_per_ha:.1f} nodes/ha")

# Run one day of smart irrigation
for hour in range(24):
    state = mesh.tick(hour)
    if state.irrigation_triggered:
        print(f"Hour {hour}: Irrigation zone {state.zone} — {state.volume_m3:.1f} m³")

# Season summary
report = mesh.season_report()
print(f"Water used      : {report.water_used_m3:.0f} m³")
print(f"Water saved     : {report.water_saved_pct:.0f}% vs traditional")
print(f"Yield estimate  : {report.yield_estimate_t_ha:.1f} t/ha")
```

---

## 🌍 Impact for Algeria

| Metric | Value |
|--------|-------|
| Agricultural area | 8.5 million ha |
| Current water waste | ~7 billion m³/year |
| Expected savings with mesh | **40–55% reduction** |
| Water saved | **3–4 billion m³/year** |
| Economic value (at $0.05/m³) | **$150–200M/year** |
| Aquifer depletion rate reduction | Significant in Mitidja, Souf, Saoura |

---

## 🗺️ Roadmap

- [x] v1.0 — Soil bucket model + Penman-Monteith ET₀
- [x] v1.0 — LoRaWAN mesh simulation (20 nodes)
- [x] v1.0 — Irrigation decision engine
- [x] v1.0 — 24h moisture forecaster
- [x] v1.0 — Federated learning (FedAvg)
- [ ] v1.1 — Real LoRaWAN hardware (RAK Wireless / Dragino)
- [ ] v1.2 — Satellite ET₀ (Sentinel-2 integration)
- [ ] v1.3 — Arabic SMS alerts for farmers
- [ ] v2.0 — National mesh (Algeria Ministry of Agriculture API)

---

## 📄 License

MIT License — Copyright (c) 2026 swordenkisk
**github.com/swordenkisk/smart_irrigation_mesh**

*Idea score: 89.0/100 — Novelty: 97 | Feasibility: 79 | Impact: 97*
*From ideas_database.db — swordenkisk 🇩🇿 March 2026*
