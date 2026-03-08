# Showcase Guide: Aerodynamics LBM (v1)

This guide explains how to parameterize and tweak the **Aerodynamics-Fidelity** engine used in the current showcase. This engine implements a Lattice Boltzmann (LBM) D2Q9 solver with smoke advection and vorticity calculation.

## 1. Physical Parameters (`engine.rules`)

The core physics is controlled by the `params` object within the rule of type `lbm-aero-fidelity-v1`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `omega` | `number` | `1.75` | **Relaxation Frequency**. Controls viscosity. <br> - High (1.9+): Low viscosity (Turbulent, but can diverge). <br> - Low (1.0): High viscosity (Syrupy/Stable). |
| `inflowUx` | `number` | `0.15` | **Wind Speed (X)**. Horizontal velocity of the tunnel. |
| `inflowUy` | `number` | `0.0` | **Wind Speed (Y)**. Vertical velocity (for cross-winds). |

### Example: High-Speed Low-Viscosity Setup
```json
"rules": [{
  "type": "lbm-aero-fidelity-v1",
  "params": {
    "omega": 1.9,
    "inflowUx": 0.22
  }
}]
```

---

## 2. Face Dictionary

These are the memory layers you can read or write to in this engine.

| Face | Name | Type | Usage |
| :--- | :--- | :--- | :--- |
| `f0`-`f8` | Populations | `scalar`| The raw LBM directions. Internal use only. |
| `obstacles` | Obstacles | `mask` | **1.0 = Wall**. Fluid flows around these. |
| `vx`, `vy` | Velocity | `scalar`| Computed horizontal and vertical speeds. |
| `vorticity` | Vorticity | `scalar`| Rotation/Turbulence intensity (for rendering). |
| `smoke` | Smoke | `scalar`| Tracer density (Visual only). |

---

## 3. Configuring Objects

You can add any number of objects to the `config.objects` array.

### Adding a Custom Obstacle
To add a new wall, set its `properties` to `{"obstacles": 1.0}`.

```json
{
  "id": "new_pillar",
  "type": "circle",
  "position": { "x": 50, "y": 256 },
  "dimensions": { "w": 20, "h": 20 },
  "properties": { "obstacles": 1.0 }
}
```

### Adding a Smoke Source
To make an object emit smoke (without being a wall), use `{"smoke": 1.0}`.

```json
{
  "id": "smoke_generator",
  "type": "rect",
  "position": { "x": 10, "y": 256 },
  "dimensions": { "w": 5, "h": 50 },
  "properties": { "smoke": 1.0 },
  "rasterMode": "add"
}
```

---

## 4. Performance Tuning

In the `config` section:
- **Chunks**: Increase `x` and `y` to leverage more CPU cores (e.g., `4x2` for 8 chunks).
- **Resolution**: Adjust `nx` and `ny`. High values (1024+) require a powerful GPU if `mode: "gpu"` is used.

> [!TIP]
> Always keep your dimensions as multiples of 16 for optimal memory alignment in the dispatcher.
