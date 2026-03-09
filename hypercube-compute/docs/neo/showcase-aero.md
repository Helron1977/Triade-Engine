# 🌪️ Aerodynamics Showcase Guide (Neo Edition)

Welcome to the Hypercube Neo Aerodynamics showcase. This guide is designed to help you understand how to customize your fluid simulation, even if you are not a physics expert.

## 📖 Glossary of Terms

To master the simulation, you should understand these 4 key concepts:

1.  **LBM (Lattice Boltzmann Method)**: The "engine" we use. Instead of complex math for every drop of water, it treats the fluid as a collection of particles moving on a grid. It's very fast and perfect for GPUs.
2.  **Omega (Ω)**: This represents the **Relaxation Frequency**. In simple terms: **Viscosity (Thickness)**.
    -   *High Omega (1.9)*: Water or Air (Thin, fast, creates lots of swirls/vorticity).
    -   *Low Omega (1.0)*: Honey or Oil (Thick, slow, very stable).
3.  **Vorticity**: A measure of the "swirliness" or rotation in the fluid. It's what creates those beautiful red turbulent wakes behind obstacles.
4.  **Advection**: The process of "carrying" something (like smoke) along with the wind.

---

## ⚙️ Tweaking the Physics (`engine.rules`)

Find the `lbm-aero-fidelity-v1` rule in your manifest to change the behavior of the wind.

| Parameter | Impact on Simulation | Recommended Range |
| :--- | :--- | :--- |
| `omega` | Controls how "chaotic" the air is. | `1.0` (Stable) to `1.95` (Turbulent) |
| `inflowUx` | The speed of the wind from left to right. | `0.05` (Breeze) to `0.25` (Hurricane) |

### 💡 Pro Tip: Instability
If you set `omega` too high (above 1.95) and `inflowUx` too fast, the simulation might "explode" (values become infinite). If the screen turns bright red or white, lower the `omega`!

---

## 🎨 Changing the Visuals

The `visualProfile` section defines what you see on the screen.

-   **Arctic Palette**: Designed for Aero. Smoke is white/blue, and Vorticity (turbulence) is represented in varying shades of red.
-   **Layers**:
    -   `Obstacles`: The solid objects you've placed.
    -   `Smoke`: The tracer that shows where the air is moving.
    -   `Vorticity`: The heat-map of turbulence.

---

## 🧱 Building Your Scenario (`config.objects`)

You can drag and drop objects or define them in the manifest:

### Example: A Solid Pillar
```json
{
  "id": "main_pillar",
  "type": "circle",
  "position": { "x": 100, "y": 256 },
  "dimensions": { "w": 30, "h": 30 },
  "properties": { "obstacles": 1.0 }
}
```

### Example: A Smoke Source
If you want to see the wind flow, place a rectangle at the start of the tunnel:
```json
{
  "id": "emitter",
  "type": "rect",
  "position": { "x": 5, "y": 256 },
  "dimensions": { "w": 2, "h": 100 },
  "properties": { "smoke": 1.0 },
  "rasterMode": "add"
}
```

---

## 🚀 Performance
-   **CPU Mode**: High resolution is limited by your processor count. Use `4x2` chunks for a quad-core CPU.
-   **GPU Mode**: Can handle massive resolutions (2048x1024) at 60 FPS on modern hardware.

Enjoy exploring the world of fluid dynamics with Hypercube Neo!
