# Paris Accessibility SDF Tool

A premium urban analysis tool powered by the **Hypercube Neo** spatial compute engine. It generates real-time **Signed Distance Fields (SDF)** over the city of Paris to analyze accessibility to key services.

![Paris Accessibility Showcase](../../../assets/showcase_preview.png)

## Features

- **Worldwide Dynamic Data**: Powered by **Overpass API (OpenStreetMap)**. Recalculates urban metrics for ANY location on Earth instantly.
- **Improved Visualization**:
  - **Color Scale Legend**: Clear visual indicators for accessibility scores.
  - **Building Rendering**: Buildings (obstacles) are now rendered as subtle technical overlays, resolving previous visual artifacts.
  - **Refined Contours**: High-contrast golden iso-lines at **200m**, **500m**, and **1km**.

- **Dual-Backend Support**: Toggle between optimized **CPU Multithreading** and **WebGPU Acceleration**.

## How to Use

1. **Select Layers**: Use the sliders in the side panel to adjust the importance (weight) of each service.
2. **Move the Map**: Pan around Paris. The engine will "bake" the distance field for the new view automatically on `moveend`.
3. **Recalculate Bounds**: Use the button to force a re-calculation if you've moved to a new area.
4. **Switch Backend**:
   - [CPU Version](index.html) (Default)
   - [GPU Version](index.html?backend=gpu) (WebGPU required)

## Technical Architecture

- **Engine**: Hypercube Neo v4.0.
- **Algorithm**: Iterative Dilation (CPU) and Jump Flooding (GPU fallback).
- **Data**: ~3MB of Paris OSM data (`paris_data.json`) filtered and projected on-the-fly.
- **Visualization**: `CanvasAdapterNeo` using the custom `accessibility-sdf` colormap with procedural contour generation.

---
Part of the Hypercube Neo Showcase Collection.
