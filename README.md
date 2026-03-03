<div align="center">
  <img src="https://raw.githubusercontent.com/Helron1977/Hypercube-Compute/main/hypercube-compute/docs/assets/logo.png" alt="Hypercube Engine Logo" width="220" style="border-radius:24px; box-shadow: 0 10px 40px rgba(0,0,0,0.3);"/>
  <h1>🌊 Hypercube Engine V4 🚀</h1>
  <p><strong>Pure GodMode O(1) • Zero-Allocation Tensor Compute • Web & Node.js</strong></p>

  [![npm version](https://img.shields.io/npm/v/hypercube-compute.svg?style=flat-square&color=00ff80)](https://www.npmjs.com/package/hypercube-compute)
  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
  [![TypeScript](https://img.shields.io/badge/TypeScript-Ready-3178C6.svg?style=flat-square)](https://www.typescriptlang.org/)
</div>

---

## ⚡ v4 – Native 3D Compute is here (Alpha) 🧊
Hypercube V4 marks a fundamental leap from 2D grids to **native 3D volumetric compute**. Experience real-time atmospheric diffusion and fluid dynamics with zero garbage collection overhead.

👉 **[Launch the 3D Volume Demo](https://helron1977.github.io/Hypercube-Compute/)** 
*Real-time fluid interaction calculated at 60 FPS entirely on the GPU.*

![3D Volume Diffusion](https://raw.githubusercontent.com/Helron1977/Hypercube-Compute/main/hypercube-compute/docs/assets/3d_diffusion.png)
*Volumetric diffusion with Isosurface extraction (V4 Alpha).*

---

## 🔥 Why Hypercube?
Traditional physics engines treat objects as discrete logic (`[{x, y, vx, vy}, ... ]`). As your simulation grows, the CPU struggles with branching and memory cache misses.

**Hypercube turns the problem upside down.** 
It structures your world as **Contiguous Memory Tensors** (Faces). 
- **O(1) Performance**: Whether you have 10 or 10,000 interacting cells, the complexity remains constant per grid.
- **Zero Allocation**: Once the Grid is created, no memory is allocated during the `compute()` loop. No more GC pauses.
- **Hardware Agnostic**: Switch between Multi-threaded CPU and WebGPU acceleration with a single line of code.

If you are building **Fluid Dynamics (CFD), Cellular Automata, or Procedural Ecosystems**, Hypercube provides the high-performance memory layout used by modern professional solvers.

---

## 💡 Quick Start: See the "Wow" in 20 Lines
The easiest way to start is the **Organic Ecosystem** (Game of Life).

```typescript
import { HypercubeGrid, HypercubeMasterBuffer, GameOfLifeEngine, HypercubeViz } from 'hypercube-compute';

// 1. Setup Memory & Grid
const master = new HypercubeMasterBuffer(); 
const grid = await HypercubeGrid.create(1, 1, 128, master, () => new GameOfLifeEngine(), 3);

// 2. Setup Loop
const canvas = document.querySelector('canvas')!;
const loop = () => {
    grid.compute(); // Standard O(1) step
    
    // 3. One-liner "Plug & Play" Visualization
    // Renders Face 2 (Organic Density) using the Viridis colormap
    HypercubeViz.quickRender(canvas, grid.cubes[0][0], 2, 'viridis');
    
    requestAnimationFrame(loop);
};
loop();
```

---

## 🌪️ Advanced: Ocean Fluid (LBM D2Q9)
Simulate realistic vortices and currents with the **Ocean Engine**.

```typescript
import { HypercubeGrid, OceanEngine, HypercubeViz } from 'hypercube-compute';

const grid = await HypercubeGrid.create(1, 1, 256, master, () => new OceanEngine(), 23);

const loop = () => {
    grid.compute([0, 1, 2, 3, 4, 5, 6, 7, 8]); // Sync LBM populations

    const chunk = grid.cubes[0][0];
    const curl = chunk.faces[21]; // Vorticity/Curl face
    
    // Visualise with 'Plasma' colormap for extra impact
    HypercubeViz.renderToCanvas(canvas, curl, 256, 256, 'plasma');
    
    requestAnimationFrame(loop);
};
```

---

## 🏗 V4 3D Dictionary: Faces & Tensors
Hypercube uses **Faces** (layers) instead of objects. 

| Engine | Face | Description | Code Snippet |
| :--- | :--- | :--- | :--- |
| **GameOfLife** | `1` | **Discrete State** (Plant/Animal/Empty) | `chunk.faces[1]` |
| | `2` | **Organic Density** (Aging/Heatmap) | `chunk.faces[2]` |
| **Ocean (LBM)**| `18` | **Velocity X** (Horizontal flow) | `chunk.faces[18]` |
| | `21` | **Vorticity** (Rotation/Eddies) | `chunk.faces[21]` |
| | `22` | **Obstacles** (Walls/Landmass) | `chunk.faces[22]` |
| **Vol. Diffusion**| `0` | **Density Input** (3D Grid) | `chunk.faces[0]` |
| | `1` | **Diffusion Output** (t+1) | `chunk.faces[1]` |

---

## 🌐 Live Interactability
Inject data directly into the tensors while the simulation is running:

```typescript
// Add a vortex on mouse move
canvas.onmousemove = (e) => {
    const engine = grid.cubes[0][0].engine as OceanEngine;
    engine.addVortex(grid.cubes[0][0].faces, e.offsetX, e.offsetY, 15.0); 
};
```

---

## 🏁 Future Roadmap
- [ ] **LBM D3Q27 Solvers**: Full 3D fluid dynamics.
- [ ] **Three.js Direct Buffer**: Mapping tensors to `Data3DTexture`.
- [ ] **Interactive Sliders**: Integration with `lil-gui` for real-time parameter tuning.
- [ ] **WebGPU Particle Injection**: 1M+ particles reacting to the tensor flow fields.

---

## 🤖 AI-Friendly Guidelines (For Copilots)
1. **Never use `.map()`** or standard arrays for physics steps. Use `chunk.faces[n]`.
2. Map 2D/3D indices manually: `const idx = (z * ny * nx) + (y * nx) + x`.
3. Stick to **GodMode V8** loops (raw C-style) for guaranteed O(1) efficiency.

`Built with passion for high-performance creative computing.`
