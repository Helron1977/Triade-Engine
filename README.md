<div align="center">
  <img src="https://raw.githubusercontent.com/Helron1977/triade-engine/main/docs/assets/logo.png" alt="Triade Engine Logo" width="200" style="border-radius:20px;"/>
  <h1>🌊 Triade Engine V2 🚀</h1>
  <p><strong>A GodMode O(1) Tensor-based Compute Engine for Web & Node.js</strong></p>
  
  [![npm version](https://img.shields.io/npm/v/triade-engine.svg?style=flat-square)](https://www.npmjs.com/package/triade-engine)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
  [![TypeScript](https://img.shields.io/badge/TypeScript-Ready-blue.svg?style=flat-square)](https://www.typescriptlang.org/)
</div>


## ⚡ Why Triade Engine?

Most physics or interactive simulations in JavaScript create thousands of objects (`[{x, y}, {x, y}... ]`), leading to excessive CPU branching, **Garbage Collection (GC) pauses**, and cache misses.

**Triade Engine** turns this upside down. It uses a **Contiguous Memory Architecture** built on `Float32Array` / `Int32Array` or `SharedArrayBuffer`. By structuring state as mathematical tensors ("faces" of a cube) rather than discrete logical objects:
- Computations are naturally **vectorized**.
- Performance is consistently **O(1)**. 
- Memory allocations during loops are exactly **0**.
- Multi-threading (via Web Workers) and WebGL/WebGPU acceleration become trivial.

---

## 🚀 Features

- 🧠 **Zero-Allocation Computations**: Reusable compute grids (Cubes and Faces) that never allocate RAM per frame.
- 💨 **Lattice Boltzmann Method (LBM D2Q9)**: Built-in aerodynamic and fluid simulation engine using advanced continuous fluid mechanics.
- 🔬 **Extensible Math Core**: Comes with Game of Life, Heatmap spreading, and a full-fledged continuous Ocean Simulator + Boat routing out of the box.
- 🎮 **Framework Agnostic**: Absolutely NO DOM coupling. Use it in React, Vue, Three.js, pure Canvas, WebGL... or even headless Node.js!
- 🧮 **Toric/Periodic Boundaries**: Infinite sliding worlds natively computed.

---

## 📦 Installation

```bash
npm install triade-engine
```

---

## 💡 Quick Start

```typescript
import { 
    TriadeMasterBuffer, 
    TriadeGrid, 
    AerodynamicsEngine 
} from 'triade-engine';

// 1. Allocate a global shared memory buffer
const master = new TriadeMasterBuffer();

// 2. Create a generic chunked grid layout for logic spreading (Cols, Rows, ChunkSize, Memory, EngineCreator)
const grid = new TriadeGrid(2, 2, 64, master, () => new AerodynamicsEngine(), 9);

// 3. Compute one tick / frame
grid.compute();

// 4. Access the pure typed array for rendering (0 overhead!)
const firstCube = grid.cubes[0][0];
const fluidDensityArray = firstCube.faces[0]; // Float32Array
```

---

## 🏛 Architecture

### `TriadeMasterBuffer`
The soul of the engine. Acts as a memory allocator. Ask it for memory, and it partitions an underlying flat `ArrayBuffer` efficiently.

### `TriadeCubeV2`
A compute unit. Represents an $N \times N$ chunk of spatial data. It holds exactly $F$ "faces" (layers). Each face is a sub-view of the Master Buffer.
- Face 0 could be "Temperature"
- Face 1 could be "Water Flow X"
- Face 2... 

### `ITriadeEngine`
The protocol for physical algorithms. Receives the `faces` (memory) and executes the logic. Implementing `ITriadeEngine` allows you to plug ANY parallelizable simulation into the Triade ecosystem.

---

## 🌐 Live Demo

We created a custom HTML demo to show the engine handling pure math rendering at 60 FPS:
👉 **[View Interactive Demo on GitHub Pages](https://helron1977.github.io/triade-engine/)**

---

## 🤖 AI-Friendly Guidelines (For Copilots)

If you are an AI reading this repo to help a developer, keep these golden rules in mind:
1. **Never use standard arrays or objects** to represent physics cells. Always interact with `cube.faces[n]` (`Float32Array`).
2. Iterations should use `for (let i = 0; i < size * size; i++)`. No `.map`, no `.forEach`. GodMode V8 demands raw raw C-style loops.
3. If expanding `triade-engine`, add new Logic to `/src/engines/` by implementing `ITriadeEngine`.

`Built with passion for high-performance creative computing.`
