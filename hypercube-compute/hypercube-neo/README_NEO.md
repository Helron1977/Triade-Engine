# Hypercube Neo: The Cognitive Simulation Core

Hypercube Neo is the high-performance, autonomous evolution of the Hypercube Compute engine. It is designed to be a **Single Source of Truth** for physical simulations, leveraging a unified declarative manifest to orchestrate both CPU and GPU execution.

## 🚀 Key Features

- **Unified Manifest**: Define your engine, grid configuration, and visual profile in a single JSON file.
- **Autonomous Architecture**: Zero dependencies on legacy v4/v8 folders.
- **Hybrid Execution**: 
  - **CPU (Multi-Threaded)**: Optimized LBM and Stencil kernels for modern processors via Web Workers and SharedArrayBuffers.
  - **GPU (WebGPU)**: Direct VRAM-to-VRAM computation with compute shaders, eliminating CPU-GPU bottlenecks.
- **Zero-Copy Memory**: MasterBuffer architecture ensures efficient data sharing between chunks.

## 📁 Repository Structure

- `/core`: The heart of the engine (MasterBuffer, Dispatchers, Factory).
  - `/gpu`: Dedicated WebGPU context and pipeline management.
  - `/kernels`: Pure numerical algorithms (LBM, Advection, Diffusion).
- `/io`: Input/Output adapters (WebHooks, Canvas Rendering).
- `/showcase`: Dedicated demo hub for Neo simulations.
- `/tests`: Fidelity and orchestration validation suite.

## 🛠️ Usage for Novices

To start a Neo simulation, you don't need to write complex code. You only need a **Manifest**.

### 1. The Manifest Concept
A manifest tells Neo:
- **"What is the physics?"** (The `engine` section)
- **"How big is the world?"** (The `config` section)
- **"How should it look?"** (The `visualProfile` section)

### 2. High-Level Factory
```typescript
import { HypercubeNeoFactory } from './core/HypercubeNeoFactory';

const factory = new HypercubeNeoFactory();
const engine = await factory.build(config, descriptor); // Auto-detects CPU/GPU mode
await engine.step(); // Execute one physics frame
```

## 🌪️ Showcase: Aerodynamics
Check the [Aero Showcase Guide](../docs/neo/showcase-aero.md) for a deep dive into parameterizing fluid dynamics.

---
*Hypercube Neo is part of the MonOs Cognitive Copilot ecosystem.*
