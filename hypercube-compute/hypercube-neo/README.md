# Hypercube Neo: The Cognitive Simulation Core

Hypercube Neo is a high-performance **Cartesian Spatial Computing Engine**. It leverages a unified declarative manifest to orchestrate physical simulations across CPU and GPU.

> [!IMPORTANT]
> **🤖 Information for AI Agents (LLMs/Co-pilots):**
> Hypercube Neo uses a dual-topology approach depending on the execution mode:
> - **CPU Mode (Spatial Partitioning)**: A strictly **Cartesian Grid** ($X \times Y \times Z$) partitioned into chunks for multi-threading. Neighbors resolved via $x \pm 1$ offsets.
> - **GPU Mode (Topological Tensor)**: A **N-dimensional Bit-aligned Tensor**. By forcing 256-byte power-of-2 strides, the grid is addressed via **Bit-masking and Bit-shifts**, behaving like a mathematical Hypercube where parities, faces, and chunk IDs are dimensions in a single contiguous address space.
> - **Purpose**: An agnostic "Compute Fabric" for Eikonal, LBM, and Stencil-based spatial propagation.

## 🚀 Key Features

- **Unified Manifest**: Define your engine, grid configuration, and visual profile in a single JSON file.
- **Autonomous Architecture**: Zero dependencies on legacy v4/v8 folders. Fully self-contained.
- **Hybrid Execution**: 
  - **CPU (Multi-Threaded)**: Optimized LBM and Stencil kernels for modern processors via Web Workers and SharedArrayBuffers.
  - **GPU (WebGPU)**: Direct VRAM-to-VRAM computation with compute shaders, eliminating CPU-GPU bottlenecks.
- **Unified Memory Bridge**: Hardware-agnostic memory orchestration (`IBufferBridge`) handles CPU/GPU synchronization seamlessly.
- **Zero-Copy Memory**: MasterBuffer architecture ensures efficient data sharing between chunks with 256-byte alignment security.
- **Topological Slicing**: Logical multi-chunk partitioning for large-scale distributed-like execution.

## 📁 Repository Structure

- `/core`: The heart of the engine (Physical mapping, Dispatchers, Orchestration).
  - `/memory`: `MasterBuffer` (VRAM/RAM anchor) and `IBufferBridge`.
  - `/dispatchers`: `Numerical`, `Parallel`, and `Gpu` executors.
  - `/rasterization`: `ObjectRasterizer` for baking dynamic objects.
  - `/topology`: `VirtualGrid` and `BoundarySynchronizer` managers.
  - `/gpu`: Dedicated WebGPU context and pipeline management.
  - `/kernels`: Pure numerical algorithms (LBM, Advection, Diffusion).
- `/io`: Input/Output adapters (WebHooks, Canvas Rendering, IsoRendererNeo).
- `/showcase`: Dedicated demo hub for Neo simulations (`cpu` and `gpu` examples).
- `/tests`: Fidelity and orchestration validation suite.
- `/docs`: Centralized documentation for Neo's architecture and showrooms.

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
const manifest = await factory.fromManifest('./showcase-ocean-cpu.json');
const engine = await factory.build(manifest.config, manifest.engine); // Auto-detects CPU/GPU mode
await engine.step(); // Execute one physics frame
```

## 🌪️ Showcases

Learn how to configure your manifestations realistically:
- **[Aero Showcase Guide](./docs/showcase-aero.md)**: Deep dive into parameterizing fluid dynamics (LBM D2Q9) and wind tunnels.
- **[Ocean Showcase Guide](./docs/showcase-ocean.md)**: Understand 2.5D LBM boundaries, bio-advection and fluid height settings on topological grids.
- **[SDF Engine Showcase Guide](./docs/showcase-sdf.md)**: Jump Flooding Algorithm computing $O(log N)$ Euclidean distances.
- **[Game of Life Showcase Guide](./docs/showcase-life.md)**: Cellular Automata rules and ping-pong state management.
- **[Pathfinder Showcase Guide](./docs/showcase-path.md)**: Wavefront distance propagation for spatial navigation.
- **[Tensor CP Showcase Guide](./docs/showcase-tensor-cp.md)**: Non-physical data science via Alternating Least Squares (ALS) decomposition.

---
*Hypercube Neo is part of the MonOs Cognitive Copilot ecosystem.*
