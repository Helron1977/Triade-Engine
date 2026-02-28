# Triade Engine V2

A high-performance O(1) tensor-based compute engine for Web and Node.js environments.

## Features
- **O(1) Data Access**: Leverages flat `SharedArrayBuffer` (or standard `ArrayBuffer`) via `Int32Array` or `Float32Array`.
- **Zero-Allocation Computations**: Reusable grids without per-frame memory allocation.
- **Built-in Engines**: Included are complex algorithms such as LBM D2Q9 (Aerodynamics), Game of Life, Heatmaps, and Ocean Simulation parameters.
- **Ecosystem Ready**: Fully uncoupled from UI frameworks for seamless integration with React, Vue, pure Canvas, WebGL, or WebGPU.

## Installation
```bash
npm install triade-engine
```

## Usage
```typescript
import { Triade } from 'triade-engine';

const engine = new Triade(50); // Allocate 50MB of raw buffer memory
// Register maps and specific algorithms...
```

## Structure
- `TriadeMasterBuffer`: Memory controller
- `TriadeCubeV2`: Tensor logic
- `Engines`: Interfaces and logic models implementation
