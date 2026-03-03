# 07 - Volume Diffusion 3D

This example demonstrates a 3D Volume Diffusion simulation using the **Hypercube Compute** engine.

## Features
- **3D Compute**: Simulates heat/concentration diffusion in a 64x64x64 (or 128x128x128) grid.
- **Hybrid GPU/CPU**: Dynamic host/device synchronization with a toggle to switch between CPU and WebGPU.
- **Dynamic Workgroups**: WGSL shader automatically scales its thread density (8x8x4, 8x8x8, or 16x8x8) based on your hardware's `maxComputeInvocationsPerWorkgroup`.
- **Volume Rendering**: Uses `HypercubeIsoRenderer` to visualize isothermal surfaces (voxels) in real-time with depth-sorted bucket transparency.
- **Slice View**: Toggleable 2D slice view of the 3D volume.
- **Performance**: Capable of 60 FPS on 64³ systems and high-scale 128³ simulations on GPU.

## Controls
- **View Mode**: Switch between Isometric (3D) and Slice (2D) views.
- **Diffusion Rate**: Adjust the speed of the diffusion process.
- **Reset**: Inject a "hot" sphere back into the center of the volume.

## Getting Started
1. Install dependencies:
   ```bash
   npm install
   ```
2. Run the development server:
   ```bash
   npm run dev
   ```
