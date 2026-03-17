import { GpuKernelRegistry } from './GpuKernelRegistry';

import NeoAeroSource from './wgsl/NeoAero.wgsl?raw';
import NeoOceanSource from './wgsl/NeoOcean.wgsl?raw';
import NeoSDFSource from './wgsl/NeoSDF.wgsl?raw';
import NeoHeatSource from './wgsl/NeoHeat.wgsl?raw';
import NeoLifeSource from './wgsl/NeoLife.wgsl?raw';
import NeoPath from './wgsl/NeoPath.wgsl?raw';
import NeoTensor from './wgsl/NeoTensor.wgsl?raw';

/**
 * Initializes the GPU compute kernels for Hypercube Neo.
 */
export function initializeGpuKernels(): void {
    GpuKernelRegistry.setSource('lbm-aero-fidelity-v1', NeoAeroSource);
    GpuKernelRegistry.setMetadata('lbm-aero-fidelity-v1', { uniformObjectOffset: 32 });

    GpuKernelRegistry.setSource('neo-ocean-v1', NeoOceanSource);
    GpuKernelRegistry.setMetadata('neo-ocean-v1', { uniformObjectOffset: 32 });

    GpuKernelRegistry.setSource('neo-sdf', NeoSDFSource);
    GpuKernelRegistry.setMetadata('neo-sdf', { uniformObjectOffset: 32 });

    GpuKernelRegistry.setSource('neo-heat', NeoHeatSource);
    GpuKernelRegistry.setMetadata('neo-heat', { uniformObjectOffset: 32 });
    GpuKernelRegistry.setSource('neo-heat-v1', NeoHeatSource);
    GpuKernelRegistry.setMetadata('neo-heat-v1', { uniformObjectOffset: 32 });

    GpuKernelRegistry.setSource('neo-life-v1', NeoLifeSource);
    GpuKernelRegistry.setMetadata('neo-life-v1', { uniformObjectOffset: 32 });

    GpuKernelRegistry.setSource('neo-path-v1', NeoPath);
    GpuKernelRegistry.setMetadata('neo-path-v1', { uniformObjectOffset: 32 });

    GpuKernelRegistry.setSource('neo-tensor-cp-v1', NeoTensor);
    GpuKernelRegistry.setMetadata('neo-tensor-cp-v1', { uniformObjectOffset: 32 });
}
