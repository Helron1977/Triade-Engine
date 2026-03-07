/// <reference types="@webgpu/types" />
import { EngineDescriptor } from './EngineManifest';

/**
 * AbstractGpuEngine - The Boilerplate Killer.
 * Handles parity, BindGroups, and semantic mapping automatically.
 */
export abstract class AbstractGpuEngine {
    protected initialReadBuffer: GPUBuffer | null = null;
    protected initialWriteBuffer: GPUBuffer | null = null;
    protected pipeline: GPUComputePipeline | null = null;
    protected bindGroups: GPUBindGroup[] = []; // [Parity 0, Parity 1]

    constructor(protected descriptor: EngineDescriptor) { }

    protected getFaceIndex(name: string): number {
        const idx = this.descriptor.faces.findIndex(f => f.name === name);
        if (idx === -1) throw new Error(`[${this.descriptor.name}] Face inconnue: ${name}`);
        return idx;
    }

    /**
     * @description Identifie l'indice du paramètre dans le buffer Uniform (u.params[i])
     */
    protected getParamIndex(name: string): number {
        const idx = this.descriptor.parameters?.findIndex(p => p.name === name) ?? -1;
        if (idx === -1) throw new Error(`[${this.descriptor.name}] Paramètre inconnu: ${name}`);
        return idx;
    }

    /**
     * @description Initialisation générique. Crée les deux BindGroups (un par parité)
     */
    public initGPU(
        device: GPUDevice,
        readBuffer: GPUBuffer,
        writeBuffer: GPUBuffer,
        uniformBuffer: GPUBuffer,
        stride: number,
        nx: number,
        ny: number,
        nz: number
    ): void {
        this.initialReadBuffer = readBuffer;
        this.initialWriteBuffer = writeBuffer;

        const shaderSource = this.getShaderSource();
        const module = device.createShaderModule({ code: shaderSource });

        // 1. Create Layout (V8 Standard: In @ 0, Out @ 1, Uniforms @ 2)
        const layout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        this.pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
            compute: { module, entryPoint: 'main' }
        });

        // 2. Create BindGroups for both parities (Double Buffering)
        this.bindGroups[0] = device.createBindGroup({
            layout,
            entries: [
                { binding: 0, resource: { buffer: readBuffer } },
                { binding: 1, resource: { buffer: writeBuffer } },
                { binding: 2, resource: { buffer: uniformBuffer } }
            ]
        });

        this.bindGroups[1] = device.createBindGroup({
            layout,
            entries: [
                { binding: 0, resource: { buffer: writeBuffer } },
                { binding: 1, resource: { buffer: readBuffer } },
                { binding: 2, resource: { buffer: uniformBuffer } }
            ]
        });

        console.info(`[${this.descriptor.name}] GPU Initialized with semantic mapping and uniform buffer.`);
    }

    /**
     * @description Dispatch automatique du Compute Pass.
     */
    public computeGPU(
        device: GPUDevice,
        commandEncoder: GPUCommandEncoder,
        nx: number,
        ny: number,
        nz: number,
        readBuffer: GPUBuffer,
        writeBuffer: GPUBuffer
    ): void {
        if (!this.pipeline) return;

        // V8 Stateless Parity Recovery: 
        // Logic Read == Initial Read? => Parity 0 (Standard Ping-Pong)
        const currentParity = (readBuffer === this.initialReadBuffer) ? 0 : 1;

        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroups[currentParity]);

        // Alignment on 16 (V8 rule)
        pass.dispatchWorkgroups(Math.ceil(nx / 16), Math.ceil(ny / 16), nz || 1);
        pass.end();
    }

    protected abstract getShaderSource(): string;
}
