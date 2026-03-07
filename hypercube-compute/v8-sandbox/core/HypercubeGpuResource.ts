/// <reference types="@webgpu/types" />
import { HypercubeGPUContext } from '../../src/core/gpu/HypercubeGPUContext';

/**
 * Gère les ressources bas-niveau WebGPU pour un chunk.
 * Sépare la logique de buffering de la structure de données logique (HypercubeChunk).
 */
export class HypercubeGpuResource {
    public readBuffer: GPUBuffer | null = null;
    public writeBuffer: GPUBuffer | null = null;
    public stagingBuffer: GPUBuffer | null = null;
    public parity: number = 0;

    constructor(private totalSize: number, private label: string) { }

    init() {
        const device = HypercubeGPUContext.device;
        this.readBuffer = HypercubeGPUContext.createStorageBuffer(this.totalSize);
        this.writeBuffer = HypercubeGPUContext.createStorageBuffer(this.totalSize);

        this.stagingBuffer = device.createBuffer({
            size: this.totalSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            label: `Staging for ${this.label}`
        });
    }

    swap() {
        if (this.readBuffer && this.writeBuffer) {
            [this.readBuffer, this.writeBuffer] = [this.writeBuffer, this.readBuffer];
            this.parity = 1 - this.parity;
        }
    }

    destroy() {
        this.readBuffer?.destroy();
        this.writeBuffer?.destroy();
        this.stagingBuffer?.destroy();
        this.readBuffer = null;
        this.writeBuffer = null;
        this.stagingBuffer = null;
    }
}
