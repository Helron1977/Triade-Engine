import type { HypercubeChunk } from '../HypercubeChunk';

export class HypercubeGPUContext {
    private static _device: GPUDevice | null = null;
    private static _adapter: GPUAdapter | null = null;

    public static get device(): GPUDevice {
        if (!this._device) {
            throw new Error("[HypercubeGPUContext] WebGPU device n'est pas initialisé. Appelez HypercubeGPUContext.init() avant d'instancier la Factory ou la Grille.");
        }
        return this._device;
    }

    public static get isInitialized(): boolean {
        return this._device !== null;
    }

    static async init(): Promise<boolean> {
        if (!navigator.gpu) {
            console.error("[HypercubeGPUContext] WebGPU n'est pas supporté par ce navigateur.");
            return false;
        }

        this._adapter = await navigator.gpu.requestAdapter();
        if (!this._adapter) {
            console.error("[HypercubeGPUContext] Impossible d'obtenir un GPUAdapter.");
            return false;
        }

        this._device = await this._adapter.requestDevice({
            label: 'Hypercube GPU Device',
            requiredLimits: {
                maxStorageBufferBindingSize: 1_073_741_824, // 1GB (utile pour gros grids)
            }
        });

        console.info("[HypercubeGPUContext] WebGPU initialisé avec succès.");
        return true;
    }

    // ── GPU REFACTO V5.4 ──
    static createComputePipeline(wgslSource: string, label = 'Compute Pipeline'): GPUComputePipeline {
        const shaderModule = this.device.createShaderModule({
            code: wgslSource,
            label: `${label} Shader`
        });

        return this.device.createComputePipeline({
            label,
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });
    }

    static createStorageBuffer(size: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC): GPUBuffer {
        return this.device.createBuffer({
            size: Math.ceil(size / 4) * 4,
            usage,
            label: 'Hypercube Storage Buffer'
        });
    }

    // ── GPU REFACTO V5.4 ──
    static createPingPongBindGroup(
        pipeline: GPUComputePipeline,
        readBuffer: GPUBuffer,
        writeBuffer: GPUBuffer,
        uniformBuffer?: GPUBuffer
    ): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: readBuffer } },   // read only
            { binding: 1, resource: { buffer: writeBuffer } }   // read_write
        ];

        if (uniformBuffer) {
            entries.push({ binding: 2, resource: { buffer: uniformBuffer } });
        }

        return this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries
        });
    }

    // ── GPU REFACTO V5.4 ──
    /**
     * Copie optimisée de frontières entre chunks GPU (utilisé par synchronizeBoundariesGPU)
     */
    static gpuCopyBoundary(
        encoder: GPUCommandEncoder,
        srcBuffer: GPUBuffer,
        dstBuffer: GPUBuffer,
        srcByteOffset: number,
        dstByteOffset: number,
        byteSize: number,
        rowCount: number = 1,
        srcStride: number = 0,
        dstStride: number = 0
    ): number {
        if (rowCount <= 1) {
            encoder.copyBufferToBuffer(srcBuffer, srcByteOffset, dstBuffer, dstByteOffset, byteSize);
            return 1;
        } else {
            for (let i = 0; i < rowCount; i++) {
                encoder.copyBufferToBuffer(
                    srcBuffer,
                    srcByteOffset + i * srcStride,
                    dstBuffer,
                    dstByteOffset + i * dstStride,
                    byteSize
                );
            }
            return rowCount;
        }
    }

    // ── GPU REFACTO V5.4 ──
    /** Debug only - Ne jamais appeler dans la boucle principale */
    static async debugReadback(buffer: GPUBuffer, size: number): Promise<Float32Array> {
        const staging = this.device.createBuffer({
            size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            label: 'Debug Readback Staging'
        });

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
        this.device.queue.submit([encoder.finish()]);

        await staging.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();

        return data;
    }

    // ── GPU REFACTO V5.4 ──
    static destroy() {
        // À appeler lors du cleanup de l'application
        this._device?.destroy();
        this._device = null;
        this._adapter = null;
    }
}