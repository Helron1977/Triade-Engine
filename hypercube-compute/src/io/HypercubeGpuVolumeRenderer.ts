import { HypercubeGPUContext } from '../core/gpu/HypercubeGPUContext';
import { HypercubeCpuGrid } from '../core/HypercubeCpuGrid';
import { GPU_VOLUME_SHADERS } from './shaders/GpuVolumeShaders';

export interface GpuRendererOptions {
    faceIndex: number;
    obstacleFaceIndex?: number;
    vorticityFace?: number;
    minVal?: number;
    maxVal?: number;
    colormap?: 'grayscale' | 'ocean' | 'heatmap' | 'arctic';
    mode?: 'topdown' | 'isometric' | 'raymarch';
    scale?: number;
    opacity?: number;
}

/**
 * HypercubeGpuVolumeRenderer
 * 
 * Native WebGPU renderer that avoids CPU readbacks by sampling VRAM directly.
 * Replaces HypercubeIsoRenderer for maximum performance (V6.0).
 */
export class HypercubeGpuVolumeRenderer {
    private canvas: HTMLCanvasElement;
    private context: GPUCanvasContext;
    private format: GPUTextureFormat;
    private pipeline: GPURenderPipeline | null = null;
    private uniformBuffer: GPUBuffer | null = null;
    private chunkResources: Map<any, { buffer: GPUBuffer, bindGroup: GPUBindGroup }> = new Map();

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        const ctx = canvas.getContext('webgpu');
        if (!ctx) throw new Error("[HypercubeGpuVolumeRenderer] WebGPU not supported on this canvas.");
        this.context = ctx as unknown as GPUCanvasContext;

        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: HypercubeGPUContext.device,
            format: this.format,
            alphaMode: 'opaque'
        });
    }

    private initPipeline() {
        if (this.pipeline) return;

        const device = HypercubeGPUContext.device;
        const shaderModule = device.createShaderModule({
            code: GPU_VOLUME_SHADERS,
            label: 'Volume Renderer Shader'
        });

        this.pipeline = device.createRenderPipeline({
            label: 'Volume Renderer Pipeline',
            layout: 'auto',
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_main'
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
                targets: [{ format: this.format }]
            },
            primitive: {
                topology: 'triangle-strip',
                stripIndexFormat: undefined
            }
        });

        this.uniformBuffer = device.createBuffer({
            size: 128, // Enough for all uniforms
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'Volume Renderer Uniforms'
        });
    }

    private chunkUniformBuffers: GPUBuffer[] = [];
    private chunkBindGroups: GPUBindGroup[] = [];

    public render(grid: HypercubeCpuGrid, options: GpuRendererOptions) {
        if (!HypercubeGPUContext.device) return;
        this.initPipeline();

        const device = HypercubeGPUContext.device;
        const commandEncoder = device.createCommandEncoder();
        const canvasTexture = this.context.getCurrentTexture();

        // 1. Prepare Base Uniforms (nx, ny, faces, colormap, etc.)
        const uBuffer = new ArrayBuffer(128);
        const u32 = new Uint32Array(uBuffer);
        const f32 = new Float32Array(uBuffer);

        const firstChunk = grid.cubes.flat().find(c => c !== null);
        const strideFloats = firstChunk ? firstChunk.stride / 4 : grid.nx * grid.ny * grid.nz;

        u32[0] = grid.nx; u32[1] = grid.ny; u32[2] = grid.nz;
        u32[3] = strideFloats; // strideFace (respect alignment)
        u32[4] = options.faceIndex;
        u32[5] = options.obstacleFaceIndex ?? 999;
        f32[6] = options.minVal ?? 0.0;
        f32[7] = options.maxVal ?? 1.0;
        f32[8] = options.scale ?? 4.0;
        f32[9] = options.opacity ?? 1.0;
        f32[10] = performance.now() / 1000.0;

        let cmapIdx = 0;
        if (options.colormap === 'ocean') cmapIdx = 1;
        else if (options.colormap === 'heatmap') cmapIdx = 2;
        else if (options.colormap === 'arctic') cmapIdx = 3;
        u32[11] = cmapIdx;

        u32[12] = options.mode === 'isometric' ? 1 : (options.mode === 'raymarch' ? 2 : 0);
        f32[13] = this.canvas.width;
        f32[14] = this.canvas.height;
        u32[15] = grid.cols;
        u32[16] = grid.rows;
        u32[19] = options.vorticityFace ?? 999;

        // 2. Clear Pass
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: canvasTexture.createView(),
                clearValue: { r: 0.01, g: 0.02, b: 0.05, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store'
            }]
        });

        renderPass.setPipeline(this.pipeline!);

        const vnx = grid.nx - 2;
        const vny = grid.ny - 2;

        let chunkIdx = 0;
        for (let gy = 0; gy < grid.rows; gy++) {
            for (let gx = 0; gx < grid.cols; gx++) {
                const chunk = grid.cubes[gy][gx];
                if (!chunk || !chunk.gpuReadBuffer) continue;

                // Create or reuse uniform buffer for THIS chunk
                if (!this.chunkUniformBuffers[chunkIdx]) {
                    this.chunkUniformBuffers[chunkIdx] = device.createBuffer({
                        size: 128,
                        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
                    });
                }
                const chunkUBuffer = this.chunkUniformBuffers[chunkIdx];

                // Create or recreate bindgroup if buffer has changed (due to swap)
                const currentBuffer = chunk.gpuReadBuffer!;

                if (!this.chunkBindGroups[chunkIdx] || (this as any)._lastBoundBuffers?.[chunkIdx] !== currentBuffer) {
                    this.chunkBindGroups[chunkIdx] = device.createBindGroup({
                        layout: this.pipeline!.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: currentBuffer } },
                            { binding: 1, resource: { buffer: chunkUBuffer } }
                        ]
                    });
                    if (!(this as any)._lastBoundBuffers) (this as any)._lastBoundBuffers = [];
                    (this as any)._lastBoundBuffers[chunkIdx] = currentBuffer;
                }

                // Update Chunk specific uniforms
                u32[17] = gx;
                u32[18] = gy;
                device.queue.writeBuffer(chunkUBuffer, 0, uBuffer);

                renderPass.setBindGroup(0, this.chunkBindGroups[chunkIdx]);
                renderPass.draw(4, vnx * vny);

                chunkIdx++;
            }
        }

        renderPass.end();
        device.queue.submit([commandEncoder.finish()]);
    }
}
