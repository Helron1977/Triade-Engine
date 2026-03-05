import { HypercubeGPUContext } from '../core/gpu/HypercubeGPUContext';
import { HypercubeCpuGrid } from '../core/HypercubeCpuGrid';
import { HypercubeChunk } from '../core/HypercubeChunk';

/**
 * WebGpuRenderer
 * Direct-to-VRAM Rendering. Read the float StorageBuffer from the Engine, 
 * run a WGSL Compute Shader to translate into RGBA8Unorm colors,
 * and copy directly to the Canvas context via WebGPU.
 */
export class WebGpuRenderer {
    private canvas: HTMLCanvasElement;
    private context: GPUCanvasContext;
    private format: GPUTextureFormat;

    private pipeline: GPUComputePipeline | null = null;
    private blitPipeline: GPURenderPipeline | null = null;
    private bindGroups: Map<HypercubeChunk, GPUBindGroup> = new Map();
    private blitBindGroup: GPUBindGroup | null = null;
    private uniformBuffer: GPUBuffer | null = null;

    // We render into a storage texture first, since compute shaders can't write to the canvas texture directly 
    // unless the format supports storage binding (bgra8unorm often doesn't on all devices).
    private storageTexture: GPUTexture | null = null;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        const ctx = canvas.getContext('webgpu');
        if (!ctx) throw new Error("[WebGpuRenderer] Canvas does not support WebGPU.");
        this.context = ctx as unknown as GPUCanvasContext;

        // This format is required for the output texture
        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: HypercubeGPUContext.device,
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
        });
    }

    private initPipelines() {
        if (this.pipeline) return;

        const device = HypercubeGPUContext.device;
        const computeShaderCode = `
            struct Uniforms {
                nx: u32,
                ny: u32,
                nz: u32,
                strideFace: u32,
                faceIdx: u32,
                obsIdx: u32,
                minV: f32,
                maxV: f32,
                colormap: u32, // 0: grayscale, 1: heatmap, 2: vorticity
            };

            @group(0) @binding(0) var<storage, read> cube: array<f32>;
            @group(0) @binding(1) var<uniform> config: Uniforms;
            @group(0) @binding(2) var outTexture: texture_storage_2d<rgba8unorm, write>;

            @compute @workgroup_size(16, 16)
            fn compute_render(@builtin(global_invocation_id) id: vec3<u32>) {
                let lx = id.x + 1u; // Offset interior
                let ly = id.y + 1u;

                if (lx >= config.nx - 1u || ly >= config.ny - 1u) { return; }

                let srcIdx = ly * config.nx + lx;
                
                // Obstacles check
                if (config.obsIdx < 100u) {
                    let obsV = cube[config.obsIdx * config.strideFace + srcIdx];
                    if (obsV > 0.5) {
                        textureStore(outTexture, id.xy, vec4<f32>(0.2, 0.2, 0.2, 1.0));
                        return;
                    }
                }

                let rawVal = cube[config.faceIdx * config.strideFace + srcIdx];
                let norm = clamp((rawVal - config.minV) / (config.maxV - config.minV + 0.00001), 0.0, 1.0);
                
                var color = vec4<f32>(norm, norm, norm, 1.0);

                if (config.colormap == 1u) { // Heatmap
                    color.r = norm;
                    color.g = select(0.0, (norm - 0.5) * 2.0, norm > 0.5);
                    color.b = norm * 0.2;
                } else if (config.colormap == 2u) { // Vorticity
                    color.r = select(0.0, (norm - 0.5) * 2.0, norm > 0.5);
                    color.g = norm;
                    color.b = select((0.5 - norm) * 2.0, 0.0, norm > 0.5);
                }

                textureStore(outTexture, id.xy, color);
            }
        `;

        const computeModule = device.createShaderModule({ code: computeShaderCode });
        this.pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: computeModule, entryPoint: 'compute_render' }
        });

        const blitShaderCode = `
            @vertex
            fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f {
                const pos = array(
                    vec2f(-1.0, -1.0),
                    vec2f( 3.0, -1.0),
                    vec2f(-1.0,  3.0)
                );
                return vec4f(pos[vertexIndex], 0.0, 1.0);
            }

            @group(0) @binding(0) var t: texture_2d<f32>;

            @fragment
            fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
                return textureLoad(t, vec2i(pos.xy), 0);
            }
        `;

        const blitModule = device.createShaderModule({ code: blitShaderCode });
        this.blitPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: { module: blitModule, entryPoint: 'vs_main' },
            fragment: {
                module: blitModule,
                entryPoint: 'fs_main',
                targets: [{ format: this.format }]
            },
            primitive: { topology: 'triangle-list' }
        });

        const uniformSize = 9 * 4; // 9 floats/uints
        this.uniformBuffer = device.createBuffer({
            size: Math.ceil(uniformSize / 16) * 16, // align 16
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }

    public render(
        grid: HypercubeCpuGrid,
        options: {
            faceIndex: number,
            colormap: 'grayscale' | 'heatmap' | 'vorticity' | 'ocean',
            minVal?: number,
            maxVal?: number,
            sliceZ?: number,
            obstaclesFace?: number
        }
    ) {
        if (!HypercubeGPUContext.device) return;
        this.initPipelines();

        // Ensure texture exists and matches the grid's visual size
        const vnx = grid.nx - 2;
        const vny = grid.ny - 2;
        const totalW = vnx * grid.cols;
        const totalH = vny * grid.rows;

        // Ensure texture size matches canvas and logic
        if (!this.storageTexture || this.storageTexture.width !== totalW || this.storageTexture.height !== totalH) {
            if (this.storageTexture) this.storageTexture.destroy();
            this.storageTexture = HypercubeGPUContext.device.createTexture({
                size: [totalW, totalH, 1],
                format: 'rgba8unorm',
                usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
            });
            this.bindGroups.clear(); // Rebind since texture changed
            this.blitBindGroup = null;
        }

        const device = HypercubeGPUContext.device;
        const commandEncoder = device.createCommandEncoder();

        // 1. Update Uniforms
        const u32 = new Uint32Array(12);
        const f32 = new Float32Array(u32.buffer);
        u32[0] = grid.nx; u32[1] = grid.ny; u32[2] = grid.nz; u32[3] = grid.nx * grid.ny * grid.nz;
        u32[4] = options.faceIndex; u32[5] = options.obstaclesFace ?? 999;
        f32[6] = options.minVal ?? 0; f32[7] = options.maxVal ?? 1;
        u32[8] = options.colormap === 'heatmap' ? 1 : (options.colormap === 'vorticity' ? 2 : 0);

        device.queue.writeBuffer(this.uniformBuffer!, 0, u32.buffer);

        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.pipeline!);

        // 2. Dispatch chunks
        const view = this.storageTexture.createView();

        for (let gy = 0; gy < grid.rows; gy++) {
            for (let gx = 0; gx < grid.cols; gx++) {
                const chunk = grid.cubes[gy][gx];
                if (!chunk || !chunk.gpuBuffer) continue;

                // Create or reuse bindgroup (if texture/buffer hasn't changed)
                let bg = this.bindGroups.get(chunk);
                if (!bg) {
                    bg = device.createBindGroup({
                        layout: this.pipeline!.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: chunk.gpuBuffer } },
                            { binding: 1, resource: { buffer: this.uniformBuffer! } },
                            { binding: 2, resource: view }
                        ]
                    });
                    this.bindGroups.set(chunk, bg);
                }

                pass.setBindGroup(0, bg);
                pass.dispatchWorkgroups(Math.ceil(vnx / 16), Math.ceil(vny / 16), 1);
            }
        }
        pass.end();

        // 3. Blit from Storage Texture to Canvas Texture using a render pass
        const canvasTexture = this.context.getCurrentTexture();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: canvasTexture.createView(),
                loadOp: 'clear',
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                storeOp: 'store'
            }]
        });

        if (!this.blitBindGroup) {
            this.blitBindGroup = device.createBindGroup({
                layout: this.blitPipeline!.getBindGroupLayout(0),
                entries: [{ binding: 0, resource: view }]
            });
        }

        renderPass.setPipeline(this.blitPipeline!);
        renderPass.setBindGroup(0, this.blitBindGroup);
        renderPass.draw(3, 1, 0, 0);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);
    }
}
