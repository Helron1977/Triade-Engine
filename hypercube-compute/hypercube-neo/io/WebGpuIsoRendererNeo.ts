import { HypercubeGPUContext } from '../core/gpu/HypercubeGPUContext';
import { NeoEngineProxy } from '../core/NeoEngineProxy';
import { MasterBuffer as NeoMasterBuffer } from '../core/MasterBuffer';

export interface IsoRenderOptions {
    densityFaceIndex: number;
    obstacleFaceIndex?: number;
    lodStep?: number;
}

/**
 * WebGpuIsoRendererNeo
 * 2.5D Isometric volumetric renderer running ENTIRELY on the GPU using Instanced Rendering.
 * ZERO-STALL: Translates LBM physics grid directly to screen pixels without RAM readbacks.
 */
export class WebGpuIsoRendererNeo {
    private canvas: HTMLCanvasElement;
    private context: GPUCanvasContext;
    private format: GPUTextureFormat;

    private pipeline: GPURenderPipeline | null = null;
    private uniformBuffer: GPUBuffer | null = null;
    private depthTexture: GPUTexture | null = null;
    private depthTextureView: GPUTextureView | null = null;
    private neoBindGroup: GPUBindGroup | null = null;
    private scale: number;

    constructor(canvas: HTMLCanvasElement, scale: number = 4.0) {
        this.canvas = canvas;
        const ctx = canvas.getContext('webgpu');
        if (!ctx) throw new Error("[WebGpuIsoRendererNeo] Canvas does not support WebGPU.");
        this.context = ctx as unknown as GPUCanvasContext;
        this.scale = scale;

        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: HypercubeGPUContext.device,
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });
    }

    private initPipelines() {
        if (this.pipeline) return;

        const device = HypercubeGPUContext.device;

        const shaderCode = `
            struct GpuObject {
                pos: vec2<f32>,
                dim: vec2<f32>,
                isObstacle: f32,
                biology: f32,
                objType: u32,
                rho: f32,
            };

            struct Uniforms {
                nx: u32,
                ny: u32,
                step: u32,
                faceIdx: u32,
                obsIdx: u32,
                scale: f32,
                centerX: f32,
                centerY: f32,
                strideFace: u32,
                canvasWidth: f32,
                canvasHeight: f32,
                numObjects: u32,
                objects: array<GpuObject, 8>,
            };

            @group(0) @binding(0) var<storage, read> cube: array<f32>;
            @group(0) @binding(1) var<uniform> config: Uniforms;

            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) color: vec4<f32>,
            };

            @vertex
            fn vs_main(@builtin(vertex_index) vIdx: u32, @builtin(instance_index) instIdx: u32) -> VertexOutput {
                var out: VertexOutput;
                
                let lx = instIdx % config.nx;
                let ly = instIdx / config.nx;
                
                // No skipping borders here, we use the virtual coordinate lx, ly [0..nx-1]
                // But we pull from the physical buffer which has ghost cells.
                let pNx = config.nx + 2u;
                let srcIdx = (ly + 1u) * pNx + (lx + 1u);
                
                let rawVal = cube[config.faceIdx * config.strideFace + srcIdx];
                
                var isObs = false;
                if (config.obsIdx < 100u) {
                    if (cube[config.obsIdx * config.strideFace + srcIdx] > 0.5) {
                        isObs = true;
                    }
                }
                
                // Dynamic Objects Check
                for (var j = 0u; j < config.numObjects; j = j + 1u) {
                    let obj = config.objects[j];
                    if (obj.isObstacle < 0.5) { continue; }
                    var inObj = false;
                    if (obj.objType == 1u) { // Circle
                        let r = obj.dim.x * 0.5;
                        let ddx = f32(lx) - obj.pos.x;
                        let ddy = f32(ly) - obj.pos.y;
                        if (ddx*ddx + ddy*ddy <= r*r) { inObj = true; }
                    } else if (obj.objType == 2u) { // Rect
                        if (f32(lx) >= obj.pos.x && f32(lx) <= obj.pos.x + obj.dim.x &&
                            f32(ly) >= obj.pos.y && f32(ly) <= obj.pos.y + obj.dim.y) { inObj = true; }
                    }
                    if (inObj) { isObs = true; break; }
                }
                
                if (rawVal < 0.01 && !isObs) {
                    out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
                    return out;
                }
                
                let scale = config.scale;
                let h = select(rawVal * scale * 25.0, scale * 10.0, isObs);
                
                let vNX = f32(config.nx);
                let vNY = f32(config.ny);
                let midW = vNX / 2.0;
                let midH = vNY / 2.0;

                let isoXScale = scale * 0.866;
                let isoYScale = scale * 0.5;

                let worldX = f32(lx) - midW;
                let worldY = f32(ly) - midH;

                let x = config.centerX + (worldX - worldY) * isoXScale;
                let y = config.centerY + (worldX + worldY) * isoYScale;
                
                let rw = scale * f32(config.step) + 0.5;
                let rh = select(scale, h, h > 0.0);
                
                // Quad Vertex Mapping (Triangle Strip)
                // 0: top-left, 1: bottom-left, 2: top-right, 3: bottom-right
                var px = x;
                var py = y;
                if (vIdx == 0u || vIdx == 2u) { py = y - rh; }
                if (vIdx == 2u || vIdx == 3u) { px = x + rw; }
                
                let ndcX = (px / config.canvasWidth) * 2.0 - 1.0;
                let ndcY = 1.0 - (py / config.canvasHeight) * 2.0; 
                
                // Depth algorithm logically matching CPU Painter's algorithm
                let z = 1.0 - (f32(instIdx) / f32(config.nx * config.ny));
                
                out.position = vec4<f32>(ndcX, ndcY, z, 1.0);
                
                if (isObs) {
                    out.color = vec4<f32>(0.2, 0.2, 0.2, 1.0);
                } else {
                    let intensity = (rawVal - 1.0) * 800.0;
                    let r = clamp(20.0 + intensity, 0.0, 255.0) / 255.0;
                    let g = clamp(100.0 + intensity, 0.0, 255.0) / 255.0;
                    let b = clamp(200.0 + intensity, 0.0, 255.0) / 255.0;
                    out.color = vec4<f32>(r, g, b, 1.0);
                }
                
                return out;
            }

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                return in.color;
            }
        `;

        const module = device.createShaderModule({ code: shaderCode });

        this.pipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module,
                entryPoint: 'vs_main'
            },
            fragment: {
                module,
                entryPoint: 'fs_main',
                targets: [{ format: this.format }]
            },
            primitive: {
                topology: 'triangle-strip',
                cullMode: 'none'
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth32float'
            }
        });

        this.uniformBuffer = device.createBuffer({
            size: 512, // Enough for parameters + 8 objects
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }

    private ensureDepthTexture() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        if (!this.depthTexture || this.depthTexture.width !== width || this.depthTexture.height !== height) {
            if (this.depthTexture) this.depthTexture.destroy();
            this.depthTexture = HypercubeGPUContext.device.createTexture({
                size: [width, height],
                format: 'depth32float',
                usage: GPUTextureUsage.RENDER_ATTACHMENT
            });
            this.depthTextureView = this.depthTexture.createView();
        }
    }

    public render(
        proxy: NeoEngineProxy,
        options: IsoRenderOptions
    ) {
        if (!HypercubeGPUContext.device) return;
        this.initPipelines();
        this.ensureDepthTexture();

        const vGrid = proxy.vGrid;
        const bridge = (proxy.bridge as any);
        if (!bridge.gpuBuffer) return;

        // Note: Zero-Stall Ocean operates natively as 1 Chunk in GPU Mode.
        const nx = vGrid.dimensions.nx;
        const ny = vGrid.dimensions.ny;

        // Setup Uniforms
        const u32Data = new Uint32Array(128); // Expanded
        const f32Data = new Float32Array(u32Data.buffer);

        const vNX = nx - 2;
        const vNY = ny - 2;
        const isoYScale = this.scale * 0.5;
        const midH = vNY / 2.0;

        u32Data[0] = nx;
        u32Data[1] = ny;
        u32Data[2] = options.lodStep || 2;
        u32Data[3] = options.densityFaceIndex;
        u32Data[4] = options.obstacleFaceIndex ?? 999;
        f32Data[5] = this.scale;
        f32Data[6] = this.canvas.width / 2.0;
        f32Data[7] = this.canvas.height / 2.0 + (midH * isoYScale * 0.5);
        u32Data[8] = bridge.strideFace;
        f32Data[9] = this.canvas.width;
        f32Data[10] = this.canvas.height;

        const objects = (proxy.vGrid as any).config.objects || [];
        u32Data[11] = Math.min(objects.length, 8);

        // Pack Objects (base 12, each obj 8 words)
        for (let j = 0; j < Math.min(objects.length, 8); j++) {
            const objBase = 12 + j * 8;
            const obj = objects[j];
            f32Data[objBase + 0] = obj.position.x;
            f32Data[objBase + 1] = obj.position.y;
            f32Data[objBase + 2] = obj.dimensions.w;
            f32Data[objBase + 3] = obj.dimensions.h;
            f32Data[objBase + 4] = obj.properties.isObstacle ?? 0;
            f32Data[objBase + 5] = obj.properties.biology ?? 0;
            u32Data[objBase + 6] = (obj.type === 'circle' ? 1 : (obj.type === 'rect' ? 2 : 0));
            f32Data[objBase + 7] = obj.properties.rho ?? 0;
        }

        HypercubeGPUContext.device.queue.writeBuffer(this.uniformBuffer!, 0, u32Data);

        if (!this.neoBindGroup) {
            this.neoBindGroup = HypercubeGPUContext.device.createBindGroup({
                layout: this.pipeline!.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: bridge.gpuBuffer } },
                    { binding: 1, resource: { buffer: this.uniformBuffer! } }
                ]
            });
        }

        const commandEncoder = HypercubeGPUContext.device.createCommandEncoder();
        const textureView = this.context.getCurrentTexture().createView();

        const pass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 5 / 255, g: 15 / 255, b: 35 / 255, a: 1.0 }, // Deep sea dark blue
                loadOp: 'clear',
                storeOp: 'store'
            }],
            depthStencilAttachment: {
                view: this.depthTextureView!,
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store'
            }
        });

        pass.setPipeline(this.pipeline!);
        pass.setBindGroup(0, this.neoBindGroup);
        // Dispatch 1 quad (4 vertices) per grid cell
        pass.draw(4, nx * ny, 0, 0);
        pass.end();

        HypercubeGPUContext.device.queue.submit([commandEncoder.finish()]);
    }
}
