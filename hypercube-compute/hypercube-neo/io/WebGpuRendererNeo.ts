import { HypercubeGPUContext } from '../core/gpu/HypercubeGPUContext';
import { NeoEngineProxy } from '../core/NeoEngineProxy';
import { MasterBuffer as NeoMasterBuffer } from '../core/MasterBuffer';

export interface RenderOptions {
    faceIndex: number;
    colormap: 'grayscale' | 'heatmap' | 'arctic' | 'spatial-decision';
    minVal?: number;
    maxVal?: number;
    obstaclesFace?: number;
    vorticityFace?: number; // Deprecated, use auxiliaryFaces
    sliceZ?: number;
    criteriaSDF?: { xFace: number, yFace: number, weight: number, distanceThreshold: number }[];
    auxiliaryFaces?: number[]; // Slots 0-7: [0]=Vorticity, [1]=vx, [2]=vy, etc.
}

/**
 * WebGpuRendererNeo
 * Neo-native Direct-to-VRAM Rendering.
 * Efficiently assembles multi-chunk buffers directly in WebGPU.
 */
export class WebGpuRendererNeo {
    private canvas: HTMLCanvasElement;
    private context: GPUCanvasContext;
    private format: GPUTextureFormat;

    private pipeline: GPUComputePipeline | null = null;
    private blitPipeline: GPURenderPipeline | null = null;
    private blitBindGroup: GPUBindGroup | null = null;
    private uniformBuffer: GPUBuffer | null = null;
    private storageTexture: GPUTexture | null = null;
    private neoBindGroups: Map<string, GPUBindGroup> = new Map();

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        const ctx = canvas.getContext('webgpu');
        if (!ctx) throw new Error("[WebGpuRendererNeo] Canvas does not support WebGPU.");
        this.context = ctx as unknown as GPUCanvasContext;

        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: HypercubeGPUContext.device,
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
            alphaMode: 'premultiplied'
        });
    }

    private initPipelines() {
        if (this.pipeline) return;

        const device = HypercubeGPUContext.device;
        const computeShaderCode = `
            struct Criteria {
                weight: f32,
                threshold: f32,
                xIdx: u32,
                yIdx: u32
            };

            struct Uniforms {
                nx: u32,
                ny: u32,
                nz: u32,
                ghosts: u32,
                faceIdx: u32,
                obsIdx: u32,
                minV: f32,
                maxV: f32,
                colormap: u32, 
                worldX0: u32,
                worldY0: u32,
                strideFace: u32,
                readParity: u32,
                numCriteria: u32,
                pixelScale: f32, 
                _pad3: u32, 
                aux: array<u32, 8>, // Slots 0-7: [0]=Vorticity, [1]=vx, [2]=vy, etc.
                criteria: array<Criteria, 6>
            };

            @group(0) @binding(0) var<storage, read> cube: array<f32>;
            @group(0) @binding(1) var<uniform> config: Uniforms;
            @group(0) @binding(2) var outTexture: texture_storage_2d<rgba8unorm, write>;

                @compute @workgroup_size(16, 16)
                fn compute_render(@builtin(global_invocation_id) id: vec3<u32>) {
                    if (id.x >= config.nx || id.y >= config.ny) { return; }
    
                    // Dynamic padding awareness (Neo MasterBuffer structure)
                    let pNx = config.nx + 2u * config.ghosts;
                    let srcIdx = (id.y + config.ghosts) * pNx + (id.x + config.ghosts);
                    
                    let outPos = vec2<u32>(
                        id.x + config.worldX0,
                        id.y + config.worldY0
                    );
    
                    // 1. Obstacles (Priority)
                    if (config.obsIdx < 100u) {
                        let obsV = cube[config.obsIdx * config.strideFace + srcIdx];
                        if (obsV > 0.5) {
                            // For all colormaps, obstacles are now SOLID BLACK for better contrast with leaflet
                            textureStore(outTexture, outPos, vec4<f32>(0.0, 0.0, 0.0, 1.0));
                            return;
                        }
                    }

                var r: f32 = 0.0;
                var g: f32 = 0.0;
                var b: f32 = 0.0;
                var a: f32 = 1.0;

                if (config.colormap == 3u) { // Spatial Decision (SDF)
                    let gX = f32(config.worldX0 + id.x);
                    let gY = f32(config.worldY0 + id.y);
                    
                    var score: f32 = 0.0;
                    var sumW: f32 = 0.0;
                    
                    for(var c = 0u; c < config.numCriteria; c = c + 1u) {
                        sumW = sumW + config.criteria[c].weight;
                    }

                    if (sumW > 0.0) {
                        for(var c = 0u; c < config.numCriteria; c = c + 1u) {
                            let weight = config.criteria[c].weight;
                            if (weight <= 0.0) { continue; }

                            let seedX = cube[config.criteria[c].xIdx * config.strideFace + srcIdx];
                            let seedY = cube[config.criteria[c].yIdx * config.strideFace + srcIdx];

                            if (seedX < -9000.0) { continue; }

                            let dx = gX - seedX;
                            let dy = gY - seedY;
                            // EXACT same logic as CPU for physical mapping.
                            let distMeters = sqrt(dx*dx + dy*dy) * config.pixelScale;
                            let thresh = config.criteria[c].threshold;

                            if (distMeters <= thresh) {
                                // Zone satisfaction: 1.0 at center, drops to 0 at threshold
                                let sLoc = pow(1.0 - (distMeters / thresh), 0.5);
                                score = score + (weight / sumW) * sLoc; 
                            }
                        }
                    }

                    // Decisions quantization (Matching Image 2 style)
                    let steps = 6.0;
                    let qS = floor(score * steps) / steps;

                    if (qS <= 0.05) {
                        a = 0.0;
                    } else if (qS < 0.25) {
                        // Cyan / Dark Turquoise (Distance Zone)
                        r = 0.0; g = 0.55; b = 0.75; a = 0.6;
                    } else if (qS < 0.85) {
                        // Golden Yellow (Urban Influence)
                        r = 0.85; g = 0.65; b = 0.12; a = 0.8;
                    } else {
                        // Bright Green (Direct Proximity / Hotspot)
                        r = 0.2; g = 0.85; b = 0.25; a = 0.95;
                    }
                } else {
                    let pNx = config.nx + 2u * config.ghosts;
                    let srcIdx = (id.y + config.ghosts) * pNx + (id.x + config.ghosts);
                    let rawVal = cube[config.faceIdx * config.strideFace + srcIdx];
                    let norm = clamp((rawVal - config.minV) / (config.maxV - config.minV + 0.00001), 0.0, 1.0);
                    
                    if (config.colormap == 1u) { // Heatmap
                        r = norm;
                        g = select(0.0, (norm - 0.5) * 2.0, norm > 0.5);
                        b = norm * 0.2;
                    } else if (config.colormap == 2u) { // Arctic
                        // Target: 1:1 Parity with CanvasAdapterNeo.ts:120
                        // Base: Light Blue (180, 220, 255)
                        r = 0.706; g = 0.863; b = 1.0; 
                        let ts = norm * (2.0 - norm); // Linear approximation of CPU tS
                        
                        // Blend to Navy (15, 30, 80)
                        r = r * (1.0 - ts) + 0.059 * ts;
                        g = g * (1.0 - ts) + 0.118 * ts;
                        b = b * (1.0 - ts) + 0.314 * ts;

                        let vortIdx = config.aux[0];
                        if (vortIdx < 100u) {
                            let vRaw = cube[vortIdx * config.strideFace + srcIdx];
                            let vMag = clamp(abs(vRaw) * 120.0, 0.0, 1.0); // Exact CPU gain
                            if (vMag > 0.05) {
                                let tc = clamp((vMag - 0.05) * 1.5, 0.0, 1.0); // Exact CPU blend
                                r = r * (1.0 - tc) + 1.0 * tc; 
                                g = g * (1.0 - tc);
                                b = b * (1.0 - tc);
                            }
                        }
                    } else if (config.colormap == 4u) { // Debug Velocity
                        let vxIdx = config.aux[1];
                        let vyIdx = config.aux[2];
                        if (vxIdx < 100u && vyIdx < 100u) {
                            let vx = cube[vxIdx * config.strideFace + srcIdx];
                            let vy = cube[vyIdx * config.strideFace + srcIdx];
                            let vMag = sqrt(vx*vx + vy*vy) * 5.0;
                            r = vMag; g = 0.2; b = 0.2; a = 1.0;
                        } else {
                            r = 0.1; g = 0.1; b = 0.1; a = 1.0;
                        }
                    } else {
                        r = norm; g = norm; b = norm;
                    }
                }

                textureStore(outTexture, outPos, vec4<f32>(r, g, b, a));
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

        this.uniformBuffer = device.createBuffer({
            size: 1024, // Plenty for params
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }

    public render(
        proxy: NeoEngineProxy,
        options: RenderOptions
    ) {
        if (!HypercubeGPUContext.device) return;
        this.initPipelines();

        const vGrid = proxy.vGrid;
        const bridge = proxy.bridge;
        if (!bridge.gpuBuffer) return;

        const totalW = vGrid.dimensions.nx;
        const totalH = vGrid.dimensions.ny;

        // Sync canvas intrinsic size for WebGPU context
        if (this.canvas.width !== totalW || this.canvas.height !== totalH) {
            this.canvas.width = totalW;
            this.canvas.height = totalH;
        }

        // Ensure storage texture
        if (!this.storageTexture || this.storageTexture.width !== totalW || this.storageTexture.height !== totalH) {
            if (this.storageTexture) this.storageTexture.destroy();
            this.storageTexture = HypercubeGPUContext.device.createTexture({
                size: [totalW, totalH, 1],
                format: 'rgba8unorm',
                usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
            });
            this.neoBindGroups.clear();
            this.blitBindGroup = null;
        }

        const device = HypercubeGPUContext.device;
        const commandEncoder = device.createCommandEncoder();

        const bytesPerChunkAligned = HypercubeGPUContext.alignToUniform(256); // Base render params size
        const totalUniformSize = vGrid.chunks.length * bytesPerChunkAligned;
        if (this.uniformBuffer!.size < totalUniformSize) {
            this.uniformBuffer!.destroy();
            this.uniformBuffer = device.createBuffer({
                size: totalUniformSize,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                label: 'WebGpuRendererNeo Uniforms'
            });
            this.neoBindGroups.clear();
        }

        const u32Data = new Uint32Array(vGrid.chunks.length * (bytesPerChunkAligned / 4));
        const facesPerChunk = bridge.totalSlotsPerChunk;
        const strideFaceBytes = bridge.strideFace * 4;

        const descriptor = (vGrid as any).dataContract.descriptor;
        const faceMappings = (vGrid as any).dataContract.getFaceMappings();

        function getPhysicalSlot(logicalIdx: number | undefined): number {
            if (logicalIdx === undefined || logicalIdx < 0 || logicalIdx >= descriptor.faces.length) return 999;
            const faceName = descriptor.faces[logicalIdx].name;
            return proxy.parityManager.getFaceIndices(faceName).read;
        }

        const tryGetLogical = (name: string) => {
            try { return proxy.getFaceLogicalIndex(name); } catch(e) { return undefined; }
        };

        const physicalFaceIdx = getPhysicalSlot(options.faceIndex);
        const physicalObsIdx = getPhysicalSlot(options.obstaclesFace);
        const physicalVortIdx = getPhysicalSlot(options.vorticityFace); // Backwards compatibility for now

        for (let i = 0; i < vGrid.chunks.length; i++) {
            const chunk = vGrid.chunks[i];
            const base = i * (bytesPerChunkAligned / 4);
            const f32 = new Float32Array(u32Data.buffer);

            u32Data[base + 0] = chunk.localDimensions.nx;
            u32Data[base + 1] = chunk.localDimensions.ny;
            u32Data[base + 2] = chunk.localDimensions.nz;
            u32Data[base + 3] = descriptor.requirements.ghostCells;
            u32Data[base + 4] = physicalFaceIdx;
            u32Data[base + 5] = physicalObsIdx;
            f32[base + 6] = options.minVal ?? 0;
            f32[base + 7] = options.maxVal ?? 1;
            const cmap = options.colormap;
            u32Data[base + 8] = cmap === 'heatmap' ? 1 : (cmap === 'arctic' ? 2 : (cmap === 'spatial-decision' ? 3 : (typeof cmap === 'number' ? (cmap as number) : 0)));
            
            // Calculate world offsets by summing preceding chunks
            let worldX0 = 0;
            let worldY0 = 0;
            for (const c of vGrid.chunks) {
                if (c.y === chunk.y && c.z === chunk.z && c.x < chunk.x) worldX0 += c.localDimensions.nx;
                if (c.x === chunk.x && c.z === chunk.z && c.y < chunk.y) worldY0 += c.localDimensions.ny;
            }

            u32Data[base + 9] = worldX0;
            u32Data[base + 10] = worldY0;
            u32Data[base + 11] = bridge.strideFace;
            u32Data[base + 12] = proxy.parityManager.currentTick % 2; // readParity
            const criteria = options.criteriaSDF || [];
            u32Data[base + 13] = criteria.length;
            f32[base + 14] = (vGrid as any).config.params?.pixelScale ?? 2.0;
            u32Data[base + 15] = 0; // Pad for 16-byte alignment

            // Auxiliary faces at base + 16 (16 * 4 = 64 bytes offset from chunk base)
            const aux = options.auxiliaryFaces || [];
            if (aux.length === 0 && physicalVortIdx < 100) {
                // Backwards compatibility for vorticity
                u32Data[base + 16] = physicalVortIdx;
            }
            for (let j = 0; j < Math.min(aux.length, 8); j++) {
                u32Data[base + 16 + j] = getPhysicalSlot(aux[j]);
            }

            // Criteria start at base + 24 (aligned to 16-byte boundary)
            for (let c = 0; c < Math.min(criteria.length, 6); c++) {
                const cBase = base + 24 + c * 4;
                f32[cBase + 0] = criteria[c].weight;
                f32[cBase + 1] = criteria[c].distanceThreshold;
                u32Data[cBase + 2] = getPhysicalSlot(criteria[c].xFace);
                u32Data[cBase + 3] = getPhysicalSlot(criteria[c].yFace);
            }
        }
        device.queue.writeBuffer(this.uniformBuffer!, 0, u32Data);

        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.pipeline!);
        const view = this.storageTexture.createView();

        for (let i = 0; i < vGrid.chunks.length; i++) {
            const chunkBufferOffset = i * facesPerChunk * strideFaceBytes;
            const uniformOffset = i * bytesPerChunkAligned;

            const bgKey = `${i}-${chunkBufferOffset}-${uniformOffset}-${this.storageTexture.width}`;
            let bg = this.neoBindGroups.get(bgKey);
            if (!bg) {
                bg = device.createBindGroup({
                    layout: this.pipeline!.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: bridge.gpuBuffer, offset: chunkBufferOffset, size: facesPerChunk * strideFaceBytes } },
                        { binding: 1, resource: { buffer: this.uniformBuffer!, offset: uniformOffset, size: HypercubeGPUContext.alignToUniform(256) } },
                        { binding: 2, resource: view }
                    ]
                });
                this.neoBindGroups.set(bgKey, bg);
            }

            const chunk = vGrid.chunks[i];
            const nx = chunk.localDimensions.nx;
            const ny = chunk.localDimensions.ny;
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(nx / 16), Math.ceil(ny / 16), 1);
        }
        pass.end();

        // --- BLIT STORAGE TO CANVAS ---
        const canvasTexture = this.context.getCurrentTexture();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: canvasTexture.createView(),
                loadOp: 'clear',
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
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
