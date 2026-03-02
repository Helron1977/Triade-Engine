import type { HypercubeGrid } from './HypercubeGrid';

/**
 * Options de rendu pour le Compositeur (7ème Plan).
 */
export interface CompositorOptions {
    canvas: HTMLCanvasElement | OffscreenCanvas;
    /** Code Fragment Shader WGSL personnalisé pour mélanger les faces (WebGPU seulement). */
    wgslFragmentSource?: string;
    /** Fonction fallback CPU pour mélanger les 6 faces manuellement pixel par pixel (CPU Mode seulement). */
    cpuFragmentShader?: (faces: Float32Array[], index: number) => { r: number, g: number, b: number, a: number };
}

/**
 * HypercubeCompositor (Le "7ème Plan")
 * Couche abstraite prenant les données logiques des 6 Faces du HypercubeGrid
 * pour les synthétiser en une image visuelle via un fragment shader (ou fallback CPU).
 * Permet un affichage haute-performance asynchrone sans bloquer l'UI principale.
 */
export class HypercubeCompositor {
    private grid: HypercubeGrid;
    private options: CompositorOptions;

    // WebGPU Context
    private device: GPUDevice | null = null;
    private context: GPUCanvasContext | null = null;
    private renderPipeline: GPURenderPipeline | null = null;
    private bindGroup: GPUBindGroup | null = null;

    // CPU Context
    private ctx2d: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null = null;
    private imageData: ImageData | null = null;

    constructor(grid: HypercubeGrid, options: CompositorOptions) {
        this.grid = grid;
        this.options = options;
    }

    /**
     * Initialise le contexte graphique (WebGPU ou 2D) en fonction du mode de la grille.
     */
    async init(): Promise<boolean> {
        if (this.grid.mode === 'webgpu') {
            return await this.initWebGPU();
        } else {
            return this.initCPU();
        }
    }

    /**
     * Effectue le rendu des données logiques (Faces) vers le Canvas.
     */
    render(): void {
        if (this.grid.mode === 'webgpu') {
            this.renderWebGPU();
        } else {
            this.renderCPU();
        }
    }

    // --- MODE WEBGPU ---

    private async initWebGPU(): Promise<boolean> {
        if (!navigator.gpu) {
            console.error("[HypercubeCompositor] WebGPU n'est pas supporté par ce navigateur.");
            return false;
        }

        // Récupérer le device pré-initialisé s'il existe (via HypercubeGPUContext)
        // ou en recréer un pour l'affichage statique.
        const HypercubeGPUContext = (await import('./gpu/HypercubeGPUContext')).HypercubeGPUContext;
        if (!HypercubeGPUContext.device) {
            const success = await HypercubeGPUContext.init();
            if (!success) return false;
        }
        this.device = HypercubeGPUContext.device;

        this.context = (this.options.canvas as HTMLCanvasElement).getContext('webgpu') as GPUCanvasContext;
        if (!this.context) {
            console.error("[HypercubeCompositor] Impossible d'obtenir le contexte WebGPU Canvas.");
            return false;
        }

        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: format,
            alphaMode: 'premultiplied'
        });

        // 1. Récupération des Storage Buffers des Faces du PREMIER cube de la grille.
        // TODO V3: Gérer des grilles NxM entières en reconstruisant une mega-texture ou via un un bindgroup dynamique.
        // Pour l'instant, le compositeur v1 suppose un seul gros cube (1x1 Mesh).
        const firstCube = this.grid.cubes[0][0];
        if (!firstCube || !firstCube.gpuBuffer) return false;

        const wgslShader = this.options.wgslFragmentSource || this.defaultWGSLShader();
        const shaderModule = this.device.createShaderModule({ code: wgslShader });

        // Créer un pipeline de rendu Full-Screen Quad
        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: shaderModule,
                entryPoint: 'vertex_main',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragment_main',
                targets: [{ format: format }]
            },
            primitive: {
                topology: 'triangle-list'
            }
        });

        // Unform contenant la taille (mapSize) et le stride (floats)
        const uniformBuffer = this.device.createBuffer({
            size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        const strideFloats = firstCube.stride / 4;
        this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([firstCube.mapSize, strideFloats]));

        // BindGroup: Lien avec le buffer unique du Cube et l'uniform
        this.bindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: firstCube.gpuBuffer } },
                { binding: 1, resource: { buffer: uniformBuffer } },
            ]
        });

        return true;
    }

    private renderWebGPU(): void {
        if (!this.device || !this.context || !this.renderPipeline || !this.bindGroup) return;

        const commandEncoder = this.device.createCommandEncoder();
        const textureView = this.context.getCurrentTexture().createView();

        const renderPassDescriptor: GPURenderPassDescriptor = {
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        };

        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(this.renderPipeline);
        passEncoder.setBindGroup(0, this.bindGroup);
        // On dessine un Full-Screen Quad (2 triangles = 6 sommets) générés dans le Vertex Shader
        passEncoder.draw(6, 1, 0, 0);
        passEncoder.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }

    private defaultWGSLShader(): string {
        return `
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) uv: vec2<f32>,
            };

            // FullScreen Quad généré dynamiquement (sans vertex buffer)
            @vertex
            fn vertex_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
                var pos = array<vec2<f32>, 6>(
                    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
                    vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0)
                );
                var uv = array<vec2<f32>, 6>(
                    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0),
                    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0)
                );
                var output: VertexOutput;
                output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
                output.uv = uv[vertexIndex];
                return output;
            }

            @group(0) @binding(0) var<storage, read> cube: array<f32>;
            struct Config {
                mapSize: u32,
                stride: u32,
            };
            @group(0) @binding(1) var<uniform> config: Config;

            @fragment
            fn fragment_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
                let x = u32(uv.x * f32(config.mapSize));
                let y = u32(uv.y * f32(config.mapSize));
                let idx = y * config.mapSize + x;

                // --- 7ème PLAN LOGIQUE: Shader par défaut (Affiche proportionnellement la face 1, 2 et 3) ---
                let v1 = cube[0 * config.stride + idx];
                let v2 = cube[1 * config.stride + idx];
                let v3 = cube[2 * config.stride + idx];

                return vec4<f32>(v1, v2, v3, 1.0);
            }
        `;
    }

    // --- MODE CPU FALLBACK ---

    private initCPU(): boolean {
        this.ctx2d = (this.options.canvas as HTMLCanvasElement).getContext('2d', { willReadFrequently: true });
        if (!this.ctx2d) {
            console.error("[HypercubeCompositor] Impossible d'obtenir le contexte Canvas 2D (CPU Fallback).");
            return false;
        }

        const width = this.options.canvas.width;
        const height = this.options.canvas.height;
        this.imageData = this.ctx2d.createImageData(width, height);
        return true;
    }

    private renderCPU(): void {
        if (!this.ctx2d || !this.imageData) return;

        const firstCube = this.grid.cubes[0][0];
        if (!firstCube) return;

        const mapSize = firstCube.mapSize;
        const faces = firstCube.faces;
        const outData = this.imageData.data;

        // Note: Cette implémentation CPU est sub-optimale N^2, elle est fournie à titre de Fallback UI pur,
        // et n'exploite un WorkerThread que si explicitement mis dans un OffscreenCanvas par l'utilisateur.

        for (let y = 0; y < mapSize; y++) {
            for (let x = 0; x < mapSize; x++) {
                const idx = y * mapSize + x;
                const pxIdx = idx * 4;

                if (this.options.cpuFragmentShader) {
                    const color = this.options.cpuFragmentShader(faces, idx);
                    outData[pxIdx + 0] = color.r;
                    outData[pxIdx + 1] = color.g;
                    outData[pxIdx + 2] = color.b;
                    outData[pxIdx + 3] = color.a;
                } else {
                    // Fallback par défaut
                    outData[pxIdx + 0] = faces[0][idx] * 255; // Face 1 = Red
                    outData[pxIdx + 1] = faces[1][idx] * 255; // Face 2 = Green
                    outData[pxIdx + 2] = faces[2][idx] * 255; // Face 3 = Blue
                    outData[pxIdx + 3] = 255;                 // Alpha
                }
            }
        }

        this.ctx2d.putImageData(this.imageData, 0, 0);
    }
}




































