import { HypercubeNeoFactory } from '../../../../core/HypercubeNeoFactory';
import { WebGpuIsoRendererNeo } from '../../io/WebGpuIsoRendererNeo';
import { BenchmarkHUD } from '../../io/BenchmarkHUD';

/**
 * Neo Ocean (GPU) Orchestrator
 * Declarative 2.5D Isometric Wave Equation optimized for ZERO-STALL WebGPU pipeline
 */
async function main() {
    const factory = new HypercubeNeoFactory();

    // 1. Load Manifest from local showcase root
    const manifest = await factory.fromManifest('../showcase-ocean-gpu.json');
    const { config, engine: descriptor } = manifest;

    // 2. Build Engine
    const engine = await factory.build(config, descriptor);

    // IA Observability (Web MCP)
    const { DebugBridge } = await import('../../helpers/DebugBridge');
    DebugBridge.setup(engine, config);

    const NX = config.dimensions.nx;
    const NY = config.dimensions.ny;

    // 3. Setup Layout
    const container = document.getElementById('canvas-container') || document.body;
    const canvas = document.createElement('canvas');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    container.appendChild(canvas);

    const hud = new BenchmarkHUD('OceanEngine GPU (Zero-Stall)', `${NX} x ${NY}`);

    // 4. Initialize Iso Renderer (Scale dynamically based on screen and grid)
    const baseScale = Math.min(window.innerWidth / NX, window.innerHeight / NY) * 0.8;
    const isoRenderer = new WebGpuIsoRendererNeo(canvas, baseScale);

    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });

    // 5. Interaction (Mouse to Declarative Rasterization via GPU Object Dispatching)
    function addSplashObject(cx: number, cy: number, rho: number, diameter: number) {
        if (!config.objects) config.objects = [];

        // Push the dynamic splash object into the config array.
        // It will be picked up by the GpuDispatcher automatically on the next frame.
        config.objects.push({
            id: `splash_${Date.now()}_${Math.random()}`,
            type: "circle",
            position: { x: cx - diameter / 2, y: cy - diameter / 2 },
            dimensions: { w: diameter, h: diameter },
            properties: {
                biology: 1.0,
                rho: rho
            },
            rasterMode: "replace"
        });
    }

    function getGridCoordsFromMouse(e: MouseEvent) {
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;

        // Base Scale logic must exactly match IsoRenderer
        const baseScale = Math.min(window.innerWidth / NX, window.innerHeight / NY) * 0.8;
        const isoXScale = baseScale * 0.866;
        const isoYScale = baseScale * 0.5;

        // Note: GPU Ocean uses 1 Chunk. Middle offset is evaluated directly on NX/NY.
        const midW = NX / 2;
        const midH = NY / 2;
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2 + (midH * isoYScale * 0.5);

        // Inverse isometric projection
        const dx = (sx - centerX) / isoXScale;
        const dy = (sy - centerY) / isoYScale;
        const wx = (dx + dy) / 2;
        const wy = (dy - dx) / 2;

        return { gx: wx + midW, gy: wy + midH };
    }

    canvas.addEventListener('mousemove', (e: MouseEvent) => {
        if (e.buttons === 1) {
            const coords = getGridCoordsFromMouse(e);
            if (coords.gx >= -20 && coords.gx <= NX + 20 && coords.gy >= -20 && coords.gy <= NY + 20) {
                addSplashObject(coords.gx, coords.gy, 1.4, 15);
            }
        }
    });

    canvas.addEventListener('mousedown', (e: MouseEvent) => {
        const coords = getGridCoordsFromMouse(e);
        if (coords.gx >= -20 && coords.gx <= NX + 20 && coords.gy >= -20 && coords.gy <= NY + 20) {
            addSplashObject(coords.gx, coords.gy, 1.8, 25);
        }
    });

    // 6. Initialization
    await (engine as any).bridge.syncToDevice();
    let isInitialized = false;

    async function loop() {
        try {
            const start = performance.now();

            // GPU physics dispatch
            await engine.step(1);
            const ms = performance.now() - start;

            // One-time initialization — remove grid_init and splash_init after first step
            if (!isInitialized) {
                if (config.objects) {
                    config.objects = config.objects.filter((o: any) => 
                        o.id !== 'grid_init' && o.id !== 'splash_init'
                    );
                }
                isInitialized = true;
            }

            // Clean up transient splash objects after they've been injected into WGSL
            if (config.objects) {
                config.objects = config.objects.filter((o: any) => !o.id.startsWith('splash_'));
            }

            // Sync indices for rendering
            const depthIdx = engine.parityManager.getFaceIndices('rho').read;
            const obsIdx = engine.parityManager.getFaceIndices('obstacles').read;

            // Native ZERO-STALL WebGPU Rendering. We do not download the MasterBuffer to RAM.
            // The shader reads directly from the VRAM storage buffer.
            isoRenderer.render(engine as any, {
                densityFaceIndex: depthIdx,
                obstacleFaceIndex: obsIdx,
                lodStep: 1 // Full resolution GPU rendering
            });

            hud.updateCompute(ms); // Note: Includes VRAM sync overhead
            hud.tickFrame();
            requestAnimationFrame(loop);
        } catch (e) {
            console.error("Simulation loop error:", e);
        }
    }

    loop();
}

main().catch(console.error);
