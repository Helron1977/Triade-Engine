import { HypercubeNeoFactory } from '../../../../core/HypercubeNeoFactory';
import { IsoRendererNeo } from '../../io/IsoRendererNeo';
import { BenchmarkHUD } from '../../io/BenchmarkHUD';

/**
 * Neo Ocean (CPU) Orchestrator
 * Declarative 2.5D Isometric Wave Equation
 */
async function main() {
    const factory = new HypercubeNeoFactory();

    // 1. Load Manifest from local showcase root
    const manifest = await factory.fromManifest('../showcase-ocean-cpu.json');
    const { config, engine: descriptor } = manifest;

    // 2. Build Engine
    const engine = await factory.build(config, descriptor);

    // IA Observability (Web MCP)
    const { DebugBridge } = await import('../../helpers/DebugBridge');
    DebugBridge.setup(engine, config);

    const NX = config.dimensions.nx;
    const NY = config.dimensions.ny;
    const COLS = config.chunks.x;
    const ROWS = config.chunks.y;

    // 3. Setup Layout
    const container = document.getElementById('canvas-container')!;
    const canvas = document.createElement('canvas');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    container.appendChild(canvas);

    const hud = new BenchmarkHUD('OceanEngine 2.5D IsoVolume', `${NX} x ${NY}`);

    // 4. Initialize Iso Renderer (Scale dynamically based on screen and grid)
    const baseScale = Math.min(window.innerWidth / NX, window.innerHeight / NY) * 0.8;
    const isoRenderer = new IsoRendererNeo(canvas, undefined, baseScale);

    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });
    // 5. Interaction (Mouse to Declarative Rasterization)
    function addSplashObject(cx: number, cy: number, rho: number, diameter: number) {
        if (!config.objects) config.objects = [];
        const w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
        const props: Record<string, number> = { "biology": 1.0, "rho": rho };
        for (let k = 0; k < 9; k++) props[`f${k}`] = w[k] * rho;

        config.objects.push({
            id: `splash_${Date.now()}_${Math.random()}`,
            type: "circle",
            position: { x: cx - diameter / 2, y: cy - diameter / 2 },
            dimensions: { w: diameter, h: diameter },
            properties: props,
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
    async function loop() {
        try {
            const start = performance.now();
            // physics step
            await engine.step(1);
            const ms = performance.now() - start;

            // Clean up transient splash and init objects after they've been rasterized
            if (config.objects) {
                config.objects = config.objects.filter((o: any) => !o.id.startsWith('splash_') && o.id !== 'grid_init');
            }

            // Sync indices for rendering
            const depthIdx = engine.parityManager.getFaceIndices('rho').read;
            const obsIdx = engine.parityManager.getFaceIndices('obstacles').read;

            // Map 1D Chunks to 2D Array for IsoRenderer
            const gridFaces: Float32Array[][][] = Array(ROWS).fill(null).map(() => Array(COLS).fill(null).map(() => []));
            const bridge = (engine as any).bridge;
            for (const chunk of engine.vGrid.chunks) {
                gridFaces[chunk.y][chunk.x] = bridge.getChunkViews(chunk.id);
            }

            // Render via Legacy IsoRenderer
            isoRenderer.clearAndSetup(5, 15, 35); // Deep sea dark blue
            isoRenderer.renderMultiChunkVolume(
                gridFaces,
                NX / COLS + 2, // Physical chunk size with ghost cells
                NY / ROWS + 2,
                COLS,
                ROWS,
                {
                    densityFaceIndex: depthIdx,
                    obstacleFaceIndex: obsIdx,
                    lodStep: Math.max(2, Math.floor(NX / 128))
                }
            );

            hud.updateCompute(ms);
            hud.tickFrame();
            requestAnimationFrame(loop);
        } catch (e) {
            console.error("Simulation loop error:", e);
        }
    }

    loop();
}

main().catch(console.error);
