import { HypercubeNeoFactory } from '../../../../core/HypercubeNeoFactory';
import { NacaHelper } from '../../helpers/ShapeHelpers';
import { HypercubeNeo } from '../../HypercubeNeo';
import { BenchmarkHUD } from '../../io/BenchmarkHUD';

/**
 * Neo Aero (CPU) Orchestrator
 * Migrated from examples/11-neo-fluid.ts
 */
async function main() {
    const factory = new HypercubeNeoFactory();

    // 1. Generate NACA Wings (Biplane signature)
    const wingPoints = NacaHelper.generateNaca4(0.00, 0.0, 0.16, 80, 120, -12 * Math.PI / 180);

    // 2. Load Manifest from local showcase root
    const manifest = await factory.fromManifest('../showcase-aero-v1.json');
    const { config, engine: descriptor } = manifest;

    // Inject dynamic NACA points
    const wingTop = config.objects?.find((o: any) => o.id === 'wing_top');
    if (wingTop) wingTop.points = wingPoints;
    const wingBottom = config.objects?.find((o: any) => o.id === 'wing_bottom');
    if (wingBottom) wingBottom.points = wingPoints;

    // 3. Build Engine
    const engine = await factory.build(config, descriptor);

    // IA Observability (Web MCP)
    const { DebugBridge } = await import('../../helpers/DebugBridge');
    DebugBridge.setup(engine, config);

    const NX = config.dimensions.nx;
    const NY = config.dimensions.ny;

    // 4. Setup Containers
    const container = document.getElementById('canvas-container')!;
    const canvas = document.createElement('canvas');
    canvas.width = NX;
    canvas.height = NY;
    container.appendChild(canvas);

    const hud = new BenchmarkHUD('Neo Aero (CPU)', `${NX} x ${NY}`);

    // Resolve stable LOGICAL face indices — CanvasAdapterNeo/WebGpuRendererNeo resolve to physical slots internally
    const smokeFaceIdx = engine.getFaceLogicalIndex('smoke');
    const obsFaceIdx   = engine.getFaceLogicalIndex('obstacles');
    const vortFaceIdx  = engine.getFaceLogicalIndex('vorticity');
    const vxFaceIdx    = engine.getFaceLogicalIndex('vx');
    const vyFaceIdx    = engine.getFaceLogicalIndex('vy');

    let isInitialized = false;

    async function loop() {
        try {
            const start = performance.now();
            // physics step
            await engine.step(1);
            const ms = performance.now() - start;

            // One-time initialization logic to remove the static grid_init object
            if (!isInitialized) {
                if (config.objects && config.objects[0].id === 'grid_init') {
                    config.objects.shift();
                    isInitialized = true;
                }
            }

            // Render via Neo adapter using stable logical face indices
            HypercubeNeo.autoRender(engine, canvas, {
                faceIndex: smokeFaceIdx,
                colormap: 'arctic',
                minVal: 0.0,
                maxVal: 1.0,
                obstaclesFace: obsFaceIdx,
                auxiliaryFaces: [vortFaceIdx, vxFaceIdx, vyFaceIdx]
            });


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
