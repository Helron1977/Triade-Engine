import { HypercubeNeoFactory } from '../../../../core/HypercubeNeoFactory';
import { NacaHelper } from '../../helpers/ShapeHelpers';
import { BenchmarkHUD } from '../../io/BenchmarkHUD';
import { WebGpuRendererNeo } from '../../io/WebGpuRendererNeo';

/**
 * Neo Aero (GPU) Orchestrator
 */
async function main() {
    const factory = new HypercubeNeoFactory();

    // 1. Generate NACA Wings (Biplane signature)
    const wingPoints = NacaHelper.generateNaca4(0.00, 0.0, 0.16, 80, 120, -12 * Math.PI / 180);

    // 2. Load Manifest
    const manifest = await factory.fromManifest('../showcase-aero-gpu.json');
    const { config, engine: descriptor } = manifest;

    // Ensure GPU mode
    config.mode = 'gpu';

    // Inject dynamic NACA points
    const wingTop = config.objects?.find((o: any) => o.id === 'wing_top');
    if (wingTop) wingTop.points = wingPoints;
    const wingBottom = config.objects?.find((o: any) => o.id === 'wing_bottom');
    if (wingBottom) wingBottom.points = wingPoints;

    // 3. Build Engine (WebGPU)
    const engine = await factory.build(config, descriptor);

    // Expose Debug Bridge for IA Observability (WebMCP-like)
    const { DebugBridge } = await import('../../helpers/DebugBridge');
    DebugBridge.setup(engine, config);

    const NX = config.dimensions.nx;
    const NY = config.dimensions.ny;

    // 4. Setup Canvas
    const container = document.getElementById('canvas-container')!;
    const canvas = document.createElement('canvas');
    canvas.width = NX;
    canvas.height = NY;
    container.appendChild(canvas);

    const hud = new BenchmarkHUD('Neo Aero (GPU)', `${NX} x ${NY}`);

    // Native WebGPU Renderer (Direct-to-VRAM)
    const renderer = new WebGpuRendererNeo(canvas);

    // Resolve LOGICAL face indices (stable across parity swaps).
    const smokeFaceIdx = engine.getFaceLogicalIndex('smoke');
    const obsFaceIdx   = engine.getFaceLogicalIndex('obstacles');
    const vortFaceIdx  = engine.getFaceLogicalIndex('vorticity');
    const vxFaceIdx    = engine.getFaceLogicalIndex('vx');
    const vyFaceIdx    = engine.getFaceLogicalIndex('vy');

    // 6. Initialization
    await (engine as any).bridge.syncToDevice();
    let isInitialized = false;

    async function loop() {
        try {
            const start = performance.now();
            // Physics step
            await engine.step(1);
            const ms = performance.now() - start;

            // One-time initialization — remove grid_init object after first step
            if (!isInitialized) {
                if (config.objects && config.objects[0]?.id === 'grid_init') {
                    config.objects.shift();
                }
                isInitialized = true;
            }

            // Render via native WebGPU renderer
            renderer.render(engine as any, {
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
