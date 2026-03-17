import { HypercubeNeoFactory } from '../../../../core/HypercubeNeoFactory';
import { HypercubeNeo } from '../../HypercubeNeo';
import { BenchmarkHUD } from '../../io/BenchmarkHUD';

async function main() {
    const factory = new HypercubeNeoFactory();
    const manifest = await factory.fromManifest('../showcase-path.json');
    const { config, engine: descriptor } = manifest;

    config.mode = 'cpu';
    const engine = await factory.build(config, descriptor);

    const NX = config.dimensions.nx;
    const NY = config.dimensions.ny;

    const container = document.getElementById('canvas-container')!;
    const canvas = document.createElement('canvas');
    canvas.width = NX;
    canvas.height = NY;
    container.appendChild(canvas);

    const hud = new BenchmarkHUD('Pathfinder (CPU)', `${NX} x ${NY}`);

    const distFaceIdx = engine.getFaceLogicalIndex('distance');
    const obsFaceIdx  = engine.getFaceLogicalIndex('obstacles');

    async function loop() {
        try {
            const start = performance.now();
            await engine.step(1);
            const ms = performance.now() - start;

            HypercubeNeo.autoRender(engine, canvas, {
                faceIndex: distFaceIdx,
                colormap: 'heatmap',
                minVal: 0.0,
                maxVal: 200.0,
                obstaclesFace: obsFaceIdx
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
