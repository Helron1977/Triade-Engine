import { HypercubeNeoFactory } from '../../../../core/HypercubeNeoFactory';
import { HypercubeNeo } from '../../HypercubeNeo';
import { BenchmarkHUD } from '../../io/BenchmarkHUD';

async function main() {
    const factory = new HypercubeNeoFactory();
    const manifest = await factory.fromManifest('../showcase-life.json');
    const { config, engine: descriptor } = manifest;

    // Force GPU mode
    config.mode = 'gpu';
    const engine = await factory.build(config, descriptor);

    const NX = config.dimensions.nx;
    const NY = config.dimensions.ny;

    const container = document.getElementById('canvas-container')!;
    const canvas = document.createElement('canvas');
    canvas.width = NX;
    canvas.height = NY;
    container.appendChild(canvas);

    const hud = new BenchmarkHUD('Game of Life (GPU)', `${NX} x ${NY}`);

    const lifeFaceIdx = engine.getFaceLogicalIndex('life');

    async function loop() {
        try {
            const start = performance.now();
            await engine.step(1);
            const ms = performance.now() - start;

            HypercubeNeo.autoRender(engine, canvas, {
                faceIndex: lifeFaceIdx,
                colormap: 'arctic',
                minVal: 0.0,
                maxVal: 1.0
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
