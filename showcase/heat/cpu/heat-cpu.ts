import { HypercubeNeoFactory } from '../core/HypercubeNeoFactory';
import { CanvasAdapterNeo } from '../io/CanvasAdapterNeo';
import { BenchmarkHUD } from '../io/BenchmarkHUD';

async function main() {
    const resManifest = await fetch('./showcase-heat-cpu.json?v=' + Date.now());
    const manifest = await resManifest.json();

    const factory = new HypercubeNeoFactory();
    const engine = await factory.build(manifest.config, manifest.engine);

    // Resolve stable logical face indices
    const tempFaceIdx = engine.getFaceLogicalIndex('temperature');
    const obsFaceIdx  = engine.getFaceLogicalIndex('obstacles');

    // Setup Canvas
    const container = document.getElementById('canvas-container')!;
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 256;
    container.appendChild(canvas);

    const hud = new BenchmarkHUD('Neo Heat Diffusion (CPU)', '512x256');

    let frame = 0;

    const render = async () => {
        frame++;

        // MOTION: Two orbiting heat sources + mobile cooling disk
        const t1 = frame * 0.03;
        const t2 = frame * 0.02 + Math.PI;
        const obsX = 256 + Math.sin(frame * 0.01) * 120;

        // CPU path: objects are updated in the manifest and rasterized by ObjectRasterizer.
        const objects = (engine.vGrid as any).config.objects;
        
        // 1. Move Heat Eater
        const heater = objects.find((o: any) => o.id === 'heat_eater');
        if (heater) {
            heater.position.x = obsX - 25;
            heater.position.y = 128 - 25;
        }

        // 2. Orbit sources
        const s1 = objects.find((o: any) => o.id === 'source_1');
        if (s1) {
            s1.position.x = 256 + Math.cos(t1) * 140 - 12;
            s1.position.y = 128 + Math.sin(t1 * 1.5) * 80 - 12;
        }

        const s2 = objects.find((o: any) => o.id === 'source_2');
        if (s2) {
            s2.position.x = 256 + Math.cos(t2) * 100 - 8;
            s2.position.y = 128 + Math.sin(t2 * 2.1) * 110 - 8;
        }

        // Compute step (single iteration per render frame)
        await engine.step(1);

        // Render — pass LOGICAL face indices, CanvasAdapterNeo resolves physical slots
        CanvasAdapterNeo.render(engine as any, canvas, {
            faceIndex: tempFaceIdx,
            colormap: 'heatmap',
            minVal: 0,
            maxVal: 3.5,
            obstaclesFace: obsFaceIdx
        });

        hud.tickFrame();
        requestAnimationFrame(render);
    };

    render();
    console.log("Neo Heat CPU Showcase Running ☕🌍");
}

main().catch(err => console.error(err));
