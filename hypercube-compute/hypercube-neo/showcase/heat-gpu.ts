import { HypercubeNeoFactory } from '../core/HypercubeNeoFactory';
import { WebGpuRendererNeo } from '../io/WebGpuRendererNeo';
import { BenchmarkHUD } from '../io/BenchmarkHUD';

async function main() {
    const resManifest = await fetch('./showcase-heat-gpu.json?v=' + Date.now());
    const manifest = await resManifest.json();

    const factory = new HypercubeNeoFactory();
    const engine = await factory.build(manifest.config, manifest.engine);

    // IA Observability (Web MCP)
    const { DebugBridge } = await import('../helpers/DebugBridge');
    DebugBridge.setup(engine, manifest.config);

    // Resolve stable logical face indices (do NOT use getFaceIndices().read here,
    // WebGpuRendererNeo.render() calls parityManager internally via getPhysicalSlot)
    const tempFaceIdx = engine.getFaceLogicalIndex('temperature');
    const obsFaceIdx  = engine.getFaceLogicalIndex('obstacles');

    // Setup Canvas
    const container = document.getElementById('canvas-container')!;
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 256;
    container.appendChild(canvas);

    const hud = new BenchmarkHUD('Neo Heat Diffusion (GPU)', '512x256');
    const renderer = new WebGpuRendererNeo(canvas);

    let frame = 0;

    const render = async () => {
        frame++;

        // "WOW" MOTION: Two orbiting heat sources + mobile cooling disk
        const t1 = frame * 0.03;
        const t2 = frame * 0.02 + Math.PI;
        const obsX = 256 + Math.sin(frame * 0.01) * 120;

        // Objects are updated in the manifest and sent as GPU uniforms via GpuDispatcher.
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

        // Render from GPU — use LOGICAL face indices, not physical slots
        renderer.render(engine as any, {
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
    console.log("Neo Heat GPU Showcase Running 🚀");
}

main().catch(err => console.error(err));
