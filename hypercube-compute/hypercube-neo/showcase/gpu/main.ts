import { HypercubeNeoFactory } from '../../core/HypercubeNeoFactory';
import { CanvasAdapterNeo } from '../../io/CanvasAdapterNeo';

async function launch() {
    // 1. Load Manifest
    const manifestUrl = '/examples/showcase-aero-gpu.json';
    const response = await fetch(manifestUrl);
    const manifest = await response.json();

    // Ensure GPU mode
    manifest.config.mode = 'gpu';

    // 2. Build Engine via Neo Factory
    const factory = new HypercubeNeoFactory();
    const engine = await factory.build(manifest.config, manifest.engine);

    // 3. Setup Visualization
    const container = document.getElementById('canvas-container')!;
    const canvas = document.createElement('canvas');
    container.appendChild(canvas);

    // 4. Main Loop
    let lastTime = performance.now();
    let frames = 0;
    const fpsElement = document.getElementById('fps-counter')!;

    async function step(t: number) {
        await engine.step(t / 1000);

        // Static render call
        CanvasAdapterNeo.render(engine as any, canvas, {
            faceIndex: 0,
            vorticityFace: 1,
            obstaclesFace: 2,
            colormap: 'arctic'
        });

        frames++;
        const now = performance.now();
        if (now - lastTime > 1000) {
            fpsElement.innerText = `${frames} FPS`;
            frames = 0;
            lastTime = now;
        }

        requestAnimationFrame(step);
    }

    requestAnimationFrame(step);
}

launch().catch(console.error);
