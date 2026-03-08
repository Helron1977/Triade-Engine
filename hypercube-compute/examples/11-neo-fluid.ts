import { HypercubeNeoFactory } from '../hypercube-neo/core/HypercubeNeoFactory';
import { EngineDescriptor, HypercubeConfig } from '../hypercube-neo/core/types';
import { NacaHelper } from '../hypercube-neo/helpers/ShapeHelpers';
import { HypercubeNeo } from '../hypercube-neo/HypercubeNeo';

/**
 * Showcase: Neo LBM Aerodynamics (Architecture v4)
 * Phase 10: Performance Optimization & Regression Fix
 */
async function main() {
    const canvas = document.getElementById('display') as HTMLCanvasElement;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;

    // Resolution: 512x512 (Standard Legacy Performance Configuration)
    // experiment: Lower resolution (256x256) and different chunking (4x1)

    const resize = () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    };
    window.addEventListener('resize', resize);
    resize();

    const factory = new HypercubeNeoFactory();

    // NACA Wings (Biplane signature) - AOI: 12 degrees
    // Using 120 points for high-fidelity contour (reduces aliasing noise)
    const wingPoints = NacaHelper.generateNaca4(0.00, 0.0, 0.16, 80, 120, -12 * Math.PI / 180);


    // experiment: hybrid manifest strategy (JSON + TS)
    const manifest = await factory.fromManifest('showcase-aero-v1.json');
    const { config, engine: descriptor } = manifest;

    // Inject dynamic NACA points into the manifest objects
    const wingTop = config.objects?.find((o: any) => o.id === 'wing_top');
    if (wingTop) wingTop.points = wingPoints;
    const wingBottom = config.objects?.find((o: any) => o.id === 'wing_bottom');
    if (wingBottom) wingBottom.points = wingPoints;

    const engine = await factory.build(config, descriptor);

    const NX = config.dimensions.nx;
    const NY = config.dimensions.ny;

    // Internal Buffer for the adapter (Native resolution)
    const internalCanvas = document.createElement('canvas');
    internalCanvas.width = NX;
    internalCanvas.height = NY;

    let isInitialized = false;

    // Native HUD elements
    const fpsElem = document.getElementById('fps');
    const resElem = document.getElementById('resolution');
    const chunksElem = document.getElementById('chunks');

    if (resElem) resElem.innerText = `Res: ${NX}x${NY}`;
    if (chunksElem) chunksElem.innerText = `Chunks: ${config.chunks.x * (config.chunks.y || 1)}`;

    let frameCount = 0;
    let lastTime = performance.now();

    async function loop() {
        try {
            // SUB-STEPPING: 1 step per frame (Align with Legacy Aerodynamics Case 01)
            const SUBSTEPS = 1;
            for (let i = 0; i < SUBSTEPS; i++) {
                // Pass time=1 on the only sub-step to trigger vorticity/smoke updates
                await engine.step(1);

                if (!isInitialized) {
                    if (config.objects && config.objects[0].id === 'grid_init') {
                        config.objects.shift();
                        isInitialized = true;
                    }
                }
            }

            const smokeIdx = engine.parityManager.getFaceIndices('smoke').read;
            const obsIdx = engine.parityManager.getFaceIndices('obstacles').read;
            const vortIdx = engine.parityManager.getFaceIndices('vorticity').read;

            // NATIVE NEO RENDERING
            HypercubeNeo.autoRender(engine, internalCanvas, {
                faceIndex: smokeIdx,
                colormap: 'arctic',
                minVal: 0.0,
                maxVal: 1.0,
                obstaclesFace: obsIdx,
                vorticityFace: vortIdx
            });

            // Scale to display canvas
            ctx.fillStyle = '#020408';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            const scale = Math.min(canvas.width / NX, canvas.height / NY) * 0.95;
            const dw = NX * scale, dh = NY * scale;
            const dx = (canvas.width - dw) / 2, dy = (canvas.height - dh) / 2;

            ctx.imageSmoothingEnabled = true;
            ctx.drawImage(internalCanvas, dx, dy, dw, dh);

            // Update Native FPS
            frameCount++;
            const now = performance.now();
            if (now - lastTime >= 1000) {
                if (fpsElem) fpsElem.innerText = `FPS: ${frameCount}`;
                frameCount = 0;
                lastTime = now;
            }

            requestAnimationFrame(loop);
        } catch (e) {
            console.error("Showcase loop error:", e);
        }
    }

    loop();
}

main().catch(console.error);
