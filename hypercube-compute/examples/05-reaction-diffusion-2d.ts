import { Hypercube } from '../src/Hypercube';
import { BenchmarkHUD } from './shared/BenchmarkHUD';

const RESOLUTION = 512;
const COLS = 1;
const ROWS = 1;

async function bootstrap() {
    // Add description overlay
    const desc = document.createElement('div');
    desc.className = 'showcase-description';
    desc.innerHTML = `
        <h2>05: Réaction-Diffusion (Turing)</h2>
        <p>Simulation de motifs organiques (léopard, zèbre) via des équations de Gray-Scott.
        Illustre la puissance de calcul d'Hypercube pour des phénomènes de morphogénèse complexes.</p>
    `;
    document.body.appendChild(desc);

    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = './showcase.css';
    document.head.appendChild(link);

    // 1. Create Grid (Gray-Scott Reaction-Diffusion)
    const grid = await Hypercube.create({
        engine: 'GrayScottEngine',
        resolution: RESOLUTION,
        cols: COLS, rows: ROWS,
        workers: true,
        workerScript: new URL('./cpu.worker.ts', import.meta.url).href,
        params: {
            Da: 0.16,
            Db: 0.08,
            feed: 0.035, // Zebra patterns
            kill: 0.060
        }
    });

    const worldW = (RESOLUTION - 2) * COLS;
    const worldH = (RESOLUTION - 2) * ROWS;

    // Initial conditions: Disrupt center to start reaction
    grid.paintCircle(worldW / 2, worldH / 2, 0, 1, 15, 1.0); // Inject B=1.0 in Face 1

    const canvas = document.createElement('canvas');
    canvas.width = worldW;
    canvas.height = worldH;
    canvas.className = 'centered-canvas';
    document.body.appendChild(canvas);

    const hud = new BenchmarkHUD('Reaction Diffusion', `${worldW} x ${worldH}`);

    async function tick() {
        const start = performance.now();

        // 10 integration steps per frame so we can see the slow diffusion patterns grow
        for (let i = 0; i < 10; i++) {
            await grid.compute();
        }

        Hypercube.autoRender(grid, canvas, {
            faceIndex: 1, // Visualize Substance B (the pattern)
            colormap: 'heatmap',
            minVal: 0,
            maxVal: 0.5
        });

        hud.updateCompute(performance.now() - start);
        hud.tickFrame();
        requestAnimationFrame(tick);
    }
    tick();
}

bootstrap();
