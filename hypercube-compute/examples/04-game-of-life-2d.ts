import { Hypercube } from '../src/Hypercube';
import { BenchmarkHUD } from './shared/BenchmarkHUD';

const RESOLUTION = 256;
const COLS = 2;
const ROWS = 2;

async function bootstrap() {
    // Add description overlay
    const desc = document.createElement('div');
    desc.className = 'showcase-description';
    desc.innerHTML = `
        <h2>04: Game of Life Ecosystem</h2>
        <p>Automate cellulaire complexe simulant un écosystème (Vide, Plante, Herbivore, Carnivore).
        Démontre la simplicité de l'API Hypercube V5 : configuration et rendu automatique en quelques lignes.</p>
    `;
    document.body.appendChild(desc);

    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = './showcase.css';
    document.head.appendChild(link);

    // 1. Create Grid using V5 Facade (Elegant & Simple)
    const grid = await Hypercube.create({
        engine: 'GameOfLifeEngine',
        resolution: RESOLUTION,
        cols: COLS, rows: ROWS,
        workers: true,
        workerScript: new URL('./cpu.worker.ts', import.meta.url).href
    });

    // 2. Initial Seeding using V5 Global Helpers
    const worldW = (RESOLUTION - 2) * COLS;
    const worldH = (RESOLUTION - 2) * ROWS;

    // Seed a dense colony in the center so it doesn't immediately die
    // 1=Plant, 2=Herbi, 3=Carni
    grid.paintCircle(worldW / 2, worldH / 2, 0, 1, 40, 1.0); // Large forest
    grid.paintCircle(worldW / 2, worldH / 2, 0, 1, 15, 2.0); // Herd of herbivores
    grid.paintCircle(worldW / 2, worldH / 2, 0, 1, 5, 3.0);  // Few carnivores

    // Initialize dense age matrix for the colony
    grid.paintCircle(worldW / 2, worldH / 2, 0, 3, 45, 1.0);

    const canvas = document.createElement('canvas');
    canvas.width = worldW;
    canvas.height = worldH;
    canvas.className = 'centered-canvas';
    document.body.appendChild(canvas);

    const hud = new BenchmarkHUD('Game of Life V5', `${worldW} x ${worldH}`);

    async function tick() {
        const start = performance.now();

        // 3. Simple Compute & Render
        await grid.compute();
        Hypercube.autoRender(grid, canvas, {
            faceIndex: 3, // age/density for nice visuals
            colormap: 'vorticity' // gives organic green/blue/red colors
        });

        hud.updateCompute(performance.now() - start);
        hud.tickFrame();
        requestAnimationFrame(tick);
    }
    tick();
}

bootstrap();
