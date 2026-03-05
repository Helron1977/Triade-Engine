import { HypercubeCpuGrid } from '../src/core/HypercubeCpuGrid';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';
import { AerodynamicsEngine } from '../src/engines/AerodynamicsEngine';
import { BoundaryType } from '../src/core/cpu/BoundaryConditions';
import { HypercubeMath } from '../src/math/HypercubeMath';
import { Hypercube } from '../src/Hypercube';
import { BenchmarkHUD } from './shared/BenchmarkHUD';

const RESOLUTION = 256;
const ROWS = 2;
const COLS = 2;

async function bootstrap() {
    const urlParams = new URLSearchParams(window.location.search);
    const mode = urlParams.get('mode') === 'gpu' ? 'gpu' : 'cpu';

    // Add description overlay
    const desc = document.createElement('div');
    desc.className = 'showcase-description';
    desc.innerHTML = `
        <h2>01: Aérodynamique 2D</h2>
        <p>Simulation fluide utilisant la méthode Lattice Boltzmann (LBM D2Q9) à O(1). 
        Le calcul est distribué via un pool de Web Workers multithreadé (SharedArrayBuffer).</p>
        <p style="margin-top:10px; font-size: 0.8rem; border-top: 1px solid #333; padding-top:10px;">
        Vortex de Karman visibles derrière l'obstacle fixe.</p>
    `;
    document.body.appendChild(desc);

    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = './showcase.css';
    document.head.appendChild(link);

    const totalCells = RESOLUTION * RESOLUTION;
    const engineTemp = new AerodynamicsEngine();
    const numFaces = engineTemp.getRequiredFaces();

    const masterBuffer = new HypercubeMasterBuffer(totalCells * numFaces * 4 * ROWS * COLS + 1024);

    const grid = await HypercubeCpuGrid.create(
        COLS, ROWS,
        RESOLUTION,
        masterBuffer,
        () => new AerodynamicsEngine(),
        numFaces,
        false, // Not periodic
        true,   // Multithreading on
        new URL('./cpu.worker.ts', import.meta.url).href,
        mode
    );

    // Apply strict Global Boundaries
    grid.boundaryConfig = {
        left: BoundaryType.INFLOW,
        right: BoundaryType.OUTFLOW,
        top: BoundaryType.WALL,
        bottom: BoundaryType.WALL,
        inflowUx: 0.12,
        inflowUy: 0.0,
        inflowDensity: 1.0
    };

    // Draw obstacle in the center of the global grid
    // In a 2x2 grid, the center (512 total) is the boundary.
    // Let's place it at (RESOLUTION, RESOLUTION/2) global -> chunk(1, 0) side?
    // Let's just put it in chunk (0,0) near the right edge to be safe
    const cx = Math.floor(RESOLUTION * 0.7);
    const cy = Math.floor(RESOLUTION / 2);

    for (let y = 0; y < ROWS; y++) {
        for (let x = 0; x < COLS; x++) {
            const faces = grid.cubes[y][x]!.faces;
            // Obstacle logic
            if (x === 0 && y === 0) {
                for (let ly = 0; ly < RESOLUTION; ly++) {
                    for (let lx = 0; lx < RESOLUTION; lx++) {
                        const dist = Math.sqrt((lx - cx) ** 2 + (ly - cy) ** 2);
                        if (dist < 32) {
                            faces[18][ly * RESOLUTION + lx] = 1.0;
                        }
                    }
                }
            }
        }
    }

    // Prepare Rendereur sur la grille UTILE (sans ghost cells)
    const canvas = document.createElement('canvas');
    canvas.width = (RESOLUTION - 2) * COLS;
    canvas.height = (RESOLUTION - 2) * ROWS;
    canvas.style.display = 'block';
    canvas.style.width = '100vw';
    canvas.style.height = '100vh';
    canvas.style.objectFit = 'contain';
    document.body.appendChild(canvas);

    const hud = new BenchmarkHUD('Aerodynamics D2Q9 (O(1) Unrolled)', `${RESOLUTION * COLS} x ${RESOLUTION * ROWS}`);

    // Main Compute Loop
    async function tick() {
        const start = performance.now();

        // 1. Math Step (O(1) LBM Multithreaded)
        await grid.compute();

        // 2. Render Vorticité (Face 21) safely via autoRender
        Hypercube.autoRender(grid, canvas, {
            faceIndex: 21,
            colormap: 'vorticity',
            minVal: -0.05,
            maxVal: 0.05,
            obstaclesFace: 18
        });

        const ms = performance.now() - start;
        hud.updateCompute(ms);
        hud.tickFrame();

        requestAnimationFrame(tick);
    }

    tick();
}

bootstrap();
