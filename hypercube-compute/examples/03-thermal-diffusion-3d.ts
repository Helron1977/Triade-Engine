import { HypercubeCpuGrid } from '../src/core/HypercubeCpuGrid';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';
import { HeatDiffusionEngine3D } from '../src/engines/HeatDiffusionEngine3D';
import { Hypercube } from '../src/Hypercube';
import { BenchmarkHUD } from './shared/BenchmarkHUD';

const RESOLUTION = 48;
const ROWS = 2;
const COLS = 2;
const NZ = 32;

async function bootstrap() {
    const urlParams = new URLSearchParams(window.location.search);
    const mode = urlParams.get('mode') === 'gpu' ? 'gpu' : 'cpu';

    // Add description overlay
    const desc = document.createElement('div');
    desc.className = 'showcase-description';
    desc.innerHTML = `
        <h2>03: Diffusion Thermique 3D</h2>
        <p>Simulation de diffusion de chaleur dans un volume 3D (Laplacien 3D). 
        Le moteur calcule l'état de plus de 200 000 cellules en 3D à chaque frame.</p>
        <p style="margin-top:10px; font-size: 0.8rem; border-top: 1px solid #333; padding-top:10px;">
        Affichage d'une tranche 2D (Slice Z) au centre du volume.</p>
    `;
    document.body.appendChild(desc);

    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = './showcase.css';
    document.head.appendChild(link);

    const totalCells = RESOLUTION * RESOLUTION * NZ;
    const engineTemp = new HeatDiffusionEngine3D();
    const numFaces = engineTemp.getRequiredFaces();

    const masterBuffer = new HypercubeMasterBuffer(totalCells * numFaces * 4 * ROWS * COLS + 1024);

    const grid = await HypercubeCpuGrid.create(
        COLS, ROWS,
        { nx: RESOLUTION, ny: RESOLUTION, nz: NZ },
        masterBuffer,
        () => new HeatDiffusionEngine3D(),
        numFaces,
        false,
        true, // Multithreading on 
        new URL('./cpu.worker.ts', import.meta.url).href,
        mode
    );

    // Initial Heat Drop centered GLOBALLY (using V5 Helper)
    const worldW = (RESOLUTION - 2) * COLS;
    const worldH = (RESOLUTION - 2) * ROWS;

    // Paint heat at face 0
    grid.paintCircle(worldW / 2, worldH / 2, NZ / 2, 0, 15, 100.0);

    const canvas = document.createElement('canvas');
    canvas.width = (RESOLUTION - 2) * COLS;
    canvas.height = (RESOLUTION - 2) * ROWS;
    canvas.style.display = 'block';
    canvas.style.width = '100vw';
    canvas.style.height = '100vh';
    canvas.style.objectFit = 'contain';
    document.body.appendChild(canvas);

    const hud = new BenchmarkHUD('Thermal Diffusion 3D', `${RESOLUTION * COLS} x ${RESOLUTION * ROWS} x ${NZ}`);

    let currentSliceZ = Math.floor(NZ / 2);

    async function tick() {
        const start = performance.now();
        await grid.compute();

        const syncFaces = grid.cubes[0][0]?.engine?.getSyncFaces?.();
        const displayFace = (syncFaces && syncFaces.length > 0) ? syncFaces[0] : 0;

        Hypercube.autoRender(grid, canvas, {
            faceIndex: displayFace,
            colormap: 'heatmap',
            minVal: 0,
            maxVal: 80, // Slightly tighter max to see internal colors better
            sliceZ: currentSliceZ
        });

        const ms = performance.now() - start;
        hud.updateCompute(ms);
        hud.tickFrame();
        requestAnimationFrame(tick);
    }
    tick();
}
bootstrap();
