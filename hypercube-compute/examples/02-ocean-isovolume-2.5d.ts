import { HypercubeCpuGrid } from '../src/core/HypercubeCpuGrid';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';
import { OceanEngine } from '../src/engines/OceanEngine';
import { HypercubeIsoRenderer } from '../src/utils/HypercubeIsoRenderer';
import { HypercubeGPUContext } from '../src/core/gpu/HypercubeGPUContext';
import { BenchmarkHUD } from './shared/BenchmarkHUD';

const RESOLUTION = 128; // Faster for CPU Iso
const ROWS = 2;
const COLS = 2;

async function bootstrap() {
    const urlParams = new URLSearchParams(window.location.search);
    const mode = urlParams.get('mode') === 'gpu' ? 'gpu' : 'cpu';

    if (mode === 'gpu' && (navigator as any).gpu) {
        await HypercubeGPUContext.init();
    }

    // Add description overlay
    const desc = document.createElement('div');
    desc.className = 'showcase-description';
    desc.innerHTML = `
        <h2>02: Ocean 2.5D</h2>
        <p>Simulation de vagues et d'ondes de surface (Wave Equation) avec rendu isométrique volumétrique. 
        Notez la synchronisation fluide des ondes entre les 4 chunks de calcul.</p>
        <p style="margin-top:10px; font-size: 0.8rem; border-top: 1px solid #333; padding-top:10px;">
        Vue isométrique 2.5D (Painter's Algorithm).</p>
    `;
    document.body.appendChild(desc);

    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = './showcase.css';
    document.head.appendChild(link);

    const totalCells = RESOLUTION * RESOLUTION;
    const engineTemp = new OceanEngine();
    const numFaces = engineTemp.getRequiredFaces();

    const masterBuffer = new HypercubeMasterBuffer(totalCells * numFaces * 4 * ROWS * COLS + 1024);

    const grid = await HypercubeCpuGrid.create(
        COLS, ROWS,
        RESOLUTION,
        masterBuffer,
        () => new OceanEngine(),
        numFaces,
        true, // Periodic boundaries globally
        true, // Multithreading on
        new URL('./cpu.worker.ts', import.meta.url).href,
        mode
    );

    // Initial Splash (GLOBAL CENTERED using V5 Helper)
    const worldW = (RESOLUTION - 2) * COLS;
    const worldH = (RESOLUTION - 2) * ROWS;

    // applyEquilibrium automatically handles chunk boundaries 
    // and re-equilibrates LBM populations (f0-f8)
    grid.applyEquilibrium(
        worldW / 2,
        worldH / 2,
        0,    // z
        20,   // radius
        1.8,  // rho
        0.5,  // ux
        0.5   // uy
    );

    if (mode === 'gpu') {
        grid.cubes.flat().forEach(c => c?.syncFromHost());
    }

    // Prepare Advanced IsoRenderer
    const canvas = document.createElement('canvas');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    document.body.appendChild(canvas);

    const isoRenderer = new HypercubeIsoRenderer(canvas, undefined, 4.0);
    const hud = new BenchmarkHUD('OceanEngine 2.5D IsoVolume', `${RESOLUTION * COLS} x ${RESOLUTION * ROWS}`);

    async function tick() {
        const start = performance.now();
        await grid.compute();

        if ((grid as any).mode === 'gpu') {
            for (let y = 0; y < grid.rows; y++) {
                for (let x = 0; x < grid.cols; x++) {
                    const chunk = grid.cubes[y][x];
                    // On ne synchronise que la densité (22) et les obstacles (18) pour le rendu
                    if (chunk) await chunk.syncToHost([22, 18]);
                }
            }
        }

        isoRenderer.clearAndSetup(5, 15, 45); // Deep sea dark blue
        isoRenderer.renderMultiChunkVolume(
            grid.cubes.map(r => r.map(c => c!.faces)),
            grid.nx, grid.ny, COLS, ROWS,
            {
                densityFaceIndex: 22, // rho
                obstacleFaceIndex: 18 // obst
            }
        );

        const ms = performance.now() - start;
        hud.updateCompute(ms);
        hud.tickFrame();
        requestAnimationFrame(tick);
    }
    tick();
}
bootstrap();
