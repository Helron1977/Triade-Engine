import { HypercubeCpuGrid } from '../src/core/HypercubeCpuGrid';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';
import { AerodynamicsEngine } from '../src/engines/AerodynamicsEngine';
import { BoundaryType } from '../src/core/cpu/BoundaryConditions';
import { CanvasAdapter } from '../src/io/CanvasAdapter';
import { HypercubeMath } from '../src/math/HypercubeMath';
import { BenchmarkHUD } from './shared/BenchmarkHUD';

const RESOLUTION = 512;
const ROWS = 2;
const COLS = 2;

async function bootstrap() {
    const totalCells = RESOLUTION * RESOLUTION;
    const engineTemp = new AerodynamicsEngine();
    const numFaces = engineTemp.getRequiredFaces();

    // Allocate contiguous SharedArrayBuffer with V4 Header (1024 bytes buffer)
    const masterBuffer = new HypercubeMasterBuffer(totalCells * numFaces * 4 * ROWS * COLS + 1024);

    const grid = await HypercubeCpuGrid.create(
        COLS, ROWS,
        RESOLUTION,
        masterBuffer,
        () => new AerodynamicsEngine(),
        numFaces,
        false, // Not periodic
        true,   // Multithreading on
        '/cpu.worker.ts'
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

    // Draw a wind tunnel obstacle
    const cx = Math.floor(RESOLUTION / 2);
    const cy = Math.floor(RESOLUTION / 2);
    for (let y = 0; y < ROWS; y++) {
        for (let x = 0; x < COLS; x++) {
            const faces = grid.cubes[y][x]!.faces;
            // Draw a circle in the global center chunk
            const isCenterChunk = (x === Math.floor(COLS / 2) && y === Math.floor(ROWS / 2));
            if (isCenterChunk) {
                for (let ly = 0; ly < RESOLUTION; ly++) {
                    for (let lx = 0; lx < RESOLUTION; lx++) {
                        const dist = Math.sqrt((lx - cx) ** 2 + (ly - cy) ** 2);
                        if (dist < 40) {
                            faces[18][ly * RESOLUTION + lx] = 1.0;
                        }
                    }
                }
            }
        }
    }

    // Prepare Renderer
    const canvas = document.createElement('canvas');
    canvas.width = RESOLUTION * COLS;
    canvas.height = RESOLUTION * ROWS;
    canvas.style.width = '100vw';
    canvas.style.height = '100vh';
    canvas.style.objectFit = 'contain';
    document.body.appendChild(canvas);

    const adapter = new CanvasAdapter(canvas);
    const hud = new BenchmarkHUD('Aerodynamics D2Q9 (O(1) Unrolled)', `${RESOLUTION * COLS} x ${RESOLUTION * ROWS}`);

    // Main Compute Loop
    async function tick() {
        const start = performance.now();

        // 1. Math Step (O(1) LBM Multithreaded)
        await grid.compute();

        // 2. Render Vorticité (Face 21)
        adapter.renderFromFaces(
            grid.cubes.map(row => row.map(c => c!.faces)),
            grid.nx, grid.ny, COLS, ROWS,
            {
                faceIndex: 21,
                colormap: 'vorticity',
                minVal: -0.05,
                maxVal: 0.05,
                obstaclesFace: 18
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
