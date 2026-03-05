import { HypercubeCpuGrid } from '../src/core/HypercubeCpuGrid';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';
import { OceanEngine } from '../src/engines/OceanEngine';
import { HypercubeIsoRenderer } from '../src/utils/HypercubeIsoRenderer';
import { BenchmarkHUD } from './shared/BenchmarkHUD';

const RESOLUTION = 256;
const ROWS = 2; // Testing multi-chunk sync for Ocean
const COLS = 2;

async function bootstrap() {
    const totalCells = RESOLUTION * RESOLUTION;
    const engineTemp = new OceanEngine();
    const numFaces = engineTemp.getRequiredFaces();

    // Allocate contiguous SharedArrayBuffer with V4 Header
    const masterBuffer = new HypercubeMasterBuffer(totalCells * numFaces * 4 * ROWS * COLS + 1024);

    const grid = await HypercubeCpuGrid.create(
        COLS, ROWS,
        RESOLUTION,
        masterBuffer,
        () => new OceanEngine(),
        numFaces,
        true, // Periodic boundaries globally for the pool
        true, // Multithreading on
        '/cpu.worker.ts'
    );

    // Initial Splash
    const cx = Math.floor(RESOLUTION / 2);
    const cy = Math.floor(RESOLUTION / 2);
    for (let y = 0; y < ROWS; y++) {
        for (let x = 0; x < COLS; x++) {
            const faces = grid.cubes[y][x]!.faces;
            if (x === Math.floor(COLS / 2) && y === Math.floor(ROWS / 2)) {
                for (let ly = 0; ly < RESOLUTION; ly++) {
                    for (let lx = 0; lx < RESOLUTION; lx++) {
                        const dist = Math.sqrt((lx - cx) ** 2 + (ly - cy) ** 2);
                        if (dist < 30) {
                            // Density drop
                            faces[9][ly * RESOLUTION + lx] = 2.0;
                        }
                    }
                }
            }
        }
    }

    // Prepare Advanced IsoRenderer
    const canvas = document.createElement('canvas');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    document.body.appendChild(canvas);

    const isoRenderer = new HypercubeIsoRenderer(canvas, undefined, 4.0); // 4x scale factor
    const hud = new BenchmarkHUD('OceanEngine 2.5D IsoVolume', `${RESOLUTION * COLS} x ${RESOLUTION * ROWS}`);

    // Main Compute Loop
    async function tick() {
        const start = performance.now();

        // 1. Math Step
        await grid.compute();

        // 2. Render 2.5D Isometric 
        isoRenderer.clearAndSetup(0, 0, 0);
        isoRenderer.renderMultiChunkVolume(
            grid.cubes.map(r => r.map(c => c!.faces)),
            grid.nx, grid.ny, COLS, ROWS,
            {
                densityFaceIndex: 9,
                obstacleFaceIndex: 20
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
