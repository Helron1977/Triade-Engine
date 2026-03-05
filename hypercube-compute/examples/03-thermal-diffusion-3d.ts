import { HypercubeCpuGrid } from '../src/core/HypercubeCpuGrid';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';
import { HeatDiffusionEngine3D } from '../src/engines/HeatDiffusionEngine3D';
import { CanvasAdapter } from '../src/io/CanvasAdapter';
import { BenchmarkHUD } from './shared/BenchmarkHUD';

const RESOLUTION = 32;
const ROWS = 2;
const COLS = 2;
const NZ = 32; // Z-axis chunks (1 per cube naturally if NZ array is passed)
// Wait, HypercubeGrid maps 3D internally via the NZ parameter passed on compute/creation.

async function bootstrap() {
    // 32x32x32 = 32768 cells per chunk
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
        '/cpu.worker.ts'
    );

    // Initial Heat Drop in the center of the 3D volume
    for (let y = 0; y < ROWS; y++) {
        for (let x = 0; x < COLS; x++) {
            const faces = grid.cubes[y][x]!.faces;
            if (x === 0 && y === 0) { // Top left chunk
                const cz = Math.floor(NZ / 2);
                const cy = Math.floor(RESOLUTION / 2);
                const cx = Math.floor(RESOLUTION / 2);

                for (let lz = 0; lz < NZ; lz++) {
                    const zOff = lz * RESOLUTION * RESOLUTION;
                    for (let ly = 0; ly < RESOLUTION; ly++) {
                        const yOff = ly * RESOLUTION;
                        for (let lx = 0; lx < RESOLUTION; lx++) {
                            const dist = Math.sqrt((lx - cx) ** 2 + (ly - cy) ** 2 + (lz - cz) ** 2);
                            if (dist < 8) {
                                faces[0][zOff + yOff + lx] = 100.0; // 100 degrees
                            }
                        }
                    }
                }
            }
        }
    }

    const canvas = document.createElement('canvas');
    canvas.width = RESOLUTION * COLS;
    canvas.height = RESOLUTION * ROWS;
    canvas.style.width = '100vw';
    canvas.style.height = '100vh';
    canvas.style.objectFit = 'contain';
    document.body.appendChild(canvas);

    const adapter = new CanvasAdapter(canvas);
    const hud = new BenchmarkHUD('Thermal Diffusion 3D', `${RESOLUTION * COLS} x ${RESOLUTION * ROWS} x ${NZ}`);

    // Z-slice to render
    let currentSliceZ = Math.floor(NZ / 2);

    async function tick() {
        const start = performance.now();

        await grid.compute();

        // Render a 2D slice of the 3D volume
        adapter.renderFromFaces(
            grid.cubes.map(row => row.map(c => c!.faces)),
            grid.nx, grid.ny, COLS, ROWS,
            {
                faceIndex: 0,
                colormap: 'heatmap',
                minVal: 0,
                maxVal: 50,
                sliceZ: currentSliceZ
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
