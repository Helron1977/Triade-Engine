import { HypercubeGrid, HypercubeMasterBuffer, OceanEngine, HypercubeViz } from 'hypercube-compute';

const SIZE = 256;

async function init() {
    const canvas = document.getElementById('app') as HTMLCanvasElement;
    const master = new HypercubeMasterBuffer();

    // Aerodynamics Wind Tunnel (LBM engine with obstacles)
    const grid = await HypercubeGrid.create(1, 1, SIZE, master, () => new OceanEngine(), 23);

    const chunk = grid.cubes[0][0]!;
    const engine = chunk.engine as OceanEngine;
    const faces = chunk.faces;

    // Create a circular obstacle in the middle
    const cx = SIZE / 4, cy = SIZE / 2, r = 20;
    for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
            const dx = x - cx, dy = y - cy;
            if (dx * dx + dy * dy < r * r) {
                faces[22][y * SIZE + x] = 1.0; // Face 22 = Obstacles
            }
        }
    }

    const loop = async () => {
        // Add constant wind from left
        engine.addGlobalCurrent(faces, 0.12, 0.0);

        await grid.compute();

        // WOW: Heat colormap for Velocity Magnitude (extracted from Face 18/19 internally by engine or via Viz)
        // Let's use Face 21 (Curl) for the wind tunnel aesthetic
        HypercubeViz.quickRender(canvas, chunk, 21, 'viridis');

        requestAnimationFrame(loop);
    };

    loop();
}

init();
