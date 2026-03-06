import { HypercubeGrid, HypercubeMasterBuffer, AerodynamicsEngine, HypercubeViz } from 'hypercube-compute';

const SIZE = 256;

async function init() {
    const canvas = document.getElementById('app') as HTMLCanvasElement;
    const master = new HypercubeMasterBuffer();

    // Aerodynamics Wind Tunnel (LBM engine with obstacles)
    const grid = await HypercubeGrid.create(1, 1, SIZE, master, () => new AerodynamicsEngine(), 22);

    const chunk = grid.cubes[0][0]!;
    const engine = chunk.engine as AerodynamicsEngine;
    const faces = chunk.faces;

    // Create a circular obstacle in the middle
    const cx = SIZE / 4, cy = SIZE / 2, r = 20;
    for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
            const dx = x - cx, dy = y - cy;
            if (dx * dx + dy * dy < r * r) {
                faces[18][y * SIZE + x] = 1.0; // Face 18 = Obstacles in AerodynamicsEngine
            }
        }
    }

    const loop = async () => {
        await grid.compute();

        // Render SMOKE (Face 22) with viridis or plasma for a WOW effect
        HypercubeViz.quickRender(canvas, chunk, 22, 'plasma');

        requestAnimationFrame(loop);
    };

    // Interaction: Inject additional smoke!
    canvas.onmousemove = (e) => {
        const x = (e.offsetX / canvas.clientWidth) * SIZE;
        const y = (e.offsetY / canvas.clientHeight) * SIZE;
        engine.addVortex(faces, x, y, 1.0); // addVortex now injects smoke
    };

    loop();
}

init();
