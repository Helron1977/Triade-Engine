import { HypercubeGrid, HypercubeMasterBuffer, OceanEngine, HypercubeViz } from 'hypercube-compute';

const SIZE = 256;

async function init() {
    const canvas = document.getElementById('app') as HTMLCanvasElement;
    const master = new HypercubeMasterBuffer();

    // Create an Ocean Fluid Grid (LBM D2Q9)
    const grid = await HypercubeGrid.create(1, 1, SIZE, master, () => new OceanEngine(), 23);

    const chunk = grid.cubes[0][0]!;
    const engine = chunk.engine as OceanEngine;

    // Set some initial current
    engine.addGlobalCurrent(chunk.faces, 0.1, 0.0);

    const loop = async () => {
        // Sync LBM populations (Faces 0-8)
        await grid.compute([0, 1, 2, 3, 4, 5, 6, 7, 8]);

        // WOW: Magma colormap to visualize vorticity/curl (Face 21)
        HypercubeViz.quickRender(canvas, chunk, 21, 'magma');

        requestAnimationFrame(loop);
    };

    // Interaction: Dynamic vortices
    canvas.onmousemove = (e) => {
        const x = (e.offsetX / canvas.clientWidth) * SIZE;
        const y = (e.offsetY / canvas.clientHeight) * SIZE;
        engine.addVortex(chunk.faces, x, y, 10.0);
    };

    loop();
}

init();
