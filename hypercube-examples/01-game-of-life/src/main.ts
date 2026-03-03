import { HypercubeGrid, HypercubeMasterBuffer, GameOfLifeEngine, HypercubeViz } from 'hypercube-compute';

const SIZE = 256;

async function init() {
    const canvas = document.getElementById('app') as HTMLCanvasElement;
    const master = new HypercubeMasterBuffer();

    // Create an Organic Ecosystem Grid
    // Face 1: Discrete State (Plant/Herbivore/Carnivore)
    // Face 2: Density/Age (0.0 to 1.0)
    const grid = await HypercubeGrid.create(1, 1, SIZE, master, () => new GameOfLifeEngine({
        deathProb: 0.002,
        growthProb: 0.05
    }), 3);

    const chunk = grid.cubes[0][0];
    if (!chunk) throw new Error("Could not initialize HypercubeChunk. Check grid dimensions.");
    const faces = chunk.faces;

    // Initial Random Seed
    for (let i = 0; i < SIZE * SIZE; i++) {
        if (Math.random() > 0.95) faces[1][i] = 1; // Plants
        if (Math.random() > 0.99) faces[1][i] = 2; // Herbivores
    }

    const loop = () => {
        grid.compute();

        // WOW: Effortless professional visualization with Viridis colormap
        // Rendering Face 2 (Organic Density)
        HypercubeViz.quickRender(canvas, chunk, 2, 'viridis');

        requestAnimationFrame(loop);
    };

    // Interaction: Add Carnivores on Click
    canvas.onmousedown = (e) => {
        const x = Math.floor((e.offsetX / canvas.clientWidth) * SIZE);
        const y = Math.floor((e.offsetY / canvas.clientHeight) * SIZE);
        const idx = y * SIZE + x;
        faces[1][idx] = 3; // Carnivore
        faces[2][idx] = 1.0;
    };

    loop();
}

init();
