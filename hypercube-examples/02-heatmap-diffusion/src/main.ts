import { HypercubeGrid, HypercubeMasterBuffer, HeatmapEngine, HypercubeViz } from 'hypercube-compute';

const SIZE = 512;

async function init() {
    const canvas = document.getElementById('app') as HTMLCanvasElement;
    const master = new HypercubeMasterBuffer();

    // Create a Heatmap Diffusion Grid (O-N Blur)
    // Face 0: Inputs (sources)
    // Face 2: Blur output
    const grid = await HypercubeGrid.create(1, 1, SIZE, master, () => new HeatmapEngine(10, 0.1), 3);

    const chunk = grid.cubes[0][0];
    const faces = chunk.faces;

    const loop = () => {
        grid.compute();

        // WOW: Professional 'plasma' colormap for thermal rendering
        HypercubeViz.quickRender(canvas, chunk, 2, 'plasma');

        requestAnimationFrame(loop);
    };

    // Interaction: Injected heat on click/drag
    canvas.onmousemove = (e) => {
        if (e.buttons !== 1) return;
        const x = Math.floor((e.offsetX / canvas.clientWidth) * SIZE);
        const y = Math.floor((e.offsetY / canvas.clientHeight) * SIZE);

        // Inject heat directly into Face 0
        const radius = 5;
        for (let dy = -radius; dy <= radius; dy++) {
            for (let dx = -radius; dx <= radius; dx++) {
                const ix = x + dx;
                const iy = y + dy;
                if (ix >= 0 && ix < SIZE && iy >= 0 && iy < SIZE) {
                    faces[0][iy * SIZE + ix] = 1.0;
                }
            }
        }
    };

    loop();
}

init();
