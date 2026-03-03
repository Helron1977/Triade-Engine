import { HypercubeGrid, HypercubeMasterBuffer, FlowFieldEngine, HypercubeViz } from 'hypercube-compute';

const SIZE = 256;

async function init() {
    const canvas = document.getElementById('app') as HTMLCanvasElement;
    const master = new HypercubeMasterBuffer();

    // Massive Pathfinding Flowfield (V3/V4 optimized)
    const grid = await HypercubeGrid.create(1, 1, SIZE, master, () => new FlowFieldEngine(), 12);

    const chunk = grid.cubes[0][0];
    const engine = chunk.engine as FlowFieldEngine;

    // Set a target in the bottom-right
    engine.setTarget(SIZE - 20, SIZE - 20);

    const loop = () => {
        grid.compute();

        // WOW: Plasma colormap for distance potential field
        HypercubeViz.quickRender(canvas, chunk, 0, 'plasma');

        requestAnimationFrame(loop);
    };

    // Interaction: Move Target
    canvas.onmousedown = (e) => {
        const x = Math.floor((e.offsetX / canvas.clientWidth) * SIZE);
        const y = Math.floor((e.offsetY / canvas.clientHeight) * SIZE);
        engine.setTarget(x, y);
    };

    loop();
}

init();
