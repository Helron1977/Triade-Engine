import { HypercubeGrid, HypercubeMasterBuffer, AerodynamicsEngine, HypercubeViz } from 'hypercube-compute';
async function init() {
    const canvas = document.getElementById('app');
    canvas.width = 512;
    canvas.height = 256;
    const stats = document.createElement('div');
    stats.style.cssText = 'position:absolute; top:20px; left:20px; color:#fff; font-family:monospace; font-size:16px; font-weight:bold; pointer-events:none;';
    document.body.appendChild(stats);
    const master = new HypercubeMasterBuffer();
    // 512x256 via 2 chunks of 256x256 side by side
    const grid = await HypercubeGrid.create(2, 1, 256, master, () => new AerodynamicsEngine(), 23);
    // Obstacle finding
    const cx = 100, cy = 128, r = 24;
    for (let c = 0; c < 2; c++) {
        const chunk = grid.cubes[0][c];
        for (let y = cy - r; y < cy + r; y++) {
            for (let x = 0; x < 256; x++) {
                const globalX = c * 256 + x;
                const dist2 = (globalX - cx) ** 2 + (y - cy) ** 2;
                if (dist2 < r * r) {
                    chunk.faces[22][y * 256 + x] = 1.0;
                }
            }
        }
    }
    // Interactions go to the left chunk since that's where the obstacle/vortex is
    const engineConfig = grid.cubes[0][0].engine;
    canvas.onmousemove = (e) => {
        engineConfig.targetX = e.offsetX;
        engineConfig.targetY = e.offsetY;
        engineConfig.weight = e.buttons > 0 ? 1.0 : 0.0;
    };
    canvas.onmousedown = (e) => engineConfig.weight = 1.0;
    canvas.onmouseup = () => engineConfig.weight = 0.0;
    // Use absolute global min/max for the dual-chunk rendering
    const cCtx = canvas.getContext('2d');
    const c1 = document.createElement('canvas');
    c1.width = 256;
    c1.height = 256;
    const c2 = document.createElement('canvas');
    c2.width = 256;
    c2.height = 256;
    let frames = 0, lastTime = performance.now();
    const loop = async () => {
        await grid.compute();
        // Render both chunks and stitch them manually onto the main canvas
        HypercubeViz.quickRender(c1, grid.cubes[0][0], 21, 'bipolar');
        HypercubeViz.quickRender(c2, grid.cubes[0][1], 21, 'bipolar');
        cCtx.drawImage(c1, 0, 0);
        cCtx.drawImage(c2, 256, 0);
        frames++;
        if (performance.now() - lastTime > 1000) {
            stats.innerText = `LBM D2Q9 | FPS: ${frames} | Mass/rho: ${engineConfig.stats.avgRho.toFixed(5)}`;
            frames = 0;
            lastTime = performance.now();
        }
        requestAnimationFrame(loop);
    };
    loop();
}
init();
