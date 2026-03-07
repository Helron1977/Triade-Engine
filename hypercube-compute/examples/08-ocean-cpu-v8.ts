import { HypercubeMasterBuffer } from '../v8-sandbox/core/HypercubeMasterBuffer';
import { HypercubeCpuGrid } from '../v8-sandbox/core/HypercubeCpuGrid';
import { OceanEngine } from '../src/engines/OceanEngine';
import { OceanV8 } from '../v8-sandbox/engines/OceanV8';
import { V8EngineProxy } from '../v8-sandbox/core/V8EngineProxy';
import { Circle } from '../v8-sandbox/core/Shapes';
import { HypercubeIsoRenderer } from '../src/utils/HypercubeIsoRenderer';
import { BenchmarkHUD } from './shared/BenchmarkHUD';

/**
 * V8 Showcase 08: Ocean Engine (2.5D) Contract Test
 * This example proves the V8 architecture's modularity by hosting a 2.5D engine.
 */
async function main() {
    console.info("[Showcase 08] Starting V8 Ocean 2.5D (Stress Test)...");

    const RESOLUTION = 128;
    const COLS = 2;
    const ROWS = 2;
    const numFaces = 25;

    // 1. Resources allocation via V8 Contract
    const ALIGNMENT = 256;
    const totalCells = RESOLUTION * RESOLUTION * COLS * ROWS;
    const masterBuffer = new HypercubeMasterBuffer(totalCells * numFaces * 4 + 4096);

    // 2. Agnostic Grid Creation
    const grid = await HypercubeCpuGrid.create(
        COLS, ROWS,
        RESOLUTION,
        masterBuffer,
        () => new OceanEngine() as any, // Contract Injection
        numFaces,
        true, // Periodic boundaries are better for ocean
        true, // Multithreading
        undefined,
        'cpu'
    );

    // 3. Robust Initialization (LBM requires a non-zero background rho)
    // We initialize all chunks to rho=1.0 and equilibrium populations.
    const worldW = (RESOLUTION - 2) * COLS;
    const worldH = (RESOLUTION - 2) * ROWS;

    for (const cube of grid.cubes.flat()) {
        const engineInst = cube!.engine as unknown as OceanEngine;
        // Legacy engines like OceanEngine have an init method that sets up both banks
        engineInst.init(cube!.faces, cube!.nx, cube!.ny, cube!.nz);
    }

    // Initial "Momentum" Splash
    grid.applyEquilibrium(worldW / 2, worldH / 2, 0, 15, 1.4, 0.2, 0.2);

    // 4. Proxy Wrapping for Semantic Controls
    const engine = grid.cubes[0][0]?.engine as unknown as OceanEngine;
    const proxy = new V8EngineProxy(grid, OceanV8, engine as any);

    // 5. Setup Fullscreen Canvas for ISO view
    const canvas = document.createElement('canvas');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.zIndex = '-1';
    document.body.appendChild(canvas);

    const isoRenderer = new HypercubeIsoRenderer(canvas, undefined, 4.0);
    const hud = new BenchmarkHUD('V8 Ocean 2.5D (Stress Test)', `${RESOLUTION * COLS} x ${RESOLUTION * ROWS}`);

    let frame = 0;

    // 6. Main Loop
    const tick = async () => {
        const start = performance.now();
        frame++;

        // Compute step (NOW ASYNC & handles automatic border sync via V8 Proxy)
        await proxy.compute();

        // Parity update (propagated to workers on NEXT frame config)
        if (engine.parity !== undefined) {
            engine.parity++;
        }

        // Render ISO view
        isoRenderer.clearAndSetup(5, 15, 35); // Deep sea
        isoRenderer.renderMultiChunkVolume(
            grid.cubes.map(r => r.map(c => c!.faces)),
            grid.nx, grid.ny, COLS, ROWS,
            {
                densityFaceIndex: 22, // Water_Height
                obstacleFaceIndex: 18 // Obstacles
            }
        );

        // Interaction (Semantic Rasterization)
        if (frame % 60 === 0) {
            const rx = Math.random() * worldW;
            const ry = Math.random() * worldH;
            proxy.addShape(new Circle({ x: rx, y: ry, z: 0 }, 10, {
                'Biology': 1.0 // This will be painted on faces 23/24
            }));
        }

        const ms = performance.now() - start;
        hud.updateCompute(ms);
        hud.tickFrame();
        requestAnimationFrame(tick);
    };

    tick();
    console.info("[Showcase 08] Ocean V8 Contract Test Running. 🌊⚖️");
}

main();
