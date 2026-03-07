import { HypercubeFactory } from '../v8-sandbox/core/HypercubeFactory';
import { HeatDiffusionV8 } from '../v8-sandbox/engines/HeatDiffusionV8';
import { HeatDiffusionGpuV8 } from '../v8-sandbox/engines/HeatDiffusionGpuV8';
import { Circle, Box } from '../v8-sandbox/core/Shapes';
import { HypercubeGPUContext } from '../src/core/gpu/HypercubeGPUContext';
import { CanvasAdapter } from '../src/io/CanvasAdapter';

/**
 * V8 Heat Diffusion Showcase
 * Demonstrating the "Zero-Refactor" Contract in action.
 */
async function main() {
    // 0. Initialize GPU
    const ok = await HypercubeGPUContext.init();
    if (!ok) {
        alert("WebGPU not supported. Switching to CPU logic (Draft)...");
    }

    // 1. Instantiation Déclarative via Factory V8 (ORBITAL CORE)
    const proxy = await HypercubeFactory.instantiate(
        HeatDiffusionV8,
        {
            dimensions: {
                nx: 256,
                ny: 256,
                chunks: [2, 1]
            },
            boundaries: {
                // Outlets aux bords pour éviter la saturation (refroidissement passif)
                left: { role: 'outlet', factor: 0.1 },
                right: { role: 'outlet', factor: 0.1 },
                top: { role: 'outlet', factor: 0.1 },
                bottom: { role: 'outlet', factor: 0.1 }
            },
            initialState: [
                // Obstacle stylisé au centre pour casser la diffusion
                new Circle({ x: 256, y: 128, z: 0 }, 15, {
                    'Obstacles': { role: 'wall', factor: 1.0 }
                })
            ],
            params: {
                diffusionRate: 0.2, // Diffusion maximale pour le WOW
            },
            mode: 'gpu'
        },
        HeatDiffusionGpuV8
    );

    // 2. Setup Canvas
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 256;
    canvas.className = 'centered-canvas';
    document.body.appendChild(canvas);
    const adapter = new CanvasAdapter(canvas);

    let frame = 0;

    // 3. Main Loop
    const render = async () => {
        frame++;

        // ── CLEAR TRANSIENT DATA ──
        // On vide la face des obstacles (2) à chaque frame pour éviter l'effet "Capsule" (traînée)
        const flatCubes = (proxy.grid as any).cubes.flat().filter((c: any) => c !== null);
        for (const cube of flatCubes) {
            cube.clearFace(2); // Clear 'Obstacles'
            if (proxy.grid.mode === 'gpu') cube.syncFromHost([2]); // Push clear to GPU
        }

        // "WOW" MOTION: Deux cœurs orbitaux persistants
        const t1 = frame * 0.03;
        const t2 = frame * 0.02 + Math.PI;

        // OBSTACLE MOBILE : "L'Avaleur de Chaleur"
        const obsX = 256 + Math.sin(frame * 0.01) * 120;
        proxy.addShape(new Circle({ x: obsX, y: 128, z: 0 }, 25, {
            'Obstacles': { role: 'wall', factor: 1.0 },       // Physique : Blocage
            'Temperature': { role: 'inlet', value: 0.0 }      // "Puits" : Avale la chaleur
        }));

        // Cœur d'intensité 1
        proxy.addShape(new Circle({
            x: 256 + Math.cos(t1) * 140,
            y: 128 + Math.sin(t1 * 1.5) * 80,
            z: 0
        }, 12, {
            'Temperature': { role: 'inlet', value: 4.0 }
        }));

        // Cœur d'intensité 2
        proxy.addShape(new Circle({
            x: 256 + Math.cos(t2) * 100,
            y: 128 + Math.sin(t2 * 2.1) * 110,
            z: 0
        }, 8, {
            'Temperature': { role: 'inlet', value: 3.0 }
        }));

        // Compute frame
        proxy.compute();
        const activeFace = proxy.activeFaceIndex;

        // Sync and Render
        const firstChunk = flatCubes[0];

        for (const cube of flatCubes) {
            await cube.syncToHost([activeFace, 2], true);
        }

        const facesMatrix = (proxy.grid as any).cubes.map((row: any[]) => row.map((c: any) => c.faces));

        adapter.renderFromFaces(
            facesMatrix,
            256, 256,
            2, 1,
            {
                faceIndex: activeFace,
                colormap: 'heatmap',
                minVal: 0,
                maxVal: 3.5, // Ajusté pour une meilleure lisibilité
                obstaclesFace: 2
            }
        );

        requestAnimationFrame(render);
    };

    render();
    console.info("V8 Showcase Running: Point de friction résolu. Logique <=> Performance. 🚀");
}

main();
