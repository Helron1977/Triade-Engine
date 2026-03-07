import { HypercubeFactory } from '../v8-sandbox/core/HypercubeFactory';
import { HeatDiffusionV8 } from '../v8-sandbox/engines/HeatDiffusionV8';
import { Circle } from '../v8-sandbox/core/Shapes';
import { CanvasAdapter } from '../src/io/CanvasAdapter';

/**
 * V8 Heat Diffusion CPU Showcase
 * Proof of the "Zero-Refactor" Contract: Same logic, different backend.
 */
async function main() {
    // 1. Instantiation Déclarative (MODE CPU)
    const proxy = await HypercubeFactory.instantiate(
        HeatDiffusionV8,
        {
            dimensions: {
                nx: 256,
                ny: 256,
                chunks: [2, 1]
            },
            boundaries: {
                left: { role: 'outlet', factor: 0.1 },
                right: { role: 'outlet', factor: 0.1 },
                top: { role: 'outlet', factor: 0.1 },
                bottom: { role: 'outlet', factor: 0.1 }
            },
            params: {
                diffusionRate: 0.2, // Même physique
            },
            mode: 'cpu',
            useWorkers: true // Multithreading CPU
        }
        // Pas besoin de classe GPU ici
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

        // CLEAR TRANSIENT
        const flatCubes = (proxy.grid as any).cubes.flat().filter((c: any) => c !== null);
        for (const cube of flatCubes) {
            cube.clearFace(2); // Clear 'Obstacles'
        }

        // MOTION: Deux cœurs orbitaux
        const t1 = frame * 0.03;
        const t2 = frame * 0.02 + Math.PI;

        const obsX = 256 + Math.sin(frame * 0.01) * 120;
        proxy.addShape(new Circle({ x: obsX, y: 128, z: 0 }, 25, {
            'Obstacles': { role: 'wall', factor: 1.0 },
            'Temperature': { role: 'inlet', value: 0.0 }
        }));

        proxy.addShape(new Circle({
            x: 256 + Math.cos(t1) * 140,
            y: 128 + Math.sin(t1 * 1.5) * 80,
            z: 0
        }, 12, {
            'Temperature': { role: 'inlet', value: 4.0 }
        }));

        proxy.addShape(new Circle({
            x: 256 + Math.cos(t2) * 100,
            y: 128 + Math.sin(t2 * 2.1) * 110,
            z: 0
        }, 8, {
            'Temperature': { role: 'inlet', value: 3.0 }
        }));

        // Compute frame (CPU)
        proxy.compute();

        // Render (Calcul de la face active via la parité)
        const facesMatrix = (proxy.grid as any).cubes.map((row: any[]) => row.map((c: any) => c.faces));
        const activeFace = (proxy.engine as any).parity % 2 === 0 ? 0 : 1;

        adapter.renderFromFaces(
            facesMatrix,
            256, 256,
            2, 1,
            {
                faceIndex: activeFace,
                colormap: 'heatmap',
                minVal: 0,
                maxVal: 3.5,
                obstaclesFace: 2
            }
        );

        requestAnimationFrame(render);
    };

    render();
    console.info("V8 CPU Showcase Running: Zero-Refactor contract validated. 🌍☕");
}

main();
