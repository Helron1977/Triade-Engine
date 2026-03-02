import { HypercubeGrid, GameOfLifeEngine, CanvasAdapter, HypercubeMasterBuffer } from 'hypercube-compute';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const stats = document.getElementById('stats') as HTMLDivElement;

const SIZE = 256;
canvas.width = 512;
canvas.height = 512;

async function init() {
    try {
        // 1. Allocation d'un buffer global partagé
        const master = new HypercubeMasterBuffer();

        // 2. Initialisation de la grille (Mode CPU par défaut pour cet exemple simple)
        const grid = await HypercubeGrid.create(
            1, 1,            // 1x1 chunk
            SIZE,            // 256x256 cellules
            master,          // MasterBuffer requis
            () => new GameOfLifeEngine(),
            2,               // 2 faces (current, next)
            true,            // Periodique
            'cpu',           // Mode
            true             // Utiliser les Workers si possible
        );

        // 3. Initialisation aléatoire de la première face
        const masterFace = grid.cubes[0][0]!.faces[0];
        for (let i = 0; i < masterFace.length; i++) {
            masterFace[i] = Math.random() > 0.8 ? 1 : 0;
        }

        const ctx = canvas.getContext('2d')!;
        let lastTime = performance.now();
        let frameCount = 0;

        // 4. Boucle de simulation
        async function loop() {
            const now = performance.now();

            // Calcul d'une étape (Calcul tensoriel Hypercube)
            await grid.compute();

            // Rendu via l'adapter statique
            CanvasAdapter.renderFaceToCanvas(grid.cubes[0][0]!.faces[0], SIZE, ctx);

            // Stats FPS
            frameCount++;
            if (now - lastTime >= 1000) {
                stats.innerText = `FPS: ${frameCount} | Grid: ${SIZE}x${SIZE} | Mode: CPU Multithread`;
                frameCount = 0;
                lastTime = now;
            }

            requestAnimationFrame(loop);
        }

        loop();

    } catch (e) {
        console.error("Erreur d'initialisation Hypercube:", e);
        stats.innerText = "Erreur: " + (e as Error).message;
    }
}

init();
