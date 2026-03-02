import { HypercubeGrid, OceanEngine, HypercubeMasterBuffer } from 'hypercube-compute';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const stats = document.getElementById('stats') as HTMLDivElement;

const CHUNKS_X = 2;
const CHUNKS_Y = 2;
const CHUNK_SIZE = 256;
const TOTAL_SIZE = CHUNK_SIZE * CHUNKS_X;

canvas.width = TOTAL_SIZE;
canvas.height = TOTAL_SIZE;

async function init() {
    try {
        const engineFactory = () => new OceanEngine();
        const master = new HypercubeMasterBuffer();

        // 1. Initialisation Grid 2x2 (Multithreading activé)
        const grid = await HypercubeGrid.create(
            CHUNKS_X, CHUNKS_Y, CHUNK_SIZE,
            master,
            engineFactory,
            6, true, 'cpu', true
        );

        const ctx = canvas.getContext('2d')!;
        const imageData = ctx.createImageData(TOTAL_SIZE, TOTAL_SIZE);
        const data = imageData.data;

        // 2. Interaction: Génération de forçages vortex
        let isForcing = false;
        canvas.onmousedown = (e) => { isForcing = true; handleInteraction(e); };
        canvas.onmouseup = () => isForcing = false;
        canvas.onmousemove = (e) => { if (isForcing) handleInteraction(e); };

        function handleInteraction(e: MouseEvent) {
            const rect = canvas.getBoundingClientRect();
            const gx = e.clientX - rect.left;
            const gy = e.clientY - rect.top;

            // Appliquer le forçage sur le chunk concerné
            const cx = Math.floor(gx / CHUNK_SIZE);
            const cy = Math.floor(gy / CHUNK_SIZE);
            const cube = grid.cubes[cy][cx];

            if (cube && cube.engine instanceof OceanEngine) {
                const lx = Math.floor(gx % CHUNK_SIZE);
                const ly = Math.floor(gy % CHUNK_SIZE);

                // On injecte un moment de rotation (vortex)
                cube.engine.interaction.mouseX = lx;
                cube.engine.interaction.mouseY = ly;
                cube.engine.interaction.active = true;
                // vortex_force is not directly exposed but we can use vortexStrength if needed, 
                // but let's just set active for now as per the engine's interaction structure.
            }
        }

        let lastTime = performance.now();
        let frameCount = 0;

        // 3. Boucle de simulation
        async function loop() {
            const now = performance.now();

            // Calcul multithreadé (Exchange des Ghost Cells automatique)
            await grid.compute();

            // Rendu manuel multi-chunk (pour montrer la structure interne)
            for (let cy = 0; cy < CHUNKS_Y; cy++) {
                for (let cx = 0; cx < CHUNKS_X; cx++) {
                    const cube = grid.cubes[cy][cx]!;
                    const face0 = cube.faces[0]; // Densité / Hauteur

                    const startX = cx * CHUNK_SIZE;
                    const startY = cy * CHUNK_SIZE;

                    for (let y = 0; y < CHUNK_SIZE; y++) {
                        for (let x = 0; x < CHUNK_SIZE; x++) {
                            const val = face0[y * CHUNK_SIZE + x];
                            const px = ((startY + y) * TOTAL_SIZE + (startX + x)) * 4;

                            // Mapper : hauteur -> bleuet profond
                            const h = Math.min(255, Math.max(0, (val - 0.98) * 1000));
                            data[px] = h * 0.2;
                            data[px + 1] = h * 0.5;
                            data[px + 2] = h;
                            data[px + 3] = 255;
                        }
                    }
                }
            }
            ctx.putImageData(imageData, 0, 0);

            // Stats
            frameCount++;
            if (now - lastTime >= 1000) {
                stats.innerText = `Ocean Multitiles • FPS: ${frameCount}`;
                frameCount = 0;
                lastTime = now;
            }

            requestAnimationFrame(loop);
        }

        loop();

    } catch (e) {
        console.error(e);
        stats.innerText = "Error: " + (e as Error).message;
    }
}

init();
