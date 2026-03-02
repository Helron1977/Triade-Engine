import { HypercubeGrid, AerodynamicsEngine, HypercubeMasterBuffer } from 'hypercube-compute';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const stats = document.getElementById('stats') as HTMLDivElement;
const dragDisplay = document.getElementById('drag') as HTMLDivElement;

const WIDTH = 512;
const HEIGHT = 256;
canvas.width = WIDTH;
canvas.height = HEIGHT;

async function init() {
    try {
        const engine = new AerodynamicsEngine();
        const master = new HypercubeMasterBuffer();

        // 1. Initialisation Grid (22 faces pour D2Q9 LBM)
        const grid = await HypercubeGrid.create(
            1, 1, WIDTH, // On utilise WIDTH comme base pour le MasterBuffer (qui attend un cube size)
            master,
            () => engine,
            22, true, 'cpu', true
        );

        if (!grid.cubes[0][0]) throw new Error("Cube not initialized");

        const ctx = canvas.getContext('2d')!;
        const imageData = ctx.createImageData(WIDTH, HEIGHT);
        const data = imageData.data;

        // 2. Interaction : Dessin d'obstacles (Face 18)
        let isDrawing = false;
        canvas.onmousedown = (e) => { isDrawing = true; handleMouse(e); };
        canvas.onmouseup = () => isDrawing = false;
        canvas.onmousemove = (e) => { if (isDrawing) handleMouse(e); };

        function handleMouse(e: MouseEvent) {
            const rect = canvas.getBoundingClientRect();
            const x = Math.floor(e.clientX - rect.left);
            const y = Math.floor(e.clientY - rect.top);

            // On dessine une petite tache d'obstacle (Face 18)
            const obstacles = grid.cubes[0][0]!.faces[18];
            for (let dy = -3; dy <= 3; dy++) {
                for (let dx = -3; dx <= 3; dx++) {
                    const idx = (y + dy) * WIDTH + (x + dx);
                    if (idx >= 0 && idx < obstacles.length) {
                        obstacles[idx] = e.shiftKey ? 0 : 1;
                    }
                }
            }
        }

        // 3. Boucle de simulation
        let lastTime = performance.now();
        let frameCount = 0;

        async function loop() {
            const now = performance.now();

            // Calcul Physique
            await grid.compute();

            // Rendu Vorticité (Face 21) + Obstacles (Face 18)
            const curl = grid.cubes[0][0]!.faces[21];
            const obstacles = grid.cubes[0][0]!.faces[18];

            for (let i = 0; i < WIDTH * HEIGHT; i++) {
                const c = curl[i] * 1500; // Intensité vorticité
                const obs = obstacles[i];
                const px = i * 4;

                if (obs > 0) {
                    data[px] = 80; data[px + 1] = 80; data[px + 2] = 90; // Obstacle (Gris/Bleu)
                } else {
                    data[px] = c > 0 ? c : 0;        // Rouge (Vortex +)
                    data[px + 1] = 40 + Math.abs(c);   // Vert (Base)
                    data[px + 2] = c < 0 ? -c : 0;     // Bleu (Vortex -)
                }
                data[px + 3] = 255;
            }
            ctx.putImageData(imageData, 0, 0);

            // Stats
            frameCount++;
            if (now - lastTime >= 1000) {
                stats.innerText = `LBM D2Q9 • FPS: ${frameCount}`;
                dragDisplay.innerText = `Drag Score: ${engine.dragScore.toFixed(2)}`;
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
