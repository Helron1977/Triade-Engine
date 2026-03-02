import { HypercubeGrid, FlowFieldEngine, HypercubeMasterBuffer } from 'hypercube-compute';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const stats = document.getElementById('stats') as HTMLDivElement;

const SIZE = 512;
const AGENT_COUNT = 10000;
canvas.width = SIZE;
canvas.height = SIZE;

interface Agent {
    x: number;
    y: number;
    vx: number;
    vy: number;
}

const agents: Agent[] = [];
for (let i = 0; i < AGENT_COUNT; i++) {
    agents.push({
        x: Math.random() * SIZE,
        y: Math.random() * SIZE,
        vx: 0,
        vy: 0
    });
}

async function init() {
    try {
        const engine = new FlowFieldEngine();
        const master = new HypercubeMasterBuffer();

        // 1. Initialisation de la grille tensorielle (CPU Multithread)
        const grid = await HypercubeGrid.create(
            1, 1, SIZE,
            master,
            () => engine,
            6, true, 'cpu', true
        );

        if (!grid.cubes[0][0]) throw new Error("Cube not initialized");

        const ctx = canvas.getContext('2d', { alpha: false })!;
        let lastTime = performance.now();
        let frameCount = 0;

        // 2. Interaction : Mise à jour de la cible du FlowField
        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            engine.targetX = e.clientX - rect.left;
            engine.targetY = e.clientY - rect.top;
        });

        // 3. Boucle principale
        async function loop() {
            const now = performance.now();

            // Étape A: Calcul de la grille (Flow Field)
            await grid.compute();

            // Étape B: Mise à jour des agents (Lecture O(1) du tenseur)
            // On récupère les faces de ForceX (Face 3) et ForceY (Face 4)
            const forceX = grid.cubes[0][0]!.faces[3];
            const forceY = grid.cubes[0][0]!.faces[4];

            ctx.fillStyle = 'rgba(0, 0, 0, 0.2)'; // Effet de traînée
            ctx.fillRect(0, 0, SIZE, SIZE);
            ctx.fillStyle = '#4488ff';

            for (let i = 0; i < AGENT_COUNT; i++) {
                const a = agents[i];

                // Mappage coordonnées -> index grid
                const gx = Math.floor(a.x);
                const gy = Math.floor(a.y);
                const idx = gy * SIZE + gx;

                if (idx >= 0 && idx < forceX.length) {
                    // Lecture instantanée de la force à cette position
                    a.vx += forceX[idx] * 0.5;
                    a.vy += forceY[idx] * 0.5;
                }

                // Application vitesse + friction
                a.x += a.vx;
                a.y += a.vy;
                a.vx *= 0.95;
                a.vy *= 0.95;

                // Bords (Bounce)
                if (a.x < 0 || a.x >= SIZE) a.vx *= -1;
                if (a.y < 0 || a.y >= SIZE) a.vy *= -1;

                // Dessin simple pixel
                ctx.fillRect(a.x, a.y, 1.5, 1.5);
            }

            // Stats
            frameCount++;
            if (now - lastTime >= 1000) {
                stats.innerText = `${AGENT_COUNT.toLocaleString()} Agents • FPS: ${frameCount}`;
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
