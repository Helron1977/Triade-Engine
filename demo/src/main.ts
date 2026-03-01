import { HypercubeMasterBuffer, HypercubeGrid, AerodynamicsEngine } from 'Hypercube-engine';

const canvas = document.getElementById('Hypercube-canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;

// 4 chunks to keep it fast but demo the "chunking" concept
const COLS = 2;
const ROWS = 2;
const CHUNK_SIZE = 64;
const GLOBAL_SIZE_W = COLS * CHUNK_SIZE; // 128
const GLOBAL_SIZE_H = ROWS * CHUNK_SIZE; // 128

// 1. Allocate buffer (22 faces required for LBM Aerodynamics D2Q9)
const master = new HypercubeMasterBuffer();

// 2. Create continuous Toric boundary grid (true)
const grid = new HypercubeGrid(COLS, ROWS, CHUNK_SIZE, master, () => new AerodynamicsEngine(), 22, true);

// 3. Setup central circular obstacle
const cx = GLOBAL_SIZE_W / 2;
const cy = GLOBAL_SIZE_H / 2;
const radius = 15;

for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
        const cube = grid.cubes[y][x];
        if (!cube) continue;
        const obstaclesFace = cube.faces[18]; // Face 18 is obstacles in AerodynamicsEngine

        for (let ly = 0; ly < CHUNK_SIZE; ly++) {
            for (let lx = 0; lx < CHUNK_SIZE; lx++) {
                const globalX = x * CHUNK_SIZE + lx;
                const globalY = y * CHUNK_SIZE + ly;
                const distSq = (globalX - cx) ** 2 + (globalY - cy) ** 2;
                if (distSq < radius * radius) {
                    obstaclesFace[ly * CHUNK_SIZE + lx] = 1.0;
                } else {
                    obstaclesFace[ly * CHUNK_SIZE + lx] = 0.0;
                }
            }
        }
    }
}

// 4. Interaction (Draw walls with mouse)
let isDrawing = false;
canvas.addEventListener('mousedown', (e) => { isDrawing = true; paint(e); });
window.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    paint(e);
});

function paint(e: MouseEvent) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = GLOBAL_SIZE_W / rect.width;
    const scaleY = GLOBAL_SIZE_H / rect.height;

    const x = Math.floor((e.clientX - rect.left) * scaleX);
    const y = Math.floor((e.clientY - rect.top) * scaleY);

    if (x >= 0 && x < GLOBAL_SIZE_W && y >= 0 && y < GLOBAL_SIZE_H) {
        const chunkX = Math.floor(x / CHUNK_SIZE);
        const chunkY = Math.floor(y / CHUNK_SIZE);
        const localX = x % CHUNK_SIZE;
        const localY = y % CHUNK_SIZE;

        const cube = grid.cubes[chunkY]?.[chunkX];
        if (!cube) return;

        const brushSize = 2;
        for (let dy = -brushSize; dy <= brushSize; dy++) {
            for (let dx = -brushSize; dx <= brushSize; dx++) {
                const nx = localX + dx;
                const ny = localY + dy;
                // Add obstacle
                if (nx >= 0 && nx < CHUNK_SIZE && ny >= 0 && ny < CHUNK_SIZE) {
                    cube.faces[18][ny * CHUNK_SIZE + nx] = 1.0; // Draw wall
                }
            }
        }
    }
}

// Setup image data
const imgData = ctx.createImageData(GLOBAL_SIZE_W, GLOBAL_SIZE_H);
const data = imgData.data;

function tick() {
    requestAnimationFrame(tick);

    // Compute synchronizes the faces across all chunks
    // AerodynamicsEngine requires distributions [0-8] and macros [18, 19, 20, 21] synced
    grid.compute([0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21]);

    // Paint to canvas
    for (let cy = 0; cy < ROWS; cy++) {
        for (let cx = 0; cx < COLS; cx++) {
            const cube = grid.cubes[cy][cx];
            if (!cube) continue;

            const obstacles = cube.faces[18];
            const ux = cube.faces[19];
            const uy = cube.faces[20];
            const vorticity = cube.faces[21]; // Curl

            for (let ly = 0; ly < CHUNK_SIZE; ly++) {
                for (let lx = 0; lx < CHUNK_SIZE; lx++) {
                    const localIdx = ly * CHUNK_SIZE + lx;
                    const globalX = cx * CHUNK_SIZE + lx;
                    const globalY = cy * CHUNK_SIZE + ly;
                    const pixelIdx = (globalY * GLOBAL_SIZE_W + globalX) * 4;

                    if (obstacles[localIdx] > 0.5) {
                        // Wall (Solid Gray)
                        data[pixelIdx] = 150;
                        data[pixelIdx + 1] = 150;
                        data[pixelIdx + 2] = 150;
                        data[pixelIdx + 3] = 255;
                    } else {
                        // Fluid visualization based on vorticity (curl) and speed
                        const curl = vorticity[localIdx];
                        const speedSq = ux[localIdx] ** 2 + uy[localIdx] ** 2;

                        // Map curl to Red/Blue (Vortices)
                        const curlR = Math.max(0, curl * 20000);
                        const curlB = Math.max(0, -curl * 20000);

                        // Map speed to White (Laminar flow)
                        const glow = Math.min(255, speedSq * 10000);

                        data[pixelIdx] = Math.min(255, curlR + glow * 0.2);
                        data[pixelIdx + 1] = Math.min(255, glow * 0.5);
                        data[pixelIdx + 2] = Math.min(255, curlB + glow);
                        data[pixelIdx + 3] = 255;
                    }
                }
            }
        }
    }
    // Scale up rendering to fill the 512x512 canvas natively using Canvas api
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = GLOBAL_SIZE_W;
    tempCanvas.height = GLOBAL_SIZE_H;
    tempCanvas.getContext('2d')!.putImageData(imgData, 0, 0);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
}

// Start
tick();




































