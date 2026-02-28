import { TriadeMasterBuffer, TriadeGrid, GameOfLifeEngine } from 'triade-engine';

const canvas = document.getElementById('triade-canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;

const COLS = 4;
const ROWS = 4;
const CHUNK_SIZE = 64;
const GLOBAL_SIZE_W = COLS * CHUNK_SIZE; // 256
const GLOBAL_SIZE_H = ROWS * CHUNK_SIZE; // 256

// 1. Allocate a global shared memory buffer (Faces=2 for GoL: buffer + swap)
const master = new TriadeMasterBuffer();

// 2. Create continuous Toric boundary grid (true)
const grid = new TriadeGrid(COLS, ROWS, CHUNK_SIZE, master, () => new GameOfLifeEngine(), 2, true);

// 3. Setup initial state
for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
        const cube = grid.cubes[y][x];
        if (!cube) continue;
        const stateFace = cube.faces[0];
        // Random seed
        for (let i = 0; i < CHUNK_SIZE * CHUNK_SIZE; i++) {
            stateFace[i] = Math.random() > 0.85 ? 1.0 : 0.0;
        }
    }
}

// 4. Interaction
let isDrawing = false;
canvas.addEventListener('mousedown', (e) => { isDrawing = true; paint(e); });
window.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    paint(e);
});

function paint(e: MouseEvent) {
    const rect = canvas.getBoundingClientRect();

    // Scale from canvas visualization size to logic size
    const scaleX = GLOBAL_SIZE_W / rect.width;
    const scaleY = GLOBAL_SIZE_H / rect.height;

    const x = Math.floor((e.clientX - rect.left) * scaleX);
    const y = Math.floor((e.clientY - rect.top) * scaleY);

    if (x >= 0 && x < GLOBAL_SIZE_W && y >= 0 && y < GLOBAL_SIZE_H) {
        const chunkX = Math.floor(x / CHUNK_SIZE);
        const chunkY = Math.floor(y / CHUNK_SIZE);

        const localX = x % CHUNK_SIZE;
        const localY = y % CHUNK_SIZE;

        const cube = grid.cubes[chunkY] && grid.cubes[chunkY][chunkX];
        if (!cube) return;

        const brushSize = 2;
        for (let dy = -brushSize; dy <= brushSize; dy++) {
            for (let dx = -brushSize; dx <= brushSize; dx++) {
                const nx = localX + dx;
                const ny = localY + dy;
                // Simple bounds check within the chunk (ignoring borders for simplicity in drawing)
                if (nx >= 0 && nx < CHUNK_SIZE && ny >= 0 && ny < CHUNK_SIZE) {
                    // Add some randomness to the brush
                    if (Math.random() > 0.2) {
                        cube.faces[0][ny * CHUNK_SIZE + nx] = 1.0;
                    }
                }
            }
        }
    }
}

// Setup image data
const imgData = ctx.createImageData(GLOBAL_SIZE_W, GLOBAL_SIZE_H);
const data = imgData.data;

// Render loop mapped to grid faces directly (O(1) Memory read)
let lastTime = performance.now();
let acc = 0;

function tick() {
    requestAnimationFrame(tick);

    const now = performance.now();
    const dt = now - lastTime;
    lastTime = now;

    // Limit to 30 FPS logic updates so it doesn't spin too insanely fast,
    // but we let requestAnimationFrame run free.
    acc += dt;
    if (acc > 33) { // ~30 fps tick
        acc = 0;
        // Compute synchronizes the faces across all chunks
        grid.compute([0, 1]); // Faces 0 and 1
    }

    // Paint to canvas via quick raw iteration
    for (let cy = 0; cy < ROWS; cy++) {
        for (let cx = 0; cx < COLS; cx++) {
            const cube = grid.cubes[cy][cx];
            if (!cube) continue;
            const stateFace = cube.faces[0];

            for (let ly = 0; ly < CHUNK_SIZE; ly++) {
                for (let lx = 0; lx < CHUNK_SIZE; lx++) {
                    const globalX = cx * CHUNK_SIZE + lx;
                    const globalY = cy * CHUNK_SIZE + ly;

                    const localIdx = ly * CHUNK_SIZE + lx;
                    const active = stateFace[localIdx] > 0.5;

                    const pixelIdx = (globalY * GLOBAL_SIZE_W + globalX) * 4;
                    // Colors
                    if (active) {
                        data[pixelIdx] = 0;     // R
                        data[pixelIdx + 1] = 255; // G
                        data[pixelIdx + 2] = 136; // B
                        data[pixelIdx + 3] = 255; // A
                    } else {
                        data[pixelIdx] = 15;
                        data[pixelIdx + 1] = 15;
                        data[pixelIdx + 2] = 15;
                        data[pixelIdx + 3] = 255;
                    }
                }
            }
        }
    }
    ctx.putImageData(imgData, 0, 0);
}

// Start
tick();
