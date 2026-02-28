import { TriadeMasterBuffer } from '../../core/TriadeMasterBuffer';
import { TriadeGrid } from '../../core/TriadeGrid';
import { OceanEngine } from './OceanEngine';

export interface Boat {
    x: number; // Global coordinates
    y: number;
    vx: number;
    vy: number;
    length: number;
    angle: number;
}

export class OceanWorld {
    public readonly cols: number;
    public readonly rows: number;
    public readonly chunkSize: number;
    public readonly globalSizeW: number;
    public readonly globalSizeH: number;

    public grid: TriadeGrid;
    public boats: Boat[] = [];
    public keys = { up: false, down: false, left: false, right: false };

    constructor(
        masterBuffer: TriadeMasterBuffer,
        cols: number = 2,
        rows: number = 2,
        chunkSize: number = 64
    ) {
        this.cols = cols;
        this.rows = rows;
        this.chunkSize = chunkSize;
        this.globalSizeW = cols * chunkSize;
        this.globalSizeH = rows * chunkSize;

        // Create the TriadeGrid with continuous periodic boundaries
        // 24 faces are required by OceanEngine (LBM D2Q9 + Macro + Bio)
        this.grid = new TriadeGrid(cols, rows, chunkSize, masterBuffer, () => new OceanEngine(), 24, true);

        this.reset();
    }

    public reset() {
        const w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];

        // Reset distributions and macros
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                const cube = this.grid.cubes[y][x]!;
                cube.faces[20].fill(1.0);     // rho = 1.0
                cube.faces[18].fill(0.0);     // ux = 0
                cube.faces[19].fill(0.0);     // uy = 0
                for (let k = 0; k < 9; k++) {
                    cube.faces[k].fill(w[k]); // w_k * rho (1.0)
                    cube.faces[k + 9].fill(w[k]); // Init swap buffer too!
                }
            }
        }

        // Initialize Global Obstacles (Central Island)
        const cx = this.globalSizeW / 2;
        const cy = this.globalSizeH / 2;
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                const cube = this.grid.cubes[y][x]!;
                const obst = cube.faces[22];
                obst.fill(0.0); // Reset all to water
                for (let ly = 0; ly < this.chunkSize; ly++) {
                    for (let lx = 0; lx < this.chunkSize; lx++) {
                        const globalX = (x * this.chunkSize + lx);
                        const globalY = (y * this.chunkSize + ly);
                        const dx = globalX - cx;
                        const dy = globalY - cy;
                        // Island of radius 20
                        if (dx * dx + dy * dy < 400) {
                            obst[ly * this.chunkSize + lx] = 1.0;
                        }
                    }
                }
            }
        }

        // Reset boats
        for (let b of this.boats) {
            b.x = 20; b.y = 20;
            b.vx = 0; b.vy = 0;
            b.angle = 0;
        }

        // Force sync once
        this.grid.compute([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22]);
    }

    public addBoat(globalX: number, globalY: number, length: number = 20) {
        this.boats.push({ x: globalX, y: globalY, vx: 0, vy: 0, length, angle: 0 });
    }

    public step() {
        // 1. Math Step for all chunks + Boundary Exchange
        // Synchronize LBM distributions [0..8], Bio data [21], AND Macros [18, 19, 20]
        // Syncing Macros ensures the Boat doesn't read 0 velocity exactly on the boundary cell!
        this.grid.compute([0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21]);

        // 2. Global Boat Physics
        for (const boat of this.boats) {
            // Find which chunk the boat is currently in
            const gx = Math.floor(boat.x);
            const gy = Math.floor(boat.y);

            // Toric world wrap around
            const safeGx = (gx + this.globalSizeW) % this.globalSizeW;
            const safeGy = (gy + this.globalSizeH) % this.globalSizeH;

            const chunkX = Math.floor(safeGx / this.chunkSize);
            const chunkY = Math.floor(safeGy / this.chunkSize);
            const localX = safeGx % this.chunkSize;
            const localY = safeGy % this.chunkSize;

            const cube = this.grid.cubes[chunkY][chunkX];
            if (cube) {
                // Read macro velocities from the localized chunk
                const uxFace = cube.faces[18];
                const uyFace = cube.faces[19];

                const localIdx = localY * this.chunkSize + localX;
                const fux = uxFace[localIdx];
                const fuy = uyFace[localIdx];

                // --- Player Controls (WASD / ZQSD) ---
                const engineForce = 0.15;
                const turnForce = 0.05;

                if (this.keys.up) {
                    boat.vx += Math.cos(boat.angle) * engineForce;
                    boat.vy += Math.sin(boat.angle) * engineForce;
                }
                if (this.keys.down) {
                    boat.vx -= Math.cos(boat.angle) * engineForce * 0.5;
                    boat.vy -= Math.sin(boat.angle) * engineForce * 0.5;
                }
                if (this.keys.left) boat.angle -= turnForce;
                if (this.keys.right) boat.angle += turnForce;

                // Si le joueur ne contrôle pas, le moteur tourne au ralenti tout droit
                if (!this.keys.up && !this.keys.down && !this.keys.left && !this.keys.right) {
                    boat.vx += Math.cos(boat.angle) * 0.01;
                    boat.vy += Math.sin(boat.angle) * 0.01;
                }

                // Advection: Boat is carried by ocean currents
                boat.vx += fux * 0.25 - boat.vx * 0.05; // Force de l'eau + Friction hydrodynamique
                boat.vy += fuy * 0.25 - boat.vy * 0.05;

                // Move boat
                boat.x += boat.vx;
                boat.y += boat.vy;

                // --- HARD LIMIT ON ISLAND COLLISION ---
                // Island radius is approx 20. We stop the boat from entering.
                let cDx = boat.x - this.globalSizeW / 2;
                if (Math.abs(cDx) > this.globalSizeW / 2) cDx = cDx > 0 ? cDx - this.globalSizeW : cDx + this.globalSizeW;
                let cDy = boat.y - this.globalSizeH / 2;
                if (Math.abs(cDy) > this.globalSizeH / 2) cDy = cDy > 0 ? cDy - this.globalSizeH : cDy + this.globalSizeH;

                const newDist = Math.sqrt(cDx * cDx + cDy * cDy);
                const islandR = 24; // Buffer radius
                if (newDist < islandR) {
                    const angleFromIsland = Math.atan2(cDy, cDx);

                    // Repulse strictly to the border of the island
                    boat.x += Math.cos(angleFromIsland) * (islandR - newDist);
                    boat.y += Math.sin(angleFromIsland) * (islandR - newDist);

                    // Kill velocity
                    boat.vx *= 0.1;
                    boat.vy *= 0.1;
                }

                // Toric world wrap for exact coordinates
                boat.x = (boat.x + this.globalSizeW) % this.globalSizeW;
                boat.y = (boat.y + this.globalSizeH) % this.globalSizeH;

                // Update orientation
                if (Math.abs(boat.vx) > 0.01 || Math.abs(boat.vy) > 0.01) {
                    boat.angle = Math.atan2(boat.vy, boat.vx);
                }
            }
        }
    }

    // Paramètres Globaux du "Touilleur" (Tempête/Vortex)
    public setVortexParams(strength: number, radius: number) {
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                const engine = this.grid.cubes[y][x]?.engine as OceanEngine;
                if (engine) {
                    engine.params.vortexStrength = strength;
                    engine.params.vortexRadius = radius;
                }
            }
        }
    }

    /**
     * Helper to render a specific chunk to an isolated Canvas Context
     */
    public renderChunk(chunkX: number, chunkY: number, ctx: CanvasRenderingContext2D, imageData: ImageData) {
        const cube = this.grid.cubes[chunkY][chunkX];
        if (!cube) return;

        const size = this.chunkSize;
        const bio = cube.faces[21];
        const ux = cube.faces[18];
        const uy = cube.faces[19];
        const obst = cube.faces[22];
        const data = imageData.data;

        for (let i = 0; i < size * size; i++) {
            const j = i * 4;
            if (obst[i] > 0.5) {
                data[j] = 20; data[j + 1] = 20; data[j + 2] = 40; data[j + 3] = 255;
            } else {
                const speed = Math.sqrt(ux[i] * ux[i] + uy[i] * uy[i]);
                const intensity = speed * 1200;
                data[j] = Math.min(255, bio[i] * 40 + intensity * 0.1);
                data[j + 1] = Math.min(255, bio[i] * 120 + intensity * 0.4);
                data[j + 2] = Math.min(255, intensity * 2.5 + 20);
                data[j + 3] = 255;
            }
        }
        ctx.putImageData(imageData, 0, 0);

        // Draw boats that belong to THIS chunk
        ctx.fillStyle = "#fff";
        for (const boat of this.boats) {
            // Check if boat is in this chunk visually
            // (We convert global boat coords to local chunk coords to draw them relative to this Canvas)
            const localBoatX = boat.x - chunkX * size;
            const localBoatY = boat.y - chunkY * size;

            // Simple culling (draw if it's within chunk bounds or slightly outside to handle edges)
            if (localBoatX >= -boat.length && localBoatX <= size + boat.length &&
                localBoatY >= -boat.length && localBoatY <= size + boat.length) {
                ctx.save();
                ctx.translate(localBoatX, localBoatY);
                ctx.rotate(boat.angle);
                ctx.fillRect(-boat.length / 2, -2, boat.length, 4);
                ctx.restore();
            }
        }
    }

    // Interaction globally mapped
    public setInteraction(globalX: number, globalY: number, active: boolean) {
        const safeGx = (globalX + this.globalSizeW) % this.globalSizeW;
        const safeGy = (globalY + this.globalSizeH) % this.globalSizeH;

        const chunkX = Math.floor(safeGx / this.chunkSize);
        const chunkY = Math.floor(safeGy / this.chunkSize);
        const localX = safeGx % this.chunkSize;
        const localY = safeGy % this.chunkSize;

        // Reset all chunks interactions first to avoid "sticky" mouse
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                const engine = this.grid.cubes[y][x]?.engine as OceanEngine;
                if (engine) engine.interaction.active = false;
            }
        }

        const cube = this.grid.cubes[chunkY][chunkX];
        if (cube) {
            const engine = cube.engine as OceanEngine;
            engine.interaction.mouseX = localX;
            engine.interaction.mouseY = localY;
            engine.interaction.active = active;
        }
    }
}
