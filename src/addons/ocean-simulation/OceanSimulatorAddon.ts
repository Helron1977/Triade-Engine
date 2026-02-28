import { TriadeMasterBuffer } from '../../core/TriadeMasterBuffer';
import { TriadeCubeV2 } from '../../core/TriadeCubeV2';
import { OceanEngine } from './OceanEngine';

export interface Boat {
    x: number;
    y: number;
    vx: number;
    vy: number;
    length: number;
    angle: number;
}

export class OceanSimulatorAddon {
    public readonly size: number;
    private master: TriadeMasterBuffer;
    private cube: TriadeCubeV2;
    private engine: OceanEngine;

    public boats: Boat[] = [];

    constructor(size: number = 256) {
        this.size = size;

        // 1. Initialize Triade Memory System (24 faces for LBM + Bio + Temp)
        this.master = new TriadeMasterBuffer();
        this.cube = new TriadeCubeV2(size, this.master, 24);

        // 2. Initialize Engine
        this.engine = new OceanEngine();
        this.cube.setEngine(this.engine);

        // 3. Initialize distributions to perfect rest equilibrium
        const w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
        for (let k = 0; k < 9; k++) {
            this.cube.faces[k].fill(w[k]);
        }
    }

    /**
     * Engine parameters access
     */
    get params() { return this.engine.params; }
    get stats() { return this.engine.stats; }

    /**
     * High-level Boat API
     */
    addBoat(x: number, y: number, length: number = 20): void {
        this.boats.push({ x, y, vx: 0, vy: 0, length, angle: 0 });
    }

    /**
     * Global Step: Fluid -> Bio -> Boats
     */
    step(): void {
        // Run Triade Math Engine
        this.cube.compute();

        // Run Boat Physics
        const ux = this.cube.faces[18];
        const uy = this.cube.faces[19];
        const size = this.size;

        for (const boat of this.boats) {
            // Sample fluid velocity under boat
            const ix = Math.floor(boat.x);
            const iy = Math.floor(boat.y);

            if (ix >= 0 && ix < size && iy >= 0 && iy < size) {
                const i = iy * size + ix;
                const fux = ux[i];
                const fuy = uy[i];

                // Advection: Boat is carried by fluid
                boat.vx += fux * 0.15 - boat.vx * 0.05; // Force + Friction
                boat.vy += fuy * 0.15 - boat.vy * 0.05;

                // Move boat
                boat.x += boat.vx;
                boat.y += boat.vy;

                // Update orientation based on velocity
                if (Math.abs(boat.vx) > 0.01 || Math.abs(boat.vy) > 0.01) {
                    boat.angle = Math.atan2(boat.vy, boat.vx);
                }

                // Periodic bounds for boats (optional)
                boat.x = (boat.x + size) % size;
                boat.y = (boat.y + size) % size;
            }
        }
    }

    /**
     * Interaction bridge
     */
    setInteraction(x: number, y: number, active: boolean): void {
        this.engine.interaction.mouseX = x;
        this.engine.interaction.mouseY = y;
        this.engine.interaction.active = active;
    }

    /**
     * Helper to render everything to a canvas
     */
    render(ctx: CanvasRenderingContext2D, imageData: ImageData): void {
        const size = this.size;
        const bio = this.cube.faces[21];
        const ux = this.cube.faces[18];
        const uy = this.cube.faces[19];
        const obst = this.cube.faces[22];
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

        // Draw Boats
        ctx.fillStyle = "#fff";
        for (const boat of this.boats) {
            ctx.save();
            ctx.translate(boat.x, boat.y);
            ctx.rotate(boat.angle);
            ctx.fillRect(-boat.length / 2, -2, boat.length, 4);
            ctx.restore();
        }
    }

    /** Access to raw buffers for custom rendering */
    getRawFaces() {
        return this.cube.faces;
    }
}
