import { MasterBuffer } from './core/memory/MasterBuffer.js';

// The file MasterBuffer.ts has totalSlotsPerChunk logic.
// I just want to run the exact lines that setFaceData does.

const SIZE = 512;
const nChunksX = 2;
const nChunksY = 2;
const vnx = SIZE / nChunksX;
const vny = SIZE / nChunksY;
const nxPhys = vnx + 2;
const nyPhys = vny + 2;

console.log("nxPhys * nyPhys (data length): ", nxPhys * nyPhys);

const grid = {
    config: {
        dimensions: { nx: 512, ny: 512, nz: 1 },
        chunks: { x: 2, y: 2, z: 1 }
    }
};

const nx = Math.floor(grid.config.dimensions.nx / grid.config.chunks.x);
const ny = Math.floor(grid.config.dimensions.ny / grid.config.chunks.y);
const nz = Math.floor((grid.config.dimensions.nz || 1) / (grid.config.chunks.z || 1));
const cellsPerFaceRaw = (nx + 2) * (ny + 2) * (grid.config.dimensions.nz > 1 ? (nz + 2) : 1);

console.log("cellsPerFaceRaw (view length): ", cellsPerFaceRaw);

if (cellsPerFaceRaw !== nxPhys * nyPhys) {
    console.error("LENGTH MISMATCH");
} else {
    console.log("Lengths match precisely");
}
