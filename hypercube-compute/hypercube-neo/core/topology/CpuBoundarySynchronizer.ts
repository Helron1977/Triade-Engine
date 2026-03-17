import { IBoundarySynchronizer, IVirtualGrid, VirtualChunk } from './GridAbstractions';
import { IBufferBridge } from '../memory/IBufferBridge';
import { DataContract } from '../DataContract';
import { ParityManager } from '../ParityManager';

/**
 * Handles high-performance ghost cell synchronization for CPU mode.
 * Focuses on 'Joint' boundaries, including corner/diagonal transfers.
 */
export class CpuBoundarySynchronizer implements IBoundarySynchronizer {

    syncAll(vGrid: IVirtualGrid, bridge: IBufferBridge, parityManager?: ParityManager, mode: 'read' | 'write' = 'write'): void {
        const grid = vGrid as any;
        const dataContract = grid.dataContract as DataContract;
        const descriptor = dataContract.descriptor;
        const padding = descriptor.requirements.ghostCells;
        if (padding === 0) return; // Nothing to sync

        // 1. Resolve current indices for all synchronized faces in the target mode
        const syncIndices: number[] = [];
        let faceIdx = 0;
        for (const face of descriptor.faces) {
            // Smart Sync: Default to true unless explicitly disabled
            if (face.isSynchronized !== false) {
                if (parityManager) {
                    syncIndices.push(parityManager.getFaceIndices(face.name)[mode]);
                } else {
                    syncIndices.push(faceIdx); 
                }
            }
            faceIdx++;
        }

        // The MasterBuffer stride is uniform across ALL chunks based on the MAX dimensions
        let maxNx = 0, maxNy = 0;
        for (const chunk of vGrid.chunks) {
            maxNx = Math.max(maxNx, chunk.localDimensions.nx);
            maxNy = Math.max(maxNy, chunk.localDimensions.ny);
        }
        const pNx = maxNx + 2 * padding;
        const pNy = maxNy + 2 * padding;

        for (const chunk of vGrid.chunks) {
            this.syncChunkBoundaries(chunk, vGrid, bridge, syncIndices, pNx, pNy, padding);
        }
    }

    private syncChunkBoundaries(
        chunk: VirtualChunk,
        vGrid: IVirtualGrid,
        bridge: IBufferBridge,
        syncIndices: number[],
        pNx: number, pNy: number,
        padding: number
    ) {
        const views = bridge.getChunkViews(chunk.id);
        const nx = chunk.localDimensions.nx;
        const ny = chunk.localDimensions.ny;

        for (const joint of chunk.joints) {
            if (joint.role !== 'joint' || !joint.neighborId) continue;

            const neighbor = vGrid.chunks.find(c => c.id === joint.neighborId);
            if (!neighbor) continue;

            const neighborViews = bridge.getChunkViews(joint.neighborId);
            const nNx = neighbor.localDimensions.nx;
            const nNy = neighbor.localDimensions.ny;

            for (const bufIdx of syncIndices) {
                this.transferFace(
                    views[bufIdx], 
                    neighborViews[bufIdx], 
                    joint.face, 
                    nx, ny, 
                    nNx, nNy,
                    pNx, pNy, padding
                );
            }
        }

        // --- CORNER SYNCHRONIZATION ---
        this.syncCorners2D(chunk, vGrid, bridge, syncIndices, pNx, pNy, padding);
    }

    private transferFace(
        mine: Float32Array, theirs: Float32Array, 
        face: string, 
        nx: number, ny: number, 
        nNx: number, nNy: number,
        pNx: number, pNy: number, padding: number
    ) {
        if (face === 'left') {
            // My Left Ghost (x=0) <- Their Right Real (x=nNx)
            for (let y = 0; y < ny + 2 * padding; y++) {
                mine[y * pNx + 0] = theirs[y * pNx + nNx];
            }
        } else if (face === 'right') {
            // My Right Ghost (x=nx+1) <- Their Left Real (x=1)
            for (let y = 0; y < ny + 2 * padding; y++) {
                mine[y * pNx + (nx + 1)] = theirs[y * pNx + 1];
            }
        } else if (face === 'top') {
            // My Top Ghost (y=0) <- Their Bottom Real (y=nNy)
            const startMine = 0 * pNx;
            const startTheirs = nNy * pNx;
            // Note: We only copy the overlapping area. For 1D-safe LBM, pNx is identical.
            // If chunks have different widths, we'd need a loop. Assuming pNx is uniform.
            mine.set(theirs.subarray(startTheirs, startTheirs + pNx), startMine);
        } else if (face === 'bottom') {
            // My Bottom Ghost (y=ny+1) <- Their Top Real (y=1)
            const startMine = (ny + 1) * pNx;
            const startTheirs = 1 * pNx;
            mine.set(theirs.subarray(startTheirs, startTheirs + pNx), startMine);
        }
    }

    private syncCorners2D(
        chunk: VirtualChunk, 
        vGrid: IVirtualGrid, 
        bridge: IBufferBridge, 
        syncIndices: number[], 
        pNx: number, pNy: number, 
        padding: number
    ) {
        const nx = chunk.localDimensions.nx;
        const ny = chunk.localDimensions.ny;
        const dxs = [-1, 1];
        const dys = [-1, 1];

        for (const dx of dxs) {
            for (const dy of dys) {
                const neighbor = vGrid.findChunkAt(chunk.x + dx, chunk.y + dy, chunk.z);
                if (!neighbor) continue;

                const myViews = bridge.getChunkViews(chunk.id);
                const theirViews = bridge.getChunkViews(neighbor.id);
                
                const nNx = neighbor.localDimensions.nx;
                const nNy = neighbor.localDimensions.ny;

                for (const bufIdx of syncIndices) {
                    const mine = myViews[bufIdx];
                    const theirs = theirViews[bufIdx];

                    const myX = dx === -1 ? 0 : nx + 1;
                    const myY = dy === -1 ? 0 : ny + 1;
                    const theirX = dx === -1 ? nNx : 1;
                    const theirY = dy === -1 ? nNy : 1;

                    mine[myY * pNx + myX] = theirs[theirY * pNx + theirX];
                }
            }
        }
    }
}
