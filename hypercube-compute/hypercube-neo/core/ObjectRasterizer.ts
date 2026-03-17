import { IRasterizer, IVirtualGrid, VirtualChunk } from './topology/GridAbstractions';
import { IBufferBridge } from './IBufferBridge';
import { VirtualObject, EngineFace, EngineDescriptor } from './types';
import { DataContract } from './DataContract';
import { ParityManager } from './ParityManager';

/**
 * High-performance rasterizer for Hypercube Neo.
 * Bakes VirtualObjects into physical grids with influence falloff.
 */
export class ObjectRasterizer implements IRasterizer {
    constructor(private parityManager?: ParityManager) { }

    rasterizeChunk(vChunk: VirtualChunk, vGrid: IVirtualGrid, bridge: IBufferBridge, t: number, target: 'read' | 'write' = 'write'): void {
        const grid = vGrid as any;
        const config = grid.config;
        const dataContract = grid.dataContract as DataContract;
        const descriptor = dataContract.descriptor;
        const faceMappings = dataContract.getFaceMappings();

        const chunkObjects: VirtualObject[] = (vGrid as any).getObjectsInChunk(vChunk, t);
        if (chunkObjects.length === 0) return;

        const views = bridge.getChunkViews(vChunk.id);

        const nx = vChunk.localDimensions.nx;
        const ny = vChunk.localDimensions.ny;
        const padding = descriptor.requirements.ghostCells;

        // Calculate maximum dimensions and chunk world offsets
        let maxNx = 0;
        let chunkX0 = 0;
        let chunkY0 = 0;
        
        for (const c of grid.chunks) {
            maxNx = Math.max(maxNx, c.localDimensions.nx);
            // World X offset is the sum of all preceding chunks' widths along the X axis (y and z being the same)
            if (c.y === vChunk.y && c.z === vChunk.z && c.x < vChunk.x) {
                chunkX0 += c.localDimensions.nx;
            }
            // World Y offset is the sum of all preceding chunks' heights along the Y axis (x and z being the same)
            if (c.x === vChunk.x && c.z === vChunk.z && c.y < vChunk.y) {
                chunkY0 += c.localDimensions.ny;
            }
        }
        
        const pNx = maxNx + 2 * padding;
        const pNy = Math.ceil(config.dimensions.ny / config.chunks.y) + 2 * padding; // Roughly, for bounds

        for (const obj of chunkObjects) {
            if (obj.renderOnly) continue;
            if (obj.isBaked === false) continue;

            // 1. Evaluate dynamic position
            let objX = obj.position.x;
            let objY = obj.position.y;
            if (obj.animation?.velocity) {
                objX += obj.animation.velocity.x * t;
                objY += obj.animation.velocity.y * t;
            }

            // 2. Iterate through properties to find matching faces
            for (const [propName, propValue] of Object.entries(obj.properties)) {
                // Find all faces affected by this property
                // (In a real system we might have a name mapping, for now we assume property name = face name)
                const faceIdx = descriptor.faces.findIndex(f => f.name === propName);
                if (faceIdx === -1) continue;

                let bufferIdx: number;
                if (this.parityManager) {
                    const indices = this.parityManager.getFaceIndices(propName);
                    bufferIdx = target === 'read' ? indices.read : indices.write;
                } else {
                    bufferIdx = this.getBufferIndex(faceMappings, faceIdx);
                }

                const view = views[bufferIdx];

                this.rasterizeShape(obj, objX, objY, propValue as number, view, chunkX0, chunkY0, nx, ny, pNx, pNy, padding);
            }
        }
    }

    private getBufferIndex(faceMappings: any[], targetIdx: number): number {
        let bufIdx = 0;
        for (let i = 0; i < targetIdx; i++) {
            bufIdx += faceMappings[i].isPingPong ? 2 : 1;
        }
        return bufIdx;
    }

    private rasterizeShape(
        obj: VirtualObject,
        objX: number, objY: number,
        value: number,
        view: Float32Array,
        chunkX0: number, chunkY0: number,
        nx: number, ny: number,
        pNx: number, pNy: number,
        padding: number
    ) {
        // Simple 2D Circle Rasterization with Falloff
        const objRadius = obj.type === 'circle' ? obj.dimensions.w / 2 : 0;
        const influenceRadius = obj.influence?.radius ?? 0;
        const totalRadius = objRadius + influenceRadius;

        // Bounding box of the object in world coords
        let bX0 = Math.floor(objX - totalRadius);
        let bX1 = Math.ceil(objX + obj.dimensions.w + totalRadius);
        let bY0 = Math.floor(objY - totalRadius);
        let bY1 = Math.ceil(objY + obj.dimensions.h + totalRadius);

        // For polygons, compute bounds from points to avoid clipping rotated shapes
        if (obj.type === 'polygon' && obj.points) {
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            for (const p of obj.points) {
                if (p.x < minX) minX = p.x;
                if (p.x > maxX) maxX = p.x;
                if (p.y < minY) minY = p.y;
                if (p.y > maxY) maxY = p.y;
            }
            bX0 = Math.floor(objX + minX - influenceRadius);
            bX1 = Math.ceil(objX + maxX + influenceRadius);
            bY0 = Math.floor(objY + minY - influenceRadius);
            bY1 = Math.ceil(objY + maxY + influenceRadius);
        }

        // Clip to internal chunk boundaries (excluding ghost cells)
        const startX = Math.max(padding, bX0 - chunkX0 + padding);
        const endX = Math.min(nx + padding, bX1 - chunkX0 + padding);
        const startY = Math.max(padding, bY0 - chunkY0 + padding);
        const endY = Math.min(ny + padding, bY1 - chunkY0 + padding);

        for (let py = startY; py < endY; py++) {
            for (let px = startX; px < endX; px++) {
                const worldX = chunkX0 + (px - padding);
                const worldY = chunkY0 + (py - padding);

                let factor = 0;
                if (obj.type === 'circle') {
                    const dx = worldX - (objX + obj.dimensions.w / 2);
                    const dy = worldY - (objY + obj.dimensions.h / 2);
                    const dist = Math.sqrt(dx * dx + dy * dy);

                    if (obj.influence) {
                        factor = this.calculateFalloff(dist, objRadius, obj.influence);
                    } else if (dist <= objRadius) {
                        factor = 1.0;
                    }
                } else if (obj.type === 'ellipse') {
                    const cx = objX + obj.dimensions.w / 2;
                    const cy = objY + obj.dimensions.h / 2;
                    const rx = obj.dimensions.w / 2;
                    const ry = obj.dimensions.h / 2;
                    const dx = (worldX - cx) / rx;
                    const dy = (worldY - cy) / ry;
                    if (dx * dx + dy * dy <= 1.0) {
                        factor = 1.0;
                    }
                } else if (obj.type === 'rect') {
                    // Simple Box Check
                    if (worldX >= objX && worldX < objX + obj.dimensions.w &&
                        worldY >= objY && worldY < objY + obj.dimensions.h) {
                        factor = 1.0;
                    }
                } else if (obj.type === 'polygon' && obj.points) {
                    // SDF (Signed Distance Field) for Polygon with Anti-Aliasing
                    let minDistanceSq = Infinity;
                    let inside = false;
                    const pts = obj.points;

                    for (let j = 0, k = pts.length - 1; j < pts.length; k = j++) {
                        const pi = pts[j], pk = pts[k];
                        const xi = objX + pi.x, yi = objY + pi.y;
                        const xk = objX + pk.x, yk = objY + pk.y;

                        // 1. Ray Casting for In/Out check
                        if (((yi > worldY) !== (yk > worldY)) &&
                            (worldX < (xk - xi) * (worldY - yi) / (yk - yi) + xi)) {
                            inside = !inside;
                        }

                        // 2. Distance to segment (for SDF anti-aliasing)
                        const dx = xk - xi;
                        const dy = yk - yi;
                        const l2 = dx * dx + dy * dy;
                        if (l2 === 0) continue;

                        let t = ((worldX - xi) * dx + (worldY - yi) * dy) / l2;
                        t = Math.max(0, Math.min(1, t));

                        const dSq = Math.pow(worldX - (xi + t * dx), 2) + Math.pow(worldY - (yi + t * dy), 2);
                        if (dSq < minDistanceSq) minDistanceSq = dSq;
                    }

                    const dist = Math.sqrt(minDistanceSq);
                    if (inside) {
                        // Fully inside or slightly inside (gradient of 1 pixel width)
                        factor = Math.min(1.0, 0.5 + dist);
                    } else {
                        // Fully outside or slightly outside
                        factor = Math.max(0.0, 0.5 - dist);
                    }
                }

                if (factor > 0) {
                    const finalVal = value * factor;
                    const idx = py * pNx + px;

                    switch (obj.rasterMode) {
                        case 'add': view[idx] += finalVal; break;
                        case 'multiply': view[idx] *= finalVal; break;
                        case 'max': view[idx] = Math.max(view[idx], finalVal); break;
                        case 'min': view[idx] = Math.min(view[idx], finalVal); break;
                        case 'replace':
                        default:
                            view[idx] = finalVal;
                            break;
                    }
                }
            }
        }
    }

    private calculateFalloff(dist: number, bodyRadius: number, influence: any): number {
        const d = Math.max(0, dist - bodyRadius);
        if (d > influence.radius) return 0;

        switch (influence.falloff) {
            case 'linear': return 1 - (d / influence.radius);
            case 'gaussian': return Math.exp(-(d * d) / (influence.radius * influence.radius));
            case 'inverse-square': return 1 / (1 + d * d);
            case 'step':
            default:
                return d <= influence.radius ? 1.0 : 0;
        }
    }
}
