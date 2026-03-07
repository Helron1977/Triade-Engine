import { HypercubeCpuGrid } from './HypercubeCpuGrid';
import { Shape } from './Shapes';
import { BoundaryProperty } from '../engines/EngineManifest';

/**
 * V8 Rasterizer - The "Bridge" between World Objects and the Simulation Grid.
 */
export class Rasterizer {
    /**
     * @description Peint une forme dans la grille en injectant toutes ses propriétés.
     */
    static paint(grid: HypercubeCpuGrid, shape: Shape) {
        const descriptor = (grid as any)._descriptor;
        if (!descriptor) return;

        // 1. Préparer les mappings FaceName -> Index
        const faceMappings: { idx: number, value: number }[] = [];

        for (const [name, prop] of Object.entries(shape.properties)) {
            // V8 FIX: On cherche toutes les faces liées (ex: Temperature ET TemperatureNext)
            // Cela garantit que l'injection physique n'est pas écrasée par le ping-pong.
            const matches = descriptor.faces
                .map((f: any, i: number) => ({ f, i }))
                .filter(({ f }: { f: any }) => f.name === name || f.name === name + 'Next');

            if (matches.length === 0) {
                console.warn(`[Rasterizer] Face '${name}' non trouvée dans le descripteur.`);
                continue;
            }

            const val = typeof prop === 'number' ? prop : (Array.isArray(prop.value) ? prop.value[0] : (prop.value ?? 1.0));

            for (const match of matches) {
                faceMappings.push({ idx: match.i, value: val as number });
            }
        }

        if (faceMappings.length === 0) return;

        // 2. Iterate over affected Chunks (Optimized via Bounding Box)
        const bbox = shape.getBoundingBox();
        const vnx = grid.nx - 2;
        const vny = grid.ny - 2;

        const minCX = Math.max(0, Math.floor(bbox.min.x / vnx));
        const maxCX = Math.min(grid.cols - 1, Math.floor(bbox.max.x / vnx));
        const minCY = Math.max(0, Math.floor(bbox.min.y / vny));
        const maxCY = Math.min(grid.rows - 1, Math.floor(bbox.max.y / vny));

        for (let cy = minCY; cy <= maxCY; cy++) {
            for (let cx = minCX; cx <= maxCX; cx++) {
                const cube = grid.cubes[cy][cx];
                if (!cube) continue;

                const modifiedFaceIdxs = faceMappings.map(m => m.idx);

                for (let lz = 0; lz < cube.nz; lz++) {
                    const worldZ = grid.nz > 1 ? lz - 1 : 0;
                    for (let ly = 1; ly < cube.ny - 1; ly++) {
                        const worldY = cy * vny + (ly - 1);
                        for (let lx = 1; lx < cube.nx - 1; lx++) {
                            const worldX = cx * vnx + (lx - 1);

                            if (shape.contains({ x: worldX, y: worldY, z: worldZ })) {
                                const idx = lz * cube.ny * cube.nx + ly * cube.nx + lx;
                                // Injection multi-faces
                                for (const mapping of faceMappings) {
                                    cube.faces[mapping.idx][idx] = mapping.value;
                                }
                            }
                        }
                    }
                }

                // GPU SYNC : If in GPU mode, upload changes immediately
                if (grid.mode === 'gpu') {
                    cube.syncFromHost(modifiedFaceIdxs);
                }
            }
        }

        // 3. Trigger CPU Boundary Sync (For multi-chunk CPU consistency)
        for (const mapping of faceMappings) {
            grid.synchronizeBoundaries(mapping.idx);
        }

        console.info(`[Rasterizer] Shape '${shape.constructor.name}' peinte avec ${faceMappings.length} propriétés. Sync GPU: ${grid.mode === 'gpu'}`);
    }
}
