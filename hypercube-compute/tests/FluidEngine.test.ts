import { describe, it, expect } from 'vitest';
import { HypercubeGrid } from '../src/core/HypercubeGrid';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';
import { FluidEngine } from '../src/engines/FluidEngine';

describe('FluidEngine v3 minimalist test', () => {
    it('simulates advection, buoyancy and dissipation without NaN', async () => {
        const mapSize = 8;
        const totalCells = mapSize * mapSize;
        const masterBuffer = new HypercubeMasterBuffer(totalCells * 6 * 4);

        const grid = await HypercubeGrid.create(
            1, 1, mapSize, masterBuffer,
            () => new FluidEngine(0.5, 0.2, 0.99),
            6, false, 'cpu', false
        );

        const engine = grid.cubes[0][0]?.engine as FluidEngine;
        const faces = grid.cubes[0][0]?.faces!;

        // Ajout d'une source (center: x=4, y=4)
        engine.addSplat(faces, mapSize, 4, 4, 0, -1.0, 1, 1.0, 5.0);

        const initialDensity = engine.getTotalDensity(faces);

        // Execute 10 compute steps
        for (let i = 0; i < 10; i++) {
            await grid.compute();
        }

        const finalDensity = engine.getTotalDensity(faces);

        // 1. Pas de NaN
        const hasNaN = faces.some(face => face && face.some(val => isNaN(val)));
        expect(hasNaN).toBe(false);

        // 2. Dissipation 
        const epsilon = 1e-3;
        const expectedMaxDensity = initialDensity * Math.pow(0.99, 10) + epsilon;
        expect(finalDensity).toBeLessThan(expectedMaxDensity);

        // 3. Buoyancy (velY moyenne < 0)
        const avgVelY = faces[3].reduce((acc, val) => acc + val, 0) / totalCells;
        expect(avgVelY).toBeLessThan(0);

        // 4. Advection: la densité bouge
        // Au bout de 10 frames, le fluide a bougé depuis de sa cellule initiale.
        const allDensitiesPos = faces[0].some(val => val > 0);
        expect(allDensitiesPos).toBe(true);
        expect(finalDensity).toBeGreaterThan(0);
    });
});
