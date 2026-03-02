import { describe, it, expect } from 'vitest';
import { HypercubeGrid } from '../src/core/HypercubeGrid';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';
import { HeatmapEngine } from '../src/engines/HeatmapEngine';

describe('HeatmapEngine O(1) SAT', () => {
    it('correctly diffuses a single point source using a box filter', async () => {
        const mapSize = 8;
        const totalCells = mapSize * mapSize;
        const masterBuffer = new HypercubeMasterBuffer(totalCells * 5 * 4);

        const grid = await HypercubeGrid.create(
            1, 1, mapSize, masterBuffer,
            () => new HeatmapEngine(1, 1.0), // Rayon 1 (3x3), poids 1.0
            5, false, 'cpu', false
        );

        const faces = grid.cubes[0][0]?.faces!;

        // Ajout d'une source unique (center: x=3, y=3)
        // Face 1 est l'input (binaire/densité brute) selon getRequiredFaces() et compute()
        const centerIdx = 3 * mapSize + 3;
        faces[1][centerIdx] = 10.0;

        await grid.compute();

        // Le radius est de 1. La source est à (3,3). 
        // Donc le box filter doit affecter le carré de (2,2) à (4,4), et la valeur doit être 10.0 partout dans ce carré.
        const outputFace = faces[2]; // Face 2 = synth diffusion

        // Vérifie le centre
        expect(outputFace[centerIdx]).toBe(10.0);

        // Vérifie les voisins immédiats
        expect(outputFace[2 * mapSize + 2]).toBe(10.0); // Top-Left
        expect(outputFace[4 * mapSize + 4]).toBe(10.0); // Bottom-Right

        // Vérifie en dehors du radius (par exemple 0,0 ou 5,5)
        expect(outputFace[0]).toBe(0.0);
        expect(outputFace[5 * mapSize + 5]).toBe(0.0);
    });
});
