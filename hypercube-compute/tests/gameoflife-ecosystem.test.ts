import { describe, it, expect } from 'vitest';
import { HypercubeGrid } from '../src/core/HypercubeGrid';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';
import { GameOfLifeEngine } from '../src/engines/GameOfLifeEngine';

describe('Game Of Life Ecosystem (New Rules)', () => {

    it('runs an ecosystem without stagnating and maps density to Face 3', async () => {
        const mapSize = 64;
        const totalCellsStrided = mapSize * mapSize;
        const numFaces = 6;
        const masterBuffer = new HypercubeMasterBuffer(totalCellsStrided * numFaces * 4);

        const engine = new GameOfLifeEngine({
            deathProb: 0.015,
            growthProb: 0.03,
            eatThresholdBase: 3.5,
            plantEatThreshold: 2.8,
            herbiEatThreshold: 3.8,
            carniEatThreshold: 3.2,
            carniStarveThreshold: 3.5
        });

        // --- 1. SETUP ENGINE (CPU MODE) ---
        const grid = await HypercubeGrid.create(
            1, 1, mapSize, masterBuffer,
            () => engine,
            numFaces, true, 'cpu', false
        );

        const faces = grid.cubes[0][0]?.faces!;

        // INIT DENSITY Randomly (Face 1: state, Face 3: density)
        // Set everything to states 0, 1, 2, 3 randomly
        for (let i = 0; i < totalCellsStrided; i++) {
            faces[1][i] = Math.floor(Math.random() * 4);
            faces[3][i] = 1.0;
        }

        // Run simulation for 100 steps
        for (let step = 0; step < 100; step++) {
            await grid.compute();
        }

        let hasPlants = false;
        let hasHerbivores = false;
        let hasCarnivores = false;
        let hasDensityVariation = false;

        for (let i = 0; i < totalCellsStrided; i++) {
            const state = faces[1][i];
            const density = faces[3][i];

            if (state === 1) hasPlants = true;
            if (state === 2) hasHerbivores = true;
            if (state === 3) hasCarnivores = true;
            if (density > 0 && density < 1) hasDensityVariation = true;
        }

        // Ecosystem should ideally sustain itself or completely wipe if totally random, 
        // but with 64x64 grid and RPS probabilities, at least plants and density variance MUST exist.
        expect(hasPlants).toBe(true);
        expect(hasDensityVariation).toBe(true);
    });
});
