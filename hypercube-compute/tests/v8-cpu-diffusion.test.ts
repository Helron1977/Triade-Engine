import { describe, it, expect } from 'vitest';
import { V8EngineShim } from '../v8-sandbox/core/V8EngineShim';
import { HeatDiffusionV8 } from '../v8-sandbox/engines/HeatDiffusionV8';

describe('V8EngineShim CPU Diffusion', () => {
    it('should diffuse heat between two faces (Ping-Pong)', () => {
        const shim = new V8EngineShim(HeatDiffusionV8);
        const nx = 4;
        const ny = 4;
        const nz = 1;
        const size = nx * ny * nz;

        // Face 0: Temperature
        // Face 1: TemperatureNext
        // Face 2: Obstacles
        const faces = [
            new Float32Array(size),
            new Float32Array(size),
            new Float32Array(size)
        ];

        // 1. Initial State: Un point chaud au centre (1,1)
        // [0, 0, 0, 0]
        // [0, 1.0, 0, 0]
        // [0, 0, 0, 0]
        // [0, 0, 0, 0]
        faces[0][1 * nx + 1] = 1.0;

        // 2. Premier Compute (Parity 0) -> Source=Face0, Dest=Face1
        shim.parity = 0;
        shim.compute(faces, nx, ny, nz);

        // On attend de la diffusion dans Face 1
        // Laplacian à (1,1) pour rate=0.1
        // neighbors: 0+0+0+0 = 0. center = 1.0
        // laplacian = 0 - 4*1.0 = -4.0
        // newVal = 1.0 + 0.1 * (-4.0) = 0.6
        expect(faces[1][1 * nx + 1]).toBeCloseTo(0.6);
        // Les voisins INTERNES doivent avoir reçu de la chaleur
        // (1,2) -> 1 * 4 + 2 = 6
        expect(faces[1][1 * nx + 2]).toBeCloseTo(0.1);

        // 3. Second Compute (Parity 1) -> Source=Face1, Dest=Face0
        // Actuellement le shim est "hardcodé" face0 -> face1. Testons si ça échoue.
        shim.parity = 1;
        // On modifie Face 1 pour voir si le compute repart bien de là
        faces[1].fill(0);
        faces[1][1 * nx + 1] = 2.0;

        shim.compute(faces, nx, ny, nz);

        // Si le shim est buggé et ne regarde que Parity, il écrira encore dans Face 1 
        // ou lira encore de Face 0.
        // On veut que Face 0 reçoive le résultat
        expect(faces[0][1 * nx + 1], "Doit lire de Face 1 et écrire dans Face 0 lors du second tour").toBeCloseTo(1.2); // 2.0 + 0.1 * (-8)
    });

    it('should respect obstacles on CPU', () => {
        const shim = new V8EngineShim(HeatDiffusionV8);
        const nx = 4;
        const ny = 4;
        const faces = [new Float32Array(16), new Float32Array(16), new Float32Array(16)];

        faces[0].fill(1.0);
        faces[2][1 * nx + 1] = 1.0; // MOCK OBSTACLE

        shim.compute(faces, nx, 4, 1);

        // L'obstacle doit forcer la température à 0 dans le buffer de destination
        expect(faces[1][1 * nx + 1]).toBe(0.0);
    });
});
