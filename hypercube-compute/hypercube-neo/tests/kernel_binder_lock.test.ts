import { describe, it, expect, vi } from 'vitest';
import { NumericalDispatcher } from '../core/NumericalDispatcher';
import { NeoHeatmapKernel } from '../core/kernels/NeoHeatmapKernel';
import { ParityManager } from '../core/ParityManager';
import { KernelRegistry } from '../core/kernels/KernelRegistry';
import { DataContract } from '../core/DataContract';
import { ComputeContext } from '../core/kernels/ComputeContext';

describe('Kernel Binder Lock (Phase 3: Context)', () => {
    
    it('locks view resolution in NumericalDispatcher', () => {
        const mockVChunk = { id: 'c0', localDimensions: { nx: 2, ny: 2 }, x:0, y:0, z:0 };
        const descriptor = {
            faces: [
                { name: 'temp', type: 'scalar' },
                { name: 'obstacles', type: 'mask' }
            ],
            requirements: { ghostCells: 1, pingPong: true },
            rules: [{ type: 'neo-heat', source: 'temp', params: {} }]
        };
        const contract = new DataContract(descriptor as any);

        const mockVGrid = {
            chunks: [mockVChunk],
            dimensions: { nx: 2, ny: 2, nz: 1 },
            chunkLayout: { x: 1, y: 1 },
            config: { boundaries: [], objects: [] },
            dataContract: contract
        };

        const mockBuffer = {
            getChunkViews: vi.fn(() => [
                new Float32Array(16).fill(0.1), // temp read
                new Float32Array(16).fill(0.2), // temp write
                new Float32Array(16).fill(0.3)  // obstacles
            ])
        };

        const parity = new ParityManager(contract);
        const dispatcher = new NumericalDispatcher(mockVGrid as any, mockBuffer as any, parity);
        
        // Register a spy kernel
        const spyKernel = {
            metadata: { 
                roles: { 
                    source: 'temp', 
                    destination: 'temp', 
                    obstacles: 'obstacles' 
                } 
            },
            execute: vi.fn()
        };
        vi.spyOn(KernelRegistry, 'get').mockReturnValue(spyKernel as any);

        dispatcher.dispatch(1.0);

        expect(spyKernel.execute).toHaveBeenCalled();
        const [views, context] = spyKernel.execute.mock.calls[0];

        // views are still the raw physical faces (3: tempRead, tempWrite, obstacles)
        expect(views).toHaveLength(3);
        expect(context.indices['temp']).toBeDefined();
        expect(context.pNx).toBe(4); // 2 + 2*1
    });

    it('locks NeoHeatmapKernel baseline physics (1 iteration)', () => {
        const kernel = new NeoHeatmapKernel();
        
        // 2x2 grid + 1 ghost cell = 4x4 buffer (16 cells)
        const tempRead = new Float32Array(16).fill(0);
        const tempWrite = new Float32Array(16).fill(0);
        const obstacles = new Float32Array(16).fill(0);

        // Inject heat at center (1,1)
        tempRead[5] = 100.0; 

        // NumericalDispatcher passes ALL faces
        const views = [tempRead, tempWrite, obstacles];
        const indices = {
            'temp': { read: 0, write: 1 },
            'obstacles': { read: 2, write: 2 }
        };
        const scheme = { type: 'neo-heat', source: 'temp', params: { diffusion_rate: 0.1, decay_factor: 1.0 } } as any;
        const chunk = { localDimensions: { nx: 2, ny: 2 }, x: 0, y: 0, z: 0 };

        const context: ComputeContext = {
            nx: 2, ny: 2,
            pNx: 4, pNy: 4,
            padding: 1,
            scheme,
            indices,
            params: { time: 0, tick: 0 },
            chunk: chunk as any,
            gridConfig: {}
        };

        kernel.execute(views, context);

        expect(tempWrite[5]).toBeCloseTo(60);
        expect(tempWrite[6]).toBeCloseTo(10);
    });
});
