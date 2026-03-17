import { describe, it, expect, vi, beforeEach } from 'vitest';
import { CanvasAdapterNeo } from '../core/../io/CanvasAdapterNeo';
import { NeoEngineProxy } from '../core/NeoEngineProxy';

// Minimal Mock for NeoEngineProxy and its dependencies
function createMockProxy() {
    return {
        vGrid: {
            dimensions: { nx: 16, ny: 16, nz: 1 },
            chunkLayout: { x: 1, y: 1 },
            chunks: [{ id: 'c0', x: 0, y: 0, localDimensions: { nx: 16, ny: 16 } }],
            dataContract: {
                descriptor: {
                    faces: [
                        { name: 'temperature', type: 'scalar' },
                        { name: 'obstacles', type: 'mask' }
                    ],
                    requirements: { ghostCells: 1 }
                }
            }
        },
        bridge: {
            getChunkViews: () => [
                new Float32Array(18 * 18).fill(0.5), // temperature (nx+2 * ny+2)
                new Float32Array(18 * 18).fill(0)    // obstacles
            ]
        },
        parityManager: {
            getFaceIndices: (name: string) => ({ read: name === 'temperature' ? 0 : 1, write: name === 'temperature' ? 0 : 1 })
        }
    } as unknown as NeoEngineProxy;
}

describe('CanvasAdapterNeo', () => {
    let mockCanvas: any;
    let mockCtx: any;

    beforeEach(() => {
        mockCtx = {
            createImageData: vi.fn(() => ({
                data: new Uint8ClampedArray(16 * 16 * 4)
            })),
            putImageData: vi.fn(),
        };
        mockCanvas = {
            getContext: vi.fn(() => mockCtx),
            width: 0,
            height: 0,
        };
    });

    it('should fill pixelData for arctic colormap (Aero/Ocean regression check)', () => {
        const proxy = createMockProxy();
        CanvasAdapterNeo.render(proxy, mockCanvas as any, { faceIndex: 0, colormap: 'arctic' });

        const imageData = mockCtx.putImageData.mock.calls[0][0];
        const pixelData = new Uint32Array(imageData.data.buffer);
        
        // Verify that at least some pixels are not zero (transparent/black)
        let nonZero = 0;
        for(let i=0; i<pixelData.length; i++) if(pixelData[i] !== 0) nonZero++;
        expect(nonZero).toBeGreaterThan(0);
    });

    it('should fill pixelData for heatmap colormap even without criteria (Heat fix check)', () => {
        const proxy = createMockProxy();
        // Currently this is expected to FAIL (nonZero will be 0) because of the bug
        CanvasAdapterNeo.render(proxy, mockCanvas as any, { faceIndex: 0, colormap: 'heatmap' });

        const imageData = mockCtx.putImageData.mock.calls[0][0];
        const pixelData = new Uint32Array(imageData.data.buffer);
        
        let nonZero = 0;
        for(let i=0; i<pixelData.length; i++) if(pixelData[i] !== 0) nonZero++;
        
        // BUG REPRODUCTION: Currently this fails because of the 'continue' in CanvasAdapterNeo.ts
        expect(nonZero).toBeGreaterThan(0);
    });
});
