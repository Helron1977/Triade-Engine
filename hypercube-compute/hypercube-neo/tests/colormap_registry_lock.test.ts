import { describe, it, expect, vi, beforeEach } from 'vitest';
import { CanvasAdapterNeo } from '../io/CanvasAdapterNeo';
import { NeoEngineProxy } from '../core/NeoEngineProxy';

// Helper to create a precise mock proxy for a 1x1 test grid
function createPrecisionMock(value: number, vortValue: number = 0) {
    const data = new Float32Array(3 * 3).fill(value); // 1x1 + padding=1
    const vort = new Float32Array(3 * 3).fill(vortValue);
    return {
        vGrid: {
            dimensions: { nx: 1, ny: 1, nz: 1 },
            chunkLayout: { x: 1, y: 1 },
            chunks: [{ id: 'c0', x: 0, y: 0, localDimensions: { nx: 1, ny: 1 } }],
            dataContract: {
                descriptor: {
                    faces: [
                        { name: 'f1', type: 'scalar' },
                        { name: 'vorticity', type: 'scalar' }
                    ],
                    requirements: { ghostCells: 1 }
                }
            }
        },
        bridge: {
            getChunkViews: () => [data, vort]
        },
        parityManager: {
            getFaceIndices: (name: string) => ({ 
                read: name === 'vorticity' ? 1 : 0, 
                write: name === 'vorticity' ? 1 : 0 
            })
        }
    } as unknown as NeoEngineProxy;
}

describe('Colormap Logic Lock (Pre-Refacto)', () => {
    let mockCanvas: any;
    let mockCtx: any;

    beforeEach(() => {
        mockCtx = {
            createImageData: vi.fn(() => ({ data: new Uint8ClampedArray(4) })),
            putImageData: vi.fn(),
        };
        mockCanvas = { getContext: vi.fn(() => mockCtx), width: 0, height: 0 };
    });

    it('locks grayscale output for value 0.5', () => {
        const proxy = createPrecisionMock(0.5);
        CanvasAdapterNeo.render(proxy, mockCanvas, { faceIndex: 0, colormap: 'grayscale' });
        const pixelData = new Uint32Array(mockCtx.putImageData.mock.calls[0][0].data.buffer);
        // gray = 0.5 * 255 = 127
        // ABGR: 0xFF7F7F7F => 4286545791
        expect(pixelData[0]).toBe(0xFF7F7F7F);
    });

    it('locks arctic output for value 0.0 (Baseline Blue)', () => {
        const proxy = createPrecisionMock(0.0);
        CanvasAdapterNeo.render(proxy, mockCanvas, { faceIndex: 0, colormap: 'arctic' });
        const pixelData = new Uint32Array(mockCtx.putImageData.mock.calls[0][0].data.buffer);
        // Base color: r=180, g=220, b=255. ABGR: 0xFFFFDCB4 => 4294958260
        expect(pixelData[0]).toBe(0xFFFFDCB4);
    });

    it('locks arctic output for value 0.0 and vorticity 1.0 (Red Highlight)', () => {
        const proxy = createPrecisionMock(0.0, 1.0);
        // Logical index 1 is vorticity in our mock
        CanvasAdapterNeo.render(proxy, mockCanvas, { faceIndex: 0, colormap: 'arctic', vorticityFace: 1 });
        const pixelData = new Uint32Array(mockCtx.putImageData.mock.calls[0][0].data.buffer);
        // Expecting pure Red: ABGR 0xFF0000FF => 4278190335
        expect(pixelData[0]).toBe(0xFF0000FF);
    });

    it('locks heatmap output for value 0.75 (Orange-ish)', () => {
        const proxy = createPrecisionMock(0.75);
        CanvasAdapterNeo.render(proxy, mockCanvas, { faceIndex: 0, colormap: 'heatmap' });
        const pixelData = new Uint32Array(mockCtx.putImageData.mock.calls[0][0].data.buffer);
        // s=0.75 > 0.5 => r=255, g=(0.75-0.5)*2*255=127, b=0.75*0.2*255=38
        // ABGR: 0xFF267FFF => 4281237503
        expect(pixelData[0]).toBe(0xFF267FFF);
    });
});
