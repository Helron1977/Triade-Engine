/**
 * DebugBridge utility for Hypercube Neo.
 * Centralizes observability by exposing the engine state to window.__HYPERCUBE__.
 */
export class DebugBridge {
    /**
     * Set up the bridge for a specific engine and config.
     */
    public static setup(engine: any, config: any) {
        (window as any).__HYPERCUBE__ = {
            engine,
            config,
            // Core stats for the agent
            getStats: () => ({
                tick: engine.parityManager.currentTick,
                mode: config.mode,
                dimensions: config.dimensions,
                fps: (window as any).BenchmarkHUD?.instances?.[0]?.fps || 0
            }),
            // Direct access to physics data
            getFaceValue: async (faceName: string, gx: number, gy: number) => {
                const bridge = (engine as any).bridge;
                if (config.mode === 'gpu') await bridge.syncToHost();
                const faceIdx = engine.getFaceLogicalIndex(faceName);
                
                const vGrid = (engine as any).vGrid;
                const chunkX = Math.floor(gx / (vGrid.dimensions.nx / vGrid.chunkLayout.x));
                const chunkY = Math.floor(gy / (vGrid.dimensions.ny / vGrid.chunkLayout.y));
                const chunk = vGrid.chunks.find((c: any) => c.x === chunkX && c.y === chunkY);
                if (!chunk) return null;

                const views = bridge.getChunkViews(chunk.id);
                const pNx = Math.floor(vGrid.dimensions.nx / vGrid.chunkLayout.x) + 2;
                const lx = (gx % (vGrid.dimensions.nx / vGrid.chunkLayout.x)) + 1;
                const ly = (gy % (vGrid.dimensions.ny / vGrid.chunkLayout.y)) + 1;
                return views[faceIdx][ly * pNx + lx];
            },
            // Diagnostic tools for the AI agent
            diagnose: () => {
                const hubButton = document.querySelector('.hub-button') as HTMLElement;
                const canvas = document.querySelector('canvas');
                const hud = (window as any).BenchmarkHUD?.instances?.[0];

                const hubStyles = hubButton ? window.getComputedStyle(hubButton) : null;

                return {
                    hubButton: hubButton ? {
                        exists: true,
                        visible: hubButton.getBoundingClientRect().height > 0,
                        text: hubButton.textContent,
                        rect: hubButton.getBoundingClientRect(),
                        styles: {
                            position: hubStyles?.position,
                            top: hubStyles?.top,
                            right: hubStyles?.right,
                            backgroundColor: hubStyles?.backgroundColor,
                            padding: hubStyles?.padding,
                            zIndex: hubStyles?.zIndex
                        }
                    } : { exists: false },
                    canvas: canvas ? {
                        width: canvas.width,
                        height: canvas.height,
                        rect: canvas.getBoundingClientRect()
                    } : null,
                    simulation: {
                        nx: config.dimensions.nx,
                        ny: config.dimensions.ny,
                        nz: config.dimensions.nz || 1,
                        mode: config.mode,
                        execution: config.executionMode
                    },
                    performance: {
                        fps: hud?.fps || 0,
                        computeMs: hud?.computeMs || 0,
                        tick: engine.parityManager.currentTick
                    },
                    engineInitialized: !!engine
                };
            }
        };
        console.log("Hypercube: DebugBridge established 🌉");
    }
}
