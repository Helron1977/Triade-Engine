export class BenchmarkHUD {
    private container: HTMLDivElement;
    private computeMsElem: HTMLSpanElement;
    private fpsElem: HTMLSpanElement;
    private resolutionElem: HTMLSpanElement;
    private engineElem: HTMLSpanElement;

    private frames: number = 0;
    private lastTime: number = performance.now();

    constructor(engineName: string, resolution: string) {
        this.container = document.createElement('div');
        this.container.style.position = 'fixed';
        this.container.style.top = '10px';
        this.container.style.left = '10px';
        this.container.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
        this.container.style.color = '#00ffcc';
        this.container.style.padding = '15px';
        this.container.style.fontFamily = 'monospace';
        this.container.style.fontSize = '14px';
        this.container.style.borderRadius = '8px';
        this.container.style.zIndex = '9999';
        this.container.style.pointerEvents = 'none';

        this.container.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="font-weight: bold; color: #fff;">HYPERCUBE V5.3</div>
                <button id="hud-mode-switch" style="padding: 2px 8px; font-size: 10px; cursor: pointer; border: 1px solid #00ffcc; background: rgba(0,255,204,0.1); color: #00ffcc; font-weight: bold; border-radius: 4px;"></button>
            </div>
            <div style="margin-bottom: 5px;">Engine: <span id="hud-engine" style="color: #fff;">${engineName}</span></div>
            <div style="margin-bottom: 5px;">Resolution: <span id="hud-res" style="color: #fff;">${resolution}</span></div>
            <div style="margin-bottom: 5px;">Compute: <span id="hud-ms" style="color: #ff3366;">0.00</span> ms</div>
            <div>FPS: <span id="hud-fps" style="color: #33ff33;">0</span></div>
        `;

        // Interactive HUD needs pointerEvents re-enabled, but prevent default interfering everywhere
        this.container.style.pointerEvents = 'auto';
        document.body.appendChild(this.container);

        this.engineElem = this.container.querySelector('#hud-engine') as HTMLSpanElement;
        this.resolutionElem = this.container.querySelector('#hud-res') as HTMLSpanElement;
        this.computeMsElem = this.container.querySelector('#hud-ms') as HTMLSpanElement;
        this.fpsElem = this.container.querySelector('#hud-fps') as HTMLSpanElement;

        // Add Mode Switch Logic
        const urlParams = new URLSearchParams(window.location.search);
        let mode = urlParams.get('mode') || 'auto';
        const swBtn = this.container.querySelector('#hud-mode-switch') as HTMLButtonElement;

        // Display nice label
        swBtn.innerText = mode.toUpperCase() + (mode === 'auto' ? ' (M)' : '');

        swBtn.addEventListener('click', () => {
            const nextMode = mode === 'gpu' ? 'cpu' : 'gpu';
            window.location.search = `?mode=${nextMode}`;
        });
    }

    public updateCompute(ms: number) {
        // Lissage exponentiel pour la lisibilité
        const current = parseFloat(this.computeMsElem.innerText);
        const smooth = current === 0 ? ms : current * 0.9 + ms * 0.1;
        this.computeMsElem.innerText = smooth.toFixed(2);

        if (smooth > 16.6) this.computeMsElem.style.color = '#ff3366'; // Dropping frames
        else if (smooth > 8) this.computeMsElem.style.color = '#ffaa33'; // Warning
        else this.computeMsElem.style.color = '#33ff33'; // Good
    }

    public tickFrame() {
        this.frames++;
        const now = performance.now();
        if (now - this.lastTime >= 1000) {
            this.fpsElem.innerText = this.frames.toString();

            if (this.frames >= 58) this.fpsElem.style.color = '#33ff33';
            else if (this.frames >= 30) this.fpsElem.style.color = '#ffaa33';
            else this.fpsElem.style.color = '#ff3366';

            this.frames = 0;
            this.lastTime = now;
        }
    }
}
