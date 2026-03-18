import { HypercubeNeoFactory } from '../../core/HypercubeNeoFactory';
import { CanvasAdapterNeo } from '../../io/CanvasAdapterNeo';

declare const Chart: any;

const MOVIELENS_LABELS = {
    users: Array.from({length: 100}, (_, i) => `User ${100 + i}`),
    movies: ["Inception", "Titanic", "The Matrix", "Pulp Fiction", "Avatar", "Interstellar", "Gladiator", "The Godfather", "Star Wars", "The Dark Knight"],
    genres: ["Action", "Drama", "Sci-Fi", "Comedy", "Horror", "Thriller", "Romance", "Docs"]
};

let engine: any = null;
let chart: any = null;
let isRunning = false;
let currentIteration = 0;
const errorHistory: number[] = [];

async function initUI() {
    const btnRun = document.getElementById('btn-run') as HTMLButtonElement;
    const btnStop = document.getElementById('btn-stop') as HTMLButtonElement;
    const btnExport = document.getElementById('btn-export') as HTMLButtonElement;
    const btnLoadSample = document.getElementById('btn-load-sample') as HTMLButtonElement;
    const btnReset = document.getElementById('btn-reset') as HTMLButtonElement;
    const csvUpload = document.getElementById('csv-upload') as HTMLInputElement;
    const sliceSlider = document.getElementById('slice-slider') as HTMLInputElement;
    const sliceLabel = document.getElementById('slice-label');
    const sliceVal = document.getElementById('slice-val');

    btnRun.onclick = () => startDecomposition();
    btnStop.onclick = () => stopDecomposition();
    btnExport.onclick = () => exportFactors();
    btnLoadSample.onclick = () => loadSample();
    btnReset.onclick = () => resetFactors();
    
    csvUpload.onchange = async (e: any) => {
        const file = e.target.files[0];
        if (file) {
            const text = await file.text();
            handleCSV(text);
        }
    };

    sliceSlider.oninput = () => {
        const val = parseInt(sliceSlider.value);
        if (sliceLabel) sliceLabel.innerText = `Z-Slice: ${val}`;
        if (sliceVal) sliceVal.innerText = val.toString();
        renderFrame();
    };

    initChart();
}

function initChart() {
    const ctx = (document.getElementById('error-chart') as HTMLCanvasElement).getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Reconstruction Error (MSE)',
                data: [],
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: false, grid: { color: '#2d2d35' }, ticks: { color: '#94a3b8' } },
                x: { grid: { color: '#2d2d35' }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

async function loadSample() {
    try {
        const response = await fetch('../assets/sample-tensor.csv');
        const text = await response.text();
        handleCSV(text);
    } catch (e) {
        console.error("Failed to load sample CSV", e);
    }
}

async function handleCSV(text: string) {
    if (!engine) {
        await setupEngine();
    }
    
    const lines = text.trim().split('\n');
    const bridge = (engine as any).bridge;
    const chunkId = (engine as any).vGrid.chunks[0].id;
    const views = bridge.getChunkViews(chunkId);
    const target = views[3];
    const { nx, ny, nz } = (engine.vGrid as any).config.dimensions;

    target.fill(0);
    let count = 0;
    for (let i = 1; i < lines.length; i++) {
        const [u, m, g, val] = lines[i].split(',').map(Number);
        if (!isNaN(u) && u < nx && m < ny && g < nz) {
            const idx = u + m * nx + g * nx * ny;
            target[idx] = val;
            count++;
        }
    }

    const sparsity = (count / (nx * ny * nz) * 100).toFixed(2);
    document.getElementById('stats-summary')!.innerHTML = `
        <div style="display: flex; flex-direction: column; gap: 0.5rem;">
            <div><strong>Entries:</strong> ${count}</div>
            <div><strong>Dimensions:</strong> ${nx} x ${ny} x ${nz}</div>
            <div><strong>Theoretical Max:</strong> ${nx * ny * nz}</div>
            <div><strong>Sparsity:</strong> ${100 - parseFloat(sparsity)}%</div>
            <div><strong>Density:</strong> ${sparsity}%</div>
        </div>
    `;

    // Initialize Factors
    const rank = engine.vGrid.config.params.rank || 10;
    for (let f = 0; f < 3; f++) {
        const view = views[f];
        for (let i = 0; i < view.length; i++) {
            view[i] = 0.05 + Math.random() * 0.2;
        }
    }

    if (engine.vGrid.config.mode === 'gpu') {
        await bridge.syncToDevice();
    }

    renderFrame();
    updateFactorTables();
    console.log(`Loaded ${count} entries into tensor.`);
}

async function setupEngine() {
    const factory = new HypercubeNeoFactory();
    const urlParams = new URLSearchParams(window.location.search);
    const backendSelect = document.getElementById('backend-select') as HTMLSelectElement;
    const backend = urlParams.get('backend') || backendSelect.value;
    backendSelect.value = backend; 
    const rankEl = document.getElementById('param-rank') as HTMLInputElement;
    const rank = rankEl ? parseInt(urlParams.get('rank') || rankEl.value) : 10;
    if (rankEl) rankEl.value = rank.toString();

    const regEl = document.getElementById('param-reg') as HTMLInputElement;
    const reg = regEl ? parseFloat(regEl.value) : 0.05;

    const lrEl = document.getElementById('param-lr') as HTMLInputElement;
    const lr = lrEl ? parseFloat(lrEl.value) : 0.01;

    const manifest = await factory.fromManifest('./showcase-tensor-cp.json');
    manifest.config.mode = backend as any;
    manifest.config.params.rank = rank;
    manifest.config.params.regularization = reg;

    // Critical: Inject into the rule itself so dispatchers see it
    if (manifest.engine.rules[0]) {
        manifest.engine.rules[0].params = {
            rank,
            regularization: reg,
            learningRate: lr
        };
    }
    
    engine = await factory.build(manifest.config, manifest.engine);

    // Sync Details to UI
    const memMb = engine.bridge.rawBuffer ? (engine.bridge.rawBuffer.byteLength / 1024 / 1024).toFixed(1) : "0.0";
    document.getElementById('manifest-details')!.innerHTML = `
        Rank: ${rank}<br>
        Backend: ${backend.toUpperCase()}<br>
        Memory: ~${memMb} MB
    `;
    
    const canvases = ['canvas-original', 'canvas-recon', 'canvas-error'];
    const { nx, ny } = manifest.config.dimensions;
    canvases.forEach(id => {
        const c = document.getElementById(id) as HTMLCanvasElement;
        c.width = nx;
        c.height = ny;
        c.style.width = '600px';
        c.style.height = '600px';
    });

    const sliceSlider = document.getElementById('slice-slider') as HTMLInputElement;
    sliceSlider.max = (manifest.config.dimensions.nz - 1).toString();
}

async function resetFactors() {
    console.log("Attempting reset...");
    if (!engine) {
        console.log("Engine not ready, setting up...");
        await setupEngine();
    }
    const bridge = engine.bridge;
    const chunkId = engine.vGrid.chunks[0].id;
    const views = bridge.getChunkViews(chunkId);
    
    for (let f = 0; f < 3; f++) {
        const view = views[f];
        for (let i = 0; i < view.length; i++) {
            view[i] = 0.05 + Math.random() * 0.2;
        }
    }
    
    // Clear reconstruction to reset visuals
    if (views[4]) views[4].fill(0);
    
    if (engine.vGrid.config.mode === 'gpu') {
        await bridge.syncToDevice();
    }
    
    // UI Reset
    currentIteration = 0;
    errorHistory.length = 0;
    if (chart) {
        chart.data.labels = [];
        chart.data.datasets[0].data = [];
        chart.update();
    }
    document.getElementById('decomp-status')!.innerText = 'IDLE';
    document.getElementById('decomp-mse')!.innerText = 'N/A';
    
    renderFrame();
    updateFactorTables();
    console.log("Factors reset completed.");
}

async function startDecomposition() {
    console.log("Starting decomposition...");
    if (!engine) await setupEngine();
    
    // Safety check: Is there any data?
    const bridge = engine.bridge;
    const chunkId = engine.vGrid.chunks[0].id;
    const views = bridge.getChunkViews(chunkId);
    const target = views[3];
    let hasData = false;
    for (let i = 0; i < Math.min(target.length, 1000); i++) {
        if (target[i] > 0) { hasData = true; break; }
    }
    
    if (!hasData) {
        console.warn("No data in tensor.");
        alert("No data loaded in tensor. Please upload a CSV or load the sample first.");
        return;
    }

    isRunning = true;
    currentIteration = 0;
    errorHistory.length = 0;
    chart.data.labels = [];
    chart.data.datasets[0].data = [];
    chart.update();
    document.getElementById('decomp-status')!.innerText = 'RUNNING';
    document.getElementById('decomp-status')!.style.color = '#6366f1';
    console.log("Entering simulation loop...");
    loop();
}

function stopDecomposition() {
    isRunning = false;
    document.getElementById('decomp-status')!.innerText = 'STOPPED';
    document.getElementById('decomp-status')!.style.color = '#ef4444';
}

async function loop() {
    if (!isRunning || !engine) return;

    const maxIter = parseInt((document.getElementById('param-iter') as HTMLInputElement).value);
    const start = performance.now();
    
    try {
        await engine.step(currentIteration);
        const ms = performance.now() - start;

        if (engine.vGrid.config.mode === 'gpu') {
            await engine.bridge.syncToHost();
        }

        const error = calculateMSE();
        errorHistory.push(error);
        
        if (currentIteration % 10 === 0) {
            console.log(`Iteration ${currentIteration}/${maxIter}, MSE: ${error.toFixed(6)}, Time: ${ms.toFixed(2)}ms`);
        }

        updateHUD(ms, error);
        updateChart(error);
        if (currentIteration % 2 === 0) updateFactorTables();
        renderFrame();

        currentIteration++;
        if (currentIteration < maxIter && isRunning) {
            requestAnimationFrame(loop);
        } else {
            console.log("Decomposition finished.");
            isRunning = false;
            document.getElementById('decomp-status')!.innerText = currentIteration >= maxIter ? 'COMPLETED' : 'STOPPED';
            document.getElementById('decomp-status')!.style.color = currentIteration >= maxIter ? '#22c55e' : '#ef4444';
            document.getElementById('decomp-mse')!.innerText = error.toFixed(6);
            document.getElementById('decomp-converged')!.innerText = error < 0.001 ? 'Yes' : 'No';
            
            updateHUD(ms, error);
            updateFactorTables();
            renderFrame();
        }
    } catch (e) {
        console.error("Error in decomposition loop:", e);
        isRunning = false;
    }
}

function calculateMSE(): number {
    if (!engine) return 0;
    const bridge = engine.bridge;
    const chunkId = engine.vGrid.chunks[0].id;
    const views = bridge.getChunkViews(chunkId);
    const target = views[3];
    const recon = views[4];
    
    let sumErr = 0;
    let count = 0;
    for (let i = 0; i < target.length; i++) {
        if (target[i] > 0) {
            const diff = target[i] - recon[i];
            sumErr += diff * diff;
            count++;
        }
    }
    return count > 0 ? sumErr / count : 0;
}

function updateHUD(ms: number, error: number) {
    const maxIter = (document.getElementById('param-iter') as HTMLInputElement).value;
    document.getElementById('hud-compute')!.innerText = `${ms.toFixed(2)} ms`;
    document.getElementById('hud-iter')!.innerText = `${currentIteration}/${maxIter}`;
    document.getElementById('hud-error')!.innerText = error.toFixed(4);
}

function updateChart(error: number) {
    chart.data.labels.push(currentIteration);
    chart.data.datasets[0].data.push(error);
    if (chart.data.labels.length > 50) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }
    chart.update('none');
}

function renderFrame() {
    if (!engine) return;
    const sliceZ = parseInt((document.getElementById('slice-slider') as HTMLInputElement).value);
    console.log(`Rendering frame, Z-Slice: ${sliceZ}`);
    
    // 1. Original
    CanvasAdapterNeo.render(engine as any, document.getElementById('canvas-original') as HTMLCanvasElement, {
        faceIndex: 3, // Target
        colormap: 'magma',
        minVal: 0,
        maxVal: 5,
        sliceZ: sliceZ
    });

    // 2. Reconstruction
    CanvasAdapterNeo.render(engine as any, document.getElementById('canvas-recon') as HTMLCanvasElement, {
        faceIndex: 4, // Reconstruction
        colormap: 'magma',
        minVal: 0,
        maxVal: 5,
        sliceZ: sliceZ
    });

    // 3. Error (Residuals) - Match target scale to see initial residuals clearly
    CanvasAdapterNeo.render(engine as any, document.getElementById('canvas-error') as HTMLCanvasElement, {
        faceIndex: 3, // Source (Target)
        reconFaceIndex: 4, // Reconstruction to compare against
        colormap: 'inferno', // Different colormap for error
        minVal: 0,
        maxVal: 1.5, // Better sensitivity for residuals
        sliceZ: sliceZ
    });

    updateSliceMetrics(sliceZ);
}

function updateSliceMetrics(sliceZ: number) {
    if (!engine) return;
    const bridge = engine.bridge;
    const chunkId = engine.vGrid.chunks[0].id;
    const views = bridge.getChunkViews(chunkId);
    const target = views[3];
    const recon = views[4];
    const { nx, ny } = (engine.vGrid as any).config.dimensions;
    const sliceSize = nx * ny;
    const offset = sliceZ * sliceSize;

    let sliceSumSqErr = 0;
    let sliceSumAbsErr = 0;
    let sliceActiveCount = 0;

    for (let i = 0; i < sliceSize; i++) {
        const idx = offset + i;
        if (target[idx] > 0) {
            const diff = target[idx] - recon[idx];
            sliceSumSqErr += diff * diff;
            sliceSumAbsErr += Math.abs(diff);
            sliceActiveCount++;
        }
    }

    const sliceMSE = sliceActiveCount > 0 ? sliceSumSqErr / sliceActiveCount : 0;
    const sliceMAE = sliceActiveCount > 0 ? sliceSumAbsErr / sliceActiveCount : 0;
    const sliceDensity = (sliceActiveCount / sliceSize) * 100;

    document.getElementById('mse-original')!.innerText = `Active Density: ${sliceDensity.toFixed(2)}%`;
    document.getElementById('mse-recon')!.innerText = `Slice MSE: ${sliceMSE.toFixed(6)}`;
    document.getElementById('mse-error')!.innerText = `Mean Absolute Error: ${sliceMAE.toFixed(6)}`;
}

function updateFactorTables() {
    if (!engine) return;
    const bridge = engine.bridge;
    const chunkId = engine.vGrid.chunks[0].id;
    const views = bridge.getChunkViews(chunkId);
    const rank = engine.vGrid.config.params.rank || 10;

    const renderTable = (view: Float32Array, id: string, name: string, labels: string[]) => {
        let html = `<table><thead><tr><th>${name}</th>`;
        for (let r = 0; r < rank; r++) html += `<th>R${r}</th>`;
        html += `</tr></thead><tbody>`;
        
        const maxRows = 15;
        const nRows = view.length / rank;
        for (let i = 0; i < Math.min(nRows, maxRows); i++) {
            const label = labels[i] || `${name} #${i}`;
            html += `<tr><td style="font-weight: 600; color: #fff;">${label}</td>`;
            for (let r = 0; r < rank; r++) {
                const val = view[i * rank + r];
                const cls = val > 0.01 ? 'val-pos' : (val < -0.01 ? 'val-neg' : '');
                html += `<td class="${cls}">${val.toFixed(3)}</td>`;
            }
            html += `</tr>`;
        }
        html += `</tbody></table>`;
        document.getElementById(id)!.innerHTML = html;
    };

    renderTable(views[0], 'table-a', 'User', MOVIELENS_LABELS.users);
    renderTable(views[1], 'table-b', 'Movie', MOVIELENS_LABELS.movies);
    renderTable(views[2], 'table-c', 'Genre', MOVIELENS_LABELS.genres);
}

function exportFactors() {
    if (!engine) return;
    const bridge = engine.bridge;
    const chunkId = engine.vGrid.chunks[0].id;
    const views = bridge.getChunkViews(chunkId);
    
    const data = {
        mode_a: Array.from(views[0]),
        mode_b: Array.from(views[1]),
        mode_c: Array.from(views[2])
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'tensor_factors.json';
    a.click();
}

initUI();
loadSample();
