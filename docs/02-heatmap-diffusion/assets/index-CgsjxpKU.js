var B=Object.defineProperty;var M=(t,e,i)=>e in t?B(t,e,{enumerable:!0,configurable:!0,writable:!0,value:i}):t[e]=i;var g=(t,e,i)=>M(t,typeof e!="symbol"?e+"":e,i);(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const a of document.querySelectorAll('link[rel="modulepreload"]'))u(a);new MutationObserver(a=>{for(const s of a)if(s.type==="childList")for(const r of s.addedNodes)r.tagName==="LINK"&&r.rel==="modulepreload"&&u(r)}).observe(document,{childList:!0,subtree:!0});function i(a){const s={};return a.integrity&&(s.integrity=a.integrity),a.referrerPolicy&&(s.referrerPolicy=a.referrerPolicy),a.crossOrigin==="use-credentials"?s.credentials="include":a.crossOrigin==="anonymous"?s.credentials="omit":s.credentials="same-origin",s}function u(a){if(a.ep)return;a.ep=!0;const s=i(a);fetch(a.href,s)}})();const E="modulepreload",G=function(t){return"/"+t},v={},z=function(e,i,u){let a=Promise.resolve();if(i&&i.length>0){document.getElementsByTagName("link");const r=document.querySelector("meta[property=csp-nonce]"),p=(r==null?void 0:r.nonce)||(r==null?void 0:r.getAttribute("nonce"));a=Promise.allSettled(i.map(f=>{if(f=G(f),f in v)return;v[f]=!0;const n=f.endsWith(".css"),h=n?'[rel="stylesheet"]':"";if(document.querySelector(`link[href="${f}"]${h}`))return;const c=document.createElement("link");if(c.rel=n?"stylesheet":E,n||(c.as="script"),c.crossOrigin="",c.href=f,p&&c.setAttribute("nonce",p),document.head.appendChild(c),n)return new Promise((o,l)=>{c.addEventListener("load",o),c.addEventListener("error",()=>l(new Error(`Unable to preload CSS for ${f}`)))})}))}function s(r){const p=new Event("vite:preloadError",{cancelable:!0});if(p.payload=r,window.dispatchEvent(p),!p.defaultPrevented)throw r}return a.then(r=>{for(const p of r||[])p.status==="rejected"&&s(p.reason);return e().catch(s)})};var _,S=(_=class{static get isSupported(){var t,e;return this._isSupported===null&&(this._isSupported=typeof navigator<"u"&&"gpu"in navigator,this.isSupported&&(this._preferredFormat=((e=(t=navigator.gpu).getPreferredCanvasFormat)==null?void 0:e.call(t))??"bgra8unorm")),this._isSupported}static get device(){if(!this._device)throw new Error("[Hypercube GPU] GPUDevice non initialisé. Appelez init() d'abord.");return this._device}static get preferredFormat(){return this._preferredFormat}static async init(t={}){if(!this.isSupported)return console.warn("[Hypercube GPU] WebGPU non supporté."),!1;if(this._device)return!0;try{if(this._adapter=await navigator.gpu.requestAdapter(t),!this._adapter)return console.warn("[Hypercube GPU] Aucun adaptateur trouvé."),!1;const e=[],i={maxComputeInvocationsPerWorkgroup:this._adapter.limits.maxComputeInvocationsPerWorkgroup,maxComputeWorkgroupSizeX:this._adapter.limits.maxComputeWorkgroupSizeX,maxComputeWorkgroupSizeY:this._adapter.limits.maxComputeWorkgroupSizeY,maxComputeWorkgroupSizeZ:this._adapter.limits.maxComputeWorkgroupSizeZ};return this._device=await this._adapter.requestDevice({requiredFeatures:e,requiredLimits:i}),this._device.lost.then(u=>{console.error(`[Hypercube GPU] Device perdu: ${u.message} (${u.reason})`),this._device=null,this._adapter=null}),!0}catch(e){return console.error("[Hypercube GPU] Init échouée:",e),!1}}static createStorageBuffer(t,e=!0){const i=this.device.createBuffer({size:t.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,mappedAtCreation:e});return e?(new Float32Array(i.getMappedRange()).set(t),i.unmap()):this.device.queue.writeBuffer(i,0,t),i}static createUniformBuffer(t){const e=this.device.createBuffer({size:Math.ceil(t.byteLength/16)*16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),i=("buffer"in t,t);return this.device.queue.writeBuffer(e,0,i),e}static createComputePipeline(t,e){const i=this.device.createShaderModule({code:t});return this.device.createComputePipeline({layout:"auto",compute:{module:i,entryPoint:e}})}static destroy(){this._device&&(this._device.destroy(),this._device=null),this._adapter=null}},g(_,"_device",null),g(_,"_adapter",null),g(_,"_isSupported",null),g(_,"_preferredFormat","bgra8unorm"),_),A=class{constructor(t,e,i,u,a=1,s,r=6,p=0){g(this,"nx");g(this,"ny");g(this,"nz");g(this,"faces",[]);g(this,"gpuBuffer",null);g(this,"offset");g(this,"stride");g(this,"engine",null);g(this,"x");g(this,"y");g(this,"z");g(this,"masterBuffer");this.x=t,this.y=e,this.z=p,this.masterBuffer=s,this.nx=i,this.ny=u,this.nz=a;const f=s.allocateCube(i,u,a,r);this.offset=f.offset,this.stride=f.stride;const n=i*u*a;for(let h=0;h<r;h++)this.faces.push(new Float32Array(s.buffer,this.offset+h*this.stride,n))}getIndex(t,e,i=0){return i*this.ny*this.nx+e*this.nx+t}getSlice(t,e){const i=this.nx*this.ny,u=e*i;return this.faces[t].slice(u,u+i)}setEngine(t){this.engine=t}initGPU(){if(!this.engine)return;const t=this.faces.length*this.stride;this.gpuBuffer=S.device.createBuffer({size:t,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC}),S.device.queue.writeBuffer(this.gpuBuffer,0,this.masterBuffer.buffer,this.offset,t),this.engine.initGPU&&this.engine.initGPU(S.device,this.gpuBuffer,this.stride,this.nx,this.ny,this.nz)}async compute(){this.engine&&await this.engine.compute(this.faces,this.nx,this.ny,this.nz,this.x,this.y,this.z)}clearFace(t){this.faces[t].fill(0)}async syncToHost(){if(!this.gpuBuffer)return;const t=this.faces.length*this.stride,e=S.device,i=e.createBuffer({size:t,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),u=e.createCommandEncoder();u.copyBufferToBuffer(this.gpuBuffer,0,i,0,t),e.queue.submit([u.finish()]),await i.mapAsync(GPUMapMode.READ);const a=i.getMappedRange();new Uint8Array(this.masterBuffer.buffer,this.offset,t).set(new Uint8Array(a)),i.unmap(),i.destroy()}syncFromHost(){if(!this.gpuBuffer)return;const t=this.faces.length*this.stride;S.device.queue.writeBuffer(this.gpuBuffer,0,this.masterBuffer.buffer,this.offset,t)}},I=class{constructor(t=10,e=1){g(this,"radius");g(this,"weight");g(this,"pipelineHorizontal",null);g(this,"pipelineVertical",null);g(this,"pipelineDiffusion",null);g(this,"bindGroup",null);this.radius=t,this.weight=e}get name(){return"Heatmap (O1 Spatial Convolution)"}getRequiredFaces(){return 5}initGPU(t,e,i,u,a,s){const r=t.createShaderModule({code:this.wgslSource}),p=t.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),f=t.createPipelineLayout({bindGroupLayouts:[p]});this.pipelineHorizontal=t.createComputePipeline({layout:f,compute:{module:r,entryPoint:"compute_sat_horizontal"}}),this.pipelineVertical=t.createComputePipeline({layout:f,compute:{module:r,entryPoint:"compute_sat_vertical"}}),this.pipelineDiffusion=t.createComputePipeline({layout:f,compute:{module:r,entryPoint:"compute_diffusion"}});const n=t.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),h=i/4,c=new ArrayBuffer(32),o=new Uint32Array(c),l=new Int32Array(c),m=new Float32Array(c);o[0]=u,o[1]=a,o[2]=s,l[3]=this.radius,m[4]=this.weight,o[5]=h,t.queue.writeBuffer(n,0,c),this.bindGroup=t.createBindGroup({layout:p,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:n}}]})}computeGPU(t,e,i,u,a){if(!this.bindGroup)return;let s;this.pipelineHorizontal&&(s=e.beginComputePass(),s.setBindGroup(0,this.bindGroup),s.setPipeline(this.pipelineHorizontal),s.dispatchWorkgroups(1,u,a),s.end(),t.queue.submit([e.finish()]),e=t.createCommandEncoder()),this.pipelineVertical&&(s=e.beginComputePass(),s.setBindGroup(0,this.bindGroup),s.setPipeline(this.pipelineVertical),s.dispatchWorkgroups(i,1,a),s.end(),t.queue.submit([e.finish()]),e=t.createCommandEncoder()),this.pipelineDiffusion&&(s=e.beginComputePass(),s.setBindGroup(0,this.bindGroup),s.setPipeline(this.pipelineDiffusion),s.dispatchWorkgroups(Math.ceil(i/16),Math.ceil(u/16),a),s.end())}compute(t,e,i,u){const a=t[0],s=t[2],r=t[4];r&&r.fill(0);for(let p=0;p<u;p++){const f=p*i*e;for(let n=0;n<i;n++)for(let h=0;h<e;h++){const c=f+n*e+h,o=a[c],l=n>0?r[f+(n-1)*e+h]:0,m=h>0?r[f+n*e+(h-1)]:0,d=n>0&&h>0?r[f+(n-1)*e+(h-1)]:0;r[c]=o+l+m-d}for(let n=0;n<i;n++)for(let h=0;h<e;h++){const c=Math.max(0,h-this.radius),o=Math.max(0,n-this.radius),l=Math.min(e-1,h+this.radius),m=Math.min(i-1,n+this.radius),d=c>0&&o>0?r[f+(o-1)*e+(c-1)]:0,y=o>0?r[f+(o-1)*e+l]:0,b=c>0?r[f+m*e+(c-1)]:0,x=r[f+m*e+l]-y-b+d;s[f+n*e+h]=x*this.weight}}}get wgslSource(){return`
            struct Uniforms {
                mapSize: u32,
                radius: i32,
                weight: f32,
                stride: u32,
            };

            @group(0) @binding(0) var<storage, read_write> cube: array<f32>;
            @group(0) @binding(1) var<uniform> config: Uniforms;

            // Face 0: Input, Face 4: SAT, Face 2: Output
            const FACE_IN = 0u;
            const FACE_SAT = 4u;
            const FACE_OUT = 2u;

            // --- PASS 1: Prefix Sum Horizontal (par ligne, parallèle avec shared mem) ---
            @compute @workgroup_size(256)
            fn compute_sat_horizontal(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
                let y = global_id.y;
                if (y >= config.mapSize) { return; }

                let base_in = FACE_IN * config.stride;
                let base_sat = FACE_SAT * config.stride;
                let mapSize = config.mapSize;

                var shared: array<f32, 256>;  // Shared memory pour workgroup
                let lid = local_id.x;

                // Charge initiale (un thread par colonne dans la ligne)
                let x = global_id.x;
                if (x < mapSize) {
                    shared[lid] = cube[base_in + y * mapSize + x];
                } else {
                    shared[lid] = 0.0;
                }
                workgroupBarrier();

                // Hillis-Steele scan parallèle (O(log N) steps)
                var offset: u32 = 1u;
                while (offset < 256u) {
                    if (lid >= offset) {
                        shared[lid] += shared[lid - offset];
                    }
                    workgroupBarrier();
                    offset *= 2u;
                }

                // Écriture finale dans SAT (seulement si x < mapSize)
                if (x < mapSize) {
                    cube[base_sat + y * mapSize + x] = shared[lid];
                }
            }

            // --- PASS 2: Prefix Sum Vertical (par colonne, parallèle avec shared mem) ---
            @compute @workgroup_size(256)
            fn compute_sat_vertical(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
                let x = global_id.x;
                if (x >= config.mapSize) { return; }

                let base_sat = FACE_SAT * config.stride;
                let mapSize = config.mapSize;

                var shared: array<f32, 256>;
                let lid = local_id.y;  // Note : on swap sur y pour colonnes

                let y = global_id.y;
                if (y < mapSize) {
                    shared[lid] = cube[base_sat + y * mapSize + x];
                } else {
                    shared[lid] = 0.0;
                }
                workgroupBarrier();

                // Hillis-Steele scan parallèle
                var offset: u32 = 1u;
                while (offset < 256u) {
                    if (lid >= offset) {
                        shared[lid] += shared[lid - offset];
                    }
                    workgroupBarrier();
                    offset *= 2u;
                }

                if (y < mapSize) {
                    cube[base_sat + y * mapSize + x] = shared[lid];
                }
            }

            // --- PASS 3: Extraction Box Filter (Diffusion) ---
            @compute @workgroup_size(16, 16)
            fn compute_diffusion(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let x = i32(global_id.x);
                let y = i32(global_id.y);
                let mapSize = i32(config.mapSize);
                let stride = config.stride;

                if (x >= mapSize || y >= mapSize) { return; }

                let min_x = max(0, x - config.radius);
                let min_y = max(0, y - config.radius);
                let max_x = min(mapSize - 1, x + config.radius);
                let max_y = min(mapSize - 1, y + config.radius);

                var A: f32 = 0.0;
                var B: f32 = 0.0;
                var C: f32 = 0.0;
                
                let base = FACE_SAT * stride;
                if (min_x > 0 && min_y > 0) { A = cube[base + u32((min_y - 1) * mapSize + (min_x - 1))]; }
                if (min_y > 0) { B = cube[base + u32((min_y - 1) * mapSize + max_x)]; }
                if (min_x > 0) { C = cube[base + u32(max_y * mapSize + (min_x - 1))]; }
                
                let D: f32 = cube[base + u32(max_y * mapSize + max_x)];

                let sum = D - B - C + A;
                cube[FACE_OUT * stride + u32(y * mapSize + x)] = sum * config.weight;
            }
        `}},O=class{constructor(t=100*1024*1024){g(this,"buffer");g(this,"offset",0);typeof SharedArrayBuffer<"u"?this.buffer=new SharedArrayBuffer(t):(console.warn("[HypercubeMasterBuffer] SharedArrayBuffer n'est pas supporté (vérifiez vos headers COOP/COEP). Fallback sur ArrayBuffer (pas de multi-threading CPU possible)."),this.buffer=new ArrayBuffer(t))}allocateCube(t,e,i=1,u=6){this.offset=Math.ceil(this.offset/256)*256;const s=this.offset,p=t*e*i*4,f=Math.ceil(p/256)*256,n=f*u;if(this.offset+n>this.buffer.byteLength)throw new Error(`[HypercubeMasterBuffer] Out Of Memory. Impossible d'allouer ${n} bytes supplémentaires (3D: ${t}x${e}x${i}).`);return this.offset+=n,{offset:s,stride:f}}getUsedMemoryInMB(){return(this.offset/(1024*1024)).toFixed(2)+" MB"}},k=class{constructor(t){g(this,"workers",[]);g(this,"maxThreads");this.maxThreads=t??(typeof navigator<"u"&&navigator.hardwareConcurrency||4)}async init(t="./cpu.worker.js"){return new Promise(e=>{let i=0;const u=()=>{i++,i===this.maxThreads&&e()};for(let a=0;a<this.maxThreads;a++){const s=new Worker(t,{type:"module"});s.onmessage=r=>{r.data.type},this.workers.push(s),u()}})}async computeAll(t,e,i){if(this.workers.length===0){console.warn("[HypercubeWorkerPool] Pool vide, fallback sur l'éxecution séquentielle (Main Thread).");for(const u of t)u.compute();return}return new Promise(u=>{let a=0;const s=t.length;if(s===0)return u();let r=0,p=0;const f=n=>{const h=this.workers[n];if(a===s){p===0&&u();return}if(r<s){const c=t[r];r++,p++,h.onmessage=o=>{o.data.type==="DONE"&&(a++,p--,f(n))},h.postMessage({type:"COMPUTE",cubeOffset:c.offset,stride:c.stride||c.nx*c.ny*c.nz*4,numFaces:c.faces.length,nx:c.nx,ny:c.ny,nz:c.nz,engineName:i.name,engineConfig:i.config,sharedBuffer:e,chunkX:c.x,chunkY:c.y})}};for(let n=0;n<this.maxThreads&&n<s;n++)f(n)})}terminate(){for(const t of this.workers)t.terminate();this.workers=[]}},F=class U{constructor(e,i,u,a,s,r=6,p=!0,f="cpu"){g(this,"cubes",[]);g(this,"cols");g(this,"rows");g(this,"nx");g(this,"ny");g(this,"nz");g(this,"isPeriodic");g(this,"mode");g(this,"masterBuffer");g(this,"_engineFactory");g(this,"workerPool",null);this.cols=e,this.rows=i,typeof u=="number"?(this.nx=u,this.ny=u,this.nz=1):(this.nx=u.nx,this.ny=u.ny,this.nz=u.nz??1),this.masterBuffer=a,this.isPeriodic=p,this.mode=f,this._engineFactory=s;const n=s(),h=n.getRequiredFaces(),c=Math.max(r,h);for(let o=0;o<i;o++){this.cubes[o]=[];for(let l=0;l<e;l++){const m=new A(l,o,this.nx,this.ny,this.nz,a,c);m.setEngine(o===0&&l===0?n:s()),this.cubes[o][l]=m}}}static async create(e,i,u,a,s,r=6,p=!0,f="cpu",n=!0){var c;f==="webgpu"&&(await(await z(async()=>{const{HypercubeGPUContext:m}=await Promise.resolve().then(()=>C);return{HypercubeGPUContext:m}},void 0)).HypercubeGPUContext.init()?console.info("[HypercubeGrid] Initialisation asynchrone du contexte WebGPU : Succès."):(console.warn("[HypercubeGrid] WebGPU init n'a pas réussi. Fallback implicite vers le mode 'cpu'."),f="cpu"));const h=new U(e,i,u,a,s,r,p,f);if(f==="webgpu")for(let o=0;o<i;o++)for(let l=0;l<e;l++)(c=h.cubes[o][l])==null||c.initGPU();else if(f==="cpu"&&n&&typeof SharedArrayBuffer<"u"&&a.buffer instanceof SharedArrayBuffer){h.workerPool=new k;try{await h.workerPool.init(),console.info("[HypercubeGrid] WorkerPool CPU instanciée avec succès (Zero-Copy prêt).")}catch(o){console.warn("[HypercubeGrid] Échec de l'initialisation de la WorkerPool.",o),h.workerPool=null}}return h}async compute(e){var u,a,s,r,p,f,n,h,c,o,l,m;if(this.mode==="webgpu"){const d=(await z(async()=>{const{HypercubeGPUContext:b}=await Promise.resolve().then(()=>C);return{HypercubeGPUContext:b}},void 0)).HypercubeGPUContext,y=d.device.createCommandEncoder();for(let b=0;b<this.rows;b++)for(let P=0;P<this.cols;P++){const x=this.cubes[b][P];x&&x.engine&&x.engine.computeGPU&&x.engine.computeGPU(d.device,y,x.nx,x.ny,x.nz)}d.device.queue.submit([y.finish()]);return}if(this.workerPool&&this.masterBuffer.buffer instanceof SharedArrayBuffer){const d=this.cubes.flat().filter(P=>P!==null),y=((a=(u=d[0])==null?void 0:u.engine)==null?void 0:a.name)||"Unknown",b={radius:((r=(s=d[0])==null?void 0:s.engine)==null?void 0:r.radius)||10,weight:((f=(p=d[0])==null?void 0:p.engine)==null?void 0:f.weight)||1,targetX:((h=(n=d[0])==null?void 0:n.engine)==null?void 0:h.targetX)??256,targetY:((o=(c=d[0])==null?void 0:c.engine)==null?void 0:o.targetY)??256};await this.workerPool.computeAll(d,this.masterBuffer.buffer,{name:y,config:b})}else for(let d=0;d<this.rows;d++)for(let y=0;y<this.cols;y++)await((l=this.cubes[d][y])==null?void 0:l.compute());if(this.cols===1&&this.rows===1)return;let i;if(e!==void 0)i=Array.isArray(e)?e:[e];else{const d=(m=this.cubes[0][0])==null?void 0:m.engine;i=d&&d.getSyncFaces?d.getSyncFaces():[0]}for(const d of i)this.synchronizeBoundaries(d)}synchronizeBoundaries(e){const i=this.nx,u=this.ny,a=this.nz;for(let s=0;s<this.rows;s++)for(let r=0;r<this.cols;r++){const p=this.cubes[s][r],f=p.faces[e];if(r<this.cols-1||this.isPeriodic){const n=this.cubes[s][(r+1)%this.cols],h=n.faces[e];for(let c=0;c<a;c++)for(let o=1;o<u-1;o++)h[n.getIndex(0,o,c)]=f[p.getIndex(i-2,o,c)]}if(r>0||this.isPeriodic){const n=this.cubes[s][(r-1+this.cols)%this.cols],h=n.faces[e];for(let c=0;c<a;c++)for(let o=1;o<u-1;o++)h[n.getIndex(i-1,o,c)]=f[p.getIndex(1,o,c)]}}for(let s=0;s<this.rows;s++)for(let r=0;r<this.cols;r++){const p=this.cubes[s][r],f=p.faces[e];if(s<this.rows-1||this.isPeriodic){const n=this.cubes[(s+1)%this.rows][r],h=n.faces[e];for(let c=0;c<a;c++)for(let o=1;o<i-1;o++)h[n.getIndex(o,0,c)]=f[p.getIndex(o,u-2,c)]}if(s>0||this.isPeriodic){const n=s===0?this.rows-1:s-1,h=this.cubes[n][r],c=h.faces[e];for(let o=0;o<a;o++)for(let l=1;l<i-1;l++)c[h.getIndex(l,u-1,o)]=f[p.getIndex(l,1,o)]}}for(let s=0;s<this.rows;s++)for(let r=0;r<this.cols;r++){const p=this.cubes[s][r],f=p.faces[e],n=(r+1)%this.cols,h=(r-1+this.cols)%this.cols,c=(s+1)%this.rows,o=(s-1+this.rows)%this.rows;for(let l=0;l<a;l++)(this.isPeriodic||r<this.cols-1&&s<this.rows-1)&&(this.cubes[c][n].faces[e][this.cubes[c][n].getIndex(0,0,l)]=f[p.getIndex(i-2,u-2,l)]),(this.isPeriodic||r>0&&s<this.rows-1)&&(this.cubes[c][h].faces[e][this.cubes[c][h].getIndex(i-1,0,l)]=f[p.getIndex(1,u-2,l)]),(this.isPeriodic||r<this.cols-1&&s>0)&&(this.cubes[o][n].faces[e][this.cubes[o][n].getIndex(0,u-1,l)]=f[p.getIndex(i-2,1,l)]),(this.isPeriodic||r>0&&s>0)&&(this.cubes[o][h].faces[e][this.cubes[o][h].getIndex(i-1,u-1,l)]=f[p.getIndex(1,1,l)])}}destroy(){this.workerPool&&(this.workerPool.terminate(),this.workerPool=null)}},H=class{static getSliceZ(t,e,i){if(i<0||i>=t.nz)throw new Error(`Slice index ${i} out of bounds (0-${t.nz-1})`);const u=t.nx,a=t.ny,s=t.faces[e],r=u*a,p=i*r;return s.slice(p,p+r)}static projectIso(t,e,i="max"){const u=t.nx,a=t.ny,s=t.nz,r=t.faces[e],p=new Float32Array(u*a);for(let f=0;f<a;f++)for(let n=0;n<u;n++){let h=0,c=-1/0;for(let l=0;l<s;l++){const m=r[l*a*u+f*u+n];h+=m,m>c&&(c=m)}const o=f*u+n;p[o]=i==="max"?c:h/s}return p}static exportVolume(t,e){const i=t.faces[e],u=new Uint8Array(i.length);for(let a=0;a<i.length;a++)u[a]=Math.max(0,Math.min(255,i[a]*255));return u}static injectSphere(t,e,i,u,a,s,r=1){const p=t.faces[e],{nx:f,ny:n,nz:h}=t,c=s*s;for(let o=0;o<h;o++){const l=o*n*f,m=o-a;for(let d=0;d<n;d++){const y=d*f,b=d-u;for(let P=0;P<f;P++){const x=P-i;x*x+b*b+m*m<=c&&(p[l+y+P]=r)}}}}static injectSlice(t,e,i,u,a=1){const s=t.faces[e],{nx:r,ny:p,nz:f}=t;for(let n=0;n<f;n++){if(i==="z"&&n!==u)continue;const h=n*p*r;for(let c=0;c<p;c++){if(i==="y"&&c!==u)continue;const o=c*r;for(let l=0;l<r;l++)i==="x"&&l!==u||(s[h+o+l]=a)}}}static renderToCanvas(t,e,i,u,a="green",s=!0){const r=t.getContext("2d");if(!r)return;(t.width!==i||t.height!==u)&&(t.width=i,t.height=u);const p=r.createImageData(i,u),f=new Uint32Array(p.data.buffer);let n=1,h=0;if(s){n=1e-4,h=-1e-4;for(let o=0;o<e.length;o++)e[o]>n&&(n=e[o]),e[o]<h&&(h=e[o])}const c=Math.max(Math.abs(n),Math.abs(h));for(let o=0;o<e.length;o++){let l=e[o]/n;if(a==="bipolar"&&(l=e[o]/c),l=Math.max(-1,Math.min(1,l)),a==="bipolar")if(l<0){const m=-l,d=0,y=Math.floor(255*m),b=255;f[o]=4278190080|b<<16|y<<8|d}else{const m=l,d=255,y=Math.floor(255*(1-m)),b=Math.floor(255*(1-m));f[o]=4278190080|b<<16|y<<8|d}else if(a==="viridis"){const m=Math.floor(l*l*255),d=Math.floor(l*255),y=Math.floor((1-l)*128+l*32);f[o]=4278190080|y<<16|d<<8|m}else if(a==="plasma"){const m=Math.floor(l*255),d=Math.floor(Math.pow(l,3)*255),y=Math.floor((1-l)*255);f[o]=4278190080|y<<16|d<<8|m}else if(a==="magma"){const m=Math.floor(l*255),d=Math.floor(Math.pow(l,2)*200),y=Math.floor(Math.pow(l,4)*100);f[o]=4278190080|y<<16|d<<8|m}else if(a==="green"){const m=Math.floor(l*255);f[o]=-16777216|m<<8|0}else if(a==="heat"){const m=Math.floor(l*255),d=Math.floor(Math.max(0,l-.5)*510);f[o]=-16777216|d<<8|m}else{const m=Math.floor(l*255);f[o]=4278190080|m<<16|m<<8|m}}r.putImageData(p,0,0)}static quickRender(t,e,i=0,u="viridis"){const a=e.faces[i];this.renderToCanvas(t,a,e.nx,e.ny,u,!0)}};const w=512;async function D(){const t=document.getElementById("app"),e=new O,i=await F.create(1,1,w,e,()=>new I(10,.1),3),u=i.cubes[0][0],a=u.faces,s=async()=>{await i.compute(),H.quickRender(t,u,2,"plasma"),requestAnimationFrame(s)};t.onmousemove=r=>{if(r.buttons!==1)return;const p=Math.floor(r.offsetX/t.clientWidth*w),f=Math.floor(r.offsetY/t.clientHeight*w),n=5;for(let h=-n;h<=n;h++)for(let c=-n;c<=n;c++){const o=p+c,l=f+h;o>=0&&o<w&&l>=0&&l<w&&(a[0][l*w+o]=1)}},s()}D();const C=Object.freeze(Object.defineProperty({__proto__:null,HypercubeGPUContext:S},Symbol.toStringTag,{value:"Module"}));
