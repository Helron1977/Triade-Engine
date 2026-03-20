import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { HypercubeNeoFactory } from '../../../core/HypercubeNeoFactory';
import { SharkEntity } from './shark';

/**
 * LIFE NEBULA XL (V77.0)
 * Scaled 8x (Volume) and 4x (Physics Density).
 * Randomized organic environment.
 */
const NX = 256; const NY = 256;
const TANK_W = 80; const TANK_L = 80; const TANK_H = 20;
const SURFACE_Y = 10;

class LifeNebulaXL {
    private scene!: THREE.Scene;
    private camera!: THREE.PerspectiveCamera;
    private renderer!: THREE.WebGLRenderer;
    private controls!: OrbitControls;

    private oceanEngine: any = null;
    private pathEngine: any = null;
    private tensorEngine: any = null;

    private shark!: SharkEntity;
    private sharkMat!: THREE.MeshPhysicalMaterial;
    private preyCount = 180;
    private preyList: THREE.Mesh[] = [];
    private preyVels: THREE.Vector3[] = [];
    private obstacles: THREE.Mesh[] = [];
    private bubblePoints: THREE.Points | null = null;
    private bubbleSpeeds: number[] = [];
    private prevSharkPos = new THREE.Vector3();

    private splashCooldown = 0;
    private eatFlashCounter = 0;
    private surfaceGeo!: THREE.PlaneGeometry;
    private catchCount = 0;
    private vCatches = 0;
    private tCatches = 0;
    private currentAI: 'VECTOR' | 'TENSOR' = 'VECTOR';

    constructor() {
        this.init3D().then(() => this.initEngines()).catch(console.error);
    }

    private async init3D() {
        const container = document.getElementById('canvas-container')!;
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x010810);
        this.camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 2000);
        this.camera.position.set(45, 40, 45);

        this.renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(this.renderer.domElement);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;

        this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const sun = new THREE.DirectionalLight(0xffffff, 3.0); sun.position.set(15, 50, 15); this.scene.add(sun);
        
        const surfLight = new THREE.PointLight(0xffffff, 2.5, 150);
        surfLight.position.set(0, SURFACE_Y + 10, 0);
        this.scene.add(surfLight);

        this.setupModels();
        this.setupFloor();
    }

    private setupModels() {
        const texLoader = new THREE.TextureLoader();

        // 1. Scaled Volume & Frame (80x80x20)
        const volumeMat = new THREE.MeshBasicMaterial({ color: 0x0ea5e9, transparent: true, opacity: 0.01, depthWrite: false });
        this.scene.add(new THREE.Mesh(new THREE.BoxGeometry(TANK_W, TANK_H, TANK_L), volumeMat));
        const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(TANK_W, TANK_H, TANK_L));
        this.scene.add(new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.15 })));

        // 2. RANDOMIZED OBSTACLES (Organic Clusters)
        const coralTexture = texLoader.load('assets/coral_texture_seamless_1773999580516.png');
        coralTexture.wrapS = coralTexture.wrapT = THREE.RepeatWrapping; coralTexture.repeat.set(1, 4);
        const cryMat = new THREE.MeshStandardMaterial({ map: coralTexture, roughness: 0.9, metalness: 0.1 });

        const addPillar = (x:number, z:number, h:number, radius = 2.2) => {
            const p = new THREE.Mesh(new THREE.CylinderGeometry(radius * 0.7, radius, h, 12), cryMat);
            p.position.set(x, -TANK_H/2 + h/2, z); 
            this.scene.add(p); this.obstacles.push(p);
        };

        const addBlock = (x:number, y:number, z:number, sx:number, sy:number, sz:number) => {
            const p = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), cryMat);
            p.position.set(x, y, z); 
            this.scene.add(p); this.obstacles.push(p);
        };

        // Generative environment - Spread things out (80x80)
        for (let i=0; i<15; i++) {
            const rx = (Math.random() - 0.5) * (TANK_W - 10);
            const rz = (Math.random() - 0.5) * (TANK_L - 10);
            const rh = 8 + Math.random() * 12;
            const rr = 1.0 + Math.random() * 2.8;
            const p = new THREE.Mesh(new THREE.CylinderGeometry(rr * 0.4, rr, rh, 12), cryMat);
            p.position.set(rx, -10 + rh/2, rz); // Level with glass floor (V77.7)
            p.rotation.x = (Math.random()-0.5) * 0.4;
            p.rotation.z = (Math.random()-0.5) * 0.4;
            this.scene.add(p); this.obstacles.push(p);
            
            // Randomized blocks at the base
            if (Math.random() > 0.4) {
                const bx = rx + (Math.random()-0.5)*8;
                const bz = rz + (Math.random()-0.5)*8;
                addBlock(bx, -TANK_H/2 + 2, bz, 4+Math.random()*6, 4, 4+Math.random()*6);
            }
        }

        // 3. Leafy Algae (Randomized Field)
        const algaeTex = texLoader.load('assets/algue.jpg'); 
        const algaeMat = new THREE.ShaderMaterial({
            uniforms: { map: { value: algaeTex }, color: { value: new THREE.Color(0x166534) } },
            vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
            fragmentShader: `
                varying vec2 vUv; uniform sampler2D map; uniform vec3 color;
                void main() {
                    vec4 tex = texture2D(map, vUv);
                    float d1 = distance(tex.rgb, vec3(0.8)); // Grey checkerboard
                    float d2 = distance(tex.rgb, vec3(1.0)); // White checkerboard
                    if (d1 < 0.1 || d2 < 0.1) discard;
                    gl_FragColor = vec4(tex.rgb * color * 1.5, 1.0); // Exact non-XL tint
                }
            `,
            side: THREE.DoubleSide
        });

        for (let i=0; i<300; i++) {
            const a = new THREE.Mesh(new THREE.PlaneGeometry(3, 6), algaeMat);
            const rx = (Math.random()-0.5)*TANK_W*0.9; const rz = (Math.random()-0.5)*TANK_L*0.9;
            a.position.set(rx, -TANK_H/2 + 3.0, rz);
            a.rotation.y = Math.random()*Math.PI; this.scene.add(a);
        }

        // 4. SHARK
        this.sharkMat = new THREE.MeshPhysicalMaterial({
            color: 0x475569, roughness: 0.1, metalness: 0.7, transmission: 0.1, thickness: 1.0,
            emissive: 0x0ea5e9, emissiveIntensity: 0.15
        });
        this.shark = new SharkEntity(this.scene, this.sharkMat);

        // 5. PREY (XL Population)
        const preyColors = [0xf43f5e, 0x38bdf8, 0x10b981, 0xffd700, 0xffa500];
        for (let i=0; i<this.preyCount; i++) {
            const p = new THREE.Mesh(new THREE.DodecahedronGeometry(0.12), new THREE.MeshStandardMaterial({ color: preyColors[i%preyColors.length], emissive: preyColors[i%preyColors.length], emissiveIntensity: 3.5 }));
            p.scale.set(1.0, 0.5, 1.8);
            p.position.set((Math.random()-0.5)*TANK_W*0.95, (Math.random()-0.5)*TANK_H*0.9, (Math.random()-0.5)*TANK_L*0.95);
            this.scene.add(p); this.preyList.push(p);
            this.preyVels.push(new THREE.Vector3((Math.random()-0.5)*0.2, (Math.random()-0.5)*0.1, (Math.random()-0.5)*0.2));
        }

        // 6. HIGH-RES SURFACE (256 Segments)
        this.surfaceGeo = new THREE.PlaneGeometry(TANK_W, TANK_L, NX - 1, NY - 1);
        const surfMat = new THREE.MeshStandardMaterial({ 
            color: 0x0ea5e9, transparent: true, opacity: 0.6, 
            roughness: 0.1, metalness: 0.8, vertexColors: true, flatShading: true 
        });
        const surface = new THREE.Mesh(this.surfaceGeo, surfMat);
        surface.rotation.x = -Math.PI / 2; surface.position.y = SURFACE_Y;
        this.scene.add(surface);

        // Add vertex color attribute for foam
        const foamColors = new Float32Array(this.surfaceGeo.attributes.position.count * 3);
        this.surfaceGeo.setAttribute('color', new THREE.BufferAttribute(foamColors, 3));

        // 7. BUBBLES (V77.8 - 5000 Density Match)
        const bubbleGeo = new THREE.BufferGeometry();
        const bPos = new Float32Array(5000 * 3);
        for (let i=0; i<5000; i++) {
            bPos[i*3] = (Math.random()-0.5)*TANK_W;
            bPos[i*3+1] = (Math.random()-0.5)*TANK_H;
            bPos[i*3+2] = (Math.random()-0.5)*TANK_L;
            this.bubbleSpeeds.push(0.04 + Math.random() * 0.12);
        }
        bubbleGeo.setAttribute('position', new THREE.BufferAttribute(bPos, 3));
        const bubbleMat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.15, transparent: true, opacity: 0.7 });
        this.bubblePoints = new THREE.Points(bubbleGeo, bubbleMat);
        this.scene.add(this.bubblePoints);
    }

    private setupFloor() {
        // 1. Scaled Blue Floor Circle (Aesthetic)
        const floorGeo = new THREE.CircleGeometry(TANK_W * 2.5, 64);
        const floorMat = new THREE.ShaderMaterial({
            uniforms: { uTime: { value: 0 }, uRadius: { value: TANK_W * 15 } },
            transparent: true, depthWrite: false, side: THREE.DoubleSide,
            vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
            fragmentShader: `
                uniform float uTime; uniform float uRadius; varying vec2 vUv;
                void main() {
                    float dist = length(vUv - 0.5);
                    float mask = smoothstep(0.5, 0.25, dist);
                    vec3 col = mix(vec3(0.1, 0.4, 0.7), vec3(0.98, 0.99, 1.0), mask * (0.45 + 0.1 * sin(uTime * 0.4)));
                    gl_FragColor = vec4(col, mask * 0.3);
                }
            `
        });
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.rotation.x = -Math.PI / 2; floor.position.y = -TANK_H/2 - 0.2;
        this.scene.add(floor);

        // 2. Glass Side Panels (Structure)
        const wallMat = new THREE.MeshStandardMaterial({ 
            color: 0x082f49, transparent: true, opacity: 0.08, 
            side: THREE.DoubleSide, depthWrite: false 
        });
        const createWall = (w:number, h:number, px:number, py:number, pz:number, rotX=0, rotY=0) => {
            const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), wallMat);
            m.position.set(px, py, pz); m.rotation.x = rotX; m.rotation.y = rotY;
            this.scene.add(m);
        };
        const offW = TANK_W/2 + 0.01;
        const offL = TANK_L/2 + 0.01;
        createWall(TANK_W, TANK_H, 0, 0, -offL); // Back
        createWall(TANK_W, TANK_H, 0, 0, offL);  // Front
        createWall(TANK_L, TANK_H, -offW, 0, 0, 0, Math.PI/2); // Left
        createWall(TANK_L, TANK_H, offW, 0, 0, 0, -Math.PI/2); // Right
        createWall(TANK_W, TANK_L, 0, -TANK_H/2 - 0.01, 0, -Math.PI/2);  // Floor glass
    }

    private async initEngines() {
        const factory = new HypercubeNeoFactory();
        
        // ENGINE 1: OCEAN
        const oManifest = await factory.fromManifest('./nebula-xl-manifest.json');
        oManifest.config.dimensions = { nx: NX, ny: NY, nz: 1 };
        oManifest.config.mode = 'gpu';
        this.oceanEngine = await factory.build(oManifest.config, oManifest.engine);

        // ENGINE 2: PATHFINDER
        const pManifest = await factory.fromManifest('../../path/cpu/path-cpu.json');
        pManifest.config.mode = 'gpu'; 
        pManifest.config.dimensions = { nx: NX, ny: NY, nz: 1 };
        pManifest.engine.requirements = { ghostCells: 1, pingPong: true }; 
        this.pathEngine = await factory.build(pManifest.config, pManifest.engine);
        
        // Init Path Data
        const bridge = this.pathEngine.bridge;
        const chunk = this.pathEngine.vGrid.chunks[0];
        const indices = this.pathEngine.parityManager.getFaceIndices('distance');
        bridge.getChunkViews(chunk.id)[indices.read].fill(1000);
        bridge.syncToDevice();

        // ENGINE 3: TENSOR (Sync 32x32x8 factors)
        const tManifest = await factory.fromManifest('../../tensor-cp/manifest-tensor-cp-gpu.json');
        tManifest.config.mode = 'gpu'; tManifest.config.dimensions = { nx: 32, ny: 32, nz: 8 };
        this.tensorEngine = await factory.build(tManifest.config, tManifest.engine);

        if (this.oceanEngine && this.pathEngine) {
            if (document.getElementById('loader')) document.getElementById('loader')!.style.display = 'none';
            this.animate();
        }
    }

    private worldToGrid(pos: THREE.Vector3) {
        return {
            x: Math.floor((pos.x / TANK_W + 0.5) * (NX - 1)),
            y: Math.floor((pos.z / TANK_L + 0.5) * (NY - 1))
        };
    }

    private async animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        await this.updateSimulation();
        this.syncVisuals();
        this.renderer.render(this.scene, this.camera);
    }

    private async updateSimulation() {
        const t = performance.now() * 0.001;
        
        // 1. PATHFINDER STEP (256x256 Grid)
        const pConfig = (this.pathEngine as any).vGrid.config;
        pConfig.objects = [];
        
        const worldToGridScaleX = (NX - 1) / TANK_W;
        const worldToGridScaleZ = (NY - 1) / TANK_L;

        // Obstacles in SDF
        this.obstacles.forEach((obs, i) => {
            const go = this.worldToGrid(obs.position);
            if (obs.geometry.type === 'BoxGeometry') {
                const geom = obs.geometry as THREE.BoxGeometry;
                const gw = geom.parameters.width * worldToGridScaleX;
                const gh = geom.parameters.depth * worldToGridScaleZ;
                pConfig.objects.push({ 
                    id: 'box' + i, type: 'rect', 
                    position: { x: go.x - gw/2, y: go.y - gh/2 }, 
                    dimensions: { w: gw, h: gh }, properties: { obstacles: 1.0 } 
                });
            } else if (obs.geometry.type === 'CylinderGeometry') {
                const geom = obs.geometry as THREE.CylinderGeometry;
                const gr = geom.parameters.radiusBottom * worldToGridScaleX * 2.2;
                pConfig.objects.push({ 
                    id: 'cyl' + i, type: 'circle', 
                    position: { x: go.x, y: go.y }, dimensions: { w: gr, h: gr }, properties: { obstacles: 1.0 } 
                });
            }
        });
        
        // Target in SDF
        if (this.currentTargetIndex !== -1 && this.preyList[this.currentTargetIndex].visible) {
            const tGo = this.worldToGrid(this.preyList[this.currentTargetIndex].position);
            pConfig.objects.push({ id:'prey', type:'circle', position:{x:tGo.x, y:tGo.y}, dimensions:{w:1, h:1}, properties:{distance:0.0}, rasterMode:'replace' });
        }
        await this.pathEngine.step(1);

        // 2. OCEAN STEP
        const oConfig = (this.oceanEngine as any).vGrid.config;
        oConfig.objects = [];

        const sharkPos = this.shark.getPosition();
        if (Math.abs(sharkPos.y - SURFACE_Y) < 2.0 && this.splashCooldown-- <= 0) {
            const go = this.worldToGrid(sharkPos);
            const vel = sharkPos.clone().sub(this.prevSharkPos);
            const vx = (vel.x / TANK_W) * 0.8;
            const vz = (vel.z / TANK_L) * 0.8;
            
            oConfig.objects.push({ 
                id: 'splash', type: 'circle', 
                position: { x: go.x - 5, y: go.y - 5 }, 
                dimensions: { w: 10, h: 10 }, 
                properties: { 
                    rho: 3.5,
                    vx: vx * 10, vy: vz * 10 
                }, 
                rasterMode: 'add' // Back to add for momentum (V77.8)
            });
            this.splashCooldown = 6;
        }
        this.prevSharkPos.copy(sharkPos);
        await this.oceanEngine.step(1);

        // 3. TENSOR SCENT INJECTION (V77.3)
        const tConfig = (this.tensorEngine as any).vGrid.config;
        tConfig.objects = this.preyList.filter(p => (p as any).visible).map((p, i) => ({
            id: 'prey'+i, type: 'circle', 
            position: { x: (p.position.x/TANK_W+0.5)*31, y: (p.position.z/TANK_L+0.5)*31 }, 
            dimensions: { w: 1, h: 1 },
            properties: { scent: 1.0 }
        }));
        await this.tensorEngine.step(30);

        // --- THE BRAIN (V77.3 Refinement) ---
        const forward = new THREE.Vector3(0,0,1).applyQuaternion(this.shark.group.quaternion);
        
        // 1. TARGET SELECTION & PREDATION (RADIUS 5.5)
        const snout = this.shark.getSnoutPosition();
        for (let i=0; i<this.preyCount; i++) {
            const p = this.preyList[i];
            if (p.visible && p.position.distanceTo(snout) < 5.5) {
                p.visible = false;
                this.eatFlashCounter = 25;
                if (this.currentAI === 'TENSOR') {
                    this.tCatches++;
                    document.getElementById('t-catches')!.innerText = this.tCatches.toString();
                } else {
                    this.vCatches++;
                    document.getElementById('v-catches')!.innerText = this.vCatches.toString();
                }
                setTimeout(() => { 
                    p.visible = true; 
                    p.position.set((Math.random()-0.5)*TANK_W*0.9, (Math.random()-0.5)*TANK_H*0.9, (Math.random()-0.5)*TANK_L*0.9); 
                }, 8000);
            }
        }

        if (this.currentTargetIndex === -1 || (this.preyList[this.currentTargetIndex] && !this.preyList[this.currentTargetIndex].visible) || Math.random() < 0.005) {
            let mD = 2000;
            this.preyList.forEach((p, idx) => {
                if(!p.visible) return;
                const toPrey = p.position.clone().sub(sharkPos).normalize();
                const d = p.position.distanceTo(sharkPos);
                if (d < mD && forward.dot(toPrey) > -0.5) { mD = d; this.currentTargetIndex = idx; }
            });
        }
        
        const target = this.preyList[this.currentTargetIndex]?.position || new THREE.Vector3(0,0,0);
        let steer = target.clone().sub(sharkPos).normalize();

        // 2. OBSTACLE AVOIDANCE (LOOK-AHEAD & HARDENED GRADIENTS)
        await this.pathEngine.bridge.syncToHost();
        const pData = this.pathEngine.bridge.getChunkViews(this.pathEngine.vGrid.chunks[0].id)[0];
        
        const getSDF = (pos: THREE.Vector3) => {
            const go = this.worldToGrid(pos);
            if (go.x < 0 || go.x >= NX || go.y < 0 || go.y >= NY) return 10.0;
            return pData[(go.y + 1) * (NX + 2) + (go.x + 1)];
        };

        // Proactive Look-ahead (4 units in front)
        const lookAhead = sharkPos.clone().addScaledVector(forward, 4.0);
        const dist = getSDF(sharkPos);
        const distAhead = getSDF(lookAhead);

        // BRAKING LOGIC (V77.3)
        this.shark.speedMultiplier = (dist < 3.5 || distAhead < 4.5) ? 0.05 : 1.0;

        if (dist < 6.5 || distAhead < 8.5) {
            const eps = 1.0;
            const gradX = getSDF(sharkPos.clone().add(new THREE.Vector3(eps,0,0))) - getSDF(sharkPos.clone().add(new THREE.Vector3(-eps,0,0)));
            const gradZ = getSDF(sharkPos.clone().add(new THREE.Vector3(0,0,eps))) - getSDF(sharkPos.clone().add(new THREE.Vector3(0,0,-eps)));
            const avoid = new THREE.Vector3(gradX, 0, gradZ).normalize();
            steer.addScaledVector(avoid, 7.5); // Even stronger for 256 res
        }

        // 3. TENSOR INSIGHT (SCENT PATTERNS)
        await this.tensorEngine.bridge.syncToHost();
        const tD = this.tensorEngine.bridge.getChunkViews(this.tensorEngine.vGrid.chunks[0].id)[4];
        const tx = Math.floor((sharkPos.x/TANK_W+0.5)*31);
        const tz = Math.floor((sharkPos.z/TANK_L+0.5)*31);
        const ty = Math.floor((sharkPos.y/TANK_H+0.5)*7);
        let scent = new THREE.Vector3(0,0,0);
        if (tx>1 && tx<30 && tz>1 && tz<30 && ty>1 && ty<6) {
            const vX = tD[(tx+1) + tz*32 + ty*32*32] - tD[(tx-1) + tz*32 + ty*32*32];
            const vY = tD[tx + tz*32 + (ty+1)*32*32] - tD[tx + tz*32 + (ty-1)*32*32];
            const vZ = tD[tx + (tz+1)*32 + ty*32*32] - tD[tx + (tz-1)*32 + ty*32*32];
            scent.set(vX, vY, vZ).normalize();
        }

        const aiStatus = document.getElementById('ai-status');
        if (scent.lengthSq() > 0.0001 && scent.dot(steer) > 0.5) {
            steer.addScaledVector(scent, 0.4);
            this.currentAI = 'TENSOR';
            if (aiStatus) aiStatus.innerText = "TENSOR SCENT";
        } else {
            this.currentAI = 'VECTOR';
            if (aiStatus) aiStatus.innerText = "VECTOR PURSUIT";
        }

        this.shark.update(steer.normalize());
        this.shark.animate(t);
    }

    private currentTargetIndex = -1;

    private syncVisuals() {
        const time = performance.now() * 0.001;
        
        // 1. PREY XL AI
        for (let i=0; i<this.preyCount; i++) {
            const p = this.preyList[i];
            if (!p.visible) continue;

            p.position.add(this.preyVels[i]);
            
            const halfX = TANK_W / 2 - 1;
            const halfY = TANK_H / 2 - 1;
            const halfZ = TANK_L / 2 - 1;
            if (p.position.x > halfX || p.position.x < -halfX) this.preyVels[i].x *= -1;
            if (p.position.y > halfY || p.position.y < -halfY) this.preyVels[i].y *= -1;
            if (p.position.z > halfZ || p.position.z < -halfZ) this.preyVels[i].z *= -1;
            
            
            p.lookAt(p.position.clone().add(this.preyVels[i]));
        }

        // 2. PREDATION EAT FLASH
        if (this.eatFlashCounter > 0) {
            this.sharkMat.emissive.setHex(0xff3b30);
            this.sharkMat.emissiveIntensity = this.eatFlashCounter / 4.0;
            this.eatFlashCounter--;
        } else {
            this.sharkMat.emissive.setHex(0x0ea5e9);
            this.sharkMat.emissiveIntensity = 0.25;
        }

        // 3. OCEAN SURFACE SYNC (128x128 Displacement)
        if (this.oceanEngine) {
            this.oceanEngine.bridge.syncToHost().then(() => {
                const rhoIdx = this.oceanEngine.parityManager.getFaceIndices('rho').read;
                const oData = this.oceanEngine.bridge.getChunkViews(this.oceanEngine.vGrid.chunks[0].id)[rhoIdx];
                
                const pAttr = this.surfaceGeo.attributes.position;
                const cAttr = this.surfaceGeo.attributes.color;
                
                const H_LIMIT = 2.5;

                for (let i=0; i<pAttr.count; i++) {
                    const gx = i % NX;
                    const gy = Math.floor(i / NX);
                    const v = oData[(gy + 1) * (NX + 2) + (gx + 1)];
                    // Exact non-XL 150.0 gain with edge smoothing
                    const edgeSmooth = Math.min(1.0, Math.min(gx, gy, (NX-gx), (NY-gy)) / 8);
                    let h = (v - 1.0) * 150.0 * edgeSmooth;
                    h = Math.max(-1.0, Math.min(1.0, h));
                    
                    pAttr.setZ(i, h);
                    if (h > 0.4) cAttr.setXYZ(i, 1.0, 1.0, 1.0); // Exact non-XL foam threshold
                    else cAttr.setXYZ(i, 0.05, 0.4 + h, 0.82);
                }
                pAttr.needsUpdate = true;
                cAttr.needsUpdate = true;
            });
        }

        // 4. BUBBLES ANIMATION (V77.8)
        if (this.bubblePoints) {
            const bAttr = this.bubblePoints.geometry.attributes.position;
            for (let i=0; i<5000; i++) {
                let y = bAttr.getY(i) + this.bubbleSpeeds[i];
                if (y > SURFACE_Y - 0.5) {
                    y = -9.8;
                    bAttr.setX(i, (Math.random()-0.5)*TANK_W);
                    bAttr.setZ(i, (Math.random()-0.5)*TANK_L);
                }
                bAttr.setY(i, y);
            }
            bAttr.needsUpdate = true;
        }
    }
}

new LifeNebulaXL();
