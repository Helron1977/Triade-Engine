import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { HypercubeNeoFactory } from '../../../core/HypercubeNeoFactory';

/**
 * LIFE NEBULA V46.0 - THE PERFECT ALIGNMENT
 * Goal: Axis-Aligned Ripples. Dynamic Panel Culling. Transparent Clarity.
 */
const NX = 64; const NY = 64;
const TANK_SIZE = 20;
const SURFACE_Y = 10;
const H_LIMIT = 0.5;

class LifeNebula {
    private scene!: THREE.Scene;
    private camera!: THREE.PerspectiveCamera;
    private renderer!: THREE.WebGLRenderer;
    private controls!: OrbitControls;
    
    private engine: any = null;
    private shark!: THREE.Group;
    private sharkMat!: THREE.MeshPhysicalMaterial;
    private preyCount = 65;
    private preyList: THREE.Group[] = [];
    private preyVels: THREE.Vector3[] = [];

    private bubbles!: THREE.Points;
    private bubbleGeos!: THREE.BufferGeometry;

    private sidePanels: { mesh: THREE.Mesh, normal: THREE.Vector3 }[] = [];

    private sharkVel = new THREE.Vector3(0, 0, 0);
    private currentTargetIndex = -1;
    private splashCooldown = 0;
    private eatFlashCounter = 0;
    
    private surfaceGeo!: THREE.PlaneGeometry;

    constructor() {
        this.init3D().then(() => this.initEngine()).catch(console.error);
    }

    private async init3D() {
        const container = document.getElementById('canvas-container')!;
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x011422);
        this.camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(24, 22, 24);

        this.renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(this.renderer.domElement);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;

        this.scene.add(new THREE.HemisphereLight(0x38bdf8, 0x011422, 2.5));
        const sun = new THREE.DirectionalLight(0xffffff, 8.0);
        sun.position.set(10, 40, 20);
        this.scene.add(sun);
        
        this.setupModels();
    }

    private setupModels() {
        // 1. Crystal Water Volume (Faint)
        const cubeMat = new THREE.MeshPhysicalMaterial({ 
            color: 0x0ea5e9, transparent: true, opacity: 0.15, transmission: 0.99,
            metalness: 0, roughness: 0.01, ior: 1.1, thickness: 1.0
        });
        this.scene.add(new THREE.Mesh(new THREE.BoxGeometry(TANK_SIZE, TANK_SIZE, TANK_SIZE), cubeMat));

        // 2. Dynamic Side Panels (Walls)
        const wallMat = new THREE.MeshStandardMaterial({ 
            color: 0x082f49, transparent: true, opacity: 0.25, side: THREE.DoubleSide 
        });
        
        const createWall = (nx: number, ny: number, nz: number, px: number, py: number, pz: number, rotX = 0, rotY = 0) => {
            const mesh = new THREE.Mesh(new THREE.PlaneGeometry(TANK_SIZE, TANK_SIZE), wallMat.clone());
            mesh.position.set(px, py, pz);
            mesh.rotation.x = rotX; mesh.rotation.y = rotY;
            this.scene.add(mesh);
            this.sidePanels.push({ mesh, normal: new THREE.Vector3(nx, ny, nz) });
        };

        createWall(0, 0, 1, 0, 0, -10.05); // Back
        createWall(0, 0, -1, 0, 0, 10.05); // Front
        createWall(1, 0, 0, -10.05, 0, 0, 0, Math.PI/2); // Left
        createWall(-1, 0, 0, 10.05, 0, 0, 0, -Math.PI/2); // Right
        createWall(0, 1, 0, 0, -10.05, 0, -Math.PI/2); // Bottom

        // 3. Reactive Surface
        this.surfaceGeo = new THREE.PlaneGeometry(TANK_SIZE, TANK_SIZE, NX - 1, NY - 1);
        const colorAttr = new THREE.BufferAttribute(new Float32Array(this.surfaceGeo.attributes.position.count * 3), 3);
        this.surfaceGeo.setAttribute('color', colorAttr);
        const sMat = new THREE.MeshPhysicalMaterial({ 
            color: 0x0ea5e9, transparent: true, opacity: 0.8, transmission: 0.7,
            vertexColors: true, metalness: 0.1, roughness: 0.02, clearcoat: 1.0, side: THREE.DoubleSide
        });
        const surfaceMesh = new THREE.Mesh(this.surfaceGeo, sMat);
        surfaceMesh.rotation.x = -Math.PI / 2;
        surfaceMesh.position.y = SURFACE_Y + 0.05;
        this.scene.add(surfaceMesh);

        // 4. Bubbles
        this.bubbleGeos = new THREE.BufferGeometry();
        const count = 200;
        const pos = new Float32Array(count * 3);
        for(let i=0; i<count; i++) {
            pos[i*3] = (Math.random()-0.5)*19;
            pos[i*3+1] = (Math.random()-0.5)*19;
            pos[i*3+2] = (Math.random()-0.5)*19;
        }
        this.bubbleGeos.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        this.bubbles = new THREE.Points(this.bubbleGeos, new THREE.PointsMaterial({ color: 0xffffff, size: 0.08, transparent: true, opacity: 0.6 }));
        this.scene.add(this.bubbles);

        // 5. Entities
        this.shark = new THREE.Group();
        this.sharkMat = new THREE.MeshPhysicalMaterial({ color: 0x334155, metalness: 0.9, roughness: 0.1, clearcoat: 1.0 });
        const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.5, 1.4, 4, 16), this.sharkMat);
        body.rotation.x = Math.PI / 2;
        this.shark.add(body); this.scene.add(this.shark);

        const colors = [0xf43f5e, 0x38bdf8, 0x10b981];
        for (let i = 0; i < this.preyCount; i++) {
            const mat = new THREE.MeshStandardMaterial({ color: colors[i%3], emissive: colors[i%3], emissiveIntensity: 2.5 });
            const p = new THREE.Mesh(new THREE.DodecahedronGeometry(0.18), mat);
            p.scale.set(1, 0.5, 1.8);
            p.position.set((Math.random()-0.5)*18, (Math.random()-0.5)*18, (Math.random()-0.5)*18);
            this.scene.add(p);
            this.preyList.push(p as any);
            this.preyVels.push(new THREE.Vector3((Math.random()-0.5)*0.1, (Math.random()-0.5)*0.1, (Math.random()-0.5)*0.1));
        }
    }

    private async initEngine() {
        const factory = new HypercubeNeoFactory();
        const manifest = await factory.fromManifest('./nebula-manifest.json');
        this.engine = await factory.build(manifest.config, manifest.engine);
        if (document.getElementById('loader')) document.getElementById('loader')!.style.display = 'none';
        this.animate();
    }

    private worldToGrid(v: THREE.Vector3) {
        // AXIAL SYMMETRY FIX (Invert both axis as requested by user's mirror observation)
        return {
            x: Math.floor((1.0 - (v.x / TANK_SIZE + 0.5)) * 63),
            y: Math.floor((1.0 - (v.z / TANK_SIZE + 0.5)) * 63)
        };
    }

    private isUpdating = false;
    private updateAI = async () => {
        if (!this.engine || this.isUpdating) return;
        this.isUpdating = true;
        try {
            const bridge = this.engine.bridge;
            const chunk = this.engine.vGrid.chunks[0];
            const gPos = this.worldToGrid(this.shark.position);
            const config = (this.engine as any).vGrid.config;
            if (!config.objects) config.objects = [];
            
            // Interaction
            const dSurf = Math.abs(this.shark.position.y - SURFACE_Y);
            if (dSurf < 1.0 && this.splashCooldown-- <= 0) {
                config.objects.push({
                    id: 'impact', type: 'circle',
                    position: { x: gPos.x - 7, y: gPos.y - 7 },
                    dimensions: { w: 14, h: 14 },
                    properties: { rho: 1.55 }, rasterMode: "replace" 
                });
                this.splashCooldown = 15;
            }

            await this.engine.step(1);
            await bridge.syncToHost();
            config.objects = config.objects.filter((o: any) => o.id !== 'impact');

            const views = bridge.getChunkViews(chunk.id);
            const rData = views[this.engine.parityManager.getFaceIndices('rho').read];

            const pAttr = this.surfaceGeo.attributes.position;
            const cAttr = this.surfaceGeo.attributes.color;
            const stride = 66; 

            for (let i = 0; i < pAttr.count; i++) {
                const px = pAttr.getX(i); const pz = pAttr.getY(i);
                // DISPLAY MAP MUST ALSO MATCH INVERSION TO ALIGN
                const gx = Math.floor((1.0 - (px / TANK_SIZE + 0.5)) * 63);
                const gy = Math.floor((1.0 - (pz / TANK_SIZE + 0.5)) * 63);
                const v = rData[(gy + 1) * stride + (gx + 1)];
                const win = Math.min(1.0, Math.min(gx, gy, 63-gx, 63-gy) / 4.0);
                
                let h = isNaN(v) ? 0 : (v - 1.0) * 15.0 * win; 
                h = Math.max(-H_LIMIT, Math.min(H_LIMIT, h));
                pAttr.setZ(i, h);
                
                if (h > H_LIMIT * 0.8) cAttr.setXYZ(i, 1, 1, 1);
                else if (h < -0.05) cAttr.setXYZ(i, 0.05, 0.25, 0.5);
                else cAttr.setXYZ(i, 0.2, 0.65, 0.95);
            }
            pAttr.needsUpdate = true; cAttr.needsUpdate = true; this.surfaceGeo.computeVertexNormals();

            // Bubbles Move
            const bPos = this.bubbleGeos.attributes.position.array as Float32Array;
            for(let i=0; i<bPos.length/3; i++){
                bPos[i*3+1] += 0.05;
                if(bPos[i*3+1] > 10) bPos[i*3+1] = -10;
            }
            this.bubbleGeos.attributes.position.needsUpdate = true;

            // Hunting Logic
            if (this.currentTargetIndex === -1 || Math.random() < 0.005) {
                let mD = 1000;
                this.preyList.forEach((p, idx) => {
                    const d = p.position.distanceTo(this.shark.position);
                    if (d < mD) { mD = d; this.currentTargetIndex = idx; }
                });
            }
            if (this.currentTargetIndex !== -1) {
                const target = this.preyList[this.currentTargetIndex];
                const delta = target.position.clone().sub(this.shark.position);
                if (delta.length() < 1.3) {
                    target.position.set((Math.random()-0.5)*18, (Math.random()-0.5)*18, (Math.random()-0.5)*18);
                    this.currentTargetIndex = -1; this.eatFlashCounter = 12;
                } else {
                    this.sharkVel.lerp(delta.normalize().multiplyScalar(0.24), 0.07);
                }
            }
            if (this.eatFlashCounter > 0) {
                this.sharkMat.emissive.setHex(0xf43f5e);
                this.sharkMat.emissiveIntensity = this.eatFlashCounter / 6.0;
                this.eatFlashCounter--;
            } else { this.sharkMat.emissiveIntensity = 0; }

        } catch (e) { console.error("Nebula V46 Error:", e); }
        finally { this.isUpdating = false; }
    }

    private animate = async () => {
        const start = performance.now();
        await this.updateAI();
        this.shark.position.add(this.sharkVel);
        if (this.sharkVel.lengthSq() > 0.001) this.shark.lookAt(this.shark.position.clone().add(this.sharkVel));
        this.shark.position.clamp(new THREE.Vector3(-9.8, -9.8, -9.8), new THREE.Vector3(9.8, 10, 9.8));

        this.preyList.forEach((p, i) => {
            const v = this.preyVels[i];
            v.add(new THREE.Vector3((Math.random()-0.5)*0.015, (Math.random()-0.5)*0.015, (Math.random()-0.5)*0.015));
            if (Math.abs(p.position.x) > 9.8 || Math.abs(p.position.y) > 9.8 || Math.abs(p.position.z) > 9.8) v.add(p.position.clone().multiplyScalar(-0.25));
            if (p.position.distanceTo(this.shark.position) < 4.5) v.lerp(p.position.clone().sub(this.shark.position).normalize().multiplyScalar(0.35), 0.22);
            v.clampLength(0.04, 0.22); p.position.add(v);
            if (v.lengthSq() > 0.001) p.lookAt(p.position.clone().add(v));
        });

        // 6. Dynamic Culling of Panels
        const camPos = this.camera.position.clone().normalize();
        this.sidePanels.forEach(p => {
            // If dot product of camera dir and plane normal is > 0, the plane is facing "with" the camera (Far Wall)
            const dot = p.normal.dot(camPos);
            p.mesh.visible = (dot > 0); 
        });

        this.renderer.render(this.scene, this.camera);
        this.controls.update();
        const hud = document.getElementById('stat-fps');
        if (hud) hud.innerHTML = `${(performance.now() - start).toFixed(1)}ms`;
        requestAnimationFrame(this.animate);
    }
}

new LifeNebula();
