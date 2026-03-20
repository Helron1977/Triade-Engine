import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { HypercubeNeoFactory } from '../../../core/HypercubeNeoFactory';

/**
 * LIFE NEBULA V41.0 - THE CLAMPED WHITE-CAP MASTERPIECE
 * Goal: 3D Deforming Surface. Max 5px (0.5 units) Height. White Peak Colors.
 */
const NX = 64; const NY = 64;
const TANK_SIZE = 20;
const SURFACE_Y = 10;
const H_LIMIT = 0.5; // "5 pixels" relative limit

class LifeNebula {
    private scene!: THREE.Scene;
    private camera!: THREE.PerspectiveCamera;
    private renderer!: THREE.WebGLRenderer;
    private controls!: OrbitControls;
    
    private engine: any = null;
    private shark!: THREE.Group;
    private preyCount = 65;
    private preyList: THREE.Group[] = [];
    private preyVels: THREE.Vector3[] = [];

    private sharkVel = new THREE.Vector3(0, 0, 0);
    private currentTargetIndex = -1;
    private splashCooldown = 0;
    
    private surfaceMesh!: THREE.Mesh;
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
        // Enclosure Frame
        const frame = new THREE.BoxGeometry(TANK_SIZE, TANK_SIZE, TANK_SIZE);
        const edges = new THREE.EdgesGeometry(frame);
        this.scene.add(new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.3 })));

        // Deep Water Block
        const dGeo = new THREE.BoxGeometry(TANK_SIZE, TANK_SIZE, TANK_SIZE);
        const dMat = new THREE.MeshPhysicalMaterial({ 
            color: 0x0ea5e9, transparent: true, opacity: 0.35, transmission: 0.95,
            metalness: 0, roughness: 0.05, ior: 1.1, thickness: 2.0
        });
        this.scene.add(new THREE.Mesh(dGeo, dMat));

        // Clamped Reactive Surface
        this.surfaceGeo = new THREE.PlaneGeometry(TANK_SIZE, TANK_SIZE, NX - 1, NY - 1);
        const colorAttr = new THREE.BufferAttribute(new Float32Array(this.surfaceGeo.attributes.position.count * 3), 3);
        this.surfaceGeo.setAttribute('color', colorAttr);
        const sMat = new THREE.MeshPhysicalMaterial({ 
            color: 0x0ea5e9, transparent: true, opacity: 0.8, transmission: 0.7,
            metalness: 0.1, roughness: 0.02, clearcoat: 1.0, 
            vertexColors: true, ior: 1.33, side: THREE.DoubleSide
        });
        this.surfaceMesh = new THREE.Mesh(this.surfaceGeo, sMat);
        this.surfaceMesh.rotation.x = -Math.PI / 2;
        this.surfaceMesh.position.y = SURFACE_Y + 0.1;
        this.scene.add(this.surfaceMesh);

        // Shark
        this.shark = new THREE.Group();
        const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.5, 1.4, 4, 16), new THREE.MeshPhysicalMaterial({ color: 0x334155, metalness: 0.9, roughness: 0.1, clearcoat: 1.0 }));
        body.rotation.x = Math.PI / 2;
        this.shark.add(body); this.scene.add(this.shark);

        // Prey
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
        return {
            gx: (v.x / TANK_SIZE + 0.5) * 64,
            gy: (v.z / TANK_SIZE + 0.5) * 64
        };
    }

    private isUpdating = false;
    private updateAI = async () => {
        if (!this.engine || this.isUpdating) return;
        this.isUpdating = true;
        try {
            const bridge = this.engine.bridge;
            const chunk = this.engine.vGrid.chunks[0];
            const pos = this.worldToGrid(this.shark.position);
            const config = (this.engine as any).vGrid.config;
            if (!config.objects) config.objects = [];
            
            // Interaction
            const dist = Math.abs(this.shark.position.y - SURFACE_Y);
            if (dist < 1.0 && this.splashCooldown-- <= 0) {
                config.objects.push({
                    id: 'impact', type: 'circle',
                    position: { x: pos.gx - 6, y: pos.gy - 6 }, dimensions: { w: 12, h: 12 },
                    properties: { rho: 1.45 }, rasterMode: "replace" 
                });
                this.splashCooldown = 15;
            }

            await this.engine.step(1);
            await bridge.syncToHost();
            config.objects = config.objects.filter((o: any) => o.id !== 'impact');

            const views = bridge.getChunkViews(chunk.id);
            const rIdx = this.engine.parityManager.getFaceIndices('rho').read;
            const rData = views[rIdx];

            const pAttr = this.surfaceGeo.attributes.position;
            const cAttr = this.surfaceGeo.attributes.color;
            const stride = 66; 

            for (let i = 0; i < pAttr.count; i++) {
                const px = pAttr.getX(i); const py = pAttr.getY(i);
                const gx = Math.floor((px / TANK_SIZE + 0.5) * 63);
                const gy = Math.floor((py / TANK_SIZE + 0.5) * 63);
                const v = rData[(gy + 1) * stride + (gx + 1)];
                const win = Math.min(1.0, Math.min(gx, gy, 63-gx, 63-gy) / 4.0);
                
                // 3D Deformation with Strict 5px/0.5u Limit
                let h = isNaN(v) ? 0 : (v - 1.0) * 15.0 * win; 
                h = Math.max(-H_LIMIT, Math.min(H_LIMIT, h));
                pAttr.setZ(i, h);
                
                // Color Mapping: Peak White
                if (h > H_LIMIT * 0.8) {
                    cAttr.setXYZ(i, 1.0, 1.0, 1.0); // Pure White Peak
                } else if (h < -0.1) {
                    cAttr.setXYZ(i, 0.05, 0.15, 0.4); // Deep Blue Trough
                } else {
                    cAttr.setXYZ(i, 0.2, 0.65, 0.95); // Azure Base
                }
            }
            pAttr.needsUpdate = true; cAttr.needsUpdate = true; this.surfaceGeo.computeVertexNormals();

            // AI
            if (this.currentTargetIndex === -1 || Math.random() < 0.005) {
                let mD = 1000;
                this.preyList.forEach((p, idx) => {
                    const d = p.position.distanceTo(this.shark.position);
                    if (d < mD) { mD = d; this.currentTargetIndex = idx; }
                });
            }
            if (this.currentTargetIndex !== -1) {
                const target = this.preyList[this.currentTargetIndex];
                this.sharkVel.lerp(target.position.clone().sub(this.shark.position).normalize().multiplyScalar(0.2), 0.06);
            }
        } catch (e) { console.error("Nebula V41 Error:", e); }
        finally { this.isUpdating = false; }
    }

    private animate = async () => {
        const start = performance.now();
        await this.updateAI();
        const b = 9.8; 
        
        this.shark.position.add(this.sharkVel);
        if (this.sharkVel.lengthSq() > 0.001) this.shark.lookAt(this.shark.position.clone().add(this.sharkVel));
        this.shark.position.clamp(new THREE.Vector3(-b, -b, -b), new THREE.Vector3(b, SURFACE_Y, b));

        this.preyList.forEach((p, i) => {
            const v = this.preyVels[i];
            v.add(new THREE.Vector3((Math.random()-0.5)*0.015, (Math.random()-0.5)*0.015, (Math.random()-0.5)*0.015));
            const pos = p.position;
            if (Math.abs(pos.x) > b || Math.abs(pos.y) > b || Math.abs(pos.z) > b) v.add(pos.clone().multiplyScalar(-0.25));
            const dS = p.position.distanceTo(this.shark.position);
            if (dS < 4.5) v.lerp(p.position.clone().sub(this.shark.position).normalize().multiplyScalar(0.3), 0.2);
            v.clampLength(0.04, 0.22); p.position.add(v);
            if (v.lengthSq() > 0.001) p.lookAt(p.position.clone().add(v));
        });

        this.renderer.render(this.scene, this.camera);
        this.controls.update();
        const ms = performance.now() - start;
        const hud = document.getElementById('val-frametime');
        if (hud) hud.innerHTML = `${ms.toFixed(1)}ms`;
        requestAnimationFrame(this.animate);
    }
}

new LifeNebula();
