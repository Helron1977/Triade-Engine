import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { HypercubeNeoFactory } from '../../../core/HypercubeNeoFactory';

/**
 * LIFE NEBULA V33.0 - THE BI-MATERIAL AQUARIUM
 * Goal: Textured Top Surface, Clear Non-Mirror Sides. Slightly bluish volume.
 */
const TANK_SIZE = 20;

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
    
    private waterBox!: THREE.Mesh;

    constructor() {
        this.init3D().then(() => this.initEngine()).catch(console.error);
    }

    private async init3D() {
        const container = document.getElementById('canvas-container')!;
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x011422);
        this.camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(24, 18, 24);

        this.renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(this.renderer.domElement);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;

        this.scene.add(new THREE.HemisphereLight(0x38bdf8, 0x011422, 3));
        const sun = new THREE.DirectionalLight(0xffffff, 8.0);
        sun.position.set(10, 40, 20);
        this.scene.add(sun);
        
        const rim = new THREE.PointLight(0x38bdf8, 100, 50);
        rim.position.set(-20, 10, -20);
        this.scene.add(rim);

        this.setupModels();
    }

    private setupModels() {
        // Enclosure Lines
        const frameGeo = new THREE.BoxGeometry(TANK_SIZE, TANK_SIZE, TANK_SIZE);
        const edges = new THREE.EdgesGeometry(frameGeo);
        const frame = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.25 }));
        this.scene.add(frame);

        // Shark
        this.shark = new THREE.Group();
        const sMat = new THREE.MeshPhysicalMaterial({ color: 0x334155, metalness: 0.9, roughness: 0.1, clearcoat: 1.0 });
        const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.5, 1.4, 4, 16), sMat);
        body.rotation.x = Math.PI / 2;
        this.shark.add(body);
        this.scene.add(this.shark);

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

        // --- THE BI-MATERIAL WATER CUBE ---
        const waterGeo = new THREE.BoxGeometry(TANK_SIZE, TANK_SIZE, TANK_SIZE);
        
        // 1. Clear Side Material (Non-Mirror)
        const sideMat = new THREE.MeshPhysicalMaterial({ 
            color: 0x0ea5e9, transparent: true, opacity: 0.35, transmission: 0.95,
            metalness: 0.0, roughness: 0.05, ior: 1.1, thickness: 1.0, 
            side: THREE.FrontSide 
        });

        // 2. Pretty Top Surface Material
        const topMat = new THREE.MeshPhysicalMaterial({ 
            color: 0x0ea5e9, transparent: true, opacity: 0.7, transmission: 0.8,
            metalness: 0.1, roughness: 0.02, clearcoat: 1.0, clearcoatRoughness: 0.05,
            emissive: 0x0284c7, emissiveIntensity: 0.6,
            side: THREE.DoubleSide
        });

        // Box Material Indices: 0: +X, 1: -X, 2: +Y (TOP), 3: -Y, 4: +Z, 5: -Z
        const mats = [sideMat, sideMat, topMat, sideMat, sideMat, sideMat];
        this.waterBox = new THREE.Mesh(waterGeo, mats);
        this.scene.add(this.waterBox);
    }

    private async initEngine() {
        const factory = new HypercubeNeoFactory();
        const manifest = await factory.fromManifest('./nebula-manifest.json');
        this.engine = await factory.build(manifest.config, manifest.engine);
        if (document.getElementById('loader')) document.getElementById('loader')!.style.display = 'none';
        this.animate();
    }

    private updateAI = async () => {
        if (!this.engine) return;
        try {
            await this.engine.step(1); 
            if (this.currentTargetIndex === -1 || Math.random() < 0.005) {
                let mD = 1000;
                this.preyList.forEach((p, idx) => {
                    const d = p.position.distanceTo(this.shark.position);
                    if (d < mD) { mD = d; this.currentTargetIndex = idx; }
                });
            }
            if (this.currentTargetIndex !== -1) {
                const target = this.preyList[this.currentTargetIndex];
                const delta = target.position.clone().sub(this.shark.position).normalize();
                this.sharkVel.lerp(delta.multiplyScalar(0.2), 0.06);
            }
        } catch (e) { console.warn("Nebula AI Active Passive"); }
    }

    private animate = async () => {
        const start = performance.now();
        await this.updateAI();
        const b = 9.8; 
        
        this.shark.position.add(this.sharkVel);
        if (this.sharkVel.lengthSq() > 0.001) this.shark.lookAt(this.shark.position.clone().add(this.sharkVel));
        this.shark.position.clamp(new THREE.Vector3(-b, -b, -b), new THREE.Vector3(b, 10, b));

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
