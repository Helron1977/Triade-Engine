import { BoundaryProperty } from '../engines/EngineManifest';

export interface Point3D {
    x: number;
    y: number;
    z: number;
}

export abstract class Shape {
    constructor(
        public position: Point3D,
        public properties: Record<string, number | BoundaryProperty> = {}
    ) { }

    /**
     * @description Teste si un point (monde) est à l'intérieur de la forme.
     */
    abstract contains(p: Point3D): boolean;

    /**
     * @description Retourne l'emprise (Box) de la forme dans le monde.
     */
    abstract getBoundingBox(): { min: Point3D, max: Point3D };
}

export class Circle extends Shape {
    constructor(
        position: Point3D,
        public radius: number,
        properties: Record<string, number | BoundaryProperty> = {}
    ) {
        super(position, properties);
    }

    contains(p: Point3D): boolean {
        const dx = p.x - this.position.x;
        const dy = p.y - this.position.y;
        const dz = p.z - this.position.z;
        return (dx * dx + dy * dy + dz * dz) <= (this.radius * this.radius);
    }

    getBoundingBox(): { min: Point3D, max: Point3D } {
        return {
            min: { x: this.position.x - this.radius, y: this.position.y - this.radius, z: this.position.z - this.radius },
            max: { x: this.position.x + this.radius, y: this.position.y + this.radius, z: this.position.z + this.radius }
        };
    }
}

export class Box extends Shape {
    constructor(
        position: Point3D, // Center
        public width: number,
        public height: number,
        public depth: number = 1,
        properties: Record<string, number | BoundaryProperty> = {}
    ) {
        super(position, properties);
    }

    contains(p: Point3D): boolean {
        const halfW = this.width / 2;
        const halfH = this.height / 2;
        const halfD = this.depth / 2;

        return p.x >= (this.position.x - halfW) && p.x <= (this.position.x + halfW) &&
            p.y >= (this.position.y - halfH) && p.y <= (this.position.y + halfH) &&
            p.z >= (this.position.z - halfD) && p.z <= (this.position.z + halfD);
    }

    getBoundingBox(): { min: Point3D, max: Point3D } {
        const halfW = this.width / 2;
        const halfH = this.height / 2;
        const halfD = this.depth / 2;
        return {
            min: { x: this.position.x - halfW, y: this.position.y - halfH, z: this.position.z - halfD },
            max: { x: this.position.x + halfW, y: this.position.y + halfH, z: this.position.z + halfD }
        };
    }
}
