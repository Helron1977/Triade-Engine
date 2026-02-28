import { TriadeMasterBuffer } from './core/TriadeMasterBuffer';
import { TriadeCubeV2 } from './core/TriadeCubeV2';
import type { ITriadeEngine } from './engines/ITriadeEngine';

/**
 * Chef d'orchestre de la V2. 
 * Il possède la VRAM globale partagée et gère les instanciations de Cubes sans fragmentation RAM.
 */
export class Triade {
    private _masterBuffer: TriadeMasterBuffer;
    public cubes: Map<string, TriadeCubeV2> = new Map();

    get masterBuffer() {
        return this._masterBuffer;
    }

    /**
     * @param vRamAllocMegabytes Taille du ArrayBuffer en Mega-Octets (par defaut 50MB)
     */
    constructor(vRamAllocMegabytes: number = 50) {
        this._masterBuffer = new TriadeMasterBuffer(vRamAllocMegabytes * 1024 * 1024);
        console.log(`[Triade.js SDK] Initialized with ${vRamAllocMegabytes}MB of Raw Buffer Memory.`);
    }

    /**
     * Forge un nouveau Cube (View Paging) depuis le Buffer Maître.
     */
    public createCube(name: string, mapSize: number, engine: ITriadeEngine, numFaces: number = 6): TriadeCubeV2 {
        if (this.cubes.has(name)) {
            throw new Error(`Cube avec le nom ${name} existe déjà.`);
        }

        const cube = new TriadeCubeV2(mapSize, this._masterBuffer, numFaces);
        cube.setEngine(engine);
        this.cubes.set(name, cube);
        return cube;
    }

    /**
     * Accès sécurisé à un cube pour UI/Render logic.
     */
    public getCube(id: string): TriadeCubeV2 | undefined {
        return this.cubes.get(id);
    }
}
