import { VirtualChunk } from './GridAbstractions';
import { GridBoundaries } from '../types';

/**
 * Universal Boundary Role Identifiers.
 * These are passed as U32 to WGSL kernels.
 */
export enum BoundaryRoleID {
    WALL = 0,
    CONTINUITY = 1, // Joint between chunks or Periodic wrap
    INFLOW = 2,
    OUTFLOW = 3,
    SYMMETRY = 4,
    ABSORBING = 5,
    DIRICHLET = 6,
    NEUMANN = 7,
    CLAMPED = 8
}

export interface ResolvedTopology {
    leftRole: number;
    rightRole: number;
    topRole: number;
    bottomRole: number;
    frontRole: number;
    backRole: number;
}

/**
 * TopologyResolver: Standardizes chunk neighborhood logic.
 * Ensures the engine remains agnostic of its absolute world position.
 */
export class TopologyResolver {
    /**
     * Resolve the topology roles for a specific chunk.
     */
    resolve(
        vChunk: VirtualChunk,
        chunkLayout: { x: number; y: number; z?: number },
        globalBoundaries: GridBoundaries
    ): ResolvedTopology {
        const topology: ResolvedTopology = {
            leftRole: BoundaryRoleID.WALL,
            rightRole: BoundaryRoleID.WALL,
            topRole: BoundaryRoleID.WALL,
            bottomRole: BoundaryRoleID.WALL,
            frontRole: BoundaryRoleID.WALL,
            backRole: BoundaryRoleID.WALL
        };

        const faces: Array<'left' | 'right' | 'top' | 'bottom' | 'front' | 'back'> = 
            ['left', 'right', 'top', 'bottom', 'front', 'back'];

        for (const face of faces) {
            const joint = vChunk.joints.find(j => j.face === face);
            let role = BoundaryRoleID.WALL;

            if (joint) {
                // If it's a joint, it's either an internal connection or a periodic wrap.
                // Both act as CONTINUITY for the numerical solver.
                if (joint.role === 'joint') {
                    role = BoundaryRoleID.CONTINUITY;
                } else {
                    role = this.mapRoleToID(joint.role);
                }
            } else {
                // Fallback to global boundaries if no joint descriptor found for this face
                const bounds = globalBoundaries || {};
                const boundarySide = (bounds as any)[face] || (bounds as any).all || { role: 'wall' };
                role = this.mapRoleToID(boundarySide.role);
            }

            switch (face) {
                case 'left': topology.leftRole = role; break;
                case 'right': topology.rightRole = role; break;
                case 'top': topology.topRole = role; break;
                case 'bottom': topology.bottomRole = role; break;
                case 'front': topology.frontRole = role; break;
                case 'back': topology.backRole = role; break;
            }
        }

        return topology;
    }

    private mapRoleToID(role: string): number {
        switch (role) {
            case 'wall': return BoundaryRoleID.WALL;
            case 'inflow': return BoundaryRoleID.INFLOW;
            case 'outflow': return BoundaryRoleID.OUTFLOW;
            case 'periodic': return BoundaryRoleID.CONTINUITY; // Maps to continuity for indexing
            case 'joint': return BoundaryRoleID.CONTINUITY;
            case 'symmetry': return BoundaryRoleID.SYMMETRY;
            case 'absorbing': return BoundaryRoleID.ABSORBING;
            case 'dirichlet': return BoundaryRoleID.DIRICHLET;
            case 'neumann': return BoundaryRoleID.NEUMANN;
            case 'clamped': return BoundaryRoleID.CLAMPED;
            default: return BoundaryRoleID.WALL;
        }
    }
}
