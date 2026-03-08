# Engine Layer: Physics Definition

The `engine` section defines the "Hardware Abstracted" physics logic. It tells Hypercube how to allocate memory (Faces) and what math to run (Rules).

## 1. Faces (`faces`)

Faces are individual memory layers (tensors).

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `name` | `string` | Unique name (e.g., `smoke`, `vx`). |
| `type` | `string` | `scalar`, `vector`, `mask`. |
| `isSynchronized`| `boolean`| `true` to exchange ghost cells between chunks. |
| `isPersistent` | `boolean`| `true` to keep data between steps (Default). If `false`, it saves memory copies. |
| `isReadOnly` | `boolean`| `true` for static obstacles or masks. |

## 2. Rules (`rules`)

Rules define the numerical kernels to execute on the faces.

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `type` | `string` | The kernel ID (e.g., `lbm-aero-fidelity-v1`, `diffusion`). |
| `method` | `string` | Discretization method (e.g., `BGK`, `Upwind`). |
| `params` | `object` | Physical constants (e.g., `omega`, `viscosity`, `inflow`). |

### Example: Aerodynamics LBM

```json
"engine": {
  "name": "Aerodynamics-Fidelity",
  "faces": [
    { "name": "f0", "type": "scalar", "isSynchronized": true, "isPersistent": false },
    { "name": "obstacles", "type": "mask", "isSynchronized": true, "isReadOnly": true }
  ],
  "rules": [
    {
      "type": "lbm-aero-fidelity-v1",
      "method": "BGK",
      "params": { "omega": 1.75, "inflowUx": 0.15 }
    }
  ],
  "requirements": {
    "ghostCells": 1,
    "pingPong": true
  }
}
```
