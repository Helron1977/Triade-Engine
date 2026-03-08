# Config Layer: Execution & Grid

The `config` section defines how the simulation is mapped to your hardware (CPU/GPU) and the world geometry.

## 1. Dimensions & Chunks

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `dimensions` | `object` | Total resolution (e.g., `nx: 512, ny: 512`). Must be multiples of 16. |
| `chunks` | `object` | Parallel partitioning (e.g., `x: 3, y: 2` = 6 parallel units). |

## 2. Execution Modes

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `mode` | `string` | `cpu` or `gpu`. |
| `executionMode`| `string` | `mono` or `parallel`. Use `parallel` for high-performance CPU workers. |

## 3. World Boundaries

Defines what happens at the edges of the global grid.

| Role | Description |
| :--- | :--- |
| `wall` | Solid no-slip boundary. |
| `periodic` | Toroidal wrap-around. |
| `absorbing` | Outflow/Infinite domain. |

### Example

```json
"config": {
  "dimensions": { "nx": 512, "ny": 512, "nz": 1 },
  "chunks": { "x": 3, "y": 2 },
  "boundaries": { "all": { "role": "wall" } },
  "mode": "cpu",
  "executionMode": "parallel"
}
```
