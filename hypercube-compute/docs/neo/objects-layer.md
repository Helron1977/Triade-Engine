# Objects Layer: Spatially Distributed Items

The `objects` layer is where you "draw" your simulation content. Objects are baked into the grid (rasterized) at the start of each step.

## 1. Shape Types

| Type | Dimensions | Extra Attributes |
| :--- | :--- | :--- |
| `circle` | `w` (Diameter) | - |
| `rect` | `w`, `h` | - |
| `polygon` | `w`, `h` (Bounds) | `points`: `[{x, y}, ...]` |
| `ellipse` | `w`, `h` | - |

## 2. Rasterization (`rasterMode`)

How the object interacts with existing values in the grid:
- `replace`: Overwrites with `properties` values (Standard for walls).
- `add`: Sums up (Useful for multi-layered heat sources).
- `max`/`min`: Keeps the peak value.

## 3. Animation & Velocity

Objects can have built-in momentum.

| Attribute | Description |
| :--- | :--- |
| `velocity` | Physical shift per unit of time (e.g., `{x: 0.1, y: 0}`). |
| `pathExpression`| Functional pathing (e.g., `x: "128 + sin(t) * 10"`). |

### Example: Animated Wing

```json
{
  "id": "wing_moving",
  "type": "polygon",
  "position": { "x": 100, "y": 100 },
  "points": [...],
  "properties": { "obstacles": 1.0 },
  "animation": {
    "velocity": { "x": 0.05, "y": 0.0 }
  },
  "rasterMode": "replace"
}
```
