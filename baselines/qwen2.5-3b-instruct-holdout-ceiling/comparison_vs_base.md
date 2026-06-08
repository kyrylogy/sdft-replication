# Movement: `baselines/qwen2.5-3b-instruct-holdout-base` → `baselines/qwen2.5-3b-instruct-holdout-ceiling`

- n = **100**
- base accuracy:    **18.00%**
- ceiling accuracy: **50.00%**
- Δ = **+32.00** points

## Movement table

| transition | count |
|---|---:|
| wrong → right (gained) | 36 |
| right → right (kept)   | 14 |
| right → wrong (lost)   | 4 |
| wrong → wrong (still)  | 46 |

**Right→Wrong row indices** (regressions, worth inspecting): `[1, 33, 42, 83]`