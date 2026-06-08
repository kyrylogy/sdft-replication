# Movement: `baselines/qwen2.5-3b-instruct` → `baselines/qwen2.5-3b-instruct-teacher-ceiling`

- n = **97**
- base accuracy:    **27.84%**
- ceiling accuracy: **78.35%**
- Δ = **+50.52** points

## Movement table

| transition | count |
|---|---:|
| wrong → right (gained) | 54 |
| right → right (kept)   | 22 |
| right → wrong (lost)   | 5 |
| wrong → wrong (still)  | 16 |

**Right→Wrong row indices** (regressions, worth inspecting): `[44, 45, 68, 78, 81]`