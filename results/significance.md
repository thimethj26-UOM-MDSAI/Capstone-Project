## Macro-F1 across seeds (test)

| Model | Seeds | Macro-F1 per seed | Mean ± std |
|---|---|---|---|
| baseline | 1 | 42: 0.7734 | 0.7734 (single run) |
| sinbert | 3 | 1: 0.8156, 2: 0.8021, 42: 0.8069 | 0.8082 ± 0.0068 |
| xlmr | 3 | 1: 0.8403, 2: 0.8330, 42: 0.8185 | 0.8306 ± 0.0111 |

## Pairwise comparison, seed-42 runs (A − B)

| A | B | ΔMacro-F1 | A right / B wrong | B right / A wrong | McNemar p | Holm-adj. p | 95% bootstrap CI of Δ |
|---|---|---|---|---|---|---|---|
| baseline | sinbert | -0.0335 | 182 | 268 | 0.0001 | 0.0001 | [-0.0510, -0.0159] |
| baseline | xlmr | -0.0451 | 211 | 307 | 0.0000 | 0.0001 | [-0.0635, -0.0271] |
| sinbert | xlmr | -0.0116 | 207 | 217 | 0.6621 | 0.6621 | [-0.0282, +0.0050] |

Reading guide: a difference is only claimed if Holm-adjusted p < 0.05 AND the bootstrap interval excludes 0. Otherwise write "comparable".
