(ga-comparators)=

# Comparators

Comparators decide whether two relaxed candidates represent the same minimum.
This prevents duplicate structures from dominating the population.

| Method | Description |
| --- | --- |
| `interatomic_distance` | Compares energy and sorted interatomic-distance fingerprints. `pair_cor_cum_diff` and `pair_cor_max` control the cumulative and maximum fingerprint differences; `dE` controls the energy tolerance. |
| `ofp` | Uses Oganov fingerprints and an energy threshold. It is useful when radial environments provide a better similarity measure than direct distance-list comparison. |
| `nnmat` | Compares nearest-neighbour matrices to detect differences in atomic distribution and structure. |

`interatomic_distance` derives its minimum-image behavior from
`population.periodic` and supports parallel fingerprint generation with
`n_jobs`. Its default thresholds are
`pair_cor_cum_diff: 0.015`, `pair_cor_max: 0.7`, and `dE: 0.02` eV.
