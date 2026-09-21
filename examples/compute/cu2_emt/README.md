# Minimize three copper dimers

This small `gdp compute` example relaxes three Cu2 structures with initial bond
lengths of 2.0, 2.5, and 3.0 Å using ASE's built-in EMT potential. No model download
or external simulation software is needed. Each dimer is centered in a periodic
20 × 20 × 20 Å cell.

From the repository root:

```sh
gdp -d cu2-compute -r examples/compute/cu2_emt/runtime.yaml compute examples/compute/cu2_emt/structures.xyz
```

The three minimizations run locally in one batch. Each stops at a maximum force
below 0.05 eV/Å or after 100 optimization steps. The worker box summarizes
calculation counts and the minimum, average (`avg`), and maximum steps, energy, and `maxfrc`.

Relaxed structures are written to `cu2-compute/results/end_frames.xyz`. Inspect
or collect the same run again with:

```sh
gdp -d cu2-compute compute status
gdp -d cu2-compute compute collect
```

Use a new output directory if you change the runtime or input structures: the
saved compute plan is immutable.
