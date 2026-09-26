# Periodic water MD comparison

`benchmark.py` creates a reproducible water box and runs xreac ReaxFF or
MatterSim using GDPy's provider/materialization interface and ASE integrators.
Both models run sequentially in separate processes under a **180-second total
wall-time cap**, including scientific imports, model loading, and minimization.
This measures sustained calculator/MD
cost, including the same diagnostics and trajectory writes for both models;
it does not measure GDPy's batch-worker overhead.

Install from the repository root with `python -m pip install -e '.[reax,mattersim]'`.
Run the complete comparison:

```sh
python benchmarks/water_md/benchmark.py --output tmp/water-md-quick
python benchmarks/water_md/plot.py tmp/water-md-quick
```

Use new output directories on repeat runs. `plot.py` requires matplotlib.
The supervisor sets `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and
`CUDA_VISIBLE_DEVICES=''` for both workers. Use `--provider xreac` or
`--provider mattersim` to run just one model; outputs still go in the named
provider subdirectory. `--max-seconds` can shorten the cap but cannot exceed 180.
The benchmark's `xreac` choice builds `potential.provider: reax` with an ASE
executor. The same potential provider uses `reax/c` with a LAMMPS executor;
these are the defaults when `potential.backend` is omitted.
Historical JSON results retain the labels used when they were recorded.

The default requests 20 minimization, 50 warmup, and 200 NVE steps. Step counts
can be changed with `--min-steps`, `--equil-steps`, and `--steps`, but the wall-time
cap still applies. Each worker gets a share of the remaining time, with early
stage deadlines reserving time for NVE. It stops between steps and saves actual
counts; a supervisor kills a worker that overruns its allocation, including a
blocked import or force call. `comparison.json` records `complete`, `time_limit`,
or `failed` for each model. A hard kill retains the last atomic summary checkpoint
and flushed CSV records, but may interrupt the final trajectory frame. A model
that times out before reaching NVE has no NVE timing result to plot.

Step-limited or wall-time-limited warmup is not evidence of thermal equilibration.

## Current result: xreac 0.7.0 (2026-09-24)

The complete two-model benchmark finished in **87.1 seconds**, including startup
and minimization. Both models completed all 20 minimization, 50 NVT, and 200 NVE
steps on the same 81-atom initial box, using a 0.25 fs timestep. Recorded settings
and results are in [results-v070.json](results-v070.json).

| Measurement | ReaxFF / xreac 0.7.0 | MatterSim 1M |
| --- | ---: | ---: |
| 50-step NVT wall time | 2.48 s | 12.02 s |
| 200-step NVE wall time | 9.43 s | 48.03 s |
| Mean wall time per NVE step | 47.1 ms | 240.2 ms |
| Mean NVE temperature | 322.0 K | 360.9 K |
| NVE energy peak-to-peak range | 0.0819 meV/atom | 0.0330 meV/atom |

**xreac 0.7.0 was 5.10× faster in this short NVE comparison.** All sampled values
were finite and the original O–H distances remained bounded. Neither initial
minimization converged within 20 steps; temperature rose as the structures
continued to relax during MD. The short warmup and 50 fs NVE trajectory are
adequate for a quick timing check, not an equilibrated liquid-water study.
The protocol is shorter than the historical run below, so it is not a controlled
version-to-version speedup measurement on identical trajectories.

## Historical longer run: xreac 0.6.0 (2026-09-23)

Before the three-minute cap was introduced, both models completed 100 requested
minimization, 500 NVT, and 2,000 NVE steps on macOS ARM64 CPU, with the thread
limits above, xreac 0.6.0, MatterSim 1.2.3, ASE 3.27.0, and NumPy 2.0.2.
The initial geometry files were byte-identical. All 252 saved trajectory
frames per model had 81 atoms and full periodicity, and all sampled diagnostics
were finite. Full settings and numerical summaries are in [results.json](results.json).

| Measurement | ReaxFF / xreac | MatterSim 1M |
| --- | ---: | ---: |
| 500-step NVT wall time | 330.5 s | 133.9 s |
| 2,000-step NVE wall time | 1,315.0 s | 526.6 s |
| Mean wall time per NVE step | 657.5 ms | 263.3 ms |
| Mean NVE temperature | 256.9 K | 257.7 K |
| NVE temperature range | 210.5–299.1 K | 214.8–300.8 K |
| NVE energy slope | −0.0590 meV/atom/ps | +0.000205 meV/atom/ps |
| NVE energy peak-to-peak range | 0.1486 meV/atom | 0.03295 meV/atom |
| Sampled original O–H distance range in NVE | 0.883–1.135 Å | 0.928–1.088 Å |

**MatterSim was 2.50× faster in sustained NVE dynamics.** Both trajectories
remained stable over this short test, with bounded temperatures and small
energy fluctuations. This differs from the startup-dominated four-water GA
demo. This result applies to xreac 0.6.0, not the current 0.7.0 dependency.

ReaxFF met the initial minimization tolerance after 90 steps. MatterSim reached
the 100-step limit at 0.227 eV/Å without meeting the 0.1 eV/Å target. Each model
then followed its own trajectory. The 125 fs warmup did not establish a 300 K
equilibrium state: report the measured NVE temperatures, not the thermostat
target, when using these results. No timestep-convergence or long-time liquid
property validation was performed.

## Current short protocol

- 27 H₂O molecules (81 atoms), randomly oriented on a 3×3×3 grid, seed 731.
- Density 0.997 g/cm³; cubic periodic cell approximately 9.32 Å on each side.
- Each model independently relaxes the same initial geometry with BFGS,
  `maxstep=0.05 Å`, `fmax=0.1 eV/Å`, at most 20 steps (or its stage time budget).
- Maxwell–Boltzmann velocities at 300 K, the same random seed, zero total
  momentum. No rigid bonds or other constraints.
- Up to 50 Langevin NVT warmup steps at 300 K, friction 0.01 fs⁻¹.
- Up to 200 velocity-Verlet NVE steps, retaining the warmup's positions/velocities.
- Timestep 0.25 fs: at most 0.0125 ps warmup + 0.05 ps NVE = 0.0625 ps per model.
- Diagnostics and trajectory frames every 10 steps, including stage endpoints.

ReaxFF uses xreac 0.7.0's bundled `ffield.reax.HO.2015`, ASE image-resolved neighbors
with the default 0.3 Å skin for topology reuse,
QEq at every geometry, and the default fixed-charge force convention.
MatterSim uses its 1M checkpoint on CPU, with stress disabled. Cell dimensions
stay fixed. Runtime construction and minimization are reported separately
from the two MD stages.

The box is smaller than twice the ReaxFF nonbonded cutoff. xreac's ASE neighbor
backend includes multiple periodic images within the cutoff; this setup must
not be evaluated using a single minimum-image approximation for interactions.
Minimum-image O–H distances are used only as trajectory diagnostics.

## Outputs and interpretation

Each run saves its settings, package versions, timings, temperatures, and
NVE energy changes/slopes to `summary.json`. Full sampled diagnostics are in
`nvt.csv` and `nve.csv`; trajectories are in `nvt.traj` and `nve.traj`.
Initial, relaxed, and final configurations are also saved as extended XYZ.
`plot.py` produces `comparison.png`.

The O–H diagnostic follows the original molecular labels; changes can indicate
distortion or proton transfer and are not a chemical-species analysis. NVE
energy drift is reported per atom; NVT energy changes include thermostat work
and should not be interpreted as conservation errors. xreac's default
fixed-charge forces are not exactly the total derivative through its QEq
solution, so any measured drift includes that convention as well as timestep
and numerical effects.

This is a small-box performance/stability test starting from a constructed
configuration. The warmup does not establish equilibration, and a sub-ps
trajectory cannot validate liquid structure, diffusion, or thermodynamic
accuracy. Absolute energies from the two models are not comparable.
