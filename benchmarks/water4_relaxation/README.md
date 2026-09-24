# Water-cluster ReaxFF / MatterSim benchmark

## Measurements

ReaxFF is the suggested runtime for the short water4 GA demo. User-reported
complete CLI runs on 2026-09-23 in the local `catorch3` environment measured:

| Complete water4 GA (six candidates) | xreac | MatterSim |
| --- | ---: | ---: |
| GDPy exploration elapsed time | 2.5 s | 8.4 s |
| Initial four-candidate worker | 1.3 s | 3.0 s |
| Two-offspring worker | 0.6 s | 1.4 s |

ReaxFF was about 3.4 times faster overall and 2.3 times faster in the workers
for these runs. Both completed all six candidates with 20 relaxation steps
each. The commands did not explicitly set thread limits. These are individual
workflow measurements; they do not establish a general force-evaluation
speedup or equivalent optimization quality. MatterSim remains an alternative.

A separate [27-water periodic MD test](../water_md/README.md) now uses
xreac 0.7.0 and a three-minute total wall-time cap. In the updated 200-step NVE
comparison, xreac was **5.10× faster** than MatterSim. The earlier, longer test
with xreac 0.6.0 favored MatterSim; its historical results remain documented.

The isolated benchmark below produced a different ordering. Its fixed input
geometries, explicit thread limits, and direct ASE relaxations differ from the
CLI workflow. The cause of the timing difference has not been isolated, so
the isolated timings should not determine the demo's runtime recommendation.

Measured on macOS ARM64 CPU, 2026-09-23, with `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1`, xreac 0.6.0 and MatterSim 1.2.3 (cached 1M model):

| Measurement | xreac | MatterSim |
| --- | ---: | ---: |
| Runtime construction, including imports | 0.076 s | 6.06 s |
| Median of five warmed energy/force calls | 21.8 ms | 13.0 ms |
| Four 20-step BFGS relaxations, total | 1.91 s | 0.893 s |

Four identical 12-atom starting structures were generated with the water4
recipe's builder and seed 17. Every timed force call changed coordinates to
avoid ASE's result cache. Relaxation timings exclude model loading and search
overhead. ReaxFF took about 2.1 times as long for these relaxations, although
its startup savings can benefit very short jobs. Neither potential reached
the force tolerance within 20 steps on any of the four structures. This is
an equal-step timing comparison, not a comparison of accuracy or time to
convergence. Both potentials follow different energy surfaces.

The complete water4 GA was also exercised with ReaxFF: all six candidates
finished successfully. Existing BH examples lack compatible bundled parameters;
they retain their current runtimes.

Raw results are in `benchmark_reax_results.json`. To repeat the benchmark with
both extras installed, run from the repository root:

```shell
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
  benchmarks/water4_relaxation/benchmark_reax.py --output reax-benchmark.json
```

