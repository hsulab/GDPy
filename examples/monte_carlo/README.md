# EMT Monte Carlo examples

Run from the GDPy repository root after installing GDPy (`python -m pip install
-e .`). ASE's EMT requires no downloaded model or external executable.

```shell
gdp -d run-mc-canonical --runtime examples/monte_carlo/emt.yaml \
    explore examples/monte_carlo/canonical.yaml
gdp -d run-mc-semi-grand-canonical --runtime examples/monte_carlo/emt.yaml \
    explore examples/monte_carlo/semi-grand-canonical.yaml
gdp -d run-mc-grand-canonical --runtime examples/monte_carlo/emt.yaml \
    explore examples/monte_carlo/grand-canonical.yaml
```

| Input | System | Expected changes |
| --- | --- | --- |
| `canonical.yaml` | Periodic Cu32 | Positions; fixed composition and cell |
| `semi-grand-canonical.yaml` | Periodic Cu16Ni16 | Cu/Ni identities and positions; fixed 32 atoms and cell |
| `grand-canonical.yaml` | Cu8 plus one stationary Au in a periodic box | Cu insertion/removal; fixed Au count and cell |

`swap_type` changes an atom's
identity; `swap` would preserve species counts. These three examples run 100 steps at
1200 K using single-point energies (`spc`), without minimisation. Chemical
potentials are in eV per atom and follow `particles` order. The grand-canonical
2.0 eV value is an illustrative EMT setting, not an experimental calibration.

These are workflow demos, not validated equilibrium studies. The current
`swap_type` species-first selection lacks a proposal-count correction, and
`exchange` lacks a correction for forced insertion at zero exchangeable count.
Disabling geometry checks and retries in the inputs does not fix those issues.
See the [example guides](../../docs/source/explorations/mc/examples/index.md)
for the individual ensembles and the [MC guide](../../docs/source/explorations/mc.md) for acceptance formulas,
parameter meanings, limitations, and restart details.

## Hybrid MD/MC

Two self-contained inputs alternate 20 Langevin NVT MD steps with five MC
proposals per cycle, for five cycles at 1200 K:

```shell
gdp -d run-mc-hybrid-canonical explore examples/monte_carlo/hybrid-canonical.yaml
gdp -d run-mc-hybrid-semi-grand-canonical \
    explore examples/monte_carlo/hybrid-semi-grand-canonical.yaml
```

`hybrid-canonical.yaml` uses Cu32 with displacement moves;
`hybrid-semi-grand-canonical.yaml` uses Cu16Ni16 with identity changes. Both
contain EMT runtimes for initialization, MD, and MC, so no `--runtime` is needed.
Hybrid settings currently use a flat input rather than a `recipe` wrapper.

MD endpoints are accepted directly; only individual MC proposals receive a
potential-energy Metropolis test. This is mixed MD/MC, not Hamiltonian Monte
Carlo. See the [hybrid guide](../../docs/source/explorations/hmc.md)
for parameters, limitations, and restart details.

Each hybrid demo produces six `mc.xyz` frames (initial state plus five cycles)
and 25 MC decision rows in `opstat.txt`. Intermediate MD trajectories and MC
trials live under `calculations/step.NNNN/procedure.NNNN/`. Hybrid
`mc_attempts.xyz` contains only the initial structure, and its top-level
`dump_period` does not prune calculation directories.

## Structures and inspection

Initial structures are included. Regenerate them with:

```shell
python examples/monte_carlo/prepare.py
```

Inspect a run with:

```shell
python examples/monte_carlo/inspect_run.py run-mc-semi-grand-canonical
```

For the three plain MC examples, `mc.xyz` contains the initial state and every completed step, including repeated
states on rejection. `mc_attempts.xyz` contains evaluated trials and
`opstat.txt` reports acceptance. `dump_period` controls retained calculation
directories, not trajectory thinning. Repeat the same command and output
folder to resume; use a new output folder after changing inputs or runtime.
