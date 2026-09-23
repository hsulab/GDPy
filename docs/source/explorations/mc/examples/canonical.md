# Canonical: displacements in periodic Cu

The initial structure is a 32-atom FCC Cu supercell with a fixed 7.2 Å cubic
cell. Only Cu positions change; the cell and Cu count remain constant.
The energy-only Metropolis rule is
$P = \min(1, \exp[-\Delta E/(k_\mathrm{B}T)])$, with
$\Delta E = E_\mathrm{trial} - E_\mathrm{current}$.

```{literalinclude} ../../../../../examples/monte_carlo/canonical.yaml
:language: yaml
```

`temperature` is in kelvin and `max_disp` is in Å. The small displacement
keeps the demonstration near the initial lattice. No volume-changing move is
used: setting an operator's `pressure` does not turn this into NPT sampling.

## Run

From the repository root, use the shared {doc}`EMT single-point runtime <index>`:

```shell
gdp -d run-mc-canonical --runtime examples/monte_carlo/emt.yaml \
    explore examples/monte_carlo/canonical.yaml
python examples/monte_carlo/inspect_run.py run-mc-canonical
```

See the {ref}`Monte Carlo guide <monte-carlo>` for sampling limitations,
output interpretation, and restart instructions.
