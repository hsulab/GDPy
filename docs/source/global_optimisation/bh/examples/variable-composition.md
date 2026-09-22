(bh-variable-composition-example)=

# Variable-composition Cu₆Niₓ clusters

This example changes composition during a single basin-hopping search. It
starts with six copper atoms and one to three nickel atoms, then attempts Ni
insertion and removal alongside ordinary displacement moves. All calculations
use EMT, so no model download is needed.

## Input

```{literalinclude} ../../../../../examples/global_optimisation/explorations/basin_hopping/cu6_nix.yaml
:language: yaml
```

The initial builder samples `Ni: "1:3"` while keeping `Cu: 6`. Four candidates
are relaxed, then two chains each attempt eight moves. The default periodic
setting uses a 20 Å vacuum cell.

- `exchange` selects nickel insertion or removal and has probability 0.7.
- `move` displaces Cu or Ni atoms and has probability 0.3.
- Both operators use 1000 K. Insertions sample a sphere of radius 3 Å centered
  at `[10, 10, 10]`, with distance checks before relaxation.
- Only nickel is exchanged, so every candidate retains six copper atoms.

The builder's range applies **only to initialization**. It does not impose a
bound on later exchange moves: chains can lose all nickel or grow beyond three
nickel atoms. The finite move budget keeps this demonstration short.

## Chemical potentials and ranking

The exchange operator uses `chempots: [-0.5]`, in eV per Ni atom, for acceptance.
Population ranking uses `objective.target: formation_energy` with the same Ni
chemical potential and a zero Cu reference:

```text
target = E - 6 μCu - x μNi = E + 0.5 x
```

Lower targets rank better. Exchange acceptance uses the evaluated energy
change, chemical potential, and particle-count/volume factors; it does not
apply the population ranking score a second time. These chemical potentials
are illustrative search settings, not calibrated reservoir conditions. The
short run demonstrates variable composition rather than an equilibrium
composition or a converged global minimum.

## Run

Pair the recipe with the shared EMT relaxation runtime:

```{literalinclude} ../../../../../examples/global_optimisation/runtimes/emt.yaml
:language: yaml
```

From the repository root:

```shell
OMP_NUM_THREADS=1 gdp -d run-cu6-nix \
  --runtime examples/global_optimisation/runtimes/emt.yaml \
  explore examples/global_optimisation/explorations/basin_hopping/cu6_nix.yaml
```

This is one expedition, so its outputs are directly under `run-cu6-nix/`.
Rerun the same command to resume. Use a fresh directory when changing the
chemical potentials, recipe, or runtime.

## Inspect composition changes

`candidates.db` stores initial structures and evaluated trials, including
rejected ones. Compare each trial with its recorded parent to distinguish
attempted insertions/removals from accepted composition changes:

```python
from collections import Counter
from ase.db import connect

with connect("run-cu6-nix/candidates.db") as database:
    rows = list(database.select(relaxed=1))
counts = {
    row.confid: Counter(row.toatoms().get_chemical_symbols())
    for row in rows
}
assert all(count["Cu"] == 6 for count in counts.values())
for row in rows:
    parents = row.data.get("parents", [])
    if not parents:
        continue
    delta = counts[row.confid]["Ni"] - counts[parents[0]]["Ni"]
    if delta:
        print(row.confid, row.formula, "delta Ni:", delta,
              "accepted:", row.data["accepted"])
```

The checked seed-7 run produced insertion and removal trials; accepted removals
took its selected chains from Cu₆Ni₂ through Cu₆Ni to Cu₆. Different software
versions may change the trajectory. The committed chain history is recorded
in `tmp_folder/gen1/rounds/events.jsonl`; the candidate database supplies the
structures for those IDs.
