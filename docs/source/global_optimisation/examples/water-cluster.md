(ga-water-cluster-example)=

# Molecular cluster

This example searches for low-energy structures of a four-water cluster. The
builder inserts four intact H2O molecules in a spherical region at the centre
of a large periodic vacuum box and assigns one tag to each molecule. Structures
are relaxed with the 1-million-parameter MatterSim checkpoint, the fastest
pretrained MatterSim model.

## Input

The complete example is available at
`examples/global_optimisation/water4_mattersim.yaml`:

```{literalinclude} ../../../../examples/global_optimisation/water4_mattersim.yaml
:language: yaml
```

The periodic cut-and-splice implementation preserves tagged molecular
fragments and their inherited orientations, while rattle translates whole
water molecules. The MatterSim relaxation may change intramolecular coordinates
because the model evaluates and relaxes all atoms.

## Run

Install MatterSim in the GDPy environment, then run from the repository root:

```shell
python -m pip install mattersim
gdp -d ./run-water4-mattersim explore \
    ./examples/global_optimisation/water4_mattersim.yaml
```

On its first use, MatterSim downloads the `MatterSim-v1.0.0-1M` checkpoint. The
search is stored under `run-water4-mattersim/expedition-0`. The small population
and single generation keep this example quick; increase both for a production
search.

:::{note}
MatterSim is substantially more suitable for demonstrating H/O relaxation than
ASE EMT, whose H and O parameters are intended only for testing. This compact
example still demonstrates the workflow rather than providing a converged or
validated water-cluster study.
:::
