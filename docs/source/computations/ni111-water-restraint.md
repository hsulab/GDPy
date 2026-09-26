(compute-ni-water-restraint-example)=

# O-H restraint windows on H2O/Ni(111)

This example adds harmonic O-H distance restraints to ReaxFF molecular
dynamics for molecular water on Ni(111). It demonstrates how a runtime
`modifier` is added to a physical potential and broadcast over several
restraint centers. The supplied structure is a two-layer p(2x2) slab with the
bottom four Ni atoms fixed.

Install the ReaxFF extra, which provides xreac and its bundled
`ffield.reax.PtNiCHO.2016` parameters:

```shell
python -m pip install -e '.[reax]'
```

The complete runtime is:

```{literalinclude} ../../../examples/compute/ni111_water_restraint/runtime.yaml
:language: yaml
```

Atoms 0-7 are Ni, atom 8 is O, and atoms 9-10 are H. The group expression
`` `index 8 10` `` selects O and one H by their zero-based indices. The modifier
restrains that O-H distance with a spring constant of 5.0 eV/Angstrom squared.
Its `broadcast` creates four independent windows centered at 0.95, 1.15, 1.35,
and 1.55 Angstrom. The bias contribution in each window is

```{math}
V_\mathrm{bias}(r) = \frac{1}{2} k (r-r_0)^2.
```

From the repository root, run:

```shell
gdp -d run-ni-water-restraint \
    -r examples/compute/ni111_water_restraint/runtime.yaml \
    compute examples/compute/ni111_water_restraint/structure.xyz
```

The command creates `w0` through `w3` in broadcast order. Each worker runs a
20-step NVT trajectory with a 0.25 fs timestep and takes the ReaxFF energy and
forces as the host contribution. The modifier adds its energy and forces before
ASE advances the dynamics. For example, inspect the 1.55 Angstrom window with:

```shell
cat run-ni-water-restraint/w3/cand0/01.DistanceHarmonicCalculator/calc.log
```

The initial O-H distance is about 0.962 Angstrom, so the first bias energy is
about 0.864 eV in that window. Its saved structures are in
`run-ni-water-restraint/w3/cand0/traj.xyz`.

This deliberately short trajectory is a mechanics test, not equilibrated
sampling and not a free-energy calculation. Broadcast supplies the independent
windows that can precede umbrella sampling. For automatic equilibration,
multiple replicas, restart metadata, and active-learning selection, see the
{doc}`umbrella-sampling exploration <../explorations/umbrella-sampling>`. Neither
workflow performs replica exchange or free-energy reconstruction. A free-energy
workflow must also verify overlap between neighboring distance distributions and
combine them with an appropriate unbiased estimator. The bundled force field is
subject to its own noncommercial license, and the example does not claim a
DFT-quality water-dissociation barrier.
