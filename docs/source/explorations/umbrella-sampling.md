(umbrella-sampling)=

# Umbrella sampling

`umbrella_sampling` runs independent harmonic-distance windows for rare-event
sampling and active-learning candidate generation. Each window is equilibrated
before its production trajectory begins. This implementation does not perform
replica exchange or reconstruct a free-energy profile.

The reaction-coordinate motivation follows Bucko's discussion of finite-
temperature reaction barriers and constrained molecular dynamics in
[Bucko (2008)](https://doi.org/10.1088/0953-8984/20/6/064211). That work uses an
SN2 model reaction and thermodynamic integration; the runnable example below
instead uses H2O dissociation on Ni(111) and stops at biased trajectory
generation.

## H2O dissociation on Ni(111)

The example uses xreac with the bundled `ffield.reax.PtNiCHO.2016` parameters.
Install the ReaxFF extra first:

```shell
python -m pip install -e '.[reax]'
```

```{literalinclude} ../../../examples/exploration/umbrella_sampling/ni111_water.yaml
:language: yaml
```

Run it from the repository root:

```shell
gdp -d run-umbrella explore \
    examples/exploration/umbrella_sampling/ni111_water.yaml
```

The distance group must select exactly two atoms. Centers retain their declared
order. For every center, GDPy assigns the supplied seed whose initial
minimum-image distance is nearest to that center. Each replica removes seed
momenta and receives deterministic, independently persisted MD seeds.

The base runtime must be an unbiased ASE MD runtime. The exploration adds the
`distance_harmonic` modifier itself. Its `steps` setting is the production
length; `strategy.equilibration_steps` controls the preceding equilibration.

## Outputs and restart

Replica calculations use:

```text
windows/w000/r000/equilibration/
windows/w000/r000/production/
```

`windows.json` records the center, selected input seed, replica, RNG seeds, and
directory for every calculation. Rerunning the same command resumes those
calculations without redrawing seeds. A changed configuration requires a new
output directory.

Only production workers are exposed downstream. Every production frame stores
`umbrella_center`, `umbrella_kspring`, `umbrella_window`, `umbrella_replica`,
and `umbrella_seed_index`. `samples.csv` summarizes the actual distance, bias
energy, and committee force deviation when available.

## Active-learning selection

Use a committee potential as the base potential, extract the production
workers, and select uncertain structures independently for each center:

```yaml
resources:
  umbrella:
    __type__: exploration
    options:
      method: umbrella_sampling
      random_seed: 7
      system:
        builder:
          method: read_stru
          fname: structure.xyz
        collective_variable:
          method: distance
          group: "`index 8 10`"
      strategy:
        centers: [0.95, 1.15, 1.35, 1.55]
        kspring: 5.0
        replicas: 2
        equilibration_steps: 500

  uncertainty:
    __type__: selector
    options:
      selection:
        method: property
        group_by: info umbrella_center
        name: max_devi_f
        sparsify:
          method: hist
          range: [0.05, 0.25]
          nbins: 20
        number: [16, 1.0]

steps:
  sample:
    __type__: explore
    inputs:
      exploration: umbrella
      runtime: committee_md

  trajectories:
    __type__: extract
    inputs:
      compute: sample

  candidates:
    __type__: select
    inputs:
      structures: trajectories
      selector: uncertainty
```

`max_devi_f` comes from the unbiased committee host calculator; the harmonic
restraint is an additive modifier. The trajectory energies and forces include
the bias and must not be used as training labels. Evaluate selected geometries
with the reference runtime before adding them to a training dataset. Dataset
updates and retraining remain responsibilities of the repeated workflow.
