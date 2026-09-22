(ga-operations)=

# operations

GA operations control how GDPy recognises duplicate structures, combines
parents, and modifies offspring. Population comparison is configured under
`population.comparator`; `operators` contains crossover and mutation. See the
shared {ref}`global-optimisation-population` reference.

```yaml
population:
  comparator:
    method: interatomic_distance
operators:
  crossover:
    method: cut_and_splice
  mutation:
    - method: rattle
      probability: 1.0
    - method: cluster_rotation
      probability: 0.5
```

When several mutations are configured, `probability` gives their relative selection
weights. Builder-derived values such as minimum bond distances, the substrate,
and the number of optimised atoms are supplied to compatible operations
automatically. Periodicity and fragment preservation are configured once as
`population.periodic` and `population.preserve_fragments`. Both default to
`true`; set either value explicitly to `false` when the searched system or its
operations require it.

The standard GA operation interfaces are GDPy-owned implementations inspired
by the algorithms and configuration surface in ASE-GA 1.0.3. They use explicit
NumPy `Generator` streams; GDPy does not import the legacy `ase.ga` package at
runtime.

```{toctree}
:maxdepth: 2
:titlesonly:
:includehidden:

operations/comparators/index
operations/crossovers/index
operations/mutations/index
```
