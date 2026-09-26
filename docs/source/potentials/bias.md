(potential-bias)=

# bias

The `bias` provider exposes a built-in bias as a standalone ASE calculator.

## Requirements

The built-in distance restraint uses gdpx and its base scientific dependencies; no external model is required.

## Backends

| Potential backend | Executor | Default | Description |
| --- | --- | --- | --- |
| `ase` | `ase` | Yes | ASE-compatible calculator. |

Backend defaults depend on the executor. The comments in each configuration
show whether `potential.backend` can be omitted.

## Configurations

### ase + ase

```yaml
potential:
  provider: bias
  backend: ase  # Optional; default for the ase executor.
  parameters:
    method: distance_harmonic
    group: "`index 0 1`"
    center: 1.5
    kspring: 0.1
executor:
  provider: ase
  method: spc
```

The example applies a harmonic distance restraint to the two atoms selected by
a group expression. The `index` selector uses zero-based atom indices. An
explicit two-index list such as `[0, 1]` remains supported. `center` is the
target distance in Å and `kspring` is the spring constant in eV/Å². These
values must be floating-point numbers.
`parameters.method` selects an entry from gdpx’s bias registry.

As a potential, this evaluates only the bias contribution. See
{doc}`../computations/index` for adding restraints to a physical potential.
Other built-in bias methods are declared in `gdpx.modifiers.bias.REGISTER`.

For a complete physical-potential example, see the {ref}`H2O/Ni(111)
ReaxFF trajectory <compute-ni-water-restraint-example>`, which places
`distance_harmonic` in the runtime's `modifiers` list.
