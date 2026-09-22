(potential-bias)=

# bias

The `bias` provider exposes a built-in bias as a standalone ASE calculator.

## Requirements

The built-in distance restraint uses gdpx and its base scientific dependencies; no external model is required.

## Configuration

```yaml
potential:
  provider: bias
  parameters:
    method: distance_harmonic
    group: [0, 1]
    center: 1.5
    kspring: 0.1
```

The example applies a harmonic distance restraint to two zero-based atom
indices. `center` is the target distance in Å and `kspring` is the spring
constant in eV/Å². These values must be floating-point numbers.
`parameters.method` selects an entry from gdpx’s bias registry.

As a potential, this evaluates only the bias contribution. See
{doc}`../computations/index` for adding restraints to a physical potential.
Other built-in bias methods are declared in `gdpx.modifiers.bias.REGISTER`.
