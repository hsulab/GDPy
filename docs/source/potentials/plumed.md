(potential-plumed)=

# plumed

The `plumed` provider exposes gdpx’s PLUMED bias calculator through the ASE materializer.

## Requirements

Install the PLUMED Python bindings and a compatible PLUMED kernel library.

## Configuration

```yaml
potential:
  provider: plumed
  parameters:
    inp: ./plumed.inp
    kT: 0.02585
    use_charge: false
    update_charge: false
```

`inp` accepts an existing file (default `./plumed.inp`) or a list of PLUMED
input lines. File input has comments and blank lines removed. `kT` supplies
thermal energy in ASE energy units; the adapter default is `1.0`, so set it
explicitly for the intended temperature. Charge options default to `false`.

This calculator supplies only the bias contribution. Combining it with a
physical host potential requires an explicit integration; see
{doc}`../computations/index`.
