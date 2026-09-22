(potential-vasp)=

# vasp

The `vasp` provider configures VASP electronic settings and pseudopotentials.

## Requirements

Provide a VASP executable, pseudopotential directory, and INCAR. The ASE interface additionally requires `vasp_interactive`.

## Configuration

```yaml
potential:
  provider: vasp
  parameters:
    command: mpirun -n 32 vasp_std
    pp_path: /path/to/potentials
    incar: ./INCAR
    kpts: [1, 1, 1]
```

Additional parameters are passed to the ASE VASP calculator;
`vdw_path` sets the van der Waals kernel directory when needed.

The optional `interface` parameter can explicitly select
`vasp_interactive` or `vasp_interactive_disp`. The latter requires `dftd3.ase`
and a `dispersion` mapping under `potential.parameters`, for example:

```yaml
dispersion:
  type: dftd3
  method: PBE
  damping: d3bj
```
