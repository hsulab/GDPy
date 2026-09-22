(potential-cp2k)=

# cp2k

The `cp2k` provider configures CP2K input templates and electronic settings.

## Requirements

Provide a CP2K executable, input template, and the basis/pseudopotential data referenced by the input.

## Configuration

```yaml
potential:
  provider: cp2k
  parameters:
    command: cp2k.psmp -i cp2k.inp -o cp2k.out
    template: ./cp2k-template.inp
    basis_set_file: /path/to/BASIS_MOLOPT
    potential_file: /path/to/GTH_POTENTIALS
```

`template` supplies the CP2K input text. When present, the adapter leaves
`cutoff`, `max_scf`, and `xc` to the template unless explicitly supplied as
calculator parameters. Basis and potential file paths are resolved absolutely.

`interface: cp2k_shell` selects the shell calculator interface; supply a
shell-capable CP2K `command` for that interface. The default interface is `cp2k`.
