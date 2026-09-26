(potential-espresso)=

# espresso

The `espresso` provider configures Quantum ESPRESSO from a PW input template.

## Requirements

Provide `pw.x`, a compatible ASE Espresso interface, a PW input template, and the pseudopotentials named in the configuration.

## Backends

| Potential backend | Executor | Default | Description |
| --- | --- | --- | --- |
| `espresso` | `ase` | Yes | Quantum ESPRESSO calculator driven by ASE. |

Backend defaults depend on the executor. The comments in each configuration
show whether `potential.backend` can be omitted.

## Configurations

### espresso + ase

```yaml
potential:
  provider: espresso
  backend: espresso  # Optional; default for the ase executor.
  parameters:
    command: pw.x -in PREFIX.pwi > PREFIX.pwo
    template: ./espresso.pwi
    pp_path: /path/to/pseudopotentials
    pp_name:
      Si: Si.UPF
    kpts: [2, 2, 2]
executor:
  provider: ase
  method: spc
```

`pp_path` must exist, `pp_name` must map elements to pseudopotential filenames,
and `template` must name an existing input file (default `./espresso.pwi`).
The template supplies the electronic input parameters. Ensure its `pseudo_dir`
points to the correct data directory; `pp_path` is checked by the manager but
is not itself passed to the calculator as `pseudo_dir`.

Set either `kpts` or `kspacing`, never both. `koffset` defaults to zero.
This adapter uses the command-based Espresso interface, so ASE
API compatibility must be checked in the environment used to execute it.
