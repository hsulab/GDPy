(potential-tace)=

# tace

The `tace` provider loads TACE checkpoints and foundation models.

## Requirements

From the repository root (Git is required):

```shell
python -m pip install -e '.[tace]'
```

The extra installs the upstream commit pinned in `pyproject.toml`. See
{doc}`../installation` for the recommended combined GPU installation and
verification.

## Backends

| Potential backend | Executor | Default | Description |
| --- | --- | --- | --- |
| `ase` | `ase` | Yes | ASE-compatible calculator. |

Backend defaults depend on the executor. The comments in each configuration
show whether `potential.backend` can be omitted.

## Configurations

### ase + ase

```yaml
schema_version: 3
potential:
  provider: tace
  backend: ase  # Optional; default for the ase executor.
  parameters:
    model: TACE-OAM-7M
    precision: float32
    device: cpu
    fidelity_idx: 0
executor:
  provider: ase
  method: spc
```

`model` is required and accepts an exact upstream foundation name, an existing
checkpoint path, or a list of either. Local paths are resolved absolutely;
foundation names remain portable names in the configuration. TACE downloads
named checkpoints on first use into `~/.cache/tace/`. Subsequent runs reuse them.
For an offline run, supply a previously downloaded checkpoint path.

`precision` defaults to `float32`; an explicit upstream `dtype` takes precedence.
An explicit `device` is respected. When omitted, gdpx uses CUDA if available,
otherwise CPU. `fidelity_idx` selects the checkpoint's fidelity head; omission
retains its stored default. The example explicitly selects head 0.
Other upstream calculator options, including `neighborlist_backend`, pass through
to TACE. CUDA acceleration packages are not part of this extra.

With multiple models, `estimate_uncertainty: true` enables gdpx's committee
calculator; otherwise only the first model is evaluated. Checkpoint loading
uses upstream's EMA policy. Invalid checkpoints and download errors retain
their original exception rather than being reported as missing installations.

See {ref}`bh-cuox-tace-example` for the Cu₄O₄ basin-hopping example and measured
7M model timings. Upstream documents the available models in its
[foundation registry](https://github.com/xvzemin/tace/blob/90e241bc9c74f7ed5c1e0be42fe7aee4bf5e9896/tace/foundations/download_link.py)
and the [ASE interface](https://tace.readthedocs.io/en/latest/guide/ase.html).

## Integration test

The ordinary tests do not download models. To run real checkpoint loading,
energy/force comparisons against upstream, and minimisation:

```shell
GDPX_TEST_TACE=1 OMP_NUM_THREADS=1 python -m pytest -q tests/providers/test_tace_integration.py
```
