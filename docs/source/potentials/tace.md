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

(bh-supported-nanoparticle-tace-example)=

## User-provided model for supported clusters

For the {ref}`supported-cluster exploration <bh-supported-nanoparticle-example>`,
supply a TACE checkpoint supporting Cu, Al, and O. Place it at
`./models/tace-omat24-7m.pt`, or update `potential.parameters.model` in your
runtime. This configuration loads the local file directly without downloading
weights or requiring a DeepMD export. It uses fidelity head 0 and requests
only the energy and forces needed by the minimizer:

```{literalinclude} ../../../examples/global_optimisation/runtimes/tace_omat24_7m.yaml
:language: yaml
```

This runtime can also be used for other systems supported by the checkpoint.
Add system-specific constraints to your own runtime copy when needed.

## CPU inference and profiling

| TACE-OMat24-7M path | Time per energy-and-force evaluation |
| --- | --- |
| Standard ASE, float32, head 0 | 10,312.2 ms |
| Compiled ASE with scalar code generation | 6,104.7 ms |

See {ref}`potential-cpu-inference-comparison` for the shared protocol and
other providers. Both paths used the default matscipy neighbor list;
optional specialized acceleration operators were disabled.

A lightweight profile of TACE-OMat24-7M located the cost in model evaluation
and force differentiation. Its 6.94-million-parameter model used a 6 Å
cutoff and 12,864 directed neighbor edges for this 184-atom structure.
Graph construction took about 2.4 ms in an unprofiled stage measurement,
while the model took about 11.4 s.

In the profiled evaluation, the force-derivative routine accounted for about
8.0 s of an 11.5 s run, including 3.9 s in autograd's backward engine. The
forward path also spent about 1.5 s in 481 `einsum` calls, 0.7 s in scatter
operations, and 0.4 s in tensor contractions. These timings apply to the
uncompiled reference implementation on one CPU thread. TACE's
[documented acceleration workflow](https://tace.readthedocs.io/en/latest/guide/acceleration.html)
uses specialized operators and compiled inference on NVIDIA GPUs; that path
was not measured here. See the {ref}`shared CPU comparison <potential-cpu-inference-comparison>`
for the benchmark protocol and results from other providers.

### CPU compilation

The installed TACE 0.2.2 supports `enable_compile=True`. A compilation test in
`gdp3` reached PyTorch Inductor but failed when Clang compiled the generated
ARM CPU kernel:
`decltype(tmp56)::blendv(...)` was emitted with `tmp56` typed as scalar `float`.
This is a generated C++ compilation failure observed with PyTorch 2.11.0 and
Python 3.12, rather than evidence that TACE prohibits CPU compilation.

Disabling Inductor CPU vectorization before constructing the calculator allowed
the same checkpoint to compile:

```python
import torch._inductor.config as cfg
from tace.interface.ase import TACEAseCalc

cfg.cpp.simdlen = 1

calc = TACEAseCalc(
    model="./models/tace-omat24-7m.pt",  # User-provided checkpoint
    device="cpu",
    dtype="float32",
    fidelity_idx=0,
    target_property=["energy", "forces"],
    enable_compile=True,
)
```

This uses a private PyTorch configuration setting and is a workaround for this
specific environment, rather than a portable TACE runtime requirement. The
first evaluation includes tracing and compilation and must be excluded from
steady-state timings.

The first call took about 161 s including compilation. After three warmups,
the median of ten energy-and-force evaluations was **6,104.7 ms**, about
1.7 times faster than the uncompiled TACE path. Every evaluation returned
finite energy and forces. It remained about 16.3 times slower than compiled
MACE-OMAT-0 Small and 209 times slower than the DPA4C-mini export on this
structure.
