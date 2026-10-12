(potential-prophet)=

# prophet

The `prophet` provider uses Kairos Materials' `prophet.KairosCalculator` to
evaluate energies, forces, and stresses with local Prophet checkpoints.

## Requirements

Python 3.11 or newer and Git are required. From the repository root:

```shell
python -m pip install -e '.[prophet]'
```

For the existing `gdp3` environment, run commands without activating it:

```shell
mamba run -n gdp3 python -m pip install -e '.[prophet]'
mamba run -n gdp3 python -c 'from prophet import KairosCalculator; print("Prophet import passed")'
```

The `prophet` extra installs ASE >=3.29, e3nn >=0.6, PyTorch >=2.7, and
torch_geometric >=2.7 through upstream's dependencies. The `prophet-kernel`
extra adds openequivariance >=0.6. See {doc}`../installation` for combined
CPU/CUDA installations and MACE's incompatible e3nn requirement.

The extra pins Prophet 1.0.1 to an upstream commit. Download a checkpoint such
as `prophet-oame-mbd.pt` from the model repository linked in the
[upstream README](https://github.com/kairosmaterial/prophet).

## Backends

| Potential backend | Executor | Default | Description |
| --- | --- | --- | --- |
| `ase` | `ase` | Yes | ASE-compatible calculator. |

## Configurations

### ase + ase

```yaml
potential:
  provider: prophet
  backend: ase  # Optional; default for the ase executor.
  parameters:
    model: ./models/prophet-oame-mbd.pt
    device: cpu
    use_kernel: false
executor:
  provider: ase
  method: spc
```

`model` is required and accepts an existing checkpoint path or a list of paths.
Paths are resolved absolutely for scheduler execution; the manager does not
download checkpoints. With multiple models, `estimate_uncertainty: true`
enables GDPy's committee calculator; otherwise only the first model is used.

An explicit `device` is respected. When omitted or null, GDPy selects CUDA
if available, otherwise CPU. `use_kernel` defaults to `false`, using the pure
e3nn path. To enable kernels, install upstream's optional `kernel` extra and
its required CUDA toolchain, set `device: cuda`, and set `use_kernel: true`.
Other upstream calculator options, including `use_compile`, pass through.

This provider loads the ordinary Prophet energy/force model. The separate
`prophet_spin.SpinCalculator` and magnetic-moment prediction bundle are not
interfaces of this provider.

(prophet-supported-cluster)=

## Supported cluster exploration and GPU acceleration

The {ref}`Cu₄/alumina basin-hopping example <bh-supported-nanoparticle-example>`
includes `examples/global_optimisation/runtimes/prophet_oame_mbd.yaml` with
the Prophet-OAME-MBD checkpoint, CPU inference, and the bottom substrate
planes fixed during local relaxation. The same runtime works with the GA
recipe on that page.

For the upstream accelerated CUDA path, install the kernel extra (a CUDA
toolchain with `nvcc` and headers is required for runtime JIT compilation):

```shell
mamba run -n gdp3 python -m pip install -e '.[prophet-kernel]'
```

In a copy of the runtime, retain the executor settings and replace the
potential parameters with:

```yaml
model: ./models/prophet-oame-mbd.pt
device: cuda
use_kernel: true
use_compile: true
```

Upstream recommends the CUDA kernels. Compilation is lazy and occurs on the
first evaluation, so measure steady-state inference after warmup separately
from startup. The best timing for a short search depends on hardware and
compilation overhead; these configurations do not establish a measured speed
ranking. See the [upstream calculator implementation](https://github.com/kairosmaterial/prophet/blob/62bc2769fa83214fd564c7df263ee254e73fc6e9/prophet/calculator.py).

In Prophet 1.0.1, `use_compile: true` activates tracing only when CUDA is
available. It does not enable a compiled CPU inference path in the measured
macOS environment.

(prophet-cpu-timing)=

## CPU timing on the supported cluster

Prophet-OAME-MBD took **156.763 s (156,763 ms)** per uncached warmed
energy-and-force evaluation (median of two samples) of the 184-atom Cu₄/alumina fixture
`examples/global_optimisation/assets/cu4_alumina111_supported.xyz`.
The warmed samples took 172.832 s and 140.694 s; the preceding warmup took
205.159 s. All three returned finite energies and forces.
Model loading is excluded from these times. The calculator also computes
stress internally on this periodic structure.

This was a shorter check: **one warmup and two warmed samples**, rather than
the three warmups and ten samples in the shared CPU comparison. The longer
run was interrupted after the third completed evaluation. Treat this as a
timing estimate, not a ten-sample median or a complete relaxation time.
No GPU timing was measured.

The environment was macOS 26.6 ARM, Python 3.12.15 (`gdp3`), Prophet 1.0.1,
PyTorch 2.11.0, ASE 3.29.0, e3nn 0.6.0, torch_geometric 2.8.0.post1, and
matscipy 1.3.0. PyTorch intra-op and inter-op thread counts were both one;
`device=cpu`, `use_kernel=False`, and `use_compile=False`. Each evaluation
reset the ASE cache and perturbed the original positions with independent
Gaussian displacements of standard deviation 0.001 Å (seed 20261012).

Recorded values and environment details are in
`benchmarks/prophet_supported_cluster/results.json`. To reproduce the short
protocol with your downloaded checkpoint, run from the repository root:

```shell
mamba run -n gdp3 python -u benchmarks/prophet_supported_cluster/benchmark.py \
    --model ./models/prophet-oame-mbd.pt \
    --warmups 1 --samples 2 --output /tmp/prophet-supported-cluster.json
```

Omit `--warmups` and `--samples` to use three warmups and ten timed samples.
On the measured CPU this could take tens of minutes.
