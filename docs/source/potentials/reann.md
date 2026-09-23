(potential-reann)=

# reann

The `reann` provider loads REANN TorchScript models with gdpx’s ASE calculator.

## Installation

### Evaluating an exported model

From the repository root, install the optional REANN dependencies:

```shell
python -m pip install -e '.[reann]'
```

The extra installs PyTorch and `opt_einsum`. It does not install the upstream
REANN training code or model checkpoints. Training requires the separate
source setup described below.

To select CPU execution on Linux or Windows, install the CPU PyTorch build
first, then the extra:

```shell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[reann]'
```

On macOS, use `python -m pip install torch` instead of the CPU wheel index.
For NVIDIA GPU execution, install a CUDA-enabled PyTorch build using the
command from the [PyTorch installation selector](https://pytorch.org/get-started/locally/)
for your platform and driver before installing `.[reann]`. Choose a PyTorch version compatible with the
exported model; reproducing its training/export environment is a useful
starting point.

Supply an exported REANN TorchScript potential, typically `PES.pt`.
gdpx loads it directly with `torch.jit.load`; the upstream REANN Python source
is not needed for this inference path. gdpx uses ASE neighbour lists, so the
upstream Fortran neighbour-list extension does not need to be compiled.
A training checkpoint such as `REANN.pth` must be exported before use here.

Check the installation and GPU visibility:

```shell
python -c "from gdpx.providers.reann.calculators.reann import REANN; print('REANN calculator import OK')"
python -c "import torch; print(torch.__version__); print('GPU available:', torch.cuda.is_available())"
```

The provider selects CUDA when available and otherwise uses CPU. It does not
select Apple's MPS device. On a cluster, check GPU visibility inside a GPU
allocation using the same environment as the calculation.

### Training and exporting models

Obtain the upstream training code separately (Git is required). To use the
`v_1.0` tag:

```shell
git clone --branch v_1.0 --depth 1 https://github.com/zhangylch/REANN.git /path/to/REANN
```

This tag has no `setup.py` or `pyproject.toml`, so it must be set up from source
rather than installed as a pip dependency.

Replace `/path/to/REANN` with your chosen source directory. The `reann` extra
already includes PyTorch and `opt_einsum`, but you must set up the training
and export scripts following the [REANN documentation](https://github.com/zhangylch/REANN).
Use the README and manual from that checkout to select compatible dependency
versions and entry points; instructions on the upstream default branch may
describe a newer version.

When using gdpx's REANN trainer, configure `command` and `freeze_command` to
invoke your installed training and export scripts. Their defaults, `train`
and `freeze`, must resolve to suitable executables in the job environment;
installing gdpx does not create them. The trainer expects `REANN.pth` as the
training checkpoint and `PES.pt` as the exported potential.

The gdpx REANN provider supports the ASE interface. Its LAMMPS interface is
not implemented; upstream LAMMPS installation instructions apply to separate
use of REANN outside this provider.

## Configuration

```yaml
potential:
  provider: reann
  parameters:
    model: ./PES.pt
    type_list: [H, O]
    precision: float32
    compute_stress: false
    estimate_uncertainty: false
```

`type_list` is passed as the model’s atom-type ordering. `precision` must be
`float32` (default) or `float64`. `compute_stress` defaults to `false`; enable
it for calculations that require stress. CUDA is selected when available,
otherwise CPU.

`model` accepts one path or a list of existing paths. Multiple models with
`estimate_uncertainty: true` form a committee; otherwise only the first model
is evaluated.
