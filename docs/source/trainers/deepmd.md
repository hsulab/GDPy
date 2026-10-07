# deepmd

## Installation

Install the optional DeepMD dependencies from the repository root:

```shell
# Choose the major version matching your training configuration:
python -m pip install -e '.[deepmd2]'
# Or, in a separate environment:
python -m pip install -e '.[deepmd3]'
```

This includes `deepmd-kit` and `dpdata`, which the trainer uses to convert
structures into DeepMD datasets. Install a matching backend following
{doc}`../potentials/deepmd` and ensure the `dp` command is available in the
training environment. DPA4 requires the DeepMD 3 PyTorch backend; install the
`deepmd3-torch` extra or use one of the complete DeepMD 3 commands in
{doc}`../installation`.

## Configuration

**gdp** converts structures into the DeepMD format stored in two folders `train`
and `valid` based on `dataset` and writes a training configuration `deepmd.json`.
The default training command is `dp train deepmd.json`. For a DPA4
configuration, the trainer selects the PyTorch backend and runs
`dp --pt train deepmd.json`. DPA4C selects the exportable PyTorch backend
and runs `dp --pt-expt train deepmd.json`.

Some parameters in the `deepmd.json` will be filled automatically by **gdp**.
training.training_data and training.validation_data will be the folder paths generated
by **gdp**. Moreover, deepmd uses numb_steps instead of epochs. **gdp** will compute
the number of batches based on the input dataset and multiply it with `train_epochs`
to give the value of `numb_steps`.

See [DEEPMD] doc for more info about configuration parameters. Example Configuration:

```yaml
dataset:
  name: xyz
  dataset_path: ./dataset
  train_ratio: 0.9
  batchsize: 16
  random_seed: 1112
trainer:
  provider: deepmd
  method: default
  parameters:
    config: ./dpconfig.json
    type_list: ["H", "O"]
    train_epochs: 10
    random_seed: 1112
init_model: ../model.ckpt
```

:::{note}
Deepmd Trainer in **gdp** supports a `init_model` keyword that allows one to
initialise model parameters from a previous checkpoint. This is useful when
training models iteratively in an active learning loop.
:::

## Run the training examples

The repository includes small CPU examples for `se_e2_a`, DPA4, and DPA4C
under `examples/training/`. See the [training examples README](https://github.com/hsulab/GDPy/blob/main/examples/training/README.md)
for the configuration provenance, backend requirements, and dataset layout.

Use the existing `gdp3` mamba environment. From the repository root, generate
24 periodic Cu3Au1 structures labeled with ASE EMT energies, forces, and virials:

```shell
mamba run -n gdp3 python examples/training/prepare.py

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export DP_INTRA_OP_PARALLELISM_THREADS=2
export DP_INTER_OP_PARALLELISM_THREADS=1
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1

(cd examples/training/se_e2_a && mamba run -n gdp3 gdp -d _train train train.yaml)
(cd examples/training/dpa4 && mamba run -n gdp3 gdp -d _train train train.yaml)
(cd examples/training/dpa4c && mamba run -n gdp3 gdp -d _train train train.yaml)
```

Run from the model directories to resolve relative paths. Each `train.yaml`
uses `../dataset`, batch size 4, train ratio 0.9, seed 1112, and:

```yaml
trainer:
  provider: deepmd
  method: default
  parameters:
    config: ./dpa4.json
    train_epochs: 10
    print_epochs: 1
    train_batches: null
    random_seed: 1112
```

`train_batches: null` disables the default 200,000-step minimum. GDPy's
checkpoint/display frequency rounding makes these examples run for 100
steps. They demonstrate the workflow and do not establish model accuracy.
For real training, replace the labeled dataset, keep `type_map` consistent
with it, and choose an appropriate duration and validation split.

The `se_e2_a` example uses plain `dp` commands, whose default backend is
TensorFlow, and exports `_train/deepmd-c.pb`. DPA4 automatically selects
`dp --pt` and exports the
latest `model.ckpt.pt` checkpoint as `_train/deepmd.pt2`, without compression.
DPA4C automatically selects `dp --pt-expt`, freezes with `--lower-kind graph`,
and compresses to `_train/deepmd-c.pt2`. The `.pt2` archives target the device
type used during export, so freeze on the CPU or GPU type that will run inference.

The DPA4 and DPA4C examples adapt the supplied OMat24 Mini architectures to Cu3Au1
and CPU execution. They do not require OMat24 statistics files or pretrained
weights. AMP, training compilation, TF32, EMA, and distributed training are
disabled. The `.pt2` export still compiles an AOTInductor package and can take
several minutes. The DPA4C CPU example also enables
fitting-network residual timesteps (`resnet_dt: true`) to use general graph
compression, because the compact export path requires an operator unavailable
in the `gdp3` CPU build. Its fitting parameterization therefore differs from
the supplied OMat24 configuration.

To fine-tune DPA4 or DPA4C instead of training from scratch, add a local
PyTorch checkpoint to the top level of `train.yaml`:

```yaml
init_model: ./pretrained-model.pt
```

The trainer passes this checkpoint to DeepMD with `--finetune`. Keep the
checkpoint's element ordering and compatible architecture when fine-tuning.

[deepmd]: https://docs.deepmodeling.com/projects/deepmd/en/latest/
