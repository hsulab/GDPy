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
`dp --pt train deepmd.json`.

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

## Train a DPA4 model

The repository includes a DPA4-Mini template in
`examples/training/dpa4/`, adapted from the DeepMD-kit 3.2.0 water example.
Copy its two configuration files into a working directory:

```shell
cp examples/training/dpa4/dpa4.json .
cp examples/training/dpa4/train.yaml .
```

Keep the model's `type_map` consistent with the labeled structures. **gdp**
replaces the training and validation system paths, batch sizes, step count,
display frequency, checkpoint frequency, and random seeds in this template.

The included `train.yaml` contains:

```yaml
dataset:
  name: xyz
  dataset_path: ./dataset
  train_ratio: 0.9
  batchsize: 4
  random_seed: 1112
trainer:
  provider: deepmd
  method: default
  parameters:
    config: ./dpa4.json
    train_epochs: 10
    random_seed: 1112
```

Place labeled extended XYZ files below system directories whose names include
the composition, for example `dataset/water-H2O-molecule/set-000/data.xyz`.
The files must contain reference energies and forces. Run the training workflow
from the directory containing both configuration files:

```shell
gdp -d dpa4-train train train.yaml
```

The trainer detects `model.type: dpa4`, trains with `dp --pt`, and exports the
latest `model.ckpt.pt` checkpoint as `dpa4-train/deepmd.pt2`. DPA4 does not
support the standard DeepMD compression step. The `.pt2` archive targets the
device type used during export, so freeze it on the CPU or GPU type that will
run inference.

To fine-tune instead of training from scratch, add a local PyTorch checkpoint
to the top level of `train.yaml`:

```yaml
init_model: ./pretrained-dpa4.pt
```

For DPA4, the trainer passes this checkpoint to DeepMD with `--finetune`.

[deepmd]: https://docs.deepmodeling.com/projects/deepmd/en/latest/
