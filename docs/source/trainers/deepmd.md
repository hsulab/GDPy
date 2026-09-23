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
training environment.

## Configuration

**gdp** converts structures into the deepmd format stored in two folders `train`
and `valid` based on `dataset` and writes a training configuration `deepmd.json`.
The training will be performed by `dp train deepmd.json`.

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

[deepmd]: https://docs.deepmodeling.com/projects/deepmd/en/latest/
