Trainers
========

Related Commands
----------------

.. code-block:: shell

    # - train a mode in the current directory
    $ gdp train ./config.yaml

    # - explore configuration space defined by `config.yaml` 
    #   training outputs will be written to the `m0` folder
    #   a log file will be written to `m0/gdp.out` as well
    $ gdp -d m0 train ./config.yaml

Configuration File
------------------

The `config.yaml` requires two sections `dataset` and `trainer` to define a training process as

.. code-block:: yaml

    dataset:
      name: xyz
      dataset_path: ./dataset
      train_ratio: 0.9
      batchsize: 16
    trainer:
      name: deepmd
      command: dp
      freeze_command: dp
      config: ./dpconfig.json
      type_list: ["H", "O"]
      train_epochs: 100
      random_seed: 1112

In the `dataset` section, a `dataset` is defined. `gdp` will load the structures in the 
dataset and convert to the proper format required by the trainer.

- name:             Dataset format. (Only `xyz` is supported now.)
- dataset_path:     Dataset filepath.
- train_ratio:      Train-valid-split ratio.
- batchsize:        Training batchsize.

In the `trainer` section, a `trainer` is defined. The parameters related to the model 
architecture is defined in `config`, which may be different by models. 
`gdp` will automatically update some parameters in the `config`, which include 
the training dataset section and training epochs.

For example, if one is training `deepmd`, `training.training_data` and `training.validation_data` 
in the `./dpconfig.json` can be left empty. `gdp` will convert `dataset` into deepmd-format and update the file path. 
Moreover, `deepmd` uses `numb_steps` instead of `epochs`. `gdp` will compute the 
number of batches based on the input dataset and multiply it with `train_epochs` to give 
the value of `numb_steps`.

- name:           Trainer target.
- commad:         Command to train.
- freeze_command: Command to freeze/deploy the trained model.
- config:         Model architecture configuration.
- type_list:      Type list of the model.
- train_epochs:   Number of training epochs.
- random_seed:    Random number generator to generate random numbers in the training.

Use Scheduler
-------------

If the training is too time-consuming, one can use `gdp session` to access a workflow that 
defines a training operation. See instructions in the `Session` section.
