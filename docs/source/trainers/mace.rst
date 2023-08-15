mace
====

**gdp** writes `./_train.xyz` and `./_test.xyz` into the training directory based on 
`dataset` and generates a command line based on `trainer`.

Notice some parameters are override by **gdp** based on the `dataset` and the `trainer` 
parameters. The `trainer.config` section will be converted to a command line as 
`python ./run_train.py --name='MACE_model' ...`, which is the current training command 
supported by MACE.

- seed:           Override by `trainer.seed`
- max_num_epochs: Override by `trainer.train_epochs`.
- batch_size:     Override by `dataset`.
- train_file:     Override as `./_train.xyz`
- valid_file:     Override as `./_test.xyz`
- valid_fraction: Always 0.
- device:         Automatically detected (either cpu or cuda). No Apple Silicon!
- config_type_weights: Must be a string instead of a dictionary.

.. note::

    Train set are data used to optimise model parameters. Validation set are data 
    that helps us monitor the training progress and decide to save the model at which 
    epoch. Test set are data that neither are trained nor affect our decision on the 
    model. Some training simplifies these complex concepts and just use one `test` set 
    for both the validation and the test purposes.

See MACE_ doc for more info about configuration parameters. Example Configuration:

.. _MACE: https://github.com/ACEsuit/mace

.. code-block:: yaml

    dataset:
      name: xyz
      dataset_path: ./dataset
      train_ratio: 0.9
      batchsize: 16
      random_seed: 1112
    trainer:
      name: mace
      command: python ./run_train.py
      config: # This section can be put into a separate file e.g. `./config.yaml`
        name: MACE_model
        valid_fraction: 0.05
        config_type_weights: '{"Default": 1.0}'
        E0s: {1: -12.6261, 8: -428.5812}
        model: MACE
        default_dtype: float32
        hidden_irreps: "128x0e + 128x1o"
        r_max: 4.0
        swa: true
        start_swa: 10
        ema: true
        ema_decay: 0.99
        amsgrad: true
        restart_latest: true
      type_list: ["H", "O"]
      train_epochs: 10
      random_seed: 1112

.. warning::

    If one uses `swa`, **gdp** will not check if `start_swa` is smaller than 
    `max_num_epochs`. If `start_swa` is larger than `max_num_epochs`, there will 
    be an error when saving the model.