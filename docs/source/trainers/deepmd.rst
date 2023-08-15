deepmd
======

.. warning:: 

    This trainer requires an extra package `dpdata`. Use `conda install dpdata -c deepmodeling` to 
    install it.

**gdp** converts structures into the deepmd format stored in two folders `train` 
and `valid` based on `dataset` and writes a training configuration `deepmd.json`. 
The training will be performed by `dp train deepmd.json`.

Some parameters in the `deepmd.json` will be filled automatically by **gdp**. 
training.training_data and training.validation_data will be the folder paths generated 
by **gdp**. Moreover, deepmd uses numb_steps instead of epochs. **gdp** will compute 
the number of batches based on the input dataset and multiply it with `train_epochs`
to give the value of `numb_steps`.

See DEEPMD_ doc for more info about configuration parameters. Example Configuration:

.. _DEEPMD: https://docs.deepmodeling.com/projects/deepmd/en/master/index.html

.. code-block:: yaml

    dataset:
      name: xyz
      dataset_path: ./dataset
      train_ratio: 0.9
      batchsize: 16
      random_seed: 1112
    trainer:
      name: deepmd
      config: ./dpconfig.json
      type_list: ["H", "O"]
      train_epochs: 10
      random_seed: 1112
    init_model: ../model.ckpt

.. note::

    Deepmd Trainer in **gdp** supports a `init_model` keyword that allows one to 
    initialise model parameters from a previous checkpoint. This is useful when 
    training models iteratively in an active learning loop.
