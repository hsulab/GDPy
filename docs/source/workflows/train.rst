Train
=====

We can access the training by a `train` operation. This operation accepst four input 
variables and forwards a `potter` (AbstractPotentialManager) object. 

For the input variables,

- potter:

    The potential manager. See :ref:`Potential Examples` for more details.

- dataset:

    The dataset. See :ref:`Trainers` for more details.

- trainer:

    The trainer configuration that defines the commands and the model configuration.

- scheduler:

    Any scheduler. In general, the training needs a GPU-scheduler.

.. note::

    The name in `potter` and `trainer` should be the same.

Extra parameters,

- size:

    Number of models trained at the same time. This is useful when a committee needs 
    later for uncertainty estimation.

- init_models:

    A List of model checkpoints to initialise model parameters. 
    The number should be the same as size.

Session Configuration
---------------------

.. code-block:: yaml

    variables:
      dataset:
        type: dataset
        name: xyz
        dataset_path: ./dataset
        train_ratio: 0.9
        batchsize: 16
        # random_seed: 1112 # Set this if one wants to reproduce results
      potter:
        type: potter
        name: deepmd
        params:
          backend: lammps
          command: "lmp -in in.lammps 2>&1 > lmp.out"
          type_list: ["H", "O"]
      trainer:
        type: trainer
        name: deepmd
        command: dp
        config: ${json:./config.json}
        train_epochs: 500
        # random_seed: 1112 # Set this if one wants to reproduce results
      scheduler_gpu:
        type: scheduler
        backend: slurm
        partition: k2-gpu
        time: "6:00:00"
        ntasks: 1
        cpus-per-task: 4
        mem-per-cpu: 4G
        gres: gpu:1
        environs: "conda activate deepmd\n"
    operations:
      train:
        type: train
        potter: ${vx:potter}
        dataset: ${vx:dataset}
        trainer: ${vx:trainer}
        scheduler: ${vx:scheduler_gpu}
        size: 4
        init_models:
          - ./model.ckpt
    sessions:
      _train: train
