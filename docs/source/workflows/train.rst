Train
=====

We can access the training by a `train` operation. This operation accepst four input 
variables and forwards a `potter` (AbstractPotentialManager) object. 

For the input variables,

- potter:

    The potential manager. See :ref:`Potential Examples` for more details.

- dataset:

    The dataset.

- trainer:

    The trainer configuration that defines the commands and the model configuration.

- scheduler:

    Any gdp-supported scheduler. In general, the training needs a GPU-scheduler.

.. note::

    The name in `potter` and `trainer` should be the same.

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
      dpmd:
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
        potter: ${vx:dpmd}
        dataset: ${vx:dataset}
        trainer: ${vx:trainer}
        scheduler: ${vx:scheduler_gpu}
        size: 4
    sessions:
      _ensemble: train
