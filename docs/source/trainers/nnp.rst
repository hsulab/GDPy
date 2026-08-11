NumPy NNP
=========

The ``nnp`` trainer is a lightweight Behler--Parrinello-style potential using
ACSF descriptors and one atomic-energy network for each central element.  It
requires energies and, when ``force_weight`` is nonzero, forces on every input
structure.

Training options
----------------

The main options under ``trainer.config`` are:

``n_epochs``
    Maximum number of passes through the training set.
``learning_rate``
    Adam learning rate.
``energy_weight`` and ``force_weight``
    Relative task priorities.  With ``gradient_balance: true``, the raw task
    gradients are normalized before these priorities are applied.
``batch_size``
    Number of structures per Adam update.  Omitting it uses the full training
    set.
``validation_fraction``
    Seeded holdout fraction.  The default is zero, so all structures train the
    model.
``early_stopping_patience``
    Stop and restore the best validation checkpoint after this many epochs
    without improvement.  It has an effect only with a validation split.
``max_grad_norm``
    Optional norm limit applied after combining energy and force gradients.

Descriptors and their coordinate Jacobians are cached before optimization.
``train_config.json`` records preprocessing time and memory, while
``training_history.json`` contains per-epoch errors, gradient norms, and
timings.

Model format
------------

The current ``nn_weights.npz`` format is version 2.  It stores per-element
networks, descriptor normalization, and fitted atomic energy offsets.  Models
created by the earlier shared-network format must be retrained.

For meaningful validation, avoid splitting adjacent frames from one
trajectory between training and validation.  Use decorrelated configuration
groups covering the intended compositions, temperatures, strain states,
defects, and high-force environments.
