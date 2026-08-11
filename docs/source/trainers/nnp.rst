NumPy NNP
=========

The ``nnp`` trainer is a lightweight Behler--Parrinello-style potential using
ACSF descriptors and one atomic-energy network for each central element.  It
uses a DeepMD-style scheduled energy and force loss.

Training options
----------------

The main options under ``trainer.config`` are:

``n_epochs``
    Maximum number of passes through the training set.
``learning_rate.start`` and ``learning_rate.stop``
    Endpoints of the smooth exponential Adam learning-rate schedule.
``loss.start_pref_e`` and ``loss.limit_pref_e``
    Energy-loss prefactors at the start and low-learning-rate limit.
``loss.start_pref_f`` and ``loss.limit_pref_f``
    Force-loss prefactors at the start and low-learning-rate limit.  Force
    labels are required when either value is nonzero.
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

For optimizer step ``t``, each prefactor is coupled to the learning rate by

.. math::

    p(t) = p_0 \frac{lr(t)}{lr(0)}
         + p_\infty \left(1 - \frac{lr(t)}{lr(0)}\right).

The component losses are :math:`(E-E^*)^2/N` and the mean squared Cartesian
force error.  Gradients are multiplied directly by the scheduled prefactors;
they are not independently normalized.

Descriptors and their coordinate Jacobians are cached before optimization.
``train_config.json`` records preprocessing time and memory, while
``training_history.json`` contains per-epoch errors, gradient norms, and
timings.  It reports raw and prefactored gradient norms plus their cosine
similarity so conflicts between energy and force fitting remain visible.  The
prefactored norms are measured before optional ``max_grad_norm`` clipping.

Model format
------------

The current ``nn_weights.npz`` format is version 2.  It stores per-element
networks, descriptor normalization, and fitted atomic energy offsets.  Models
created by the earlier shared-network format must be retrained.

For meaningful validation, avoid splitting adjacent frames from one
trajectory between training and validation.  Use decorrelated configuration
groups covering the intended compositions, temperatures, strain states,
defects, and high-force environments.
