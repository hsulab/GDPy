#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Optional

import numpy as np


def compute_num_training_batches(
    cum_batchsizes: int,
    train_epochs: int = 200,
    print_epochs: int = 5,
    train_batches: Optional[int] = 200_000,
    min_freq_unit: int = 100,
) -> tuple[int, int]:
    """"""
    save_freq = int(np.ceil(cum_batchsizes * print_epochs / min_freq_unit) * min_freq_unit)

    # Currently, we check whether the training is fininished by steps in lcurve.out.
    # Thus, we need make sure the last step (numb_steps) is displayed in lcurve.out
    # by making numb_steps can be divided by disp_freq.

    numb_steps = cum_batchsizes * train_epochs
    num_checkpoints = int(np.ceil(cum_batchsizes * train_epochs / save_freq))
    numb_steps = num_checkpoints * save_freq

    # Check if the training steps are too small, which happens in the early stage of
    # active learning, and increase it to the default `training_batches`.
    # We observed the model accuracy increases nonlinearly with the dataset size,
    # which means we need a 'minimum' training steps even for an extremely small dataset
    # may have few tens of structures.
    if train_batches is not None and numb_steps < train_batches:
        num_chekpoints = int(np.ceil(train_epochs / print_epochs))
        save_freq = int(np.ceil(train_batches / num_chekpoints / min_freq_unit) * min_freq_unit)
        numb_steps = save_freq * num_chekpoints

    return numb_steps, save_freq


if __name__ == "__main__":
    ...
