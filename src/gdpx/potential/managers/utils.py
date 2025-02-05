#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import Union


def canonicalise_input_models(model: Union[str, list[str]]) -> list[str]:
    """Convert input models to a list of resolved path strings.

    Args:
        model: The input model(s).

    Returns:
        The resolved path strings. An error is raised if the model does not exist.

    """
    if not isinstance(model, list):
        assert isinstance(model, str)
        model_ = [model]
    else:
        model_ = model

    models = []
    for m in model_:
        m = pathlib.Path(m).resolve()
        if not m.exists():
            raise FileNotFoundError(f"The model {str(m)} does not exist.")
        models.append(str(m))

    return models


if __name__ == "__main__":
    ...
