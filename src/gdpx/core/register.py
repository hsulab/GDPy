#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import logging
import warnings

from .. import config


class Register:

    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        if key is None:
            key = value.__name__
        if key in self._dict:
            warnings.warn(
                "Key %s already in registry %s." % (key, self._name), UserWarning
            )
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class."""

        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            return add(None, target)

        return lambda x: add(target, x)

    def __getitem__(self, key):
        if key not in self._dict:
            raise Exception(f"No {key} in {self._dict.keys()}")
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()

    def __repr__(self) -> str:
        """"""
        content = f"{self._name.upper()}:\n"

        keys = sorted(list(self._dict.keys()))
        nkeys = len(keys)
        ncols = 5
        nrows = int(nkeys / ncols)
        for i in range(nrows):
            content += ("  " + "{:<24s}" * ncols + "\n").format(
                *keys[i * ncols : i * ncols + ncols]
            )

        nrest = nkeys - nrows * ncols
        if nrest > 0:
            content += ("  " + "{:<24s}" * nrest + "\n").format(*keys[nrows * ncols :])

        return content


class registers:

    #: Session operations.
    operation: Register = Register("operation")

    #: Session variables.
    variable: Register = Register("variable")

    #: Session placeholder
    placeholder: Register = Register("placeholder")

    #: Schedulers.
    scheduler: Register = Register("scheduler")

    #: Managers (Potentials).
    manager: Register = Register("manager")

    #: Trainers (Potential Trainers).
    trainer: Register = Register("trainer")

    #: Dataloaders (Datasets).
    dataloader: Register = Register("dataloader")

    #: Regions.
    region: Register = Register("region")

    #: Builders.
    builder: Register = Register("builder")

    #: Colvars.
    colvar: Register = Register("colvar")

    #: Modifiers.
    modifier: Register = Register("modifier")

    #: Reactors.
    reactor: Register = Register("reactor")

    #: Expeditions.
    expedition: Register = Register("expedition")

    #: Selectors.
    selector: Register = Register("selector")

    #: Describers.
    describer: Register = Register("describer")

    #: Comparators.
    comparator: Register = Register("comparator")

    #: Validators.
    validator: Register = Register("validator")

    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")

    @staticmethod
    def get(mod_name: str, cls_name: str, convert_name: bool = True, *args, **kwargs):
        """Acquire the target class from modules."""
        # - convert the cls_name by the internal convention
        if convert_name:
            # cls_name = cls_name.capitalize() + mod_name.capitalize()
            cls_name = (
                "".join([x.capitalize() for x in cls_name.strip().split("_")])
                + mod_name.capitalize()
            )

        # - get the class
        curr_register = getattr(registers, mod_name)
        target_cls = curr_register[cls_name]

        return target_cls

    @staticmethod
    def create(
        mode_name: str, cls_name: str, convert_name: bool = True, *args, **kwargs
    ):
        """"""
        target_cls = registers.get(mode_name, cls_name, convert_name, *args, **kwargs)
        instance = target_cls(*args, **kwargs)

        return instance


ALL_MODULES = [
    # - working components.
    # -- schedulers
    ("gdpx", ["scheduler"]),
    # -- managers (potentials)
    ("gdpx.potential", ["managers"]),
    # -- dataloaders (datasets)
    ("gdpx.data", ["dataset"]),
    # -- builders
    ("gdpx", ["builder"]),
    # -- genetic-algorithm-related
    ("gdpx.builder", ["crossover", "mutation"]),
    # -- colvar
    ("gdpx", ["colvar"]),
    # -- selectors
    ("gdpx", ["selector"]),
    # -- describer
    ("gdpx", ["describer"]),
    # -- comparators
    ("gdpx", ["comparator"]),
    # -- expeditions
    ("gdpx", ["expedition"]),
    # -- reactors
    # -- validators
    ("gdpx", ["validator"]),
    # - session operations + variables.
    ("gdpx.builder", ["interface"]),
    ("gdpx.computation", ["interface"]),
    ("gdpx", ["data"]),
    ("gdpx.data", ["interface"]),
    ("gdpx.describer", ["interface"]),
    ("gdpx.potential", ["interface"]),
    ("gdpx.reactor", ["interface"]),
    ("gdpx.comparator", ["interface"]),
    ("gdpx.trainer", ["interface"]),
    ("gdpx.selector", ["interface"]),
    ("gdpx.scheduler", ["interface"]),
    ("gdpx.validator", ["interface"]),
    ("gdpx.worker", ["interface"]),
]


def _handle_errors(errors):
    """Log out and possibly reraise errors during import."""
    names = []  # unimported module names
    if errors:
        for name, err in errors:
            warnings.warn("Module {} import failed: {}".format(name, err), UserWarning)
            names.append(name)
    else:
        ...

    return names


def import_all_modules_for_register(custom_module_paths=None) -> str:
    """Import all modules for register."""
    modules = []
    for base_dir, submodules in ALL_MODULES:
        for name in submodules:
            full_name = base_dir + "." + name
            modules.append(full_name)
    if isinstance(custom_module_paths, list):
        modules += custom_module_paths
    # print("ALL MODULES: ", modules)
    errors = []
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as error:
            errors.append((module, error))
    names = _handle_errors(errors)

    # - some imported packages change `logging.basicConfig`
    #   and accidently add a StreamHandler to logging.root
    #   so remove it...
    for h in logging.root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(
            h, logging.FileHandler
        ):
            logging.root.removeHandler(h)

    keys = sorted(names)
    nkeys = len(keys)
    ncols = 3
    nrows = int(nkeys / ncols)

    lines = ["FAILED TO IMPORT OPTIONAL MODULES: "]
    for i in range(nrows):
        lines.append(
            ("  " + "{:<48s}" * ncols + "").format(*keys[i * ncols : i * ncols + ncols])
        )

    nrest = nkeys - nrows * ncols
    if nrest > 0:
        lines.append(("  " + "{:<48s}" * nrest + "").format(*keys[nrows * ncols :]))

    for line in lines:
        config._print(line)

    return


if __name__ == "__main__":
    ...
