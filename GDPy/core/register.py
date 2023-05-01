#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import importlib

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
            logging.warning("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class."""
        
        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            # @reg.register
            return add(None, target)
        # @reg.register('alias')
        return lambda x: add(target, x)

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()

class registers:

    #: Session operations.
    operation: Register = Register("operation")

    #: Session variables.
    variable: Register = Register("variable")

    #: Session placeholder
    placeholder: Register = Register("placeholder")

    #: Managers (Potentials).
    manager: Register = Register("manager")

    #: Builders.
    builder: Register = Register("builder")

    #: Modifiers.
    modifier: Register = Register("modifier")

    #: Selectors.
    selector: Register = Register("selector")

    #: Validators.
    validator: Register = Register("validator")

    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")
    
    @staticmethod
    def get(mod_name: str, cls_name: str, convert_name: bool=True, *args, **kwargs):
        """Acquire the target class from modules."""
        # - convert the cls_name by the internal convention
        if convert_name:
            cls_name = cls_name.capitalize() + mod_name.capitalize()

        # - get the class
        curr_register = getattr(registers, mod_name)
        target_cls = curr_register[cls_name]

        return target_cls
    
    @staticmethod
    def create(mode_name: str, cls_name: str, convert_name: bool=True, *args, **kwargs):
        """"""
        target_cls = registers.get(mode_name, cls_name, convert_name, *args, **kwargs)
        instance = target_cls(*args, **kwargs)

        return instance


VALIDATOR_MODULES = ["singlepoint"]
DATA_OPS = ["operations"]

ALL_MODULES = [
    # - working components.
    # -- managers (potentials)
    ("GDPy.potential.managers", ["vasp", "cp2k", "xtb", "reax", "eann", "deepmd", "lasp", "nequip"]),
    # -- builders
    ("GDPy.builder", ["direct", "species"]),
    # -- modifiers
    ("GDPy.builder", ["perturbator"]),
    # -- validators
    ("GDPy.validator", VALIDATOR_MODULES),
    # - session operations.
    ("GDPy.data", DATA_OPS), 
    ("GDPy.builder", ["interface"]),
    ("GDPy.computation", ["operations"]),
    ("GDPy.validator", ["interface"]),
    ("GDPy.validator", ["meltingpoint"]),
    ("GDPy.selector", ["interface"]),
    # - session variables.
    ("GDPy.validator", ["interface"]),
    ("GDPy.scheduler", ["interface"]),
    ("GDPy.computation", ["interface"]),
    ("GDPy.potential", ["interface"]),
]


def _handle_errors(errors):
    """Log out and possibly reraise errors during import."""
    if not errors:
        return
    for name, err in errors:
        logging.warning("Module {} import failed: {}".format(name, err))


def import_all_modules_for_register(custom_module_paths=None):
    """Import all modules for register."""
    modules = []
    for base_dir, submodules in ALL_MODULES:
        for name in submodules:
            full_name = base_dir + "." + name
            modules.append(full_name)
    if isinstance(custom_module_paths, list):
        modules += custom_module_paths
    #print("ALL MODULES: ", modules)
    errors = []
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as error:
            errors.append((module, error))
    _handle_errors(errors)

if __name__ == "__main__":
    ...