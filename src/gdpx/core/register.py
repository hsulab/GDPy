import importlib

from .. import config
from .registry import BaseRegister, Register, Registry


class registers:
    #: Worker implementations.
    worker: Register = Register("worker")

    def __init__(self):
        raise RuntimeError("The registers is not intended to be instantiated")

    @staticmethod
    def get(mod_name: str, cls_name: str, convert_name: bool = True):
        """Acquire the target class from modules."""
        # Convert the cls_name by the internal convention
        if convert_name:
            # cls_name = cls_name.capitalize() + mod_name.capitalize()
            cls_name = "".join([x.capitalize() for x in cls_name.strip().split("_")]) + mod_name.capitalize()

        # Get the class
        curr_register = getattr(registers, mod_name)
        target_cls = curr_register[cls_name]

        return target_cls

    @staticmethod
    def create(
        mode_name: str,
        cls_name: str,
        convert_name: bool = True,
        *args,
        **kwargs,
    ):
        """"""
        target_cls = registers.get(mode_name, cls_name, convert_name)
        instance = target_cls(*args, **kwargs)

        return instance


ALL_MODULES = [
    (
        "gdpx.nodes",
        [
            "region",
            "trainer",
            "validator",
            "dataset",
            "selector",
            "describer",
            "driver",
            "computer",
            "scheduler",
            "expedition",
            "comparator",
            "potential",
            "correction",
        ],
    ),
]


def _handle_errors(errors):
    """Log out and possibly reraise errors during import."""
    names, reasons = [], []  # unimported module names and reasons
    if errors:
        for name, err in errors:
            # warnings.warn("Module {} import failed: {}".format(name, err), UserWarning)
            names.append(name)
            reasons.append(err)
    else:
        ...

    return names, reasons


def show_failed_modules_in_rows_with_reasons(names, reasons) -> list[str]:
    """"""
    lines = []
    for name, err in zip(names, reasons):
        lines.append(f"  {name:<33s} -> require `{err.name}`.")

    return lines


def import_all_modules_for_register(custom_module_paths=None, disable_import_info: bool = False) -> None:
    """Load domain registries first, followed by workflow adapters."""
    if not disable_import_info:
        config._print("FAILED TO IMPORT OPTIONAL MODULES: ")

    errors = []
    local_module_pairs = (
        ("bias", "bias"),
        ("builder", "builder"),
        ("colvar", "colvar"),
        ("comparator", "comparator"),
        ("dataloader", "dataloader"),
        ("describer", "describer"),
        ("expedition", "expedition"),
        ("manager", "potential"),  # use alias
        ("region", "region"),
        ("scheduler", "scheduler"),
        ("selector", "selector"),
        ("trainer", "trainer"),
        ("validator", "validator"),
    )

    for register_name, module_name in local_module_pairs:
        try:
            module = importlib.import_module(f"gdpx.{module_name}")
            local_register = getattr(module, "REGISTER")
            setattr(registers, register_name, local_register)
        except ImportError as error:
            setattr(registers, register_name, Register(register_name))
            errors.append((module_name, error))

    modules = [f"{base_dir}.{name}" for base_dir, submodules in ALL_MODULES for name in submodules]
    if isinstance(custom_module_paths, list):
        modules.extend(custom_module_paths)
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as error:
            errors.append((module, error))

    if not disable_import_info:
        names, reasons = _handle_errors(errors)
        lines = show_failed_modules_in_rows_with_reasons(names, reasons)
        for line in lines:
            config._print(line)

    return
