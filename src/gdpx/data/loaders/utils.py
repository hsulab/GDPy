#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ase.formula import Formula


def get_atomic_number_list(system_name, type_list):
    """"""
    name_parts = system_name.split("-")
    if len(name_parts) == 1:
        composition = name_parts[0]
    elif len(name_parts) == 2:
        desc, composition = name_parts
    elif len(name_parts) == 3:
        desc, composition, substrate = name_parts
    else:
        raise ValueError(
            f"System name must be as xxx, xxx-xxx, or xxx-xxx-xxx instead of `{system_name}`."
        )

    # print(composition)
    formula = Formula(composition)
    count = formula.count()

    return [count.get(s, 0) for s in type_list][::-1]


def get_system_name_components(system_name: str):
    """"""
    name_parts = system_name.split("-")
    if len(name_parts) == 1:
        composition = name_parts[0]
        description, substrate = "desc", "mixed"
    elif len(name_parts) == 2:
        description, composition = name_parts
        substrate = "mixed"
    elif len(name_parts) == 3:
        description, composition, substrate = name_parts
    else:
        description, composition, substrate = "desc", "", "mixed"

    return description, composition, substrate


def get_composition_from_system_tree(system_tree):
    """"""
    system_name = system_tree[-1]

    description, composition, substrate = get_system_name_components(system_name)

    return composition


def is_a_valid_system_name(system_name: str) -> bool:
    """"""
    is_valid = True

    description, composition, substrate = get_system_name_components(system_name)

    try:
        formula = Formula(composition)
    except:
        is_valid = False

    return is_valid


if __name__ == "__main__":
    ...
