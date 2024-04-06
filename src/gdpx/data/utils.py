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


def is_a_valid_system_name(system_name: str) -> bool:
    """"""
    is_valid = True

    name_parts = system_name.split("-")
    if len(name_parts) == 1:
        composition = name_parts[0]
    elif len(name_parts) == 2:
        desc, composition = name_parts
    elif len(name_parts) == 3:
        desc, composition, substrate = name_parts
    else:
        composition = ""

    try:
        formula = Formula(composition)
    except:
        is_valid = False

    return is_valid


if __name__ == "__main__":
    ...
