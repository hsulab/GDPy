#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from itertools import groupby
from typing import Union

from ase import Atoms

from gdpx.utils.strconv import string_to_integers

from ..core.register import registers

# from gdpx.graph.creator import find_molecules


def create_a_molecule_group(
    atoms: Atoms, group_command: str, use_tags=True
) -> list[list[int]]:
    """Find molecules in the structure."""
    args = group_command.strip().split()
    # assert args[0] in ["tag", "molecule"], f"{args[0]} is not implemented."

    if args[0] in ["tags", "molecule"]:
        groups = []
        if args[0] == "tag":
            if "tags" in atoms.arrays:
                # --- find molecuels based on tags
                natoms = len(atoms)
                tags = atoms.get_tags()  # copied already
                for k, g in groupby(range(natoms), key=lambda x: tags[x]):
                    atomic_indices = list(g)
                    # symbols = [atoms[i].symbol for i in atomic_indices]
                    # formula = Formula.from_list(symbols).format("hill")
                    if str(k) in args[1:]:
                        groups.append(atomic_indices)
            else:
                raise RuntimeError("Cant find tags in atoms.")

        # if args[0] == "molecule":
        #    target_molecule = args[1]
        #    molecules = [target_molecule]
        #    # --- find molecules with graph connectivity
        #    #raise RuntimeError("No tags in atoms.")
        #    valid_symbols = []
        #    for m in molecules:
        #        valid_symbols.extend(list(Formula(m).count().keys()))
        #    valid_symbols = set(valid_symbols)
        #    atomic_indices = create_a_group(atoms, "symbol {}".format(" ".join(valid_symbols)))
        #    fragments = find_molecules(atoms, atomic_indices)
        #    if target_molecule in fragments:
        #        groups = fragments[target_molecule]
        #    else:
        #        raise RuntimeError(f"Cant find molecule {target_molecule} in atoms.")
    else:
        # NOTE: use atomic group that equals one molecule
        groups = [create_a_group(atoms, group_command)]

    return groups


if __name__ == "__main__":
    ...
