#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ase import units


def compute_molecule_number_from_density(
    molecular_mass: float, volume: float, density: float
) -> int:
    """Compute the number of molecules in the region with a given density.

    Args:
        moleculer_mass: unit in g/mol.
        volume: unit in Ang^3.
        density: unit in g/cm^3.

    Returns:
        Number of molecules in the region.

    """
    number = (density / molecular_mass) * volume * units._Nav * 1e-24

    return int(number)


if __name__ == "__main__":
    ...
