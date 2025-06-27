#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from ase import units


def compute_thermo_wavelength(mass: float, temperature: float) -> float:
    """Compute the cubic thermo de Broglie wavelength.

    Args:
        mass: The mass of the particle in amu.
        temperature: The temperature in Kelvin.

    Returns:
        cubic_wavelength: The cubic thermo de Broglie wavelength in Angstrom^3.

    """
    # Compute the temperature in eV and then in J
    kBT_eV = units.kB * temperature
    kbT_J = kBT_eV * units._e  # J = kg*m2*s-2

    # Compute the mass in kg
    _mass = mass * units._amu

    # Planck's constant in J/Hz
    hplanck = units._hplanck  # J/Hz = kg*m2*s-1

    # Compute the cubic thermo de broglie in [Ang^3]
    cubic_wavelength = (hplanck / np.sqrt(2 * np.pi * _mass * kbT_J) * 1e10) ** 3

    return cubic_wavelength


if __name__ == "__main__":
    ...
