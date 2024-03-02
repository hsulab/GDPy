#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

from ase import units


def estimate_chemical_potential(
    temperature: float, 
    pressure: float, # pressure, 1 bar
    total_energy: float,
    zpe: float,
    dU: float,
    dS: float, # entropy
    coef: float = 1.0
) -> float:
    """Estimate Chemical Potential

    Examples:
        >>> O2 by ReaxFF
        >>>     molecular energy -5.588 atomic energy -0.109
        >>> O2 by vdW-DF spin-polarised 
        >>>     molecular energy -9.196 atomic energy -1.491
        >>>     ZPE 0.09714 
        >>>     dU 8.683 kJ/mol (exp)
        >>>     entropy@298.15K 205.147 J/mol (exp)
        >>> For two reservoirs, O and Pt
        >>> Pt + O2 -> aPtO2
        >>> mu_Pt = E_aPtO2 - G_O2
        >>> FreeEnergy = E_DFT + ZPE + U(T) + TS + pV

    References:
        Thermodynamic Data  https://janaf.nist.gov
    """
    kJm2eV = units.kJ / units.mol # from kJ/mol to eV
    # 300K, PBE-ZPE, experimental data https://janaf.nist.gov
    temp_correction = zpe + (dU*kJm2eV) - temperature*(dS/1000*kJm2eV)
    pres_correction = units.kB*temperature*np.log(pressure/1.0) # eV
    chemical_potential = coef*(
        total_energy + temp_correction + pres_correction
    )

    return chemical_potential


if __name__ == "__main__":
    ...