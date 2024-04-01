#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle

import numpy as np

from ase import Atoms
from ase import data, units

from .. import convert_string_to_atoms
from .move import MoveOperator
from .swap import SwapOperator
from .exchange import ExchangeOperator
from .react import ReactOperator

def save_operator(op, p):
    """"""
    with open(p, "wb") as fopen:
        pickle.dump(op, fopen)

    return

def load_operator(p):
    """"""
    with open(p, "rb") as fopen:
        op = pickle.load(fopen)

    return op

def select_operator(operators: list, probs: list, rng=np.random):
    """Select an operator based on the relative probabilities."""
    noperators = len(operators)
    op_idx = rng.choice(noperators, 1, probs)[0]
    op = operators[op_idx]

    return op

def parse_operators(op_params: dict):
    """Parse parameters for various operators.

    Currently, we have move, swap, and exchange (insert/remove).

    """
    operators, probs = [], []
    for param in op_params:
        name = param.pop("method", "move")
        prob = param.get("prob", 1.0)
        if name == "move":
            op = MoveOperator(**param)
        elif name == "swap":
            op = SwapOperator(**param)
        elif name == "exchange":
            op = ExchangeOperator(**param)
        elif name == "react":
            op = ReactOperator(**param)
        else:
            raise NotImplementedError(f"{name} is not supported.")
        operators.append(op)
        probs.append(prob)
    
    # - reweight probabilities
    probs = (np.array(probs) / np.sum(probs)).tolist()

    return operators, probs

def compute_thermo_wavelength(expart: str, temperature: float):
    # - beta
    kBT_eV = units.kB * temperature
    beta = 1./kBT_eV # 1/(kb*T), eV

    # - cubic thermo de broglie 
    hplanck = units._hplanck # J/Hz = kg*m2*s-1
    #_mass = np.sum([data.atomic_masses[data.atomic_numbers[e]] for e in expart]) # g/mol
    _species = convert_string_to_atoms(expart)
    _species_mass = np.sum(_species.get_masses())
    #print("species mass: ", _mass)
    _mass = _species_mass * units._amu
    kbT_J = kBT_eV * units._e # J = kg*m2*s-2
    cubic_wavelength = (hplanck/np.sqrt(2*np.pi*_mass*kbT_J)*1e10)**3 # thermal de broglie wavelength

    return _species_mass, beta, cubic_wavelength


if __name__ == "__main__":
    ...