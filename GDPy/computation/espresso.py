#!/usr/bin/env python3
# -*- coding: utf-8 -*


import pathlib
import warnings


from ase import io
from ase.calculators.calculator import FileIOCalculator, PropertyNotPresent


"""Parse Espresso input file.

There are several sections: 
    &CONTROL, &SYSTEM, &ELECTRONS, &IONS, &CELL, &FCP, &RISM,
    ATOMIC_SPECIES, ATOMIC_POSITIONS, K_POINTS, ADDITIONAL_K_POINTS,
    CELL_PARAMETERS, CONSTRAINTS, OCCUPATIONS, ATOMIC_VELOCITIES, ATOMIC_FORCES,
    SOLVENTS, HUBBARD

"""

def convert_type(v: str):
    """"""
    v = v.strip()
    if v.startswith("'"): # string
        v = v.strip("'")
    elif v.startswith("."): # bool
        if v == ".true.":
            v = True
        elif v == ".false":
            v = False
        else:
            ...
    else: # number
        if v.isdigit():
            v = int(v)
        else:
            v = float(v)

    return v


class EspressoParser():

    def __init__(self, template) -> None:
        """"""
        self.parameters = self._parse(template)

        return
    
    def _parse(self, template):
        """"""
        with open(template, "r") as fopen:
            lines = fopen.readlines()
        
        # NOTE: only read parameters in sections ...
        #       other will be written by ase.io
        parameters = {}
        is_in_section = False
        for line in lines:
            if line.startswith("&"): # start section
                is_in_section = True
                section_name = line.strip()[1:]
                parameters[section_name] = {}
            elif line.startswith("/"): # end section
                is_in_section = False
                if len(parameters[section_name]) == 0:
                    del parameters[section_name]
            else:
                if is_in_section:
                    k, v = line.strip().split("=")
                    parameters[section_name][k.strip()] = convert_type(v)
                else:
                    ...

        return parameters



error_template = 'Property "%s" not available. Please try running Quantum\n' \
                 'Espresso first by calling Atoms.get_potential_energy().'

warn_template = 'Property "%s" is None. Typically, this is because the ' \
                'required information has not been printed by Quantum ' \
                'Espresso at a "low" verbosity level (the default). ' \
                'Please try running Quantum Espresso with "high" verbosity.'


class Espresso(FileIOCalculator):
    """
    """
    implemented_properties = ['energy', 'forces', 'stress', 'magmoms']
    command = 'pw.x -in PREFIX.pwi > PREFIX.pwo'
    discard_results_on_any_change = True

    def __init__(self, restart=None,
                 ignore_bad_restart_file=FileIOCalculator._deprecated,
                 label='espresso', atoms=None, **kwargs):
        """
        All options for pw.x are copied verbatim to the input file, and put
        into the correct section. Use ``input_data`` for parameters that are
        already in a dict, all other ``kwargs`` are passed as parameters.

        Accepts all the options for pw.x as given in the QE docs, plus some
        additional options:

        input_data: dict
            A flat or nested dictionary with input parameters for pw.x
        pseudopotentials: dict
            A filename for each atomic species, e.g.
            ``{'O': 'O.pbe-rrkjus.UPF', 'H': 'H.pbe-rrkjus.UPF'}``.
            A dummy name will be used if none are given.
        kspacing: float
            Generate a grid of k-points with this as the minimum distance,
            in A^-1 between them in reciprocal space. If set to None, kpts
            will be used instead.
        kpts: (int, int, int), dict, or BandPath
            If kpts is a tuple (or list) of 3 integers, it is interpreted
            as the dimensions of a Monkhorst-Pack grid.
            If ``kpts`` is set to ``None``, only the Γ-point will be included
            and QE will use routines optimized for Γ-point-only calculations.
            Compared to Γ-point-only calculations without this optimization
            (i.e. with ``kpts=(1, 1, 1)``), the memory and CPU requirements
            are typically reduced by half.
            If kpts is a dict, it will either be interpreted as a path
            in the Brillouin zone (*) if it contains the 'path' keyword,
            otherwise it is converted to a Monkhorst-Pack grid (**).
            (*) see ase.dft.kpoints.bandpath
            (**) see ase.calculators.calculator.kpts2sizeandoffsets
        koffset: (int, int, int)
            Offset of kpoints in each direction. Must be 0 (no offset) or
            1 (half grid offset). Setting to True is equivalent to (1, 1, 1).


        .. note::
           Set ``tprnfor=True`` and ``tstress=True`` to calculate forces and
           stresses.

        .. note::
           Band structure plots can be made as follows:


           1. Perform a regular self-consistent calculation,
              saving the wave functions at the end, as well as
              getting the Fermi energy:

              >>> input_data = {<your input data>}
              >>> calc = Espresso(input_data=input_data, ...)
              >>> atoms.calc = calc
              >>> atoms.get_potential_energy()
              >>> fermi_level = calc.get_fermi_level()

           2. Perform a non-self-consistent 'band structure' run
              after updating your input_data and kpts keywords:

              >>> input_data['control'].update({'calculation':'bands',
              >>>                               'restart_mode':'restart',
              >>>                               'verbosity':'high'})
              >>> calc.set(kpts={<your Brillouin zone path>},
              >>>          input_data=input_data)
              >>> calc.calculate(atoms)

           3. Make the plot using the BandStructure functionality,
              after setting the Fermi level to that of the prior
              self-consistent calculation:

              >>> bs = calc.band_structure()
              >>> bs.reference = fermi_energy
              >>> bs.plot()

        """
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)
        self.calc = None

    def write_input(self, atoms, properties=None, system_changes=None):
        """"""
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        # - parse some system-dependant parameters
        pp = self.parameters.get("pseudopotentials", None)
        if pp is None:
            ...
        else:
            if isinstance(pp, dict):
                ...
            elif isinstance(pp, str):
                pp_suffix = pp
                pp = {}
                for sym in set(atoms.get_chemical_symbols()):
                    pp[sym] = "".join([sym, pp_suffix])
                self.parameters.update(pseudopotentials=pp)
            else:
                ...

        io.write(self.label + '.pwi', atoms, **self.parameters)

        return

    def read_results(self):
        output = io.read(self.label + '.pwo')
        self.calc = output.calc
        self.results = output.calc.results

    def get_fermi_level(self):
        if self.calc is None:
            raise PropertyNotPresent(error_template % 'Fermi level')
        return self.calc.get_fermi_level()

    def get_ibz_k_points(self):
        if self.calc is None:
            raise PropertyNotPresent(error_template % 'IBZ k-points')
        ibzkpts = self.calc.get_ibz_k_points()
        if ibzkpts is None:
            warnings.warn(warn_template % 'IBZ k-points')
        return ibzkpts

    def get_k_point_weights(self):
        if self.calc is None:
            raise PropertyNotPresent(error_template % 'K-point weights')
        k_point_weights = self.calc.get_k_point_weights()
        if k_point_weights is None:
            warnings.warn(warn_template % 'K-point weights')
        return k_point_weights

    def get_eigenvalues(self, **kwargs):
        if self.calc is None:
            raise PropertyNotPresent(error_template % 'Eigenvalues')
        eigenvalues = self.calc.get_eigenvalues(**kwargs)
        if eigenvalues is None:
            warnings.warn(warn_template % 'Eigenvalues')
        return eigenvalues

    def get_number_of_spins(self):
        if self.calc is None:
            raise PropertyNotPresent(error_template % 'Number of spins')
        nspins = self.calc.get_number_of_spins()
        if nspins is None:
            warnings.warn(warn_template % 'Number of spins')
        return nspins


if __name__ == "__main__":
    ...