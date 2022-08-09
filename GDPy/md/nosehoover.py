""" Custom Nose-Hoover NVT thermostat based on ASE.

This code was originally written by Jonathan Mailoa based on these notes:

    https://www2.ph.ed.ac.uk/~dmarendu/MVP/MVP03.pdf

It was then adapted by Simon Batzner to be used within ASE. Parts of the overall outline of the class are also based on the Langevin class in ASE.

This was further changed by Jiayan XU for a more compact formulation.

"""

import numpy as np

from ase.md.md import MolecularDynamics
from ase.md.velocitydistribution import Stationary
from ase import units


class NoseHoover(MolecularDynamics):
    """Nose-Hoover (constant N, V, T) molecular dynamics.

    Usage: NoseHoover(atoms, dt, temperature)

    atoms
        The list of atoms.

    timestep
        The time step.

    temperature
        Target temperature of the MD run in [K * units.kB]

    nvt_q
        Q in the Nose-Hoover equations

    Example Usage:

        nvt_dyn = NoseHoover(
            atoms=atoms,
            timestep=0.5 * units.fs,
            temperature=300. * units.kB,
            nvt_q=334.
        )

    """

    def __init__(
        self,
        atoms,
        timestep,
        temperature,
        nvt_q,
        trajectory=None,
        logfile=None,
        loginterval=1,
        append_trajectory=False,
    ):
        # set com momentum to zero
        # TODO: if make com fixed at each step?
        # Stationary(atoms)

        self.temp = temperature / units.kB
        self.nvt_q = nvt_q
        self.dt = timestep  # units: A/sqrt(u/eV)
        self.dtdt = np.power(self.dt, 2)
        self.nvt_bath = 0.0

        # local
        self._vel_halfstep = None

        MolecularDynamics.__init__(
            self,
            atoms,
            timestep,
            trajectory,
            logfile,
            loginterval,
            append_trajectory=append_trajectory,
        )

    def step(self, f=None):
        """ Perform a MD step. 

        """

        # TODO: we do need the f=None argument?
        atoms = self.atoms
        masses = atoms.get_masses()  # units: u

        # count actual degree of freedoms
        # count every step because sometimes constraints can be manually changed
        ndof = 3*len(atoms)
        for constraint in atoms._constraints:
            ndof -= constraint.get_removed_dof(atoms)
        ndof += 1 # bath

        if f is None:
            f = atoms.get_forces()

        # for the first step, v(t-dt/2) = v(0.5) is needed.
        if self._vel_halfstep is None:
            self._vel_halfstep = (
                atoms.get_velocities() - 
                0.5 * self.dt * (f / masses[:, np.newaxis])
            )
        else:
            pass

        # v(t-dt/2), f(t), eta(t) -> v(t)
        atoms.set_velocities(
            (
                self._vel_halfstep
                + 0.5 * self.dt * (f / masses[:, np.newaxis])
            )
            / (1 + 0.5 * self.dt * self.nvt_bath)
        )

        # v(t), f(t), eta(t) -> r(t+dt)
        modified_acc = (
            f / masses[:, np.newaxis]
            - self.nvt_bath * atoms.get_velocities()
        )

        pos_fullstep = (
            atoms.get_positions()
            + self.dt * atoms.get_velocities()
            + 0.5 * self.dtdt * modified_acc
        )

        atoms.set_positions(pos_fullstep)

        # v(t), f(t), eta(t) -> v(t+dt/2)
        self._vel_halfstep = atoms.get_velocities() + 0.5 * self.dt * modified_acc

        # eta(t), v(t) -> eta(t+dt/2)
        e_kin_diff = 0.5 * (
            np.sum(masses * np.sum(atoms.get_velocities() ** 2, axis=1))
            - (ndof) * units.kB * self.temp
        ) # number of freedoms?

        nvt_bath_halfstep = self.nvt_bath + 0.5 * self.dt * e_kin_diff / self.nvt_q

        # eta(t+dt/2), v(t+dt/2) -> eta(t+dt)
        e_kin_diff_halfstep = 0.5 * (
            np.sum(masses * np.sum(self._vel_halfstep ** 2, axis=1))
            - (ndof) * units.kB * self.temp
        )

        self.nvt_bath = (
            nvt_bath_halfstep + 0.5 * self.dt * e_kin_diff_halfstep / self.nvt_q
        )

        return

if __name__ == '__main__':
    import ase.io
    import ase.units
    from ase import Atoms
    from ase.calculators.emt import EMT
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    from ase.constraints import FixAtoms

    from .md_utils import force_temperature

    # ===== system =====
    atoms = Atoms(
        'CO2',
        positions = [
            [0.,0.,0.],
            [0.,0.,1.0],
            [0.,0.,-1.0]
        ]
    )

    cons = FixAtoms(mask=[atom.symbol == 'C' for atom in atoms])

    atoms.set_constraint(cons)

    atoms.set_calculator(EMT())

    # ===== molecular dynamics =====
    temperature = 300
    MaxwellBoltzmannDistribution(atoms, temperature*ase.units.kB)
    force_temperature(atoms, temperature)
    print('start!!!')
    print(atoms.get_velocities())
    print(atoms.get_temperature())

    nvt_dyn = NoseHoover(
        atoms = atoms,
        timestep = 2.0 * units.fs,
        temperature = temperature * units.kB,
        nvt_q = 334.
    )

    def print_temperature(atoms):
        content = 'temperature %8.4f' %atoms.get_temperature()
        print(content)

    nvt_dyn.attach(print_temperature, atoms=atoms)

    nvt_dyn.run(steps=5)
