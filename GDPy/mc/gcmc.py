#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from collections import namedtuple
from pathlib import Path
import shutil

import numpy as np

from ase import units
from ase import data
#from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase import Atom, Atoms
from ase.io import read, write
from ase.neighborlist import natural_cutoffs, NeighborList

from ase.optimize import BFGS
from ase.constraints import FixAtoms

from ase.ga.utilities import closest_distances_generator

from GDPy.builder.species import build_species


"""
# index element mass       ideal-nn rmin rmax prob-add
  0     Ag      196.966543 3        1.5  3.0  0.50
  1     O       15.9994    3        1.5  3.0  0.50

For each element, thermal de broglie wavelengths are needed.
"""

Reservior = namedtuple('Reservior', ['particle', 'temperature', 'pressure', 'mu']) # Kelvin, atm, eV
RunParam = namedtuple('RunParam', ['backend', 'calculator', 'optimiser', 'convergence', 'constraint']) # 

def estimate_chemical_potential(
    temperature: float, 
    pressure: float, # pressure, 1 bar
    total_energy: float,
    zpe: float,
    dU: float,
    dS: float, # entropy
    coef: float = 1.0
) -> float:
    """
    See experimental data
        https://janaf.nist.gov
    Examples
        O2 by ReaxFF
            molecular energy -5.588 atomic energy -0.109
        O2 by vdW-DF spin-polarised 
            molecular energy -9.196 atomic energy -1.491
            ZPE 0.09714 
            dU 8.683 kJ/mol (exp)
            entropy@298.15K 205.147 J/mol (exp)
        For two reservoirs, O and Pt
        Pt + O2 -> aPtO2
        mu_Pt = E_aPtO2 - G_O2
    Formula
        FreeEnergy = E_DFT + ZPE + U(T) + TS + pV
    """
    kJm2eV = units.kJ / units.mol # from kJ/mol to eV
    # 300K, PBE-ZPE, experimental data https://janaf.nist.gov
    temp_correction = zpe + (dU*kJm2eV) - temperature*(dS/1000*kJm2eV)
    pres_correction = units.kB*temperature*np.log(pressure/1.0) # eV
    chemical_potential = coef*(
        total_energy + temp_correction + pres_correction
    )

    return chemical_potential

class ReducedRegion():
    """
    spherical or cubic box
    """

    MAX_RANDOM_ATTEMPS = 1000

    def __init__(
        self, 
        type_list, # chemical symbols for cutoff radii
        cell, # (3x3) lattice
        caxis: list, # min and max in z-axis
        covalent_ratio = [0.8, 2.0],
        max_movedisp = 2.0,
        use_rotation = False,
        rng = None
    ):
        """"""
        # cutoff radii
        assert len(covalent_ratio) == 2, "covalent ratio should have min and max"
        self.covalent_min = covalent_ratio[0]
        self.covalent_max = covalent_ratio[1]

        unique_atomic_numbers = [data.atomic_numbers[a] for a in type_list]
        self.blmin = closest_distances_generator(
            atom_numbers=unique_atomic_numbers,
            ratio_of_covalent_radii = self.covalent_min # be careful with test too far
        )

        # move operation
        self.max_movedisp = max_movedisp

        self.use_rotation = use_rotation

        # box
        self.cell = cell.copy()
        
        assert len(caxis) == 2
        self.cmin, self.cmax = caxis
        self.cell[2,2] = self.cmax 

        self.cvect = cell[2] / np.linalg.norm(cell[2])
        self.volume = np.dot(
            self.cvect*(self.cmax-self.cmin), np.cross(self.cell[0], self.cell[1])
        )

        # random generator
        if rng is None:
            drng = np.random.default_rng()
            self.rng = drng
        else:
            self.rng = rng

        return

    def random_position(self, positions: np.ndarray):
        """ old implementation, not very efficient
        """
        mindis = 1.5
        for i in range(1000): # maximum number of attempts
            ran_frac_pos = self.rng.uniform(0,1,3)
            ran_pos = np.dot(ran_frac_pos, self.cell) 
            ran_pos[2] = self.cmin + ran_frac_pos[2] * (self.cmax-self.cmin)
            # TODO: better
            if not self.check_overlap(mindis, ran_pos, positions):
                print("random position", ran_pos)
                break
        else:
            print("Failed to generate a random position...")

        return ran_pos

    @staticmethod
    def check_overlap(mindis, ran_pos, positions):
        """"""
        st = time.time()
        # TODO: change this to the faster neigbour list construction
        # maybe use scipy
        overlapped = False
        for pos in positions:
            # TODO: use neighbour list?
            dis = np.linalg.norm(ran_pos-pos)
            if dis < mindis:
                overlapped = True
                break
        et = time.time()
        print("time for overlap: ", et-st)

        return overlapped

    def random_position_neighbour(
        self, 
        new_atoms: Atoms, 
        species_indices,
        operation # move or insert
    ):
        """"""
        print("\n---- Generate Random Position -----\n")
        st = time.time()

        # - find tag atoms
        # record original position of idx_pick
        species = new_atoms[species_indices]

        #org_pos = new_atoms[idx_pick].position.copy() # original position
        # TODO: deal with pbc, especially for move step
        org_com = np.mean(species.positions.copy(), axis=0)
        org_positions = species.positions.copy()

        chemical_symbols = new_atoms.get_chemical_symbols()
        nl = NeighborList(
            self.covalent_max*np.array(natural_cutoffs(new_atoms)), 
            skin=0.0, self_interaction=False, bothways=True
        )
        for i in range(self.MAX_RANDOM_ATTEMPS): # maximum number of attempts
            print(f"random attempt {i}")
            if operation == "insert":
                # NOTE: z-axis should be perpendicular
                ran_frac_pos = self.rng.uniform(0,1,3)
                ran_pos = np.dot(ran_frac_pos, self.cell) 
                ran_pos[2] = self.cmin + ran_frac_pos[2] * (self.cmax-self.cmin)
            elif operation == "move":
                # get random motion vector
                rsq = 1.1
                while (rsq > 1.0):
                    rvec = 2*self.rng.uniform(size=3) - 1.0
                    rsq = np.linalg.norm(rvec)
                ran_pos = org_com + rvec*self.max_movedisp
            # update position of idx_pick
            #new_atoms.positions[species_indices] += new_vec
            species_ = species.copy()
            # --- Apply a random rotation to multi-atom blocks
            if self.use_rotation and len(species_indices) > 1:
                phi, theta, psi = 360 * self.rng.uniform(0,1,3)
                species_.euler_rotate(
                    phi=phi, theta=0.5 * theta, psi=psi,
                    center=org_com
                )
            # -- Apply translation
            new_vec = ran_pos - org_com
            species_.translate(new_vec)
            new_atoms.positions[species_indices] = species_.positions.copy()
            # use neighbour list
            if not self.check_overlap_neighbour(nl, new_atoms, chemical_symbols, species_indices):
                print(f"succeed to random after {i+1} attempts...")
                print("original position: ", org_com)
                print("random position: ", ran_pos)
                break
            new_atoms.positions[species_indices] = org_positions
        else:
            new_atoms = None

        et = time.time()
        print("used time: ", et - st)

        return new_atoms
    
    def check_overlap_neighbour(
        self, nl, new_atoms, chemical_symbols, 
        species_indices
    ):
        """ use neighbour list to check newly added atom is neither too close or too
            far from other atoms
        """
        overlapped = False
        nl.update(new_atoms)
        for idx_pick in species_indices:
            print("- check index ", idx_pick)
            indices, offsets = nl.get_neighbors(idx_pick)
            if len(indices) > 0:
                print("nneighs: ", len(indices))
                # should close to other atoms
                for ni, offset in zip(indices, offsets):
                    dis = np.linalg.norm(new_atoms.positions[idx_pick] - (new_atoms.positions[ni] + np.dot(offset, self.cell)))
                    pairs = [chemical_symbols[ni], chemical_symbols[idx_pick]]
                    pairs = tuple([data.atomic_numbers[p] for p in pairs])
                    print("distance: ", ni, dis, self.blmin[pairs])
                    if dis < self.blmin[pairs]:
                        overlapped = True
                        break
            else:
                # TODO: is no neighbours valid?
                print("no neighbours, being isolated...")
                overlapped = True
                # TODO: try rotate?
                break

        return overlapped
    
    def calc_acc_volume(self, atoms) -> float:
        """calculate acceptable volume"""
        atoms_inside = [atom for atom in atoms if atom.position[2] > self.cmin]
        print("number of atoms inside the region", len(atoms_inside))
        radii = [data.covalent_radii[data.atomic_numbers[atom.symbol]] for atom in atoms_inside]
        atoms_volume = np.sum([4./3.*np.pi*r**3 for r in radii])

        acc_volume = self.volume - atoms_volume # A^3

        return acc_volume


class GCMC():

    MCTRAJ = "./miaow.xyz"
    SUSPECT_DIR = "./suspects"

    def __init__(
        self, 
        type_list: list, 
        reservior: dict, 
        restart: bool = False,
        transition_array: np.array = np.array([0.0,0.0]),
        random_seed = None
    ):
        """
        """
        self.restart = restart

        # elements
        self.type_list = type_list

        # probs
        self.trans_probs = transition_array # transition probabilities: motion, insertion, deletion
        self.accum_probs = np.cumsum(transition_array) / np.sum(transition_array)

        # set random generator TODO: move to run?
        self.set_rng(random_seed)
        print("RANDOM SEED: ", random_seed)

        # - parse reservoir
        # TODO: reservoir
        self.temperature, self.pressure = reservior["temperature"], reservior.get("pressure", -1000.) # NOTE: pressure is None

        self.exparts, self.chem_pot = [], {} # particle names, particle mus

        particle_info = reservior["species"]
        for name, data in particle_info.items():
            self.exparts.append(name)
            if isinstance(data, dict):
                self.chem_pot[name] = estimate_chemical_potential(
                    temperature=self.temperature, pressure=self.pressure,
                    **data
                )
            else:
                # directly offer chemical potential
                self.chem_pot[name] = float(data)

        # statistical mechanics
        self.species_mass = {}
        self.beta = {}
        self.cubic_wavelength = {}
        for expart in self.exparts:
            self.species_mass[expart], self.beta[expart], self.cubic_wavelength[expart] = self.compute_thermo_wavelength(
                expart, self.temperature
            )

        return
    
    @staticmethod
    def compute_thermo_wavelength(expart: str, temperature: float):
        # - beta
        kBT_eV = units.kB * temperature
        beta = 1./kBT_eV # 1/(kb*T), eV

        # - cubic thermo de broglie 
        hplanck = units._hplanck # J/Hz = kg*m2*s-1
        #_mass = np.sum([data.atomic_masses[data.atomic_numbers[e]] for e in expart]) # g/mol
        _species = build_species(expart)
        _species_mass = np.sum(_species.get_masses())
        #print("species mass: ", _mass)
        _mass = _species_mass * units._amu
        kbT_J = kBT_eV * units._e # J = kg*m2*s-2
        cubic_wavelength = (hplanck/np.sqrt(2*np.pi*_mass*kbT_J)*1e10)**3 # thermal de broglie wavelength

        return _species_mass, beta, cubic_wavelength

    def set_rng(self, seed=None):
        # - assign random seeds
        if seed is None:
            self.rng = np.random.default_rng()
        elif isinstance(seed, int):
            self.rng = np.random.default_rng(seed)

        return

    def __register_calculator(self, calc_params: dict):
        """"""
        self.minimisation = calc_params.pop("minimisation", None)

        print("\n===== Calculator INFO =====\n")
        model = calc_params["model"]
        if model == "lasp":
            from GDPy.calculator.lasp import LaspNN
            self.calc = LaspNN(**self.calc_dict["kwargs"])
        elif model == "eann": # and inteface to lammps
            from GDPy.calculator.lammps import Lammps
            self.calc = Lammps(
                command = calc_params["command"],
                directory = calc_params["directory"],
                pair_style = calc_params["model_params"]
            )
        else:
            raise ValueError("Unknown potential to calculation...")
        
        backend = calc_params.pop("backend", None)
        if backend == "ase":
            from GDPy.calculator.asedyn import AseDynamics
            self.worker = AseDynamics(self.calc, directory=self.calc.directory)
        elif backend == "lammps":
            from GDPy.calculator.lammps import LmpDynamics
            # use lammps optimisation
            self.worker = LmpDynamics(
                self.calc, directory=self.calc.directory
            )
        else:
            raise ValueError("Unknown interface to optimisation...")

        # TODO: check uncertainty control
        self.devi_tol = calc_params.pop("devi_tol", 0.015) # try few times optimisation
        calc_params["directory"] = Path(calc_params["directory"])

        # find uncertainty support
        # TODO: check uncertainty
        #model_files = calc_params["model_params"]["file"]
        #if isinstance(model_files, str):
        #    num_models = 0
        #else:
        #    num_models = len(model_files)
        #    calc_params["model_params"]["file"] = " ".join(model_files)
        num_models = 0

        self.suspects = None
        if num_models > 1:
            self.suspects = Path(self.SUSPECT_DIR)
            if self.suspects.exists():
                raise FileExistsError("{} already exists...".format(self.suspects))
            else:
                self.suspects.mkdir()

        return

    def __construct_initial_system(
        self, structure_path, 
        region_settings
    ):
        """ 
        """
        # simulation system
        self.atoms = read(structure_path) # current atoms
        self.substrate_natoms = len(self.atoms)

        # set reduced region
        use_rotation = region_settings.pop("use_rotation", True)
        self.region = ReducedRegion(
            self.type_list, self.atoms.get_cell(complete=True), 
            **region_settings, use_rotation=use_rotation, rng=self.rng
        )

        self.acc_volume = self.region.calc_acc_volume(self.atoms)
        
        # constraint
        cons_indices = [atom.index for atom in self.atoms if atom.position[2] < self.region.cmin]
        cons = FixAtoms(
            indices = cons_indices
        )
        self.atoms.set_constraint(cons)

        # few iterative properties
        # TODO: use tag for molecule species

        # - set tags for init sys
        self.atoms.set_tags(0)

        # NOTE: use tag_list to manipulate both atoms and molecules
        self.tag_list = {}
        for expart in self.exparts:
            self.tag_list[expart] = []
            
        # check restart
        if self.restart:
            if Path(self.MCTRAJ).exists():
                print("restart from last structure...")
                last_traj = read(self.MCTRAJ, ":")
                start_index = len(last_traj) - 1
                last_atoms = last_traj[-1]
                # NOTE: the atom order matters
                # update exatom_indices
                chemical_symbols = last_atoms.get_chemical_symbols()
                last_natoms = len(last_atoms)
                # TODO: update tag_list
                for i in range(self.substrate_natoms, last_natoms):
                    chem_sym = chemical_symbols[i]
                    self.exatom_indices[chem_sym].append(i)
                # set atoms and its constraint
                self.atoms = last_atoms
                self.atoms.set_constraint(cons)
                # backup traj file
                card_path = Path.cwd() / Path(self.MCTRAJ).name
                bak_fmt = ("bak.{:d}."+Path(self.MCTRAJ).name)
                idx = 0
                while True:
                    bak_card = bak_fmt.format(idx)
                    if not Path(bak_card).exists():
                        saved_card_path = Path.cwd() / bak_card
                        shutil.copy(card_path, saved_card_path)
                        break
                    else:
                        idx += 1
            else:
                print("start from new substrate...")
                start_index = 0
        else:
            start_index = 0

        return start_index

    def run(
        self, 
        substrate_path,
        region_settings,
        nattempts,
        params: dict
    ):
        """"""
        # start info
        content = "===== Simulation Information @%s =====\n\n" % time.asctime( time.localtime(time.time()) )
        content += 'Temperature %.4f [K] Pressure %.4f [atm]\n' %(self.temperature, self.pressure)
        for expart in self.exparts:
            content += "--- %s ---\n" %expart
            content += f"Mass {self.species_mass[expart]}\n"
            content += "Beta %.4f [eV-1]\n" %(self.beta[expart])
            content += "Cubic Thermal de Broglie Wavelength %f\n" %self.cubic_wavelength[expart]
            content += "Chemical Potential of is %.4f [eV]\n" %self.chem_pot[expart]
        print(content)

        start_index = self.__construct_initial_system(substrate_path, region_settings)

        # region info 
        content = "----- Region Info -----\n"
        content += "cell \n"
        for i in range(3):
            content += ("{:<8.4f}  "*3+"\n").format(*list(self.region.cell[i, :]))
        content += "covalent ratio: {}  {}\n".format(self.region.covalent_min, self.region.covalent_max)
        content += "maximum movedisp: {}\n".format(self.region.max_movedisp)
        content += "use rotation: {}\n".format(self.region.use_rotation)
        print(content)
        
        # set calculator
        self.__register_calculator(params)
        self.verbose = params["output"]["verbose"]

        # opt init structure
        self.step_index = start_index
        self.energy_stored, self.atoms = self.optimise(self.atoms)
        print("Cell Info: ", self.atoms.cell)
        print("energy_stored ", self.energy_stored)

        # add optimised initial structure
        print('\n\nrenew trajectory file')
        write(self.MCTRAJ, self.atoms, append=False)

        # start monte carlo
        self.step_index = start_index + 1
        for idx in range(start_index+1, nattempts):
            self.step_index = idx
            print('\n\n===== MC Move %04d =====\n' %idx)
            # run standard MC move
            self.step()

            # TODO: save state
            self.atoms.info["step"] = idx
            write(self.MCTRAJ, self.atoms, append=True)

            # check uncertainty
        
        print("\n\nFINISHED PROPERLY @ %s." %time.asctime( time.localtime(time.time()) ))

        return

    def step(self):
        """ various actions
        [0]: move, [1]: exchange (insertion/deletion)
        """
        st = time.time()

        # - choose species
        expart = str(self.rng.choice(self.exparts)) # each element hase same prob to chooose
        print("selected particle: ", expart)
        rn_mcmove = self.rng.uniform()
        print("prob action", rn_mcmove)

        # step for selected type of particles
        nexatoms = len(self.tag_list[expart])
        if nexatoms > 0:
            if rn_mcmove < self.accum_probs[0]:
                # atomic motion
                print('current attempt is *motion*')
                self.attempt_move_atom(expart)
            elif rn_mcmove < self.accum_probs[1]:
                # exchange (insertion/deletion)
                rn_ex = self.rng.uniform()
                print("prob exchange", rn_ex)
                if rn_ex < 0.5:
                    print('current attempt is *insertion*')
                    self.attempt_insert_atom(expart)
                else:
                    print("current attempt is *deletion*")
                    self.attempt_delete_atom(expart)
            else:
                pass # never execute here
        else:
            print('current attempt is *insertion*')
            self.attempt_insert_atom(expart)

        et = time.time()
        print("step time: ", et - st)

        return
    
    def pick_random_atom(self, expart):
        """"""
        nexpart = len(self.tag_list[expart])
        if nexpart == 0:
            idx_pick = None
        else:
            idx_pick = self.rng.choice(self.tag_list[expart])
        #print(idx_pick, type(idx_pick))

        return idx_pick
    
    def update_exlist(self):
        """update the list of exchangeable particles"""
        print("number of particles: ")
        for expart, indices in self.tag_list.items():
            print("{:<4s}  {:<8d}".format(expart, len(indices)))
            print("species tag: ", indices)

        return
    
    def attempt_move_atom(self, expart):
        """"""
        # pick an atom
        self.update_exlist()
        tag_idx_pick = self.pick_random_atom(expart)
        if tag_idx_pick is not None:
            print("select species with tag %d" %tag_idx_pick)
        else:
            print("no exchangeable atoms...")
            return 
        
        # try move
        cur_atoms = self.atoms.copy()

        tags = cur_atoms.get_tags()
        species_indices = [i for i, t in enumerate(tags) if t==tag_idx_pick]

        cur_atoms = self.region.random_position_neighbour(
            cur_atoms, species_indices, operation="move"
        )

        if cur_atoms is None:
            print(f"failed to move after {self.region.MAX_RANDOM_ATTEMPTS} attempts...")
            return

        # move info
        print("origin position: ") 
        for si in species_indices:
            print(self.atoms[si].position)
        print("-> random position: ") 
        for si in species_indices:
            print(cur_atoms[si].position)

        # TODO: change this to optimisation
        energy_after, opt_atoms = self.optimise(cur_atoms)

        beta = self.beta[expart]
        cubic_wavelength = self.cubic_wavelength[expart]

        coef = 1.0
        energy_change = energy_after - self.energy_stored
        acc_ratio = np.min([1.0, coef * np.exp(-beta*(energy_change))])

        content = "\nVolume %.4f Nexatoms %.4f CubicWave %.4f Coefficient %.4f\n" %(
            self.acc_volume, len(self.tag_list[expart]), cubic_wavelength, coef
        )
        content += "Energy Change %.4f [eV]\n" %energy_change
        content += "Accept Ratio %.4f\n" %acc_ratio
        print(content)

        rn_motion = self.rng.uniform()
        if rn_motion < acc_ratio:
            self.atoms = opt_atoms
            self.energy_stored = energy_after
        else:
            print("fail to move")
            pass
        
        print("Translation Probability %.4f" %rn_motion)

        print("energy_stored is %12.4f" %self.energy_stored)
        
        return
    
    def attempt_insert_atom(self, expart):
        """atomic insertion"""
        self.update_exlist()

        # - extend one species
        cur_atoms = self.atoms.copy()
        print("current natoms: ", len(cur_atoms))
        # --- build species and assign tag
        new_species = build_species(expart)
        new_tag = (self.exparts.index(expart)+1)*1000 + len(self.tag_list[expart])
        new_species.set_tags(new_tag)
        cur_atoms.extend(new_species)

        print("natoms: ", len(cur_atoms))
        tag_idx_pick = new_tag
        print("try to insert species with tag ", tag_idx_pick)

        tags = cur_atoms.get_tags()
        species_indices = [i for i, t in enumerate(tags) if t==tag_idx_pick]

        cur_atoms = self.region.random_position_neighbour(
            cur_atoms, species_indices, operation="insert"
        )
        if cur_atoms is None:
            print("failed to insert after {self.region.MAX_RANDOM_ATTEMPS} attempts...")
            return
        
        # insert info
        print("inserted position: ")
        for si in species_indices:
            print(cur_atoms[si].symbol, cur_atoms[si].position)

        # TODO: change this to optimisation
        energy_after, opt_atoms = self.optimise(cur_atoms)

        nexatoms = len(self.tag_list[expart])
        beta = self.beta[expart]
        chem_pot = self.chem_pot[expart]
        cubic_wavelength = self.cubic_wavelength[expart]

        # try insert
        coef = self.acc_volume/(nexatoms+1)/cubic_wavelength
        energy_change = energy_after-self.energy_stored-chem_pot
        acc_ratio = np.min([1.0, coef * np.exp(-beta*(energy_change))])

        content = '\nVolume %.4f Nexatoms %.4f CubicWave %.4f Coefficient %.4f\n' %(
            self.acc_volume, nexatoms, cubic_wavelength, coef
        )
        content += 'Energy Change %.4f [eV]\n' %energy_change
        content += 'Accept Ratio %.4f\n' %acc_ratio
        print(content)

        rn_insertion = self.rng.uniform()
        if rn_insertion < acc_ratio:
            self.atoms = opt_atoms
            self.energy_stored = energy_after
            # update exchangeable atoms
            self.tag_list[expart].append(new_tag)
        else:
            print('fail to insert...')
        print('Insertion Probability %.4f' %rn_insertion)

        print('energy_stored is %12.4f' %self.energy_stored)

        return

    def attempt_delete_atom(self, expart):
        """"""
        # pick an atom
        self.update_exlist()
        tag_idx_pick = self.pick_random_atom(expart)
        if tag_idx_pick is not None:
            print("select species with tag %d" %tag_idx_pick)
        else:
            print("no atom can be deleted...")
            return

        # try deletion
        cur_atoms = self.atoms.copy()
        
        tags = cur_atoms.get_tags()
        species_indices = [i for i, t in enumerate(tags) if t==tag_idx_pick]
        del cur_atoms[species_indices]

        # TODO: change this to optimisation
        energy_after, opt_atoms = self.optimise(cur_atoms)

        nexatoms = len(self.tag_list[expart])
        beta = self.beta[expart]
        cubic_wavelength = self.cubic_wavelength[expart]
        chem_pot = self.chem_pot[expart]

        coef = nexatoms*cubic_wavelength/self.acc_volume
        energy_change  = energy_after + chem_pot - self.energy_stored
        acc_ratio = np.min([1.0, coef*np.exp(-beta*(energy_change))])

        content = '\nVolume %.4f Nexatoms %.4f CubicWave %.4f Coefficient %.4f\n' %(
            self.acc_volume, nexatoms, cubic_wavelength, coef
        )
        content += 'Energy Change %.4f [eV]\n' %energy_change
        content += 'Accept Ratio %.4f\n' %acc_ratio
        print(content)

        rn_deletion = self.rng.uniform()
        if rn_deletion < acc_ratio:
            self.atoms = opt_atoms
            self.energy_stored = energy_after
            # reformat exchangeable atoms
            new_expart_list = [t for t in self.tag_list[expart] if t != tag_idx_pick] 
            self.tag_list[expart] = new_expart_list
        else:
            pass

        print('Deletion Probability %.4f' %rn_deletion)

        print('energy_stored is %12.4f' %self.energy_stored)

        return


    def optimise(self, atoms):
        """"""
        self.worker.reset()
        old_calc_dir = self.calc.directory
        if self.verbose == True:
            new_dir = Path(self.calc.directory) / ("opt"+str(self.step_index))
            #if not new_dir.exists():
            #    new_dir.mkdir(parents=True)
            self.worker.set_output_path(new_dir)
        else:
            self.worker.set_output_path(self.calc.directory)

        print("\n----- DYNAMICS MIN INFO -----\n")
        st = time.time()

        # run minimisation
        tags = atoms.get_tags()
        print("n_cur_atoms: ", len(atoms))
        atoms = self.worker.minimise(
            atoms,
            read_exists=False,
            extra_info = None,
            **self.minimisation
        )
        atoms.set_tags(tags)

        self.calc.directory = old_calc_dir

        en = atoms.get_potential_energy()

        et = time.time()
        print("opt time: ", et - st)

        return en, atoms
    
    @staticmethod
    def __parse_deviation(devi_file):
        # TODO: check deviation
        # max_fdevi = np.loadtxt(devi_out)[1:,4] # DP
        max_fdevi = np.loadtxt(devi_file)[1:,5] # EANN

        return


if __name__ == '__main__':
    pass