#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from pathlib import Path

import numpy as np

from ase.data import atomic_numbers, atomic_masses


class LammpsInput():

    def __init__(self, type_map, data, model: dict, ensemble='nvt', variables: dict = {}, constraint=None):
        """"""
        # form mass line
        self.mass_line = ''
        for key, value in type_map.items():
            anum = atomic_numbers[key]
            self.mass_line += 'mass %d %.4f\n' %(value+1, atomic_masses[anum])

        # structure data
        self.data_file = data

        # model line
        assert len(model) == 1
        model_type = list(model.keys())[0]
        model_value = list(model.values())[0]
        if model_type == 'deepmd':
            model_line = 'pair_style      deepmd %s out_freq ${THERMO_FREQ} out_file model_devi.out\n' \
                %(' '.join([str(p) for p in model_value]))
            model_line += 'pair_coeff    \n'
        elif model_type == 'reax/c':
            model_line = 'pair_style  reax/c NULL\n'
            model_line += 'pair_coeff  * * %s %s\n' %(model_value, ' '.join(list(type_map.keys())))
            model_line += "fix             2 all qeq/reax 1 0.0 10.0 1e-6 reax/c\n"
        else:
            raise ValueError('unsupported pair_style')
        self.model_type = model_type
        self.model_line = model_line

        # ensemble
        self.ensemble = ensemble

        # Be careful with units. 
        # dp uses metal while reax uses real
        self.nsteps = variables.get('nsteps', 1000)
        self.thermo_freq = variables.get('thermo_freq', 10)
        self.dtime = variables.get('dtime', 2) # unit ps in metal 
        self.temp = variables.get('temp', 300)
        self.pres = variables.get('pres', -1)
        self.tau_t = variables.get('tau_t', 0.1) # unit ps
        self.tau_p = variables.get('tau_p', 0.5) # ps

        # constraint
        self.constraint = constraint

        return 

    def __repr__(self):
        """"""
        # variables
        content = "variable        NSTEPS          equal %d\n" %self.nsteps
        content += "variable        THERMO_FREQ     equal %d\n" %self.thermo_freq
        content += "variable        DTIME           equal %f\n" %self.dtime
        content += "variable        TEMP            equal %f\n" %self.temp
        content += "variable        PRES            equal %f\n" %self.pres
        content += "variable        TAU_T           equal %f\n" %self.tau_t
        content += "variable        TAU_P           equal %f\n" %self.tau_p
        content += "\n"

        # md
        if self.model_type == 'deepmd':
            content += "units           metal\n"
            content += "boundary        p p p\n"
            content += "atom_style      atomic\n"
            content += "\n"
            content += "neighbor        1.0 bin\n"
        elif self.model_type == 'reax/c':
            content += "units           real\n"
            content += "boundary        p p p\n"
            content += "atom_style      charge\n"
            content += "\n"
            content += "neighbor            1.0 bin\n" # for npt, ghost atom issue
            content += "neigh_modify     every 10 delay 0 check no\n"

        content += "\n"
        content += "box          tilt large\n"
        content += "read_data    %s\n" %self.data_file
        content += "change_box   all triclinic\n"
        content += ""
        content += self.mass_line
        content += self.model_line
        content += "\n"
        if self.constraint is None:
            content += "thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz\n"
            content += "thermo          ${THERMO_FREQ}\n"
            content += "dump            1 all custom ${THERMO_FREQ} traj.dump id type x y z\n"
            content += "\n"
            content += "velocity        all create ${TEMP} %d\n" %(random.randrange(10000-1)+1)
        else:
            content += "group mobile id %s\n" %self.constraint[0]
            content += "group freezed id %s\n" %self.constraint[1]
            content += "\n"
            content += "compute mobileTemp mobile temp\n"
            content += "thermo_style    custom step c_mobileTemp pe ke etotal press vol lx ly lz xy xz yz\n"
            content += "thermo          ${THERMO_FREQ}\n"
            content += "dump            1 all custom ${THERMO_FREQ} traj.dump id type x y z\n"
            content += "\n"
            content += "velocity        mobile create ${TEMP} %d\n" %(random.randrange(10000-1)+1)
            content += "fix 3 freezed setforce 0.0 0.0 0.0\n"
        if self.ensemble == 'nvt':
            content += "fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}\n"
        elif self.ensemble == 'npt':
            content += "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P}\n"
        elif self.ensemble == 'nve':
            content += "fix             1 all nve \n"
        else:
            pass
        content += "\n"
        content += "timestep        ${DTIME}\n"
        content += "run             ${NSTEPS}\n"

        return content 

    def write(self, input_path):
        with open(input_path, 'w') as fopen:
            fopen.write(str(self))
        return


if __name__ == '__main__':
    #type_map = {'O': 0, 'Pt': 1}
    type_map = {'Pt': 0}
    model = {
        'deepmd': [
            '/users/40247882/projects/oxides/gdp-main/it-0002/ensemble-more/model-0/graph.pb',
            '/users/40247882/projects/oxides/gdp-main/it-0002/ensemble-more/model-1/graph.pb',
            '/users/40247882/projects/oxides/gdp-main/it-0002/ensemble-more/model-2/graph.pb',
            '/users/40247882/projects/oxides/gdp-main/it-0002/ensemble-more/model-3/graph.pb',
        ]
    }

    model = {
        'reax/c': '/users/40247882/projects/oxides/gdp-main/reaxff/ffield.reax.PtO'
    }

    sys_data = {
        'Pt': '/users/40247882/projects/oxides/gdp-main/init-systems/charge/O0Pt32.data',
        #'PtO': '/users/40247882/projects/oxides/gdp-main/init-systems/charge/O16Pt16.data',
        #'aPtO2': '/users/40247882/projects/oxides/gdp-main/init-systems/charge/O16Pt8.data',
        #'bPtO2': '/users/40247882/projects/oxides/gdp-main/init-systems/charge/O32Pt16.data',
        #'Pt3O4': '/users/40247882/projects/oxides/gdp-main/init-systems/charge/O64Pt48.data'
    }

    default_variables = dict(
        nsteps = 8000, 
        thermo_freq = 80, 
        dtime = 0.25, # fs
        temp = 300,
        tau_t = 10.
    )
    ensemble = 'nvt'

    # PtOx Bulk (2x2x2) system
    #temperatures = [300, 600, 1200, 1800]
    #temperatures = [100, 200, 300, 400, 500]
    #temperatures = np.arange(100,2400,300)
    temperatures = [150, 300, 450, 600,  900, 1200, 1500, 1800, 2100, 2400]

    main_dir = Path('/users/40247882/projects/oxides/gdp-main/reax-md')
    #main_dir.mkdir()

    for key, value in sys_data.items():
        data = value
        lmpin = LammpsInput(type_map, data, model, ensemble, default_variables)

        system_dir = main_dir / (key+'-'+ensemble)
        system_dir.mkdir()

        for temp in temperatures:
            temp_dir = system_dir / ('%d' %temp)
            temp_dir.mkdir()
            lmpin.temp = temp
            lmpin.write(temp_dir/'in.lammps')

        # submit
        # mpirun -n 4 lmp -in ./in.lammps 2>&1 > lmp.out
