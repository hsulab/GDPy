#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np

from ase.io import read, write
from ase.constraints import FixAtoms, FixBondLengths
from ase.calculators.emt import EMT
from ase.neb import NEB
#from ase.neb import SingleCalculatorNEB
from ase.optimize import BFGS
from ase.geometry import wrap_positions
from sklearn.preprocessing import StandardScaler
from torch import conv_transpose1d


class AccCons():

    """ initialsie random placements
        Constrained Structure Swarm Exploration
    """

    """ fort.188
        1
        3
        6
        4
        0.04
        1    3    1.766034
        0
    """

    default_parameters = dict(
        rep_infor = 3,
        precon = 4,
        step_length = 0.04,
        init_delay_times = 20,
        basic_delay = 5,
        delay_plus = 4,
        neighbourhood_radius = 1.6,
        perturbation_factor = 0.09579038,
        max_ndishis = 10
    )

    fmax = 0.05

    # indicators
    imark = 0
    imark_att = 0
    ndishis = 0 # current number of distance history
    fix_times = 0
    delaycount = 0

    # data
    old_dis = None
    cur_dis = None

    old_energy = None
    cur_energy = None

    old_forces = None
    cur_force = None

    # 0 - balance
    old_bond_status = 0
    old_force_status = 0

    cur_bond_status = 0
    cur_force_status = 0

    def __init__(
        self, atoms, cpairs: list,
        **kwargs
    ):

        self.atoms = atoms
        self.cpairs = cpairs

        # init dynamics
        self.dyn = BFGS(self.atoms, maxstep=0.2) # maxstep = POTIM

        # init constrain
        for key, value in self.default_parameters.items():
            new_value = kwargs.get(key, None)
            if new_value is not None:
                value = new_value
            setattr(self, key, value)

        return
    
    def __call__(self, maxiters=20):
        """"""
        # start optimization
        # niter = 0
        self.stopsearch = False
        self.delaycount = self.init_delay_times
        self.updated_records = []
        self.constrain_records = []

        # opt params
        isconstrain = True
        #isconstrain = False

        #iswrap = True
        iswrap = False

        # init system info
        self.init_constrain_info() # not calculation

        # wrap atoms in the first place
        scaled_positions = self.atoms.get_scaled_positions()
        pos_a, pos_b = scaled_positions[self.cpairs[0]].copy(), scaled_positions[self.cpairs[1]].copy()
        conscentre = 0.5*(pos_a + pos_b)
        poses_wrap = wrap_positions(self.atoms.get_positions().copy(), self.atoms.cell, pbc=True, center=conscentre)
        self.atoms.set_positions(poses_wrap)

        write("188.xyz", self.atoms)

        for istep in range(maxiters):
            print(f"\n\n===== step {istep} =====")
            # calculation
            # log current info and single point
            self.update_maxforces_information("----- Force Information -----\n")
            # bfgs optimization
            self.dyn.step(self.cur_forces)
            poses_opt = self.atoms.get_positions()
            print("first 0: ", poses_opt[0])
            # move atoms in cube 000
            # wrap_positions
            if iswrap:
                # maybe wrong with Hessian and dr
                scaled_positions = self.atoms.get_scaled_positions()
                print("pos 0: ", self.atoms.get_positions()[0])
                print("pos 0: ", np.dot(scaled_positions, self.atoms.cell.complete())[0])
                print(scaled_positions[0])
                pos_a, pos_b = scaled_positions[self.cpairs[0]].copy(), scaled_positions[self.cpairs[1]].copy()
                conscentre = 0.5*(pos_a + pos_b)
                print("cons: ", conscentre)
                poses_opt = wrap_positions(poses_opt.copy(), self.atoms.cell, pbc=True, center=conscentre)
                print("pos_opt: ", poses_opt[0])
                #poses_opt = self.atoms.get_positions(wrap=True, pbc=True, center=conscentre).copy()
            if isconstrain:
                # transfer info before opt to old info
                self.update_his_constrain_info()
                # because of opt
                self.cur_poses = poses_opt
                self.cur_dis = self.atoms.get_distance(*self.cpairs, mic=True, vector=False)
                print("cur_dis: {} fix_dis: {} opt_dis: {}".format(self.old_dis, self.fixed_dis, self.cur_dis))
                # fix distance and get fixed_distance
                # ----- ----- ----- ----- -----
                self.check_constrain_conditions("----- Check Constrain Conditions -----\n")
                # constrain positions
                poses_fix = self.constrain_distance(
                    poses_opt.copy(), self.cur_forces.copy()
                )
                poses_opt = poses_fix
                # if update hessian
                # set new positions 
                # ----- ----- ----- ----- -----
            self.atoms.set_positions(poses_opt) # ready to call QM calculation

            write("188.xyz", self.atoms, append=True)

        return

    def init_constrain_info(self):
        """ init constrain info
        """
        self.old_poses = self.atoms.get_positions()
        self.old_dis = self.atoms.get_distance(*self.cpairs, mic=True, vector=False)

        self.cur_poses = self.old_poses.copy()
        self.cur_dis = self.old_dis

        natoms = len(self.atoms)
        self.old_forces = np.zeros((natoms,3))
        self.old_energy = 0.0
        self.cur_forces = np.zeros((natoms,3))
        self.cur_energy = 0.0

        self.fixed_dis = self.old_dis.copy()

        # important
        self.predicted_array = np.zeros((2,self.max_ndishis))

        content = '='*100 + '\n'
        content += 'Input System Information,\n'
        #content += 'System of %d atoms, 2 constrained, %d free, %d frozen.\n'\
        #        %(natoms, self.catoms.nfreeatoms, \
        #        self.catoms.nfrozenatoms)
        content += "A index <- {}, B index <- {}.\n".format(*self.cpairs)
        content += 'Distance between A and B is %.4f Ang.\n' %self.cur_dis
        content += '{:<50}{:<50}\n'.format('POSITIONS', 'FORCES')
        #for pos, force in zip(self.cur_poses, self.cur_forces):
        #    pos_line, force_line = '  ', '  '
        #    for j in range(3):
        #        pos_line += '{:>16}'.format(round(pos[j],8))
        #        force_line += '{:>16}'.format(round(force[j],8))
        #    content += pos_line + force_line + '\n'
        content += '='*100 + '\n'
        print(content)

        return

    def get_bond_status(self, d1, d2):
        """"""
        if d2 < d1:
            bond_status = -1 # attractive
        else:
            bond_status = 1 # repulsive

        if np.abs(d2 - d1) <= 1e-20:
            bond_status = 0 # balanced

        return bond_status
    
    # Methods for bond status
    def get_constrained_bond_status(self, poses, forces):
        """
        Description:
            +1 - attractive, -1 - repulsive, 0 - balanced.
        """
        #
        eps = np.ones(3)*self.perturbation_factor

        index_a, index_b = self.cpairs

        # get positions and forces of a and b
        pos_a, for_a = poses[index_a], forces[index_a]
        pos_b, for_b = poses[index_b], forces[index_b]

        #
        d1 = np.linalg.norm(pos_a - pos_b)
        
        # perturbation
        pos_a = pos_a + for_a*eps
        pos_b = pos_b + for_b*eps

        #
        d2 = np.linalg.norm(pos_a - pos_b) # TODO: minimum image

        return self.get_bond_status(d1, d2)
    
    def update_maxforces_information(self, marksentence):
        """
        Description:
            Update the largest and second largest forces among all atoms, 
            along with their indices. The largest partial force among A 
            and B is also given.
        Called By: 
            self.cbfgs()
        IN:
            index_a, index_b, natoms, forces.
        OUT:
            self.cformax,
            self.formax1_index, self.formax2_index, 
            self.formax1_vec, self.formax2_vec, 
            self.formax1_norm, self.formax2_vec.
        """

        # ----- update current constrain info -----
        # from QM calculation
        self.cur_poses = self.atoms.get_positions()
        self.cur_forces = self.atoms.get_forces().copy() # forcecall
        self.cur_energy = self.atoms.get_potential_energy()

        self.cur_dis = self.atoms.get_distance(*self.cpairs, mic=True, vector=False)

        self.cur_bond_status = self.get_bond_status(self.old_dis, self.cur_dis)
        self.cur_force_status = self.get_constrained_bond_status(
            self.cur_poses, self.cur_forces
        )
        
        # ----- update forces info -----
        index_a, index_b = self.cpairs

        natoms = len(self.atoms)
        forces = self.cur_forces

        # find force info
        f1_index, f2_index = 0, 0
        f1_vec, f2_vec = np.array([]), np.array([])
        f1_norm, f2_norm = 0.0, 0.0

        for i in range(natoms):
            f_vec = forces[i]
            f_norm = np.linalg.norm(max(np.fabs(f_vec)))
            if f_norm > f1_norm:
                # move 1st to 2nd
                f2_norm, f2_index, f2_vec = f1_norm, f1_index, f1_vec
                # update 1st
                f1_norm, f1_index, f1_vec = f_norm, i, f_vec
            elif f_norm > f2_norm:
                f2_norm, f2_index, f2_vec = f_norm, i, f_vec

        # update
        self.cformax = max(max(forces[index_a]), max(forces[index_b]))
        self.formax1_index, self.formax2_index = f1_index, f2_index
        self.formax1_vec, self.formax2_vec = f1_vec, f2_vec
        self.formax1_norm, self.formax2_norm = f1_norm, f2_norm

        # log
        content = marksentence
        content += "Find the largest and second largest frac-force.\n"
        content += ("{:<15}  "*4+"\n").format('Max Force','Atom Index','Force Norm','Force Vector')
        content += ("{:<15}  "*2+"{:<15.4f}  "+"{:<.4f}  "*3+"\n").format(
            1,f1_index,np.linalg.norm(f1_vec),f1_vec[0],f1_vec[1],f1_vec[2]
        )
        content += ("{:<15}  "*2+"{:<15.4f}  "+"{:<.4f}  "*3+"\n").format(
            2,f2_index,np.linalg.norm(f2_vec),f2_vec[0],f2_vec[1],f2_vec[2]
        )
        print(content)

        return
    
    def update_his_constrain_info(self):
        """"""
        self.old_poses = self.cur_poses.copy()
        self.old_forces = self.cur_forces.copy()

        self.old_energy = self.cur_energy

        self.old_dis = self.cur_dis

        self.old_bond_status = self.cur_bond_status
        self.old_force_status = self.cur_force_status

        return

    
    def constrain_distance(self, poses_in, forces):
        """
        Description:
            Get atoms' positions after constrain.
        """
        poses = poses_in.copy()
        # get work variables
        natoms = poses.shape[0]
        index_a, index_b = self.cpairs

        # positions and forces
        cur_dis, fixed_dis = self.cur_dis, self.fixed_dis

        # get positions and forces of a and b
        pos_a, for_a = poses[index_a].copy(), forces[index_a].copy()
        pos_b, for_b = poses[index_b].copy(), forces[index_b].copy()

        # wrong !!! array object change in memory
        #pos_a, for_a = poses[index_a], forces[index_a].copy()
        #pos_b, for_b = poses[index_b], forces[index_b].copy()

        # calculate new positions of a and b
        lamb = -(np.linalg.norm(for_a) - np.linalg.norm(for_b)) / (np.linalg.norm(for_a) + np.linalg.norm(for_b))
        pos_a_new = (pos_a + pos_b) / 2.0 + \
                (lamb + fixed_dis/cur_dis - lamb*fixed_dis/cur_dis) * \
                (pos_a - pos_b) / 2.0
        pos_b_new = (pos_a + pos_b) / 2.0 + \
                (lamb - fixed_dis/cur_dis - lamb*fixed_dis/cur_dis) * \
                (pos_a - pos_b) / 2.0

        # move neighbours and log
        """
        radius = self.neighbourhood_radius
        poses, content = self.catoms.move_nearest_neighbours(
            natoms, poses,
            pos_a, pos_a_new, index_a, radius
        ) # move a's neighbours
        print(content)

        poses, content = self.catoms.move_nearest_neighbours(
            natoms, poses, \
            pos_b, pos_b_new, index_b, radius
        ) # move a's neighbours
        print(content)
        """

        # set new positions
        poses[index_a] = pos_a_new
        poses[index_b] = pos_b_new

        return poses

    def check_constrain_conditions(self, marksentence):
        """ constrain conditions
        """
        # force condition
        index_a, index_b = self.cpairs
        cond1 = (
            self.formax1_index == index_a and 
            self.formax2_index == index_b
        ) or (
            self.formax1_index == index_b and
            self.formax2_index == index_a
        )

        # delay condition
        cond2 = (self.delaycount <= 0)

        # consistent condition
        cond3 = (self.cur_bond_status * self.cur_force_status >= 0)

        # convergence condition
        cond4 = self.cformax > self.fmax

        # log
        content = marksentence
        content += '%6s-if constrained atoms have largest forces \n' %cond1
        content += '%6s-if delaycount is smaller than zero \n' %cond2
        content += '%6s-if bond status is consistent with force status \n' %cond3
        content += '%6s-if forces on constrained atoms are larger than ftol\n ' \
                %cond4
        print(content)

        # get predicted distance
        self.predicted_dis = self.get_predicted_distance()

        # change parameters
        old_dis, cur_dis = self.old_dis, self.cur_dis

        if not cond4:
            self.precon = 0.05

        isfixed = False
        if self.cur_bond_status == 1:
            # log
            print("NOTICE: Bond is repulsive, try to decrease distance.\n")

            # check conditions
            if cond1 and cond2 and cond3 and cond4:
                print('Now delay is done, record information.\n')
                # auto search
                self.rep_infor = self.rep_infor + 1
                print('Current RepTag is %d. Once 6, decrease distance.\n' \
                        %self.rep_infor)
                print('*'*100+'\n')

                # reach rep times
                if self.rep_infor == 6:
                    # initialize self.rep_infor
                    self.rep_infor = 3
                    # very close to TS
                    self.check_predicted_distance('repulsive')
                    # update
                    self.updated_records.append({'fix_times': self.fix_times, \
                            'distance_in': self.old_dis, \
                            'distance_out': self.predicted_dis})
                    self.fix_times = 0
                else:
                    self.fix_times += 1
            else:
                isfixed = True

        elif self.cur_bond_status == -1:
            # log
            print('NOTICE: Bond is attractive, try to increase distance.\n')

            if cond1 and cond2 and cond3 and cond4:
                # log
                self.rep_infor -= 1
                print('Current RepTag is %d. Once 0, increase distance.\n' \
                        %self.rep_infor)
                print('*'*100+'\n')
                if self.rep_infor == 0:
                    self.rep_infor = 3
                    # very close to TS
                    self.check_predicted_distance('attractive')
                    # update
                    self.updated_records.append({'fix_times': self.fix_times, \
                            'distance_in': self.old_dis, \
                            'distance_out': self.predicted_dis})
                    self.fix_times = 0
                else:
                    self.fix_times += 1
            else:
                isfixed = True

        elif self.cur_bond_status == 0:
            print('NOTICE: Bond is banlanced, do nothing.\n')
            isfixed = True
        
        if isfixed:
            # update
            self.fix_times += 1
            
            # log
            content = '*'*100+'\n' 
            content += 'Constrain: Remain distance.\n' 
            content += 'Fixed Distance: %.4f at %d delay times.\n' \
                    %(self.fixed_dis, self.delaycount)
            content += '*'*100+'\n'
            print(content)

        # set delaycount
        self.delaycount -= 1
    
        # distance
    def get_predicted_distance(self):
        """
        use history to predict distance
        Here, old_dis - input, cur_dis - after opt
        """
        #
        old_dis, cur_dis = self.old_dis, self.cur_dis
        step_length = self.step_length

        # array 0 1 2 3 4 5 6 7 8 9
        # dis   new <- old
        predicted_array = self.predicted_array

        # update predicted array
        self.ndishis += 1
        if self.ndishis >= self.max_ndishis:
            self.ndishis = self.max_ndishis
        if self.ndishis > 1:
            predicted_array = self.shift_predicted_array(predicted_array)

        p1 = self.formax1_norm**2
        p2 = old_dis - (cur_dis - old_dis)*(1.0 + step_length*10.0)
        predicted_array[0,0] = p1 
        predicted_array[1,0] = p2

        #  get predicted distance
        #predicted_dis = old_dis
        for nhis in range(self.ndishis,0,-1):
            dtmp, dtmp1 = 0.0, 0.0
            for i in range(nhis):
                dtmp += 1.0/predicted_array[0,i]
            for i in range(nhis):
                dtmp1 += 1.0/predicted_array[0,i]/dtmp*predicted_array[1,i]
            # check consistent
            if (old_dis - dtmp1)*(cur_dis - old_dis) > 0:
                if np.abs(dtmp1 - old_dis) < step_length:
                    predicted_dis = dtmp1
                else:
                    predicted_dis = old_dis + (dtmp1 - old_dis) / \
                            np.abs(dtmp1 - old_dis) * step_length
                break
        else:
            predicted_dis = old_dis - (cur_dis - old_dis) * \
                    (1.0 + step_length*10.0)
        
        # refine predicted distance
        ddis = predicted_dis - old_dis
        if ddis >= 0:
            ddis_sign = 1.0
        else:
            ddis_sign = -1.0

        if abs(ddis) < 0.01 and predicted_array[0,0] > 0.01:
            predicted_dis = old_dis + ddis_sign * \
                    min(max(abs(ddis*10), 0.005), 0.02) * step_length/0.05
            print('Small distance change.\n')

        ddis = predicted_dis - old_dis
        if abs(ddis) < 0.0025:
            predicted_dis = old_dis + ddis/abs(ddis)*0.0025

        # log
        content = 'Current constrained distance is %.4f.\n' %self.fixed_dis
        content += 'Has already collected %d history.\n' %self.ndishis
        content += 'Use %d distance history. Predicted Distance: %.4f Ang\n' \
                %(nhis, predicted_dis)
        print(content)

        # update
        self.predicted_array = predicted_array
        self.ndishis = nhis

        return predicted_dis
        # return self.old_dis**2 / self.cur_dis
    
    def shift_predicted_array(self, predicted_array):
        """"""
        max_ndishis = self.max_ndishis
        for i in range(1, max_ndishis):
            predicted_array[0,i] = predicted_array[0,i-1]
            predicted_array[1,i] = predicted_array[1,i-1]
        return predicted_array
    
    def check_predicted_distance(self, status_tag):
        """change search parameters to give more subtle performance."""
        # get marks
        if status_tag == 'repulsive':
            mark1, mark2 = self.imark, self.imark_att 
        elif status_tag == 'attractive':
            mark1, mark2 = self.imark_att, self.imark
        else:
            raise ValueError('Wrong status tag.')

        # update distance
        dtmp = self.fixed_dis # distance before opt
        # overwrite codes for update distance
        self.fixed_dis = self.predicted_dis
        if abs(dtmp - self.fixed_dis) > 0.005:
            self.ireset_hessian = 1

        # log
        content = 'Constrain: Update distance.\n'
        content += 'Successfully change distance from %.4f to %.4f.\n' \
                %(dtmp, self.fixed_dis)
        content += 'Predicted Distance: %.4f at %d fixed times \
        and %d delay times.\n' \
                %(self.predicted_dis, self.fix_times, self.delaycount)

        # update
        mark1 += 1 # change tag

        # change precon
        if self.precon != 2:
            self.delaycount = self.basic_delay + self.delay_plus
        self.precon /= 2.0
        if self.precon < 0.005:
            self.precon = 0.01

        # refine precon, which controls the step length for bond change
        # check if bond is oscillating between attractive and repulsive
        if mark2 >= 1:
            content += 'Overshoot! Too %s the bond.\n' %status_tag
            mark2 = 0
        # reduce precon after continuing increase or decrease
        if mark1 == 2 and self.precon == 0.5:
            content += 'Remain %s after change at precon==2.\n' %status_tag
            mark1 = 0
            self.precon = 2
        # still after 3 changes, give more delay times to find MEP
        if mark1 == 3:
            content += 'still %s after 3 changes.\n' %status_tag
            mark1 = 0
            self.ireset_hessian = 1
            self.delaycount = self.basic_delay + self.delay_plus
            if self.cformax > 0.1:
                self.precon = 0.4
            else:
                self.delaycount = self.basic_delay + self.delay_plus + 1

        content += 'Delay times %d. Precon %.4f\n' \
                %(self.delaycount, self.precon)

        # log
        content += '*'*100+'\n'
        print(content)

        # update marks
        if status_tag == 'repulsive':
            self.imark, self.imark_att = mark1, mark2
        elif status_tag == 'attractive':
            self.imark_att, self.imark = mark1, mark2
        else:
            raise ValueError('Wrong status tag.')

        return


def test_cons(atoms, calc):
    atoms.calc = calc
    atoms.set_constraint(
        FixBondLengths(
            pairs = [[0, 23]],
            bondlengths = [2.426]
        )
    )

    qn = BFGS(atoms, trajectory="cons.traj")
    qn.run(fmax=0.05, steps=50)

    return



if __name__ == "__main__":
    # test LASP-NEB
    from gdpx.computation.lasp import LaspNN
    pot_path = "/mnt/scratch2/users/40247882/catsign/lasp-main/ZnCrOCH.pot"
    pot = dict(
        C  = pot_path,
        O  = pot_path,
        Cr = pot_path,
        Zn = pot_path
    )
    calc = LaspNN(
        directory = "./LaspNN-Worker",
        command = "mpirun -n 4 lasp",
        pot=pot
    )

    initial = read("/mnt/scratch2/users/40247882/catsign/ZnOx/P188/CO-IS.xyz")
    transition = read("/mnt/scratch2/users/40247882/catsign/ZnOx/P188/CO-TS.xyz")
    final = read("/mnt/scratch2/users/40247882/catsign/ZnOx/P188/CO-FS.xyz")

    transition = read("/mnt/scratch2/users/40247882/catsign/ZnOx/cases/TStest.xyz")

    cons_indices = list(range(1,13)) + list(range(24,36))
    print("indices: ", cons_indices)
    constraint = FixAtoms(indices=cons_indices) # first atom is O
    #constraint = FixAtoms(indices=[1,2,3,4,5,6,7,8]) # first atom is O

    transition.set_constraint(constraint)
    transition.calc = calc

    #dyn = BFGS(transition, maxstep=0.2) # maxstep = POTIM
    #dyn.run(0.05, 200)
    #exit()

    maxiters = int(sys.argv[1])

    # test_cons(transition, calc)
    constrdyn = AccCons(transition, cpairs=[0,23])
    constrdyn(maxiters)