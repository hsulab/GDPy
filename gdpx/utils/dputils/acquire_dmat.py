#!/usr/bin/env python3

import numpy as np
from deepmd.env import tf
from deepmd.env import default_tf_session_config
from deepmd.common import make_default_mesh
from deepmd.DeepEval import DeepEval
from deepmd.DataModifier import DipoleChargeModifier
from tensorflow.python.framework.ops import _DTYPES_INTERN_TABLE
from tensorflow.python.ops.gen_sparse_ops import take_many_sparse_from_tensors_map
from tensorflow.python.ops.variable_scope import variable_op_scope

class DeepDescriptor (DeepEval) :
    def __init__(self, 
                 model_file, 
                 default_tf_graph = False) :
        DeepEval.__init__(self, model_file, default_tf_graph = default_tf_graph)
        # self.model_file = model_file
        # self.graph = self.load_graph (self.model_file)
        # checkout input/output tensors from graph
        self.t_ntypes = self.graph.get_tensor_by_name ('load/descrpt_attr/ntypes:0')
        self.t_rcut   = self.graph.get_tensor_by_name ('load/descrpt_attr/rcut:0')
        self.t_dfparam= self.graph.get_tensor_by_name ('load/fitting_attr/dfparam:0')
        self.t_daparam= self.graph.get_tensor_by_name ('load/fitting_attr/daparam:0')
        self.t_tmap   = self.graph.get_tensor_by_name ('load/model_attr/tmap:0')
        # inputs
        self.t_coord  = self.graph.get_tensor_by_name ('load/t_coord:0')
        self.t_type   = self.graph.get_tensor_by_name ('load/t_type:0')
        self.t_natoms = self.graph.get_tensor_by_name ('load/t_natoms:0')
        self.t_box    = self.graph.get_tensor_by_name ('load/t_box:0')
        self.t_mesh   = self.graph.get_tensor_by_name ('load/t_mesh:0')
        # outputs
        self.t_energy = self.graph.get_tensor_by_name ('load/o_energy:0')
        self.t_force  = self.graph.get_tensor_by_name ('load/o_force:0')
        self.t_virial = self.graph.get_tensor_by_name ('load/o_virial:0')
        self.t_ae     = self.graph.get_tensor_by_name ('load/o_atom_energy:0')
        self.t_av     = self.graph.get_tensor_by_name ('load/o_atom_virial:0')
        self.t_fparam = None
        self.t_aparam = None
        # feature tensors 
        self.t_neilist = self.graph.get_tensor_by_name ('load/o_nlist:0')
        self.t_rmat = self.graph.get_tensor_by_name ('load/o_rmat:0')
        self.t_rmat_deriv = self.graph.get_tensor_by_name ('load/o_rmat_deriv:0')
        self.t_descriptor = self.graph.get_tensor_by_name ('load/o_descriptor:0')
        #self.t_dmat = self.graph.get_tensor_by_name ('load/o_dmat:0')
        #self.t_dmat_deriv = self.graph.get_tensor_by_name ('load/o_dmat_deriv:0')
        # check if the graph has fparam
        for op in self.graph.get_operations():
            if op.name == 'load/t_fparam' :
                self.t_fparam = self.graph.get_tensor_by_name ('load/t_fparam:0')
        self.has_fparam = self.t_fparam is not None
        # check if the graph has aparam
        for op in self.graph.get_operations():
            if op.name == 'load/t_aparam' :
                self.t_aparam = self.graph.get_tensor_by_name ('load/t_aparam:0')
        self.has_aparam = self.t_aparam is not None
        # start a tf session associated to the graph
        self.sess = tf.Session (graph = self.graph, config=default_tf_session_config)        
        [self.ntypes, self.rcut, self.dfparam, self.daparam, self.tmap] = self.sess.run([self.t_ntypes, self.t_rcut, self.t_dfparam, self.t_daparam, self.t_tmap])
        self.tmap = self.tmap.decode('UTF-8').split()
        # setup modifier
        try:
            t_modifier_type = self.graph.get_tensor_by_name('load/modifier_attr/type:0')
            self.modifier_type = self.sess.run(t_modifier_type).decode('UTF-8')
        except ValueError:
            self.modifier_type = None
        except KeyError:
            self.modifier_type = None
        if self.modifier_type == 'dipole_charge':
            t_mdl_name = self.graph.get_tensor_by_name('load/modifier_attr/mdl_name:0')
            t_mdl_charge_map = self.graph.get_tensor_by_name('load/modifier_attr/mdl_charge_map:0')
            t_sys_charge_map = self.graph.get_tensor_by_name('load/modifier_attr/sys_charge_map:0')
            t_ewald_h = self.graph.get_tensor_by_name('load/modifier_attr/ewald_h:0')
            t_ewald_beta = self.graph.get_tensor_by_name('load/modifier_attr/ewald_beta:0')
            [mdl_name, mdl_charge_map, sys_charge_map, ewald_h, ewald_beta] = self.sess.run([t_mdl_name, t_mdl_charge_map, t_sys_charge_map, t_ewald_h, t_ewald_beta])
            mdl_charge_map = [int(ii) for ii in mdl_charge_map.decode('UTF-8').split()]
            sys_charge_map = [int(ii) for ii in sys_charge_map.decode('UTF-8').split()]
            self.dm = DipoleChargeModifier(mdl_name, mdl_charge_map, sys_charge_map, ewald_h = ewald_h, ewald_beta = ewald_beta)
        

    def get_ntypes(self) :
        return self.ntypes

    def get_rcut(self) :
        return self.rcut

    def get_dim_fparam(self) :
        return self.dfparam

    def get_dim_aparam(self) :
        return self.daparam

    def get_type_map(self):
        return self.tmap

    def eval(self,
             coords,
             cells,
             atom_types,
             fparam = None,
             aparam = None,
             atomic = False, 
             feature = False, 
        ) :
        if atomic :
            if self.modifier_type is not None:
                raise RuntimeError('modifier does not support atomic modification')
            ret = self.eval_inner(coords, cells, atom_types, fparam = fparam, aparam = aparam, atomic = atomic, feature=feature)
        else :
            ret = self.eval_inner(coords, cells, atom_types, fparam = fparam, aparam = aparam, atomic = atomic, feature=feature)
            e, f, v = ret[0], ret[1], ret[2] 
            if self.modifier_type is not None:
                me, mf, mv = self.dm.eval(coords, cells, atom_types)
                e += me.reshape(e.shape)
                f += mf.reshape(f.shape)
                v += mv.reshape(v.shape)
            ret[0], ret[1], ret[2] = e, f, v 
        return ret 

    def eval_inner(self,
             coords, 
             cells, 
             atom_types, 
             fparam = None, 
             aparam = None, 
             atomic = False, 
             feature = False, 
        ) :
        # standarize the shape of inputs
        atom_types = np.array(atom_types, dtype = int).reshape([-1])
        natoms = atom_types.size
        coords = np.reshape(np.array(coords), [-1, natoms * 3])
        nframes = coords.shape[0]
        if cells is None:
            pbc = False
            # make cells to work around the requirement of pbc
            cells = np.tile(np.eye(3), [nframes, 1]).reshape([nframes, 9])
        else:
            pbc = True
            cells = np.array(cells).reshape([nframes, 9])
        
        if self.has_fparam :
            assert(fparam is not None)
            fparam = np.array(fparam)
        if self.has_aparam :
            assert(aparam is not None)
            aparam = np.array(aparam)

        # reshape the inputs 
        if self.has_fparam :
            fdim = self.get_dim_fparam()
            if fparam.size == nframes * fdim :
                fparam = np.reshape(fparam, [nframes, fdim])
            elif fparam.size == fdim :
                fparam = np.tile(fparam.reshape([-1]), [nframes, 1])
            else :
                raise RuntimeError('got wrong size of frame param, should be either %d x %d or %d' % (nframes, fdim, fdim))
        if self.has_aparam :
            fdim = self.get_dim_aparam()
            if aparam.size == nframes * natoms * fdim:
                aparam = np.reshape(aparam, [nframes, natoms * fdim])
            elif aparam.size == natoms * fdim :
                aparam = np.tile(aparam.reshape([-1]), [nframes, 1])
            elif aparam.size == fdim :
                aparam = np.tile(aparam.reshape([-1]), [nframes, natoms])
            else :
                raise RuntimeError('got wrong size of frame param, should be either %d x %d x %d or %d x %d or %d' % (nframes, natoms, fdim, natoms, fdim, fdim))

        # sort inputs
        coords, atom_types, imap = self.sort_input(coords, atom_types)

        # make natoms_vec and default_mesh
        natoms_vec = self.make_natoms_vec(atom_types)
        assert(natoms_vec[0] == natoms)

        # evaluate
        energy = []
        force = []
        virial = []
        ae = []
        av = []

        descrs = [] 
        dmats = []
        dmats_deriv = []
        neilists = [] 
        rmats = []
        rmats_deriv = []

        feed_dict_test = {}
        feed_dict_test[self.t_natoms] = natoms_vec
        feed_dict_test[self.t_type  ] = atom_types
        t_out = [self.t_energy, 
                 self.t_force, 
                 self.t_virial]
        if atomic :
            t_out += [self.t_ae, 
                      self.t_av]
        if feature:
            t_out += [self.t_neilist]
            t_out += [self.t_rmat, self.t_rmat_deriv]
            t_out += [self.t_descriptor] 

        for ii in range(nframes) :
            feed_dict_test[self.t_coord] = np.reshape(coords[ii:ii+1, :], [-1])
            feed_dict_test[self.t_box  ] = np.reshape(cells [ii:ii+1, :], [-1])
            if pbc:
                feed_dict_test[self.t_mesh ] = make_default_mesh(cells[ii:ii+1, :])
            else:
                feed_dict_test[self.t_mesh ] = np.array([], dtype = np.int32)
            if self.has_fparam:
                feed_dict_test[self.t_fparam] = np.reshape(fparam[ii:ii+1, :], [-1])
            if self.has_aparam:
                feed_dict_test[self.t_aparam] = np.reshape(aparam[ii:ii+1, :], [-1])
            v_out = self.sess.run (t_out, feed_dict = feed_dict_test) # actually compute the graph 
            energy.append(v_out[0])
            force .append(v_out[1])
            virial.append(v_out[2])
            if atomic:
                ae.append(v_out[3])
                av.append(v_out[4])
            if feature: 
                #print(v_out[-3].shape)
                neilists.append(v_out[-4])
                rmats.append(v_out[-3])
                rmats_deriv.append(v_out[-2])
                descrs.append(v_out[-1]) 

        # reverse map of the outputs
        force  = self.reverse_map(np.reshape(force, [nframes,-1,3]), imap)
        if atomic :
            ae  = self.reverse_map(np.reshape(ae, [nframes,-1,1]), imap)
            av  = self.reverse_map(np.reshape(av, [nframes,-1,9]), imap)
        # reverse dmats 
        neilists = self.reverse_map(np.reshape(neilists, [nframes,-1,240*2]), imap) # sel*axis_neuronesc 
        rmats = self.reverse_map(np.reshape(rmats, [nframes,-1,240*4*2]), imap) # sel*axis_neuronesc 
        rmats_deriv = self.reverse_map(np.reshape(rmats_deriv, [nframes,-1,240*4*3*2]), imap) # sel*axis_neurons 
        descrs = self.reverse_map(np.reshape(descrs, [nframes,-1,16*100]), imap) # sel*axis_neurons 

        energy = np.reshape(energy, [nframes, 1])
        force = np.reshape(force, [nframes, natoms, 3])
        virial = np.reshape(virial, [nframes, 9])
        if atomic:
            ae = np.reshape(ae, [nframes, natoms, 1])
            av = np.reshape(av, [nframes, natoms, 9])
            ret = [energy, force, virial, ae, av] 
        else :
            ret = [energy, force, virial] 

        if feature:
            ret.extend([neilists,rmats,rmats_deriv,descrs])

        return ret
    

def atoms2dpinput(type_map, frames):
    """"""
    coords, cells, atypes = [], [], []
    for atoms in frames:
        coord = atoms.positions.reshape([1,-1])
        cell = atoms.cell.reshape([1,-1])
        atype = [type_map[s] for s in atoms.get_chemical_symbols()]
        coords.append(coord)
        cells.append(cell)
        atypes.append(atype)

    dpinput = {
        'coords': coords,
        'cells': cells,
        'atom_types': atypes[0]
    }

    return dpinput

def find_neighbours(atoms, xdp):
    """"""
    type_map = {'O': 0, 'Pt': 1}
    dpinput = atoms2dpinput(type_map, [atoms])

    e, f, v, ae, av, *rets = xdp.eval(**dpinput, atomic=True, feature=True)
    descrs = rets[2] # descriptor

    print('energy', e)
    """
    print(descrs[0].shape)
    environ_oxygen = descrs[0][0]
    # print(ae)
    rmat = rets[0][0][0]
    print(rmat.shape)
    #print(rmat[0::4])
    print(np.sum(rmat > 0))
    """
    neilist = rets[0][0][0]
    print(np.sum(neilist >= 0))
    print(rets[0][0].shape)
    print('neighbours: ', neilist[240:300])

    return

if __name__ == '__main__':
    model = '/users/40247882/projects/oxides/gdp-main/it-0011/ensemble/model-0n/graph.pb'
    xdp = DeepDescriptor(model)
    print('rcut', xdp.get_rcut())

    from ase.io import read, write
    print('O on FCC')
    atoms = read('/users/40247882/repository/gdpx/templates/structures/surf-100_opt.xyz')
    find_neighbours(atoms, xdp)


    print('O on HCP')
    atoms = read('/users/40247882/repository/gdpx/templates/structures/surf-1O-2_opt.xyz')
    type_map = {'O': 0, 'Pt': 1}
    dpinput = atoms2dpinput(type_map, [atoms])

    e, f, v, ae, av, *rets = xdp.eval(**dpinput, atomic=True, feature=True)
    descrs = rets[2] # descriptor

    print('energy', e)
    print(descrs[0].shape)
    environ_oxygen2 = descrs[0][0]
    # print(ae)
    rmat = rets[0][0][0]
    print(rmat.shape)
    #print(rmat[0::4])

    print('absolute value')
    print(np.linalg.norm(environ_oxygen))
    print(np.linalg.norm(environ_oxygen2))
    env_diff = np.linalg.norm(environ_oxygen - environ_oxygen2)
    print(env_diff)
    print('Difference Statistics')
    print(np.mean(environ_oxygen - environ_oxygen2))
    print(np.var(environ_oxygen - environ_oxygen2))
    print(np.max(environ_oxygen - environ_oxygen2))
    print(np.min(environ_oxygen - environ_oxygen2))

    exit() 
    #print(xdp.graph)
    #for op in xdp.graph.get_operations():
    #    print(op.name, op.values())
    #exit()
    from ase.io import read, write 
    #frames = read('./dp_raw.xyz', '-3:')
    frames = read('../dp_raw.xyz', ':10')

    coords, cells, atypes = [], [], []
    for idx in range(1):
        atoms = frames[idx]
        natoms = len(atoms)
        coord = atoms.positions.reshape([1,-1])
        cell = atoms.cell.reshape([1,-1])
        atype = [0]*natoms 
        coords.append(coord)
        cells.append(cell)
        atypes.append(atype)
    #print(coord)
    #print(cell)
    e, f, v, *rets = xdp.eval(coords, cells, atypes, feature=True)
    #print(e)
    #print(f)
    #print(v)
    dmats = rets[-3]  # my dmat_reshaped
    print(dmats.shape)
    print(dmats)
    #print(neilists[0][0])
    #print(neilists[0][1])
    
