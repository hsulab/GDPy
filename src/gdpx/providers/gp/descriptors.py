import itertools

import numpy as np
from ase.neighborlist import NeighborList


def _build_species_array(frames):
    species = []
    for atoms in frames:
        species.extend(atoms.get_chemical_symbols())
    return np.array(species)


def _species_pair(sp_i, sp_j):
    a, b = sorted((sp_i, sp_j))
    return f"{a}-{b}"


def _species_triple(sp_i, sp_j, sp_k):
    a, b, c = sorted((sp_i, sp_j, sp_k))
    return f"{a}-{b}-{c}"


def compute_2body_descriptor(frames, r_cut, species_array):
    dimer_list = []
    curr_atomic_index = 0
    for i_frame, atoms in enumerate(frames):
        natoms = len(atoms)
        nl = NeighborList(
            cutoffs=[r_cut / 2.0] * natoms, skin=0.0, sorted=False,
            self_interaction=False, bothways=True,
        )
        nl.update(atoms)
        for i in range(natoms):
            nei_indices, nei_offsets = nl.get_neighbors(i)
            for j, offset in zip(nei_indices, nei_offsets):
                pos_i = atoms.positions[i]
                pos_j = atoms.positions[j]
                shift = np.dot(offset, atoms.get_cell())
                dimer_list.append(
                    (i_frame, curr_atomic_index + i, curr_atomic_index + j, pos_i, pos_j, shift)
                )
        curr_atomic_index += natoms

    mapping = np.array([[d[0], d[1], d[2]] for d in dimer_list])
    vectors = np.array([d[3] - (d[4] + d[5]) for d in dimer_list])
    distances = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_vecs = vectors / np.maximum(distances, 1e-12)
    gradients = np.concatenate([unit_vecs, -unit_vecs], axis=-1)
    species_pairs = np.array(
        [_species_pair(species_array[d[1]], species_array[d[2]]) for d in dimer_list],
        dtype=object,
    )

    return mapping, distances, gradients, species_pairs


def compute_3body_descriptor(frames, r_cut, species_array):
    dimer_data = []
    curr_atomic_index = 0
    for i_frame, atoms in enumerate(frames):
        natoms = len(atoms)
        nl = NeighborList(
            cutoffs=[r_cut / 2.0] * natoms, skin=0.0, sorted=False,
            self_interaction=False, bothways=True,
        )
        nl.update(atoms)
        for i in range(natoms):
            nei_indices, nei_offsets = nl.get_neighbors(i)
            for j, offset in zip(nei_indices, nei_offsets):
                pos_i = atoms.positions[i]
                pos_j = atoms.positions[j]
                shift = np.dot(offset, atoms.get_cell())
                dimer_data.append(
                    (i_frame, curr_atomic_index + i, curr_atomic_index + j, pos_i, pos_j, shift)
                )
        curr_atomic_index += natoms

    trimer_data = []
    for k, v in itertools.groupby(dimer_data, key=lambda x: x[1]):
        vlist = list(v)
        for pair0, pair1 in itertools.combinations(vlist, 2):
            pos_j = pair0[4] + pair0[5]
            pos_k = pair1[4] + pair1[5]
            dist_jk = np.linalg.norm(pos_j - pos_k)
            if 1e-8 < dist_jk <= r_cut:
                trimer_data.append((pair0, pair1))

    mapping = []
    vectors = []
    species_triples = []
    for p0, p1 in trimer_data:
        mapping.append([p0[0], p0[1], p0[2], p1[2]])
        v0 = p0[3] - (p0[4] + p0[5])
        v1 = p1[3] - (p1[4] + p1[5])
        v2 = (p0[4] + p0[5]) - (p1[4] + p1[5])
        vectors.append([v0, v1, v2])
        species_triples.append(
            _species_triple(species_array[p0[1]], species_array[p0[2]], species_array[p1[2]])
        )

    if vectors:
        vectors = np.array(vectors)
        distances = np.linalg.norm(vectors, axis=2)
        unit_vecs = vectors / np.maximum(distances[:, :, np.newaxis], 1e-12)
        gradients = np.concatenate([unit_vecs, -unit_vecs], axis=-1)
    else:
        distances = np.empty((0, 3))
        gradients = np.empty((0, 3, 9))

    return np.array(mapping), distances, gradients, np.array(species_triples, dtype=object)


def compute_descriptors(frames, r_cut_2b, r_cut_3b):
    species_array = _build_species_array(frames)
    b2_map, b2_dist, b2_grad, b2_sp = compute_2body_descriptor(frames, r_cut_2b, species_array)
    b3_map, b3_dist, b3_grad, b3_sp = compute_3body_descriptor(frames, r_cut_3b, species_array)
    return {
        "body2_mapping": b2_map,
        "body2_features": b2_dist,
        "body2_gradients": b2_grad,
        "body2_species": b2_sp,
        "body3_mapping": b3_map,
        "body3_features": b3_dist,
        "body3_gradients": b3_grad,
        "body3_species": b3_sp,
        "species_array": species_array,
    }
