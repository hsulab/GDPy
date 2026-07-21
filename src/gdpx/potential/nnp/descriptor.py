from collections import namedtuple

import numpy as np
from ase.neighborlist import neighbor_list


G2Param = namedtuple("G2Param", ["eta", "Rs"])
G4Param = namedtuple("G4Param", ["eta", "zeta", "lambda_"])


def _normalize_params(params, param_type):
    out = []
    for p in params:
        if isinstance(p, dict):
            out.append(param_type(**p))
        elif isinstance(p, (tuple, list)):
            out.append(param_type(*p))
        elif isinstance(p, param_type):
            out.append(p)
        else:
            raise TypeError(
                f"Expected {param_type.__name__}, dict, or tuple, got {type(p)}"
            )
    return out


def _cutoff_fn(r, r_cut):
    r = np.asarray(r, dtype=float)
    result = np.zeros_like(r)
    mask = (0.0 < r) & (r < r_cut)
    result[mask] = 0.5 * (np.cos(np.pi * r[mask] / r_cut) + 1.0)
    return result


def compute_n_features(elements, g2_params, g4_params):
    n_elem = len(elements)
    n_g2 = len(g2_params)
    n_g4 = len(g4_params)
    n_pairs = n_elem * (n_elem + 1) // 2
    return n_elem * n_g2 + n_pairs * n_g4


def compute_symmetry_functions(atoms, elements, g2_params, g4_params, r_cut):
    g2_params = _normalize_params(g2_params, G2Param)
    g4_params = _normalize_params(g4_params, G4Param)

    natoms = len(atoms)
    symbols = np.array(atoms.get_chemical_symbols())
    n_elem = len(elements)
    n_g2 = len(g2_params)
    n_g4 = len(g4_params)
    n_pairs = n_elem * (n_elem + 1) // 2
    n_features = n_elem * n_g2 + n_pairs * n_g4
    desc = np.zeros((natoms, n_features))

    i, j, rij, rij_vec = _get_neighbor_info(atoms, r_cut)
    fc = _cutoff_fn(rij, r_cut)

    _fill_g2(desc, i, j, rij, fc, symbols, elements, g2_params, n_g2)
    _fill_g4(desc, i, j, rij, rij_vec, fc, r_cut, symbols, elements,
             g4_params, n_g2, n_g4)

    return desc


def _get_neighbor_info(atoms, r_cut):
    i, j, Dij, S = neighbor_list("ijDS", atoms, cutoff=r_cut,
                                  self_interaction=False)
    cell = atoms.get_cell()
    shifts = S @ cell
    rij_vec = atoms.positions[j] - atoms.positions[i] + shifts
    rij = np.linalg.norm(rij_vec, axis=1)
    return i, j, rij, rij_vec


def _fill_g2(desc, i, j, rij, fc, symbols, elements, g2_params, n_g2):
    for ie, ne in enumerate(elements):
        nei_mask = symbols[j] == ne
        offset = ie * n_g2
        for ip, p in enumerate(g2_params):
            contrib = np.exp(-p.eta * (rij - p.Rs) ** 2) * fc
            for idx in range(len(i)):
                if nei_mask[idx]:
                    desc[i[idx], offset + ip] += contrib[idx]


def _pair_index(elements, ej, ek):
    a = elements.index(ej)
    b = elements.index(ek)
    i, j = (a, b) if a <= b else (b, a)
    n = len(elements)
    return i * n - i * (i - 1) // 2 + (j - i)


def _fill_g4(desc, i, j, rij, rij_vec, fc, r_cut, symbols, elements,
             g4_params, n_g2, n_g4):
    natoms = len(symbols)
    n_elem = len(elements)
    g4_offset = n_elem * n_g2

    for ci in range(natoms):
        nei_mask = i == ci
        nei_idx = np.where(nei_mask)[0]
        n_n = len(nei_idx)
        if n_n < 2:
            continue
        for a in range(n_n):
            for b in range(a + 1, n_n):
                ia = nei_idx[a]
                ib = nei_idx[b]
                ja = j[ia]
                jb = j[ib]
                ej, ek = symbols[ja], symbols[jb]
                if ej not in elements or ek not in elements:
                    continue
                pid = _pair_index(elements, ej, ek)
                rija = rij[ia]
                rikb = rij[ib]
                rjk_vec = rij_vec[ia] - rij_vec[ib]
                rjk = np.linalg.norm(rjk_vec)
                if rjk > r_cut:
                    continue
                fcij = fc[ia]
                fcik = fc[ib]
                fcjk = _cutoff_fn(rjk, r_cut)
                dot = np.dot(rij_vec[ia], rij_vec[ib])
                cos_theta = np.clip(dot / (rija * rikb + 1e-15), -1.0, 1.0)

                for ip, p in enumerate(g4_params):
                    val = (2.0 ** (1.0 - p.zeta)
                           * (1.0 + p.lambda_ * cos_theta) ** p.zeta
                           * np.exp(-p.eta * (rija ** 2 + rikb ** 2 + rjk ** 2))
                           * fcij * fcik * fcjk)
                    desc[ci, g4_offset + pid * n_g4 + ip] += val
