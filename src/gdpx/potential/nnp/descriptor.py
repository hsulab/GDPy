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


def _d_cutoff_fn(r, r_cut):
    r = np.asarray(r, dtype=float)
    result = np.zeros_like(r)
    mask = (0.0 < r) & (r < r_cut)
    result[mask] = -0.5 * np.pi / r_cut * np.sin(np.pi * r[mask] / r_cut)
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


def compute_forces(atoms, elements, g2_params, g4_params, r_cut, dE_dG):
    g2_params = _normalize_params(g2_params, G2Param)
    g4_params = _normalize_params(g4_params, G4Param)

    natoms = len(atoms)
    symbols = np.array(atoms.get_chemical_symbols())
    n_elem = len(elements)
    n_g2 = len(g2_params)
    n_g4 = len(g4_params)
    n_pairs = n_elem * (n_elem + 1) // 2
    forces = np.zeros((natoms, 3))

    i, j, rij, rij_vec = _get_neighbor_info(atoms, r_cut)
    fc = _cutoff_fn(rij, r_cut)
    dfc = _d_cutoff_fn(rij, r_cut)

    _g2_forces(forces, i, j, rij, rij_vec, fc, dfc, symbols,
               elements, g2_params, n_g2, dE_dG)
    _g4_forces(forces, i, j, rij, rij_vec, fc, dfc, r_cut, symbols,
               elements, g4_params, n_g2, n_g4, dE_dG)

    return forces


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


def _g2_forces(forces, i, j, rij, rij_vec, fc, dfc, symbols,
               elements, g2_params, n_g2, dE_dG):
    for idx in range(len(i)):
        ci = i[idx]
        cj = j[idx]
        if ci >= cj:
            continue
        rij_val = rij[idx]
        r_hat = rij_vec[idx] / rij_val
        fcij = fc[idx]
        dfcij = dfc[idx]
        dEdr = 0.0

        for ie, ne in enumerate(elements):
            if symbols[cj] != ne:
                continue
            for ip, p in enumerate(g2_params):
                g2 = np.exp(-p.eta * (rij_val - p.Rs) ** 2)
                dg2_dr = (-2.0 * p.eta * (rij_val - p.Rs) * g2 * fcij
                          + g2 * dfcij)
                feat_idx = ie * n_g2 + ip
                dEdr += dE_dG[ci, feat_idx] * dg2_dr

        for ie, ne in enumerate(elements):
            if symbols[ci] != ne:
                continue
            for ip, p in enumerate(g2_params):
                g2 = np.exp(-p.eta * (rij_val - p.Rs) ** 2)
                dg2_dr = (-2.0 * p.eta * (rij_val - p.Rs) * g2 * fcij
                          + g2 * dfcij)
                feat_idx = ie * n_g2 + ip
                dEdr += dE_dG[cj, feat_idx] * dg2_dr

        forces[ci] += dEdr * r_hat
        forces[cj] -= dEdr * r_hat


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


def _g4_forces(forces, i_idx, j_idx, rij, rij_vec, fc, dfc, r_cut, symbols,
               elements, g4_params, n_g2, n_g4, dE_dG):
    natoms = len(symbols)
    n_elem = len(elements)
    g4_offset = n_elem * n_g2

    for ci in range(natoms):
        nei_mask = i_idx == ci
        nei_idx = np.where(nei_mask)[0]
        n_n = len(nei_idx)
        if n_n < 2:
            continue
        for a in range(n_n):
            for b in range(a + 1, n_n):
                ia = nei_idx[a]
                ib = nei_idx[b]
                ja = j_idx[ia]
                jb = j_idx[ib]
                ej, ek = symbols[ja], symbols[jb]
                if ej not in elements or ek not in elements:
                    continue
                pid = _pair_index(elements, ej, ek)

                r_ci_ja = rij[ia]
                r_ci_jb = rij[ib]
                u_vec = rij_vec[ia]
                v_vec = rij_vec[ib]
                w_vec = u_vec - v_vec
                r_ja_jb = np.linalg.norm(w_vec)
                if r_ja_jb > r_cut:
                    continue
                fc_ci_ja = fc[ia]
                fc_ci_jb = fc[ib]
                fc_ja_jb = _cutoff_fn(r_ja_jb, r_cut)
                dfc_ci_ja = dfc[ia]
                dfc_ci_jb = dfc[ib]
                dfc_ja_jb = _d_cutoff_fn(r_ja_jb, r_cut)

                dot = np.dot(u_vec, v_vec)
                cos_theta = np.clip(dot / (r_ci_ja * r_ci_jb + 1e-15), -1.0, 1.0)
                u_hat = u_vec / r_ci_ja
                v_hat = v_vec / r_ci_jb
                w_hat = w_vec / r_ja_jb if r_ja_jb > 1e-15 else np.zeros(3)

                for ip, p in enumerate(g4_params):
                    w = dE_dG[ci, g4_offset + pid * n_g4 + ip]

                    A = 2.0 ** (1.0 - p.zeta)
                    B = (1.0 + p.lambda_ * cos_theta) ** p.zeta
                    C = np.exp(-p.eta * (r_ci_ja ** 2 + r_ci_jb ** 2 + r_ja_jb ** 2))
                    D = fc_ci_ja * fc_ci_jb * fc_ja_jb

                    pref_ang = A * C * D
                    ang_deriv = p.zeta * p.lambda_ * (1.0 + p.lambda_ * cos_theta) ** (p.zeta - 1.0)
                    dB_dcos = pref_ang * ang_deriv

                    pref_exp = A * B * D * C
                    exp_pre = -2.0 * p.eta

                    pref_fc_ja = A * B * C * fc_ci_jb * fc_ja_jb
                    pref_fc_jb = A * B * C * fc_ci_ja * fc_ja_jb
                    pref_fc_jab = A * B * C * fc_ci_ja * fc_ci_jb

                    dcos_dci = (cos_theta * u_hat - v_hat) / r_ci_ja \
                               + (cos_theta * v_hat - u_hat) / r_ci_jb
                    dcos_dja = (v_hat - cos_theta * u_hat) / r_ci_ja
                    dcos_djb = (u_hat - cos_theta * v_hat) / r_ci_jb

                    dri_dci = -u_hat;    dri_dja = u_hat;        dri_djb = np.zeros(3)
                    drk_dci = -v_hat;    drk_dja = np.zeros(3);   drk_djb = v_hat
                    drjk_dci = np.zeros(3); drjk_dja = w_hat;    drjk_djb = -w_hat

                    dG4_dci = _g4_gradient(dB_dcos, dcos_dci, pref_exp, exp_pre,
                        r_ci_ja, dri_dci, r_ci_jb, drk_dci, r_ja_jb, drjk_dci,
                        pref_fc_ja, dfc_ci_ja, dri_dci,
                        pref_fc_jb, dfc_ci_jb, drk_dci,
                        pref_fc_jab, dfc_ja_jb, drjk_dci)
                    dG4_dja = _g4_gradient(dB_dcos, dcos_dja, pref_exp, exp_pre,
                        r_ci_ja, dri_dja, r_ci_jb, drk_dja, r_ja_jb, drjk_dja,
                        pref_fc_ja, dfc_ci_ja, dri_dja,
                        pref_fc_jb, dfc_ci_jb, drk_dja,
                        pref_fc_jab, dfc_ja_jb, drjk_dja)
                    dG4_djb = _g4_gradient(dB_dcos, dcos_djb, pref_exp, exp_pre,
                        r_ci_ja, dri_djb, r_ci_jb, drk_djb, r_ja_jb, drjk_djb,
                        pref_fc_ja, dfc_ci_ja, dri_djb,
                        pref_fc_jb, dfc_ci_jb, drk_djb,
                        pref_fc_jab, dfc_ja_jb, drjk_djb)

                    forces[ci] -= w * dG4_dci
                    forces[ja] -= w * dG4_dja
                    forces[jb] -= w * dG4_djb


def _g4_gradient(dB_dcos, dcos, pref_exp, exp_pre,
                 r_ij, dri, r_ik, drk, r_jk, drjk,
                 pref_fc_ij, dfc_ij, dri_term,
                 pref_fc_ik, dfc_ik, drk_term,
                 pref_fc_jk, dfc_jk, drjk_term):
    return (dB_dcos * dcos
            + pref_exp * exp_pre * (r_ij * dri + r_ik * drk + r_jk * drjk)
            + pref_fc_ij * dfc_ij * dri_term
            + pref_fc_ik * dfc_ik * drk_term
            + pref_fc_jk * dfc_jk * drjk_term)
