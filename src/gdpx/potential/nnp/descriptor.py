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
    dfc = _d_cutoff_fn(rij, r_cut)

    _fill_g2(desc, i, j, rij, fc, symbols, elements, g2_params, n_g2)
    _fill_g4(desc, i, j, rij, rij_vec, fc, dfc, r_cut, symbols, elements,
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
    g2_eta = np.array([p.eta for p in g2_params])
    g2_Rs = np.array([p.Rs for p in g2_params])
    contrib = np.exp(-g2_eta[:, None] * (rij[None, :] - g2_Rs[:, None]) ** 2) * fc[None, :]
    for ie, ne in enumerate(elements):
        nei_mask = symbols[j] == ne
        offset = ie * n_g2
        for ip in range(n_g2):
            np.add.at(desc, (i[nei_mask], offset + ip), contrib[ip, nei_mask])


def compute_force_gradient_weights(atoms, elements, g2_params, g4_params,
                                   r_cut, dF):
    """Adjoint of the force computation wrt the force residual.

    Returns ``B`` of shape ``(natoms, n_features)`` with

        B[j, k] = sum_i dF[i] * dG[j, k] / dr[i]

    where ``G[j]`` is the descriptor of atom ``j`` and ``dF`` is the force
    residual ``F_pred - F_ref``.  ``B`` is contracted with the network mixed
    second derivative ``d^2E/dG dW`` to build the analytic force-gradient.
    """
    g2_params = _normalize_params(g2_params, G2Param)
    g4_params = _normalize_params(g4_params, G4Param)

    natoms = len(atoms)
    symbols = np.array(atoms.get_chemical_symbols())
    n_elem = len(elements)
    n_g2 = len(g2_params)
    n_g4 = len(g4_params)
    n_pairs = n_elem * (n_elem + 1) // 2
    B = np.zeros((natoms, n_elem * n_g2 + n_pairs * n_g4))

    i, j, rij, rij_vec = _get_neighbor_info(atoms, r_cut)
    fc = _cutoff_fn(rij, r_cut)
    dfc = _d_cutoff_fn(rij, r_cut)

    _g2_force_weights(B, i, j, rij, rij_vec, fc, dfc, symbols,
                      elements, g2_params, n_g2, dF)
    _g4_force_weights(B, i, j, rij, rij_vec, fc, dfc, r_cut, symbols,
                      elements, g4_params, n_g2, n_g4, dF)

    return B


def _g2_force_weights(B, i, j, rij, rij_vec, fc, dfc, symbols,
                      elements, g2_params, n_g2, dF):
    mask = i < j
    r_hat = rij_vec / rij[:, None]
    dF_diff = np.einsum('ij,ij->i', dF[j] - dF[i], r_hat)
    g2_eta = np.array([p.eta for p in g2_params])
    g2_Rs = np.array([p.Rs for p in g2_params])
    g2v = np.exp(-g2_eta[:, None] * (rij[None, :] - g2_Rs[:, None]) ** 2)
    dg2_dr = (-2.0 * g2_eta[:, None] * (rij[None, :] - g2_Rs[:, None]) * g2v
              * fc[None, :] + g2v * dfc[None, :])

    for ie, ne in enumerate(elements):
        mask_j = symbols[j] == ne
        mask_i = symbols[i] == ne
        offset = ie * n_g2
        for ip in range(n_g2):
            sel = mask & mask_j
            np.add.at(B, (i[sel], offset + ip), dF_diff[sel] * dg2_dr[ip, sel])
            sel = mask & mask_i
            np.add.at(B, (j[sel], offset + ip), dF_diff[sel] * dg2_dr[ip, sel])


def _g2_forces(forces, i, j, rij, rij_vec, fc, dfc, symbols,
               elements, g2_params, n_g2, dE_dG):
    mask = i < j
    r_hat = rij_vec / rij[:, None]
    g2_eta = np.array([p.eta for p in g2_params])
    g2_Rs = np.array([p.Rs for p in g2_params])
    g2v = np.exp(-g2_eta[:, None] * (rij[None, :] - g2_Rs[:, None]) ** 2)
    dg2_dr = (-2.0 * g2_eta[:, None] * (rij[None, :] - g2_Rs[:, None]) * g2v
              * fc[None, :] + g2v * dfc[None, :])

    dEdr = np.zeros(len(i))
    for ie, ne in enumerate(elements):
        mask_j = symbols[j] == ne
        mask_i = symbols[i] == ne
        offset = ie * n_g2
        for ip in range(n_g2):
            sel = mask & mask_j
            dEdr[sel] += dE_dG[i[sel], offset + ip] * dg2_dr[ip, sel]
            sel = mask & mask_i
            dEdr[sel] += dE_dG[j[sel], offset + ip] * dg2_dr[ip, sel]

    np.add.at(forces, i[mask], dEdr[mask, None] * r_hat[mask])
    np.add.at(forces, j[mask], -dEdr[mask, None] * r_hat[mask])


def _g4_pairs(i_idx, j_idx, rij, rij_vec, fc, dfc, r_cut, symbols, elements):
    """Yield per-central-atom vectorized G4 pair data.

    For each central atom with at least two valid neighbor pairs, yield a dict
    of arrays (over the ``a < b`` pairs of its neighbors) with keys:
    ``ja``, ``jb``, ``pid``, ``ru``, ``rv``, ``rw``, ``u_hat``, ``v_hat``,
    ``w_hat``, ``cos``, ``fcu``, ``fcv``, ``fcw``, ``dfcu``, ``dfcv``,
    ``dfcw``.
    """
    n_elem = len(elements)
    elem_idx = {e: k for k, e in enumerate(elements)}
    atom_elem = np.array([elem_idx.get(s, -1) for s in symbols])

    for ci in range(len(symbols)):
        nei = np.flatnonzero(i_idx == ci)
        n_n = len(nei)
        if n_n < 2:
            continue
        ai, bi = np.triu_indices(n_n, 1)
        ia = nei[ai]
        ib = nei[bi]
        ja = j_idx[ia]
        jb = j_idx[ib]
        ea = atom_elem[ja]
        eb = atom_elem[jb]
        ok = (ea >= 0) & (eb >= 0)
        if not np.any(ok):
            continue
        ia = ia[ok]
        ib = ib[ok]
        ja = ja[ok]
        jb = jb[ok]
        ea = ea[ok]
        eb = eb[ok]
        ru = rij[ia]
        rv = rij[ib]
        u_vec = rij_vec[ia]
        v_vec = rij_vec[ib]
        w_vec = u_vec - v_vec
        rw = np.linalg.norm(w_vec, axis=1)
        ok = rw < r_cut
        if not np.any(ok):
            continue
        ia = ia[ok]
        ib = ib[ok]
        ja = ja[ok]
        jb = jb[ok]
        ea = ea[ok]
        eb = eb[ok]
        ru = ru[ok]
        rv = rv[ok]
        u_vec = u_vec[ok]
        v_vec = v_vec[ok]
        w_vec = w_vec[ok]
        rw = rw[ok]
        lo = np.minimum(ea, eb)
        hi = np.maximum(ea, eb)
        pid = lo * n_elem - lo * (lo - 1) // 2 + (hi - lo)
        fcu = fc[ia]
        fcv = fc[ib]
        fcw = _cutoff_fn(rw, r_cut)
        dfcu = dfc[ia]
        dfcv = dfc[ib]
        dfcw = _d_cutoff_fn(rw, r_cut)
        dot = np.einsum('ij,ij->i', u_vec, v_vec)
        cos = np.clip(dot / (ru * rv + 1e-15), -1.0, 1.0)
        u_hat = u_vec / ru[:, None]
        v_hat = v_vec / rv[:, None]
        w_hat = w_vec / rw[:, None]
        w_hat[rw <= 1e-15] = 0.0
        yield dict(
            ci=ci, ja=ja, jb=jb, pid=pid, ru=ru, rv=rv, rw=rw,
            u_hat=u_hat, v_hat=v_hat, w_hat=w_hat, cos=cos,
            fcu=fcu, fcv=fcv, fcw=fcw, dfcu=dfcu, dfcv=dfcv, dfcw=dfcw,
        )


def _fill_g4(desc, i, j, rij, rij_vec, fc, dfc, r_cut, symbols, elements,
             g4_params, n_g2, n_g4):
    n_elem = len(elements)
    g4_offset = n_elem * n_g2
    g4_eta = np.array([p.eta for p in g4_params])
    g4_zeta = np.array([p.zeta for p in g4_params])
    g4_lambda = np.array([p.lambda_ for p in g4_params])
    g4_amp = np.array([2.0 ** (1.0 - p.zeta) for p in g4_params])

    for p in _g4_pairs(i, j, rij, rij_vec, fc, dfc, r_cut, symbols, elements):
        ci = p["ci"]
        r2 = p["ru"] ** 2 + p["rv"] ** 2 + p["rw"] ** 2
        fcm = p["fcu"] * p["fcv"] * p["fcw"]
        vals = (g4_amp[:, None]
                * (1.0 + g4_lambda[:, None] * p["cos"][None, :]) ** g4_zeta[:, None]
                * np.exp(-g4_eta[:, None] * r2[None, :])
                * fcm[None, :])
        for ip in range(n_g4):
            for pidk in np.unique(p["pid"]):
                m = p["pid"] == pidk
                desc[ci, g4_offset + pidk * n_g4 + ip] += vals[ip, m].sum()


def _g4_derivatives(eta, zeta, lam, amp, p):
    """Return (dG4_dci, dG4_dja, dG4_djb) for one G4 parameter."""
    ru, rv, rw = p["ru"], p["rv"], p["rw"]
    cos = p["cos"]
    r2 = ru ** 2 + rv ** 2 + rw ** 2
    fcm = p["fcu"] * p["fcv"] * p["fcw"]
    u_hat, v_hat, w_hat = p["u_hat"], p["v_hat"], p["w_hat"]
    c1 = cos[:, None]

    A = amp
    Bv = (1.0 + lam * cos) ** zeta
    C = np.exp(-eta * r2)
    D = fcm

    pref_ang = A * C * D
    ang_deriv = zeta * lam * (1.0 + lam * cos) ** (zeta - 1.0)
    dB_dcos = pref_ang * ang_deriv

    pref_exp = A * Bv * D * C
    exp_pre = -2.0 * eta

    pref_fc_ja = A * Bv * C * p["fcv"] * p["fcw"]
    pref_fc_jb = A * Bv * C * p["fcu"] * p["fcw"]
    pref_fc_jab = A * Bv * C * p["fcu"] * p["fcv"]

    dcos_dci = (c1 * u_hat - v_hat) / ru[:, None] \
        + (c1 * v_hat - u_hat) / rv[:, None]
    dcos_dja = (v_hat - c1 * u_hat) / ru[:, None]
    dcos_djb = (u_hat - c1 * v_hat) / rv[:, None]

    dri_dci = -u_hat
    dri_dja = u_hat
    dri_djb = np.zeros_like(u_hat)
    drk_dci = -v_hat
    drk_dja = np.zeros_like(v_hat)
    drk_djb = v_hat
    drjk_dci = np.zeros_like(w_hat)
    drjk_dja = w_hat
    drjk_djb = -w_hat

    dG4_dci = _g4_gradient_v(dB_dcos, dcos_dci, pref_exp, exp_pre,
        ru, dri_dci, rv, drk_dci, rw, drjk_dci,
        pref_fc_ja, p["dfcu"], dri_dci, pref_fc_jb, p["dfcv"], drk_dci,
        pref_fc_jab, p["dfcw"], drjk_dci)
    dG4_dja = _g4_gradient_v(dB_dcos, dcos_dja, pref_exp, exp_pre,
        ru, dri_dja, rv, drk_dja, rw, drjk_dja,
        pref_fc_ja, p["dfcu"], dri_dja, pref_fc_jb, p["dfcv"], drk_dja,
        pref_fc_jab, p["dfcw"], drjk_dja)
    dG4_djb = _g4_gradient_v(dB_dcos, dcos_djb, pref_exp, exp_pre,
        ru, dri_djb, rv, drk_djb, rw, drjk_djb,
        pref_fc_ja, p["dfcu"], dri_djb, pref_fc_jb, p["dfcv"], drk_djb,
        pref_fc_jab, p["dfcw"], drjk_djb)
    return dG4_dci, dG4_dja, dG4_djb


def _g4_gradient_v(dB_dcos, dcos, pref_exp, exp_pre,
                   r_ij, dri, r_ik, drk, r_jk, drjk,
                   pref_fc_ij, dfc_ij, dri_term,
                   pref_fc_ik, dfc_ik, drk_term,
                   pref_fc_jk, dfc_jk, drjk_term):
    return (dB_dcos[:, None] * dcos
            + pref_exp[:, None] * exp_pre
            * (r_ij[:, None] * dri + r_ik[:, None] * drk + r_jk[:, None] * drjk)
            + pref_fc_ij[:, None] * dfc_ij[:, None] * dri_term
            + pref_fc_ik[:, None] * dfc_ik[:, None] * drk_term
            + pref_fc_jk[:, None] * dfc_jk[:, None] * drjk_term)


def _g4_forces(forces, i, j, rij, rij_vec, fc, dfc, r_cut, symbols,
               elements, g4_params, n_g2, n_g4, dE_dG):
    n_elem = len(elements)
    g4_offset = n_elem * n_g2
    g4_eta = np.array([p.eta for p in g4_params])
    g4_zeta = np.array([p.zeta for p in g4_params])
    g4_lambda = np.array([p.lambda_ for p in g4_params])
    g4_amp = np.array([2.0 ** (1.0 - p.zeta) for p in g4_params])

    for p in _g4_pairs(i, j, rij, rij_vec, fc, dfc, r_cut, symbols, elements):
        ci = p["ci"]
        for ip in range(n_g4):
            w = dE_dG[ci, g4_offset + p["pid"] * n_g4 + ip]
            dG4_dci, dG4_dja, dG4_djb = _g4_derivatives(
                g4_eta[ip], g4_zeta[ip], g4_lambda[ip], g4_amp[ip], p
            )
            forces[ci] -= np.sum(w[:, None] * dG4_dci, axis=0)
            np.add.at(forces, p["ja"], -w[:, None] * dG4_dja)
            np.add.at(forces, p["jb"], -w[:, None] * dG4_djb)


def _g4_force_weights(B, i, j, rij, rij_vec, fc, dfc, r_cut, symbols,
                      elements, g4_params, n_g2, n_g4, dF):
    n_elem = len(elements)
    g4_offset = n_elem * n_g2
    g4_eta = np.array([p.eta for p in g4_params])
    g4_zeta = np.array([p.zeta for p in g4_params])
    g4_lambda = np.array([p.lambda_ for p in g4_params])
    g4_amp = np.array([2.0 ** (1.0 - p.zeta) for p in g4_params])

    for p in _g4_pairs(i, j, rij, rij_vec, fc, dfc, r_cut, symbols, elements):
        ci = p["ci"]
        n_p = len(p["pid"])
        for ip in range(n_g4):
            dG4_dci, dG4_dja, dG4_djb = _g4_derivatives(
                g4_eta[ip], g4_zeta[ip], g4_lambda[ip], g4_amp[ip], p
            )
            feat = g4_offset + p["pid"] * n_g4 + ip
            contrib = (np.einsum('ij,ij->i', dG4_dci, np.broadcast_to(dF[ci], dG4_dci.shape))
                       + np.einsum('ij,ij->i', dG4_dja, dF[p["ja"]])
                       + np.einsum('ij,ij->i', dG4_djb, dF[p["jb"]]))
            np.add.at(B, (np.full(n_p, ci), feat), contrib)
