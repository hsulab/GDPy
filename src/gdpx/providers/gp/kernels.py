import itertools

import numpy as np


def switch_function(r, r_cut):
    r_diff = r_cut - r
    mask = r_diff > 0
    v = np.where(mask, r_diff ** 2, 0.0)
    g = np.where(mask, 2.0 * r_diff, 0.0)
    return v, g


def se_kernel(x1, x2, sigma, length):
    x_diff = x1[:, np.newaxis] - x2[np.newaxis, :]
    dist_sq = np.sum(x_diff ** 2, axis=-1)
    return sigma ** 2 * np.exp(-dist_sq / (2.0 * length ** 2))


def se_grad_r(x1, x2, sigma, length):
    x_diff = x1[:, np.newaxis] - x2[np.newaxis, :]
    dist_sq = np.sum(x_diff ** 2, axis=-1, keepdims=True)
    k = sigma ** 2 * np.exp(-dist_sq / (2.0 * length ** 2))
    return k * (-x_diff / length ** 2)


def se_grad_sigma(x1, x2, sigma, length):
    x_diff = x1[:, np.newaxis] - x2[np.newaxis, :]
    dist_sq = np.sum(x_diff ** 2, axis=-1, keepdims=True)
    k = sigma ** 2 * np.exp(-dist_sq / (2.0 * length ** 2))
    return 2.0 * k / sigma


def se_grad_length(x1, x2, sigma, length):
    x_diff = x1[:, np.newaxis] - x2[np.newaxis, :]
    dist_sq = np.sum(x_diff ** 2, axis=-1, keepdims=True)
    k = sigma ** 2 * np.exp(-dist_sq / (2.0 * length ** 2))
    return k * (dist_sq / length ** 3)


def _build_2b_kernel_one_species(
    sigma, length, r_cut, nf, nat, b2_features, b2_gradients, b2_mapping,
    cluster_mask, sparse_features,
):
    cm = cluster_mask
    feats = b2_features[cm] if np.any(cm) else np.empty((0, b2_features.shape[1]))
    grads = b2_gradients[cm] if np.any(cm) else np.empty((0, b2_gradients.shape[1], b2_gradients.shape[2]))
    mapping = b2_mapping[cm] if np.any(cm) else np.empty((0, b2_mapping.shape[1]))
    n_cluster = feats.shape[0]
    if n_cluster == 0 or sparse_features.shape[0] == 0:
        return None, None, None

    sw, sw_grad = switch_function(feats, r_cut)

    k_val = se_kernel(feats, sparse_features, sigma, length)
    k_grad_r = se_grad_r(feats, sparse_features, sigma, length)

    k_val = k_val * sw
    k_grad = k_grad_r * sw[:, np.newaxis] + (k_val * sw_grad)[:, :, np.newaxis]

    Knm_grad = np.zeros((nat * 3, sparse_features.shape[0]))
    for loc, b2_grad, kernel_grad in zip(mapping, grads, k_grad):
        _, i, j = loc
        contrib = np.repeat(kernel_grad[:, :, np.newaxis], 6, axis=2) * b2_grad
        Knm_grad[i * 3: i * 3 + 3, :] += contrib[:, 0, 0:3].T
        Knm_grad[j * 3: j * 3 + 3, :] += contrib[:, 0, 3:6].T
    Knm_frc = -Knm_grad

    frame_energy_kernel = np.zeros((nf, sparse_features.shape[0]))
    for loc, kv in zip(mapping, k_val):
        frame_energy_kernel[loc[0], :] += kv

    Kmm = se_kernel(sparse_features, sparse_features, sigma, length)
    sw_sp, _ = switch_function(sparse_features, r_cut)
    Kmm = Kmm * (sw_sp @ sw_sp.T)

    return Kmm, Knm_frc, frame_energy_kernel


def compute_2b_kernel_matrices(
    sigma, length, r_cut, nf, nat,
    b2_features, b2_gradients, b2_mapping, b2_species,
    sparse_features, sparse_species,
    allowed_species=None,
):
    if allowed_species is not None:
        all_species_pairs = allowed_species
    else:
        all_species_pairs = sorted(set(b2_species) | set(sparse_species))
    Kmm_blocks = []
    Knm_parts = []
    KnmE_parts = []

    for sp in all_species_pairs:
        train_mask = b2_species == sp
        sparse_mask = sparse_species == sp
        if not np.any(train_mask) or not np.any(sparse_mask):
            continue

        sp_feats = b2_features[train_mask]
        sp_grads = b2_gradients[train_mask]
        sp_map = b2_mapping[train_mask]
        sp_sparse = sparse_features[sparse_mask]

        result = _build_2b_kernel_one_species(
            sigma, length, r_cut, nf, nat,
            b2_features, b2_gradients, b2_mapping,
            train_mask, sp_sparse,
        )
        if result is not None:
            Kmm_b, Knm_b, KnmE_b = result
            Kmm_blocks.append(Kmm_b)
            Knm_parts.append(Knm_b)
            KnmE_parts.append(KnmE_b)

    if not Kmm_blocks:
        return np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), \
               np.zeros((nf, 1)), np.zeros((nat * 3, 1))

    Kmm_total_dim = sum(b.shape[0] for b in Kmm_blocks)
    Kmm = np.zeros((Kmm_total_dim, Kmm_total_dim))
    offset = 0
    for blk in Kmm_blocks:
        n = blk.shape[0]
        Kmm[offset:offset + n, offset:offset + n] = blk
        offset += n

    Knm = np.hstack(Knm_parts)
    Knm_ene = np.hstack(KnmE_parts)
    Knm_frc = Knm

    return Kmm, Kmm, Kmm, Knm_ene, Knm_frc


def compute_3b_kernel_matrices(
    sigma, length, r_cut, nf, nat,
    b3_features, b3_gradients, b3_mapping, b3_species,
    sparse_features, sparse_species,
    allowed_species=None,
):
    if allowed_species is not None:
        all_species_triples = allowed_species
    else:
        all_species_triples = sorted(set(b3_species) | set(sparse_species))
    Kmm_blocks = []
    Knm_parts = []
    KnmE_parts = []

    for sp in all_species_triples:
        train_mask = b3_species == sp
        sparse_mask = sparse_species == sp
        if not np.any(train_mask) or not np.any(sparse_mask):
            continue

        feats = b3_features[train_mask]
        grads = b3_gradients[train_mask]
        mapping = b3_mapping[train_mask]
        sp_sparse = sparse_features[sparse_mask]
        n_cluster = feats.shape[0]
        n_sparse = sp_sparse.shape[0]

        sw3, sw3_grad = switch_function_3b(feats, r_cut)

        k_val = np.zeros((n_cluster, n_sparse))
        k_grad = np.zeros((n_cluster, n_sparse, 3))

        for p in itertools.permutations(range(3), 3):
            sp_s = sp_sparse[:, p]
            diff = feats[:, np.newaxis, :] - sp_s[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            k_se = sigma ** 2 * np.exp(-dist_sq / (2.0 * length ** 2))
            k_se_grad = k_se[:, :, np.newaxis] * (-diff / length ** 2)

            k_val += k_se * sw3[:, np.newaxis]
            k_grad += (
                k_se_grad * sw3[:, np.newaxis, np.newaxis]
                + k_se[:, :, np.newaxis] * sw3_grad[:, np.newaxis]
            )

        Knm_grad = np.zeros((nat * 3, n_sparse))
        for loc, b3_grad, kernel_grad in zip(mapping, grads, k_grad):
            _, i, j, kk = loc
            contrib = np.repeat(kernel_grad[:, :, np.newaxis], 6, axis=2) * b3_grad
            Knm_grad[i * 3: i * 3 + 3, :] += contrib[:, 0, 0:3].T
            Knm_grad[j * 3: j * 3 + 3, :] += contrib[:, 0, 3:6].T
            Knm_grad[i * 3: i * 3 + 3, :] += contrib[:, 1, 0:3].T
            Knm_grad[kk * 3: kk * 3 + 3, :] += contrib[:, 1, 3:6].T
            Knm_grad[j * 3: j * 3 + 3, :] += contrib[:, 2, 0:3].T
            Knm_grad[kk * 3: kk * 3 + 3, :] += contrib[:, 2, 3:6].T
        Knm_frc = -Knm_grad

        Kmm = np.zeros((n_sparse, n_sparse))
        for p in itertools.permutations(range(3), 3):
            sp_s = sp_sparse[:, p]
            diff = sp_sparse[:, np.newaxis, :] - sp_s[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            Kmm += sigma ** 2 * np.exp(-dist_sq / (2.0 * length ** 2))

        sw3_sp, _ = switch_function_3b(sp_sparse, r_cut)
        sw3_outer = sw3_sp[:, np.newaxis] @ sw3_sp[np.newaxis, :]
        Kmm = Kmm * sw3_outer

        # Energy kernel for 3-body
        frame_energy_kernel = np.zeros((nf, n_sparse))
        for loc, kv in zip(mapping, k_val):
            frame_energy_kernel[loc[0], :] += kv

        Kmm_blocks.append(Kmm)
        Knm_parts.append(Knm_frc)
        KnmE_parts.append(frame_energy_kernel)

    if not Kmm_blocks:
        return np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), \
               np.zeros((nf, 1)), np.zeros((nat * 3, 1))

    Kmm_total_dim = sum(b.shape[0] for b in Kmm_blocks)
    Kmm = np.zeros((Kmm_total_dim, Kmm_total_dim))
    offset = 0
    for blk in Kmm_blocks:
        n = blk.shape[0]
        Kmm[offset:offset + n, offset:offset + n] = blk
        offset += n

    Knm = np.hstack(Knm_parts)
    Knm_ene = np.hstack(KnmE_parts)

    return Kmm, Kmm, Kmm, Knm_ene, Knm


def switch_function_3b(r, r_cut):
    r_diff = r_cut - r
    mask = np.all(r_diff > 0, axis=1)
    r_diff_safe = np.where(mask[:, np.newaxis], r_diff, 1.0)
    rsq = r_diff_safe ** 2
    v = np.where(mask, np.prod(rsq, axis=1), 0.0)
    g = np.where(mask[:, np.newaxis], 2.0 * v[:, np.newaxis] / r_diff_safe, 0.0)
    return v, g
