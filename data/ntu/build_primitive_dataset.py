import argparse
import os
import h5py
import numpy as np


GROUPS = [
    [3, 20],
    [0, 1, 2, 4, 8, 12, 16],
    [8, 9, 10, 11, 23, 24],
    [4, 5, 6, 7, 21, 22],
    [16, 17, 18, 19],
    [12, 13, 14, 15],
]
PAPER_G_REF = {
    0: (20, 2),
    2: (20, 2),
    3: (20, 2),
    1: (0, 1),
    4: (0, 1),
    5: (0, 1),
}
PAPER_G_EXT = {
    0: 20,
    1: 16,
    2: 24,
    3: 22,
    4: 19,
    5: 15,
}


def _to_label(y):
    if y.ndim == 2:
        return np.argmax(y, axis=-1).astype(np.int64)
    return y.astype(np.int64)


def _one_hot(labels, num_classes):
    out = np.zeros((len(labels), num_classes), dtype=np.float32)
    if len(labels) > 0:
        out[np.arange(len(labels)), labels] = 1.0
    return out


def _smooth(seq):
    if seq.shape[0] < 3:
        return seq
    kernel = np.array([1.0, 2.0, 1.0], dtype=np.float32) / 4.0
    out = np.copy(seq)
    for j in range(seq.shape[1]):
        for c in range(seq.shape[2]):
            s = seq[:, j, c]
            p = np.pad(s, (1, 1), mode='edge')
            out[:, j, c] = np.convolve(p, kernel, mode='valid')
    return out


def _contiguous_ranges(mask):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    ranges = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            ranges.append((start, prev))
            start = i
            prev = i
    ranges.append((start, prev))
    return ranges


def _resample(seq, out_len):
    t = seq.shape[0]
    if t == out_len:
        return seq
    if t <= 1:
        return np.repeat(seq[:1], out_len, axis=0)
    x_old = np.arange(t, dtype=np.float32)
    x_new = np.linspace(0, t - 1, out_len, dtype=np.float32)
    out = np.zeros((out_len, seq.shape[1], seq.shape[2]), dtype=np.float32)
    for j in range(seq.shape[1]):
        for c in range(seq.shape[2]):
            out[:, j, c] = np.interp(x_new, x_old, seq[:, j, c])
    return out


def _compute_curvature_torsion(path):
    if path.shape[0] < 5:
        z = np.zeros((path.shape[0],), dtype=np.float32)
        return z, z
    r1 = np.gradient(path, axis=0)
    r2 = np.gradient(r1, axis=0)
    r3 = np.gradient(r2, axis=0)
    cross = np.cross(r1, r2)
    cross_norm = np.linalg.norm(cross, axis=1)
    r1_norm = np.linalg.norm(r1, axis=1)
    kappa = cross_norm / np.clip(r1_norm ** 3, 1e-8, None)
    det = np.einsum('ij,ij->i', cross, r3)
    tau = det / np.clip(cross_norm ** 2, 1e-8, None)
    return kappa.astype(np.float32), tau.astype(np.float32)


def _build_paper_feature_rows(path, primitive_idx, decim):
    if decim <= 1:
        path_d = path
    else:
        path_d = path[::decim]
    if path_d.shape[0] < 3:
        path_d = _resample(path[:, None, :], 3)[:, 0, :]
    kappa, tau = _compute_curvature_torsion(path_d)
    rows = []
    for i in range(1, path_d.shape[0] - 1):
        row = np.zeros((17,), dtype=np.float32)
        row[0:3] = path_d[i - 1]
        row[3:6] = path_d[i]
        row[6:9] = path_d[i + 1]
        row[9:12] = np.array([kappa[i - 1], kappa[i], kappa[i + 1]], dtype=np.float32)
        row[12:15] = np.array([tau[i - 1], tau[i], tau[i + 1]], dtype=np.float32)
        row[15] = float(primitive_idx)
        row[16] = 0.0
        rows.append(row)
    if len(rows) == 0:
        row = np.zeros((17,), dtype=np.float32)
        row[0:3] = path_d[0]
        row[3:6] = path_d[min(1, path_d.shape[0] - 1)]
        row[6:9] = path_d[min(2, path_d.shape[0] - 1)]
        row[9:12] = 0.0
        row[12:15] = 0.0
        row[15] = float(primitive_idx)
        row[16] = 0.0
        rows.append(row)
    n_rows = len(rows)
    for r in rows:
        r[16] = float(n_rows)
    return rows


def _normalize_indicator_columns(feat_mat):
    out = feat_mat.copy()
    for c in [15, 16]:
        col = out[:, c]
        cmin = float(col.min())
        cmax = float(col.max())
        if cmax - cmin < 1e-8:
            out[:, c] = 0.0
        else:
            out[:, c] = (col - cmin) / (cmax - cmin)
    return out


def _primitive_idx_to_group_features(samples, group_ids, decim):
    by_group = {g: [] for g in range(6)}
    for pidx, (s, gid) in enumerate(zip(samples, group_ids)):
        clip = s[:, :75].reshape(s.shape[0], 25, 3)
        ext = PAPER_G_EXT[int(gid)]
        traj = clip[:, ext, :].astype(np.float32)
        rows = _build_paper_feature_rows(traj, primitive_idx=pidx, decim=decim)
        by_group[int(gid)].append(rows)
    group_data = {}
    for gid in range(6):
        rows_nested = by_group[gid]
        if len(rows_nested) == 0:
            group_data[gid] = {
                'X': np.zeros((0, 17), dtype=np.float32),
                'prim_idx': np.zeros((0,), dtype=np.int64),
                'count_map': {},
            }
            continue
        rows_flat = []
        prim_ids = []
        count_map = {}
        for rows in rows_nested:
            pid = int(rows[0][15])
            count_map[pid] = len(rows)
            for r in rows:
                rows_flat.append(r)
                prim_ids.append(pid)
        x = np.stack(rows_flat, axis=0).astype(np.float32)
        x = _normalize_indicator_columns(x)
        group_data[gid] = {
            'X': x,
            'prim_idx': np.array(prim_ids, dtype=np.int64),
            'count_map': count_map,
        }
    return group_data


def _assign_components_to_primitives(pred_comp, prim_ids, prim_count_map, accept_ratio, drop_unassigned):
    prim_to_votes = {}
    for c, p in zip(pred_comp.tolist(), prim_ids.tolist()):
        if p not in prim_to_votes:
            prim_to_votes[p] = {}
        prim_to_votes[p][int(c)] = prim_to_votes[p].get(int(c), 0) + 1
    assigned = {}
    for pid, vote_map in prim_to_votes.items():
        best_c = max(vote_map.items(), key=lambda kv: kv[1])[0]
        best_n = vote_map[best_c]
        total_n = max(1, int(prim_count_map.get(pid, best_n)))
        ratio = float(best_n) / float(total_n)
        if ratio >= accept_ratio or (not drop_unassigned):
            assigned[pid] = int(best_c)
    return assigned


def _fit_predict_dpm(train_samples, train_gids, val_samples, val_gids, test_samples, test_gids, max_components, seed, decim, accept_ratio, drop_unassigned):
    try:
        from sklearn.mixture import BayesianGaussianMixture
    except Exception as e:
        raise RuntimeError('DPM mode requires scikit-learn with BayesianGaussianMixture support.') from e
    tr = _primitive_idx_to_group_features(train_samples, train_gids, decim=decim)
    va = _primitive_idx_to_group_features(val_samples, val_gids, decim=decim)
    te = _primitive_idx_to_group_features(test_samples, test_gids, decim=decim)

    train_labels = np.full((len(train_samples),), -1, dtype=np.int64)
    val_labels = np.full((len(val_samples),), -1, dtype=np.int64)
    test_labels = np.full((len(test_samples),), -1, dtype=np.int64)
    class_offset = 0

    for gid in range(6):
        xtr = tr[gid]['X']
        if xtr.shape[0] == 0:
            continue
        n_comp = int(min(max_components, max(2, xtr.shape[0] // 20)))
        dpm = BayesianGaussianMixture(
            n_components=n_comp,
            covariance_type='full',
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=1.0,
            max_iter=300,
            random_state=seed + gid,
            init_params='kmeans',
        )
        dpm.fit(xtr)
        active = np.where(dpm.weights_ > 1e-3)[0]
        if active.size == 0:
            active = np.array([int(np.argmax(dpm.weights_))], dtype=np.int64)
        comp_map = {int(c): i for i, c in enumerate(active.tolist())}
        n_active = len(comp_map)

        pred_tr = dpm.predict(xtr).astype(np.int64)
        tr_assign = _assign_components_to_primitives(
            pred_tr, tr[gid]['prim_idx'], tr[gid]['count_map'], accept_ratio=accept_ratio, drop_unassigned=drop_unassigned
        )
        for pid, c in tr_assign.items():
            if c in comp_map:
                train_labels[int(pid)] = class_offset + comp_map[c]

        xva = va[gid]['X']
        if xva.shape[0] > 0:
            pred_va = dpm.predict(xva).astype(np.int64)
            va_assign = _assign_components_to_primitives(
                pred_va, va[gid]['prim_idx'], va[gid]['count_map'], accept_ratio=accept_ratio, drop_unassigned=False
            )
            for pid, c in va_assign.items():
                if c in comp_map:
                    val_labels[int(pid)] = class_offset + comp_map[c]

        xte = te[gid]['X']
        if xte.shape[0] > 0:
            pred_te = dpm.predict(xte).astype(np.int64)
            te_assign = _assign_components_to_primitives(
                pred_te, te[gid]['prim_idx'], te[gid]['count_map'], accept_ratio=accept_ratio, drop_unassigned=False
            )
            for pid, c in te_assign.items():
                if c in comp_map:
                    test_labels[int(pid)] = class_offset + comp_map[c]

        class_offset += n_active

    if class_offset <= 1:
        raise RuntimeError('DPM failed to produce enough active components.')
    return train_labels, val_labels, test_labels, class_offset


def _segment_primitives_paper(seq150, out_len, min_len, max_segments, beta_v, beta_s, max_len):
    seq = seq150[:, :75].reshape(seq150.shape[0], 25, 3).astype(np.float32)
    valid = np.abs(seq).sum(axis=(1, 2)) > 1e-6
    seq = seq[valid]
    if seq.shape[0] == 0:
        seq = np.zeros((out_len, 25, 3), dtype=np.float32)
    seq = _smooth(seq)
    t = seq.shape[0]
    if t < 2:
        seq = np.concatenate([seq, np.repeat(seq[-1:], 1, axis=0)], axis=0)
        t = seq.shape[0]
    vel = np.diff(seq, axis=0, prepend=seq[:1])
    acc = np.diff(vel, axis=0, prepend=vel[:1])
    speed = np.linalg.norm(vel, axis=2)
    s_arc = np.cumsum(speed, axis=0)
    candidates = []
    global_flux = np.linalg.norm(acc, axis=2).sum(axis=1)
    t_max = seq.shape[0] - 1
    if max_len <= 0:
        max_len = seq.shape[0]

    def _gvec(gid, t_idx):
        j0, j1 = PAPER_G_REF[gid]
        g = seq[t_idx, j1] - seq[t_idx, j0]
        n = float(np.linalg.norm(g))
        if n < 1e-8:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return (g / n).astype(np.float32)

    for gid, joints in enumerate(GROUPS):
        t0 = 0
        n_seg = 0
        while t0 + min_len <= t_max and n_seg < max_segments:
            lo = t0 + min_len
            hi = min(t_max, t0 + max_len)
            if lo > hi:
                break
            best_p = -1e18
            best_rho = -1
            for rho in range(lo, hi + 1):
                phi = 0.0
                for tau in range(t0 + 1, rho + 1):
                    g_tau = _gvec(gid, tau)
                    a_tau = acc[tau, joints]
                    phi += float(np.abs(a_tau @ g_tau).sum())
                v_rho = vel[rho, joints]
                v_t0 = vel[t0, joints]
                station = 0.5 * beta_v * (float((v_rho * v_rho).sum()) + float((v_t0 * v_t0).sum()))
                seg_len = beta_s * float((s_arc[rho, joints] - s_arc[t0, joints]).sum())
                p_val = phi - station + seg_len
                if p_val > best_p:
                    best_p = p_val
                    best_rho = rho
            if best_rho < 0:
                break
            s = t0
            e = best_rho
            clip = seq[s:e + 1]
            clip = _resample(clip, out_len)
            flat = clip.reshape(out_len, 75).astype(np.float32)
            vel_g = np.linalg.norm(vel[s:e + 1, joints], axis=2)
            acc_g = np.linalg.norm(acc[s:e + 1, joints], axis=2)
            center = clip.mean(axis=1).mean(axis=0)
            center_std = clip.mean(axis=1).std(axis=0)
            disp = np.linalg.norm(clip[-1] - clip[0], axis=1).mean()
            spread = np.linalg.norm(clip.max(axis=0) - clip.min(axis=0), axis=1).mean()
            feat = np.array([
                gid / 5.0,
                (e - s + 1) / max(1.0, float(seq.shape[0])),
                float(vel_g.mean()),
                float(vel_g.std()),
                float(acc_g.mean()),
                float(acc_g.max()),
                float(disp),
                float(spread),
                float(center[0]),
                float(center[1]),
                float(center[2]),
                float(center_std.mean()),
                float(best_p / max(1.0, float(seq.shape[0]))),
            ], dtype=np.float32)
            candidates.append((best_p, gid, flat, feat))
            t0 = best_rho
            n_seg += 1

    if len(candidates) == 0:
        win = min(max(min_len, out_len // 2), seq.shape[0])
        if win <= 0:
            win = 1
        csum = np.convolve(global_flux, np.ones(win, dtype=np.float32), mode='valid')
        s = int(np.argmax(csum))
        e = min(seq.shape[0] - 1, s + win - 1)
        clip = seq[s:e + 1]
        clip = _resample(clip, out_len)
        flat = clip.reshape(out_len, 75).astype(np.float32)
        feat = np.array([0.0, (e - s + 1) / max(1.0, float(seq.shape[0])), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(csum[s])], dtype=np.float32)
        return [flat], [feat], [1]

    candidates.sort(key=lambda z: z[0], reverse=True)
    selected = candidates[:max_segments]
    prims = [x[2] for x in selected]
    feats = [x[3] for x in selected]
    gids = [int(x[1]) for x in selected]
    return prims, feats, gids


def _segment_primitives_heuristic(seq150, out_len, min_len, max_segments):
    seq = seq150[:, :75].reshape(seq150.shape[0], 25, 3).astype(np.float32)
    valid = np.abs(seq).sum(axis=(1, 2)) > 1e-6
    seq = seq[valid]
    if seq.shape[0] == 0:
        seq = np.zeros((out_len, 25, 3), dtype=np.float32)
    seq = _smooth(seq)
    t = seq.shape[0]
    if t < 2:
        seq = np.concatenate([seq, np.repeat(seq[-1:], 1, axis=0)], axis=0)
        t = seq.shape[0]
    v = np.linalg.norm(np.diff(seq, axis=0, prepend=seq[:1]), axis=2)
    a = np.abs(np.diff(v, axis=0, prepend=v[:1]))
    candidates = []
    global_flux = a.sum(axis=1)
    for gid, joints in enumerate(GROUPS):
        flux = a[:, joints].sum(axis=1)
        thr = max(np.quantile(flux, 0.6), 1e-6)
        ranges = _contiguous_ranges(flux >= thr)
        for s, e in ranges:
            if e - s + 1 < min_len:
                continue
            score = float(flux[s:e + 1].mean() * (e - s + 1))
            candidates.append((score, gid, s, e))
    if len(candidates) == 0:
        win = min(max(min_len, out_len // 2), t)
        if win <= 0:
            win = 1
        csum = np.convolve(global_flux, np.ones(win, dtype=np.float32), mode='valid')
        s = int(np.argmax(csum))
        e = min(t - 1, s + win - 1)
        candidates.append((float(csum[s]), 1, s, e))
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = candidates[:max_segments]
    prims = []
    feats = []
    for score, gid, s, e in selected:
        clip = seq[s:e + 1]
        clip = _resample(clip, out_len)
        flat = clip.reshape(out_len, 75).astype(np.float32)
        vel_g = v[s:e + 1, GROUPS[gid]]
        acc_g = a[s:e + 1, GROUPS[gid]]
        center = clip.mean(axis=1).mean(axis=0)
        center_std = clip.mean(axis=1).std(axis=0)
        disp = np.linalg.norm(clip[-1] - clip[0], axis=1).mean()
        spread = np.linalg.norm(clip.max(axis=0) - clip.min(axis=0), axis=1).mean()
        feat = np.array([
            gid / 5.0,
            (e - s + 1) / max(1.0, float(t)),
            float(vel_g.mean()),
            float(vel_g.std()),
            float(acc_g.mean()),
            float(acc_g.max()),
            float(disp),
            float(spread),
            float(center[0]),
            float(center[1]),
            float(center[2]),
            float(center_std.mean()),
            float(score / max(1.0, float(t))),
        ], dtype=np.float32)
        prims.append(flat)
        feats.append(feat)
    gids = [int(gid) for _, gid, _, _ in selected]
    return prims, feats, gids


def _extract_split(x, y, split_name, out_len, min_len, max_segments, extract_mode, beta_v, beta_s, max_len):
    samples = []
    feats = []
    metas = []
    gids = []
    y = _to_label(y)
    for i in range(x.shape[0]):
        if extract_mode == 'paper':
            prims, fts, gset = _segment_primitives_paper(
                x[i],
                out_len=out_len,
                min_len=min_len,
                max_segments=max_segments,
                beta_v=beta_v,
                beta_s=beta_s,
                max_len=max_len,
            )
        else:
            prims, fts, gset = _segment_primitives_heuristic(
                x[i],
                out_len=out_len,
                min_len=min_len,
                max_segments=max_segments,
            )
        for p, f, gid in zip(prims, fts, gset):
            sample = np.zeros((out_len, 150), dtype=np.float32)
            sample[:, :75] = p
            samples.append(sample)
            feats.append(f)
            metas.append((split_name, int(i), int(y[i])))
            gids.append(int(gid))
    return samples, feats, metas, gids


def _pairwise_sq_dist(x, centers):
    x2 = (x * x).sum(axis=1, keepdims=True)
    c2 = (centers * centers).sum(axis=1)[None, :]
    xc = x @ centers.T
    d = x2 + c2 - 2.0 * xc
    return np.maximum(d, 0.0)


def _kmeans_fit(x, n_clusters, seed, max_iter=100):
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    init_ids = rng.choice(n, size=n_clusters, replace=False)
    centers = x[init_ids].copy()
    for _ in range(max_iter):
        d = _pairwise_sq_dist(x, centers)
        labels = np.argmin(d, axis=1)
        new_centers = centers.copy()
        for k in range(n_clusters):
            idx = labels == k
            if np.any(idx):
                new_centers[k] = x[idx].mean(axis=0)
            else:
                new_centers[k] = x[rng.integers(0, n)]
        shift = np.abs(new_centers - centers).mean()
        centers = new_centers
        if shift < 1e-5:
            break
    return centers


def _kmeans_predict(x, centers):
    d = _pairwise_sq_dist(x, centers)
    return np.argmin(d, axis=1).astype(np.int64)


def _kmeans_fit_predict(x, n_clusters, seed, max_iter=100):
    centers = _kmeans_fit(x, n_clusters=n_clusters, seed=seed, max_iter=max_iter)
    labels = _kmeans_predict(x, centers)
    return centers, labels


def _calinski_harabasz_score(x, labels, centers):
    n = x.shape[0]
    k = centers.shape[0]
    if k <= 1 or n <= k:
        return -1e12
    global_center = x.mean(axis=0)
    w_trace = 0.0
    b_trace = 0.0
    for ci in range(k):
        idx = labels == ci
        if not np.any(idx):
            continue
        xi = x[idx]
        ni = float(xi.shape[0])
        diff = xi - centers[ci]
        w_trace += float((diff * diff).sum())
        c_diff = centers[ci] - global_center
        b_trace += ni * float((c_diff * c_diff).sum())
    if w_trace <= 1e-12 or b_trace <= 1e-12:
        return -1e12
    return (b_trace / (k - 1.0)) / (w_trace / (n - k))


def _select_cluster_count_auto(x, min_k, max_k, seed, sample_size):
    n = x.shape[0]
    if n < 4:
        return 2
    k_min = max(2, int(min_k))
    k_max = max(k_min, int(max_k))
    k_max = min(k_max, max(2, n - 1))
    if k_min > k_max:
        return max(2, min(8, n - 1))
    rng = np.random.default_rng(seed)
    if n > sample_size:
        ids = rng.choice(n, size=sample_size, replace=False)
        x_eval = x[ids]
    else:
        x_eval = x
    best_k = k_min
    best_score = -1e18
    for k in range(k_min, k_max + 1):
        centers, labels = _kmeans_fit_predict(x_eval, n_clusters=k, seed=seed + k, max_iter=60)
        score = _calinski_harabasz_score(x_eval, labels, centers)
        print('AutoK candidate k={} score={:.6f}'.format(k, score))
        if score > best_score:
            best_score = score
            best_k = k
    return int(best_k)


def _remap_contiguous(train_labels, val_labels, test_labels):
    all_labels = np.concatenate([train_labels, val_labels, test_labels], axis=0).astype(np.int64)
    uniq = np.unique(all_labels)
    label_map = {int(v): int(i) for i, v in enumerate(uniq.tolist())}
    remap = np.vectorize(lambda z: label_map[int(z)], otypes=[np.int64])
    return remap(train_labels), remap(val_labels), remap(test_labels), uniq


def build_primitive_h5(source_h5, out_h5, clusters, out_len, min_len, max_segments, seed, label_mode, min_clusters, max_clusters, auto_sample_size, extract_mode, beta_v, beta_s, max_len, dpm_components, dpm_decimate, dpm_accept_ratio, dpm_drop_unassigned):
    with h5py.File(source_h5, 'r') as f:
        x_train = f['x'][:]
        y_train = f['y'][:]
        x_val = f['valid_x'][:]
        y_val = f['valid_y'][:]
        x_test = f['test_x'][:]
        y_test = f['test_y'][:]
    tr_samples, tr_feats, tr_meta, tr_gids = _extract_split(x_train, y_train, 'train', out_len, min_len, max_segments, extract_mode, beta_v, beta_s, max_len)
    va_samples, va_feats, va_meta, va_gids = _extract_split(x_val, y_val, 'val', out_len, min_len, max_segments, extract_mode, beta_v, beta_s, max_len)
    te_samples, te_feats, te_meta, te_gids = _extract_split(x_test, y_test, 'test', out_len, min_len, max_segments, extract_mode, beta_v, beta_s, max_len)
    if len(tr_samples) == 0:
        raise RuntimeError('No primitive samples extracted from training split.')
    if label_mode == 'cluster':
        tr_feats_np = np.stack(tr_feats, axis=0)
        if isinstance(clusters, str) and clusters.lower() == 'auto':
            n_clusters = _select_cluster_count_auto(
                tr_feats_np,
                min_k=min_clusters,
                max_k=max_clusters,
                seed=seed,
                sample_size=auto_sample_size,
            )
            print('Auto-selected primitive cluster count:', n_clusters)
        else:
            n_clusters = int(clusters)
        n_clusters = max(2, min(n_clusters, tr_feats_np.shape[0]))
        centers, tr_labels = _kmeans_fit_predict(tr_feats_np, n_clusters=n_clusters, seed=seed, max_iter=100)
        va_labels = _kmeans_predict(np.stack(va_feats, axis=0), centers) if len(va_feats) > 0 else np.zeros((0,), dtype=np.int64)
        te_labels = _kmeans_predict(np.stack(te_feats, axis=0), centers) if len(te_feats) > 0 else np.zeros((0,), dtype=np.int64)
        class_ids = np.arange(n_clusters, dtype=np.int64)
    elif label_mode == 'dpm':
        tr_labels, va_labels, te_labels, n_clusters = _fit_predict_dpm(
            train_samples=tr_samples,
            train_gids=tr_gids,
            val_samples=va_samples,
            val_gids=va_gids,
            test_samples=te_samples,
            test_gids=te_gids,
            max_components=int(dpm_components),
            seed=seed,
            decim=int(dpm_decimate),
            accept_ratio=float(dpm_accept_ratio),
            drop_unassigned=bool(dpm_drop_unassigned),
        )
        keep_tr = tr_labels >= 0
        keep_va = va_labels >= 0
        keep_te = te_labels >= 0
        tr_labels = tr_labels[keep_tr]
        va_labels = va_labels[keep_va]
        te_labels = te_labels[keep_te]
        tr_samples = [tr_samples[i] for i in np.where(keep_tr)[0].tolist()]
        va_samples = [va_samples[i] for i in np.where(keep_va)[0].tolist()]
        te_samples = [te_samples[i] for i in np.where(keep_te)[0].tolist()]
        tr_meta = [tr_meta[i] for i in np.where(keep_tr)[0].tolist()]
        va_meta = [va_meta[i] for i in np.where(keep_va)[0].tolist()]
        te_meta = [te_meta[i] for i in np.where(keep_te)[0].tolist()]
        class_ids = np.arange(n_clusters, dtype=np.int64)
    else:
        tr_labels = np.array([m[2] for m in tr_meta], dtype=np.int64)
        va_labels = np.array([m[2] for m in va_meta], dtype=np.int64)
        te_labels = np.array([m[2] for m in te_meta], dtype=np.int64)
        tr_labels, va_labels, te_labels, class_ids = _remap_contiguous(tr_labels, va_labels, te_labels)
        n_clusters = int(len(class_ids))
    tr_x = np.stack(tr_samples, axis=0).astype(np.float32)
    va_x = np.stack(va_samples, axis=0).astype(np.float32) if len(va_samples) > 0 else np.zeros((0, out_len, 150), dtype=np.float32)
    te_x = np.stack(te_samples, axis=0).astype(np.float32) if len(te_samples) > 0 else np.zeros((0, out_len, 150), dtype=np.float32)
    os.makedirs(os.path.dirname(out_h5), exist_ok=True)
    with h5py.File(out_h5, 'w') as f:
        f.create_dataset('x', data=tr_x)
        f.create_dataset('y', data=_one_hot(tr_labels, n_clusters))
        f.create_dataset('valid_x', data=va_x)
        f.create_dataset('valid_y', data=_one_hot(va_labels, n_clusters))
        f.create_dataset('test_x', data=te_x)
        f.create_dataset('test_y', data=_one_hot(te_labels, n_clusters))
        f.create_dataset('pid', data=np.array([m[1] for m in tr_meta], dtype=np.int64))
        f.create_dataset('valid_pid', data=np.array([m[1] for m in va_meta], dtype=np.int64))
        f.create_dataset('test_pid', data=np.array([m[1] for m in te_meta], dtype=np.int64))
        f.create_dataset('aid', data=np.array([m[2] for m in tr_meta], dtype=np.int64))
        f.create_dataset('valid_aid', data=np.array([m[2] for m in va_meta], dtype=np.int64))
        f.create_dataset('test_aid', data=np.array([m[2] for m in te_meta], dtype=np.int64))
        f.create_dataset('src_action', data=np.array([m[2] for m in tr_meta], dtype=np.int64))
        f.create_dataset('valid_src_action', data=np.array([m[2] for m in va_meta], dtype=np.int64))
        f.create_dataset('test_src_action', data=np.array([m[2] for m in te_meta], dtype=np.int64))
        f.create_dataset('class_ids', data=class_ids.astype(np.int64))
    print('Saved primitive dataset to', out_h5)
    print('Primitive classes:', n_clusters)
    print('Label mode:', label_mode)
    print('Extract mode:', extract_mode)
    print('Split size train/val/test:', tr_x.shape[0], va_x.shape[0], te_x.shape[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_h5', type=str, required=True)
    parser.add_argument('--out_h5', type=str, required=True)
    parser.add_argument('--clusters', type=str, default='32')
    parser.add_argument('--min_clusters', type=int, default=8)
    parser.add_argument('--max_clusters', type=int, default=48)
    parser.add_argument('--auto_sample_size', type=int, default=20000)
    parser.add_argument('--out_len', type=int, default=20)
    parser.add_argument('--min_len', type=int, default=5)
    parser.add_argument('--max_segments', type=int, default=6)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--extract_mode', type=str, default='paper', choices=['paper', 'heuristic'])
    parser.add_argument('--beta_v', type=float, default=0.02)
    parser.add_argument('--beta_s', type=float, default=0.002)
    parser.add_argument('--max_len', type=int, default=80)
    parser.add_argument('--dpm_components', type=int, default=128)
    parser.add_argument('--dpm_decimate', type=int, default=5)
    parser.add_argument('--dpm_accept_ratio', type=float, default=0.8)
    parser.add_argument('--dpm_drop_unassigned', action='store_true')
    parser.add_argument('--label_mode', type=str, default='cluster', choices=['cluster', 'id', 'dpm'])
    args = parser.parse_args()
    np.random.seed(args.seed)
    build_primitive_h5(
        source_h5=args.source_h5,
        out_h5=args.out_h5,
        clusters=args.clusters,
        out_len=args.out_len,
        min_len=args.min_len,
        max_segments=args.max_segments,
        seed=args.seed,
        label_mode=args.label_mode,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        auto_sample_size=args.auto_sample_size,
        extract_mode=args.extract_mode,
        beta_v=args.beta_v,
        beta_s=args.beta_s,
        max_len=args.max_len,
        dpm_components=args.dpm_components,
        dpm_decimate=args.dpm_decimate,
        dpm_accept_ratio=args.dpm_accept_ratio,
        dpm_drop_unassigned=args.dpm_drop_unassigned,
    )


if __name__ == '__main__':
    main()
