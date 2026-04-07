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


def _segment_primitives(seq150, out_len, min_len, max_segments):
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
    return prims, feats


def _extract_split(x, y, split_name, out_len, min_len, max_segments):
    samples = []
    feats = []
    metas = []
    y = _to_label(y)
    for i in range(x.shape[0]):
        prims, fts = _segment_primitives(x[i], out_len=out_len, min_len=min_len, max_segments=max_segments)
        for p, f in zip(prims, fts):
            sample = np.zeros((out_len, 150), dtype=np.float32)
            sample[:, :75] = p
            samples.append(sample)
            feats.append(f)
            metas.append((split_name, int(i), int(y[i])))
    return samples, feats, metas


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


def build_primitive_h5(source_h5, out_h5, clusters, out_len, min_len, max_segments, seed, label_mode, min_clusters, max_clusters, auto_sample_size):
    with h5py.File(source_h5, 'r') as f:
        x_train = f['x'][:]
        y_train = f['y'][:]
        x_val = f['valid_x'][:]
        y_val = f['valid_y'][:]
        x_test = f['test_x'][:]
        y_test = f['test_y'][:]
    tr_samples, tr_feats, tr_meta = _extract_split(x_train, y_train, 'train', out_len, min_len, max_segments)
    va_samples, va_feats, va_meta = _extract_split(x_val, y_val, 'val', out_len, min_len, max_segments)
    te_samples, te_feats, te_meta = _extract_split(x_test, y_test, 'test', out_len, min_len, max_segments)
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
    parser.add_argument('--label_mode', type=str, default='cluster', choices=['cluster', 'id'])
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
    )


if __name__ == '__main__':
    main()
