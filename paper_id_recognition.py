import argparse
import json
from typing import Dict, List

import h5py
import numpy as np


def _parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            s, e = part.split('-', 1)
            s_i = int(s)
            e_i = int(e)
            if e_i < s_i:
                s_i, e_i = e_i, s_i
            out.extend(list(range(s_i, e_i + 1)))
        else:
            out.append(int(part))
    return sorted(list(set(out)))


def _as_int_array(x):
    if x is None:
        return None
    return np.asarray(x, dtype=np.int64)


def _get_optional(f: h5py.File, name: str):
    return f[name][:] if name in f else None


def _load_h5_auto(path: str, merge_valid_into_train: bool = True) -> Dict[str, np.ndarray]:
    with h5py.File(path, 'r') as f:
        if 'test_x' in f:
            train_x = f['x'][:]
            train_pid = _get_optional(f, 'pid')
            train_aid = _get_optional(f, 'aid')
            train_cam = _get_optional(f, 'camera')
            valid_x = _get_optional(f, 'valid_x')
            valid_pid = _get_optional(f, 'valid_pid')
            valid_aid = _get_optional(f, 'valid_aid')
            valid_cam = _get_optional(f, 'valid_camera')
            test_x = f['test_x'][:]
            test_pid = _get_optional(f, 'test_pid')
            test_aid = _get_optional(f, 'test_aid')
            test_cam = _get_optional(f, 'test_camera')
            if merge_valid_into_train and valid_x is not None:
                train_x = np.concatenate([train_x, valid_x], axis=0)
                if train_pid is not None and valid_pid is not None:
                    train_pid = np.concatenate([train_pid, valid_pid], axis=0)
                if train_aid is not None and valid_aid is not None:
                    train_aid = np.concatenate([train_aid, valid_aid], axis=0)
                if train_cam is not None and valid_cam is not None:
                    train_cam = np.concatenate([train_cam, valid_cam], axis=0)
            return {
                'train_x': train_x,
                'train_pid': _as_int_array(train_pid),
                'train_aid': _as_int_array(train_aid),
                'train_camera': _as_int_array(train_cam),
                'test_x': test_x,
                'test_pid': _as_int_array(test_pid),
                'test_aid': _as_int_array(test_aid),
                'test_camera': _as_int_array(test_cam),
            }
        x = f['x'][:]
        pid = _get_optional(f, 'pid')
        aid = _get_optional(f, 'aid')
        camera = _get_optional(f, 'camera')
        return {
            'x': x,
            'pid': _as_int_array(pid),
            'aid': _as_int_array(aid),
            'camera': _as_int_array(camera),
        }


def _to_time_major(seq: np.ndarray) -> np.ndarray:
    arr = np.asarray(seq)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 4 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0, 3))
        return arr.reshape(arr.shape[0], -1)
    if arr.ndim == 4:
        return arr.reshape(arr.shape[0], -1)
    if arr.ndim == 3:
        return arr.reshape(arr.shape[0], -1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr.reshape(arr.shape[0], -1)


def _select_single_body(seq2d: np.ndarray) -> np.ndarray:
    if seq2d.shape[1] == 75:
        out = seq2d
    elif seq2d.shape[1] >= 150:
        a = seq2d[:, :75]
        b = seq2d[:, 75:150]
        a_valid = np.any(np.abs(a) > 1e-8, axis=1)
        b_valid = np.any(np.abs(b) > 1e-8, axis=1)
        out = a.copy()
        swap = (~a_valid) & b_valid
        out[swap] = b[swap]
    else:
        trim = seq2d.shape[1] - (seq2d.shape[1] % 3)
        out = seq2d[:, :trim]
    keep = np.any(np.abs(out) > 1e-8, axis=1)
    out = out[keep]
    if out.shape[0] == 0:
        return np.zeros((0, out.shape[1] if out.ndim == 2 else 75), dtype=np.float32)
    return out.astype(np.float32)


def _extract_feature_sets(seq: np.ndarray, fd_scales: List[int]) -> Dict[str, np.ndarray]:
    seq2d = _to_time_major(seq)
    posture = _select_single_body(seq2d)
    feat = {'posture': posture}
    for t in fd_scales:
        if posture.shape[0] <= t:
            feat[f'fd{t}'] = np.zeros((0, posture.shape[1]), dtype=np.float32)
        else:
            feat[f'fd{t}'] = posture[t:] - posture[:-t]
    return feat


def _kmeans_numpy(points: np.ndarray, k: int, rng: np.random.RandomState, max_iter: int = 80) -> np.ndarray:
    n, d = points.shape
    if n == 0:
        return np.zeros((k, d), dtype=np.float32)
    if n == 1:
        return np.repeat(points, k, axis=0).astype(np.float32)
    init_idx = rng.choice(n, size=k, replace=(n < k))
    centers = points[init_idx].astype(np.float32).copy()
    for _ in range(max_iter):
        diff = points[:, None, :] - centers[None, :, :]
        dist = np.sum(diff * diff, axis=2)
        assign = np.argmin(dist, axis=1)
        new_centers = centers.copy()
        for c in range(k):
            idx = np.where(assign == c)[0]
            if idx.size == 0:
                new_centers[c] = points[rng.randint(0, n)]
            else:
                new_centers[c] = points[idx].mean(axis=0)
        shift = float(np.mean(np.linalg.norm(new_centers - centers, axis=1)))
        centers = new_centers
        if shift < 1e-5:
            break
    return centers.astype(np.float32)


def _kmeans(points: np.ndarray, k: int, seed: int, backend: str) -> np.ndarray:
    if backend == 'sklearn':
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
        model.fit(points)
        return model.cluster_centers_.astype(np.float32)
    rng = np.random.RandomState(seed)
    return _kmeans_numpy(points, k, rng)


def _compute_codebooks_per_subject(
    train_feats: List[Dict[str, np.ndarray]],
    train_labels: np.ndarray,
    keys: List[str],
    q_per_subject: int,
    seed: int,
    kmeans_backend: str,
) -> Dict[str, np.ndarray]:
    centers = {}
    subjects = sorted(np.unique(train_labels).tolist())
    for key in keys:
        all_centers = []
        for sid in subjects:
            sid_idx = np.where(train_labels == sid)[0]
            sid_points = [train_feats[i][key] for i in sid_idx if train_feats[i][key].shape[0] > 0]
            if len(sid_points) == 0:
                dim = train_feats[0][key].shape[1] if train_feats and train_feats[0][key].ndim == 2 else 75
                sid_centers = np.zeros((q_per_subject, dim), dtype=np.float32)
            else:
                data = np.concatenate(sid_points, axis=0)
                sid_centers = _kmeans(
                    data,
                    q_per_subject,
                    seed + int(sid) * 13 + len(key),
                    kmeans_backend,
                )
            all_centers.append(sid_centers)
        centers[key] = np.concatenate(all_centers, axis=0).astype(np.float32)
    return centers


def _hist_from_centers(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    c = centers.shape[0]
    hist = np.zeros((c,), dtype=np.float32)
    if points.shape[0] == 0:
        return hist
    diff = points[:, None, :] - centers[None, :, :]
    dist = np.sum(diff * diff, axis=2)
    idx = np.argmin(dist, axis=1)
    binc = np.bincount(idx, minlength=c).astype(np.float32)
    hist = binc / max(1.0, float(points.shape[0]))
    return hist


def _build_sequence_descriptors(
    feats: List[Dict[str, np.ndarray]],
    centers: Dict[str, np.ndarray],
    variant: str,
    fd_scales: List[int],
) -> np.ndarray:
    if variant == '1f':
        keys = ['posture']
    else:
        keys = ['posture'] + [f'fd{t}' for t in fd_scales]
    desc = []
    for f in feats:
        parts = [_hist_from_centers(f[k], centers[k]) for k in keys]
        desc.append(np.concatenate(parts, axis=0))
    return np.stack(desc, axis=0).astype(np.float32)


def _scores(train_x: np.ndarray, test_x: np.ndarray, metric: str) -> np.ndarray:
    eps = 1e-9
    if metric == 'hist':
        return np.minimum(train_x[None, :, :], test_x[:, None, :]).sum(axis=2)
    if metric == 'cos':
        tn = train_x / (np.linalg.norm(train_x, axis=1, keepdims=True) + eps)
        qn = test_x / (np.linalg.norm(test_x, axis=1, keepdims=True) + eps)
        return np.matmul(qn, tn.T)
    if metric == 'kl':
        p = test_x[:, None, :] + eps
        q = train_x[None, :, :] + eps
        kl1 = np.sum(p * np.log(p / q), axis=2)
        kl2 = np.sum(q * np.log(q / p), axis=2)
        d = 0.5 * (kl1 + kl2)
        return -d
    raise ValueError(metric)


def _topk_accuracy(scores: np.ndarray, train_labels: np.ndarray, test_labels: np.ndarray, topk=(1, 2, 3, 4, 5)):
    order = np.argsort(-scores, axis=1)
    ranked_labels = train_labels[order]
    out = {}
    for k in topk:
        kk = min(k, ranked_labels.shape[1])
        hit = np.any(ranked_labels[:, :kk] == test_labels[:, None], axis=1)
        out[k] = float(hit.mean() * 100.0)
    return out


def _camera_split(
    x: np.ndarray,
    pid: np.ndarray,
    aid: np.ndarray,
    camera: np.ndarray,
    train_cameras: List[int],
    test_cameras: List[int],
) -> Dict[str, np.ndarray]:
    train_mask = np.isin(camera, np.array(train_cameras, dtype=np.int64))
    test_mask = np.isin(camera, np.array(test_cameras, dtype=np.int64))
    return {
        'train_x': x[train_mask],
        'train_pid': pid[train_mask],
        'train_aid': aid[train_mask],
        'train_camera': camera[train_mask],
        'test_x': x[test_mask],
        'test_pid': pid[test_mask],
        'test_aid': aid[test_mask],
        'test_camera': camera[test_mask],
    }


def _filter_action_range(aid: np.ndarray, action_min: int, action_max: int) -> np.ndarray:
    if aid.size == 0:
        return np.zeros((0,), dtype=bool)
    if aid.min() == 0:
        aid_use = aid + 1
    else:
        aid_use = aid
    return (aid_use >= action_min) & (aid_use <= action_max)


def _remap_common_ids(train_pid: np.ndarray, test_pid: np.ndarray):
    common = sorted(list(set(train_pid.tolist()) & set(test_pid.tolist())))
    if len(common) == 0:
        return None, None, None
    lut = {int(v): i for i, v in enumerate(common)}
    train_y = np.array([lut[int(v)] for v in train_pid], dtype=np.int64)
    test_y = np.array([lut[int(v)] for v in test_pid], dtype=np.int64)
    return train_y, test_y, np.array(common, dtype=np.int64)


def run(args):
    loaded = _load_h5_auto(args.h5, merge_valid_into_train=args.merge_valid)
    if 'x' in loaded:
        if loaded['camera'] is None:
            raise ValueError('输入h5为未划分数据时必须包含camera字段。')
        split = _camera_split(
            loaded['x'],
            loaded['pid'],
            loaded['aid'],
            loaded['camera'],
            train_cameras=args.train_cameras,
            test_cameras=args.test_cameras,
        )
    else:
        split = loaded
    if split['train_pid'] is None or split['test_pid'] is None:
        raise ValueError('h5中必须包含pid/test_pid（或未划分模式下pid）。')
    if split['train_aid'] is None or split['test_aid'] is None:
        raise ValueError('h5中必须包含aid/test_aid（或未划分模式下aid）。')
    tr_keep_action = _filter_action_range(split['train_aid'], args.action_min, args.action_max)
    te_keep_action = _filter_action_range(split['test_aid'], args.action_min, args.action_max)
    train_x = split['train_x'][tr_keep_action]
    train_pid = split['train_pid'][tr_keep_action]
    test_x = split['test_x'][te_keep_action]
    test_pid = split['test_pid'][te_keep_action]
    train_aid = split['train_aid'][tr_keep_action]
    test_aid = split['test_aid'][te_keep_action]
    train_y, test_y, common_ids = _remap_common_ids(train_pid, test_pid)
    if common_ids is None:
        raise ValueError('按当前筛选条件，train/test没有共同身份，无法闭集ID识别。')
    tr_common = np.isin(train_pid, common_ids)
    te_common = np.isin(test_pid, common_ids)
    train_x = train_x[tr_common]
    test_x = test_x[te_common]
    train_pid = train_pid[tr_common]
    test_pid = test_pid[te_common]
    train_aid = train_aid[tr_common]
    test_aid = test_aid[te_common]
    train_y, test_y, common_ids = _remap_common_ids(train_pid, test_pid)
    fd_scales = [int(v) for v in args.fd_scales]
    train_feats = [_extract_feature_sets(seq, fd_scales) for seq in train_x]
    test_feats = [_extract_feature_sets(seq, fd_scales) for seq in test_x]
    keys = ['posture'] + [f'fd{t}' for t in fd_scales]
    centers = _compute_codebooks_per_subject(
        train_feats=train_feats,
        train_labels=train_y,
        keys=keys,
        q_per_subject=args.q_per_subject,
        seed=args.seed,
        kmeans_backend=args.kmeans_backend,
    )
    variants = ['1f', '4f'] if args.variant == 'both' else [args.variant]
    metrics = args.metrics
    all_results = {}
    for variant in variants:
        tr_desc = _build_sequence_descriptors(train_feats, centers, variant, fd_scales)
        te_desc = _build_sequence_descriptors(test_feats, centers, variant, fd_scales)
        all_results[variant] = {}
        for metric in metrics:
            s = _scores(tr_desc, te_desc, metric)
            acc = _topk_accuracy(s, train_y, test_y, topk=(1, 2, 3, 4, 5))
            all_results[variant][metric] = acc
    report = {
        'dataset_path': args.h5,
        'action_range': [args.action_min, args.action_max],
        'train_cameras': args.train_cameras,
        'test_cameras': args.test_cameras,
        'num_train_sequences': int(train_x.shape[0]),
        'num_test_sequences': int(test_x.shape[0]),
        'num_ids_closed_set': int(len(common_ids)),
        'ids_original': common_ids.tolist(),
        'q_per_subject': int(args.q_per_subject),
        'fd_scales': fd_scales,
        'results': all_results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out_json:
        with open(args.out_json, 'w', encoding='utf-8') as fw:
            json.dump(report, fw, ensure_ascii=False, indent=2)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', type=str, required=True)
    parser.add_argument('--out_json', type=str, default='')
    parser.add_argument('--merge_valid', dest='merge_valid', action='store_true')
    parser.add_argument('--no_merge_valid', dest='merge_valid', action='store_false')
    parser.set_defaults(merge_valid=True)
    parser.add_argument('--variant', type=str, default='both', choices=['1f', '4f', 'both'])
    parser.add_argument('--metrics', type=str, default='hist,cos,kl')
    parser.add_argument('--fd_scales', type=str, default='1,5,10')
    parser.add_argument('--q_per_subject', type=int, default=6)
    parser.add_argument('--kmeans_backend', type=str, default='numpy', choices=['numpy', 'sklearn'])
    parser.add_argument('--train_cameras', type=str, default='2,3')
    parser.add_argument('--test_cameras', type=str, default='1')
    parser.add_argument('--action_min', type=int, default=1)
    parser.add_argument('--action_max', type=int, default=40)
    parser.add_argument('--seed', type=int, default=1337)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.metrics = [x.strip().lower() for x in args.metrics.split(',') if x.strip()]
    args.fd_scales = [int(x.strip()) for x in args.fd_scales.split(',') if x.strip()]
    args.train_cameras = _parse_int_list(args.train_cameras)
    args.test_cameras = _parse_int_list(args.test_cameras)
    run(args)


if __name__ == '__main__':
    main()
