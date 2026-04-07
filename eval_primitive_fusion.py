import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

from data import NTUDataLoaders
from model import SGN


def _load_model(checkpoint_path, dataset, num_classes, seg, args, device):
    model = SGN(num_classes, dataset, seg, args).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def _collect_test_probs(loader, model, batch_size, workers, device, with_meta):
    test_loader = loader.get_test_loader(batch_size, workers)
    probs_all = []
    y_all = []
    pid_all = []
    for batch in test_loader:
        if with_meta:
            if len(batch) != 4:
                raise RuntimeError('Primitive dataset is missing pid/aid metadata. Rebuild NTU_PRIM_*.h5 with build_primitive_dataset.py --label_mode id.')
            inputs, target, pid, _ = batch
        else:
            inputs, target = batch
            pid = None
        with torch.no_grad():
            logits = model(inputs.to(device))
        view_factor = inputs.size(0) // target.size(0)
        logits = logits.view((-1, view_factor, logits.size(1))).mean(1)
        probs = F.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        probs_all.append(probs)
        y_all.append(target.numpy().astype(np.int64))
        if with_meta:
            pid_all.append(pid.numpy().astype(np.int64))
    probs_all = np.concatenate(probs_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    if with_meta:
        pid_all = np.concatenate(pid_all, axis=0)
        return probs_all, y_all, pid_all
    return probs_all, y_all, None


def _topk_acc(probs, labels, topk=(1, 5)):
    order = np.argsort(-probs, axis=1)
    out = {}
    for k in topk:
        hit = 0
        for i in range(labels.shape[0]):
            if labels[i] in order[i, :k]:
                hit += 1
        out[k] = 100.0 * hit / max(1, labels.shape[0])
    return out


def _align_probs_to_space(probs, class_ids, class_space):
    aligned = np.zeros((probs.shape[0], len(class_space)), dtype=np.float32)
    pos = {int(v): i for i, v in enumerate(class_space.tolist())}
    for i, cid in enumerate(class_ids.tolist()):
        if int(cid) in pos:
            aligned[:, pos[int(cid)]] = probs[:, i]
    return aligned


def _aggregate_primitive_probs(prim_probs, prim_pid, prim_labels, num_samples, alpha):
    c = prim_probs.shape[1]
    num = np.zeros((num_samples, c), dtype=np.float32)
    den = np.zeros((num_samples, 1), dtype=np.float32)
    y = np.full((num_samples,), -1, dtype=np.int64)
    cnt = np.zeros((num_samples,), dtype=np.int64)
    conf = np.max(prim_probs, axis=1)
    w = np.power(np.clip(conf, 1e-6, 1.0), alpha).astype(np.float32)
    for i in range(prim_probs.shape[0]):
        idx = int(prim_pid[i])
        if idx < 0 or idx >= num_samples:
            continue
        num[idx] += w[i] * prim_probs[i]
        den[idx, 0] += w[i]
        y[idx] = int(prim_labels[i])
        cnt[idx] += 1
    den = np.clip(den, 1e-8, None)
    agg = num / den
    return agg, y, cnt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, default=1)
    parser.add_argument('--seg', type=int, default=20)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--skeleton_dataset', type=str, default='NTU_ID')
    parser.add_argument('--primitive_dataset', type=str, default='NTU_PRIM')
    parser.add_argument('--skeleton_ckpt', type=str, default='')
    parser.add_argument('--primitive_ckpt', type=str, default='')
    parser.add_argument('--fusion_weight', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    args = parser.parse_args()

    if args.skeleton_ckpt == '':
        args.skeleton_ckpt = os.path.join('results', args.skeleton_dataset, 'SGN', f'{args.case}_best.pth')
    if args.primitive_ckpt == '':
        args.primitive_ckpt = os.path.join('results', args.primitive_dataset, 'SGN', f'{args.case}_best.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sk_loader = NTUDataLoaders(args.skeleton_dataset, args.case, seg=args.seg, args=args, return_meta=False)
    pr_loader = NTUDataLoaders(args.primitive_dataset, args.case, seg=args.seg, args=args, return_meta=True)
    if pr_loader.test_pid is None:
        raise RuntimeError('Primitive dataset does not contain source skeleton index metadata. Rebuild with --label_mode id.')

    sk_model = _load_model(args.skeleton_ckpt, args.skeleton_dataset, sk_loader.num_classes, args.seg, args, device)
    pr_model = _load_model(args.primitive_ckpt, args.primitive_dataset, pr_loader.num_classes, args.seg, args, device)

    sk_probs, sk_y_new, _ = _collect_test_probs(sk_loader, sk_model, args.batch_size, args.workers, device, with_meta=False)
    pr_probs, pr_y_new, pr_pid = _collect_test_probs(pr_loader, pr_model, args.batch_size, args.workers, device, with_meta=True)

    class_space = np.unique(np.concatenate([sk_loader.class_ids, pr_loader.class_ids], axis=0)).astype(np.int64)
    sk_probs_space = _align_probs_to_space(sk_probs, sk_loader.class_ids, class_space)
    pr_probs_space = _align_probs_to_space(pr_probs, pr_loader.class_ids, class_space)

    sk_y = sk_loader.class_ids[sk_y_new]
    pr_y = pr_loader.class_ids[pr_y_new]

    pr_agg_probs, pr_agg_y, pr_cnt = _aggregate_primitive_probs(
        prim_probs=pr_probs_space,
        prim_pid=pr_pid,
        prim_labels=pr_y,
        num_samples=sk_probs_space.shape[0],
        alpha=args.alpha,
    )
    valid = pr_cnt > 0
    if not np.all(valid):
        pr_agg_probs[~valid] = sk_probs_space[~valid]
        pr_agg_y[~valid] = sk_y[~valid]

    sk_conf = np.max(sk_probs_space, axis=1, keepdims=True)
    pr_conf = np.max(pr_agg_probs, axis=1, keepdims=True)
    w_s = np.power(np.clip(sk_conf, 1e-6, 1.0), args.alpha)
    w_p = args.fusion_weight * np.power(np.clip(pr_conf, 1e-6, 1.0), args.alpha)
    fused_probs = (w_s * sk_probs_space + w_p * pr_agg_probs) / np.clip(w_s + w_p, 1e-8, None)

    cls_pos = {int(v): i for i, v in enumerate(class_space.tolist())}
    y_idx = np.array([cls_pos[int(v)] for v in sk_y], dtype=np.int64)

    sk_acc = _topk_acc(sk_probs_space, y_idx, topk=(1, 5))
    pr_acc = _topk_acc(pr_agg_probs, y_idx, topk=(1, 5))
    fu_acc = _topk_acc(fused_probs, y_idx, topk=(1, 5))
    covered = float(valid.mean() * 100.0)

    print('Skeleton Top-1 {:.3f} Top-5 {:.3f}'.format(sk_acc[1], sk_acc[5]))
    print('Primitive Top-1 {:.3f} Top-5 {:.3f} Coverage {:.3f}'.format(pr_acc[1], pr_acc[5], covered))
    print('Fusion Top-1 {:.3f} Top-5 {:.3f}'.format(fu_acc[1], fu_acc[5]))


if __name__ == '__main__':
    main()
