import argparse
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from model import SGN
from data import NTUDataLoaders
from util import get_num_classes

def get_loader(args, split):
    try:
        loaders = NTUDataLoaders(dataset=args.dataset, case=args.case, seg=args.seg, args=args)
    except TypeError:
        loaders = NTUDataLoaders(dataset=args.dataset, case=args.case, seg=args.seg)
    if split == "train":
        return loaders.get_train_loader(batch_size=args.batch_size, num_workers=args.workers)
    if split == "test":
        return loaders.get_test_loader(batch_size=args.batch_size, num_workers=args.workers)
    return loaders.get_val_loader(batch_size=args.batch_size, num_workers=args.workers)

def collect_features(model, loader, device, max_samples):
    buf = {}
    def hook(mod, inp):
        buf['x'] = inp[0].detach().cpu()
    handle = model.fc.register_forward_pre_hook(lambda m, inp: hook(m, inp))
    feats = []
    labels = []
    total = 0
    for x, y in loader:
        x = x.to(device)
        with torch.no_grad():
            _ = model(x)
        f = buf['x']
        feats.append(f)
        labels.append(y)
        total += f.size(0)
        if max_samples and total >= max_samples:
            break
    handle.remove()
    return torch.cat(feats, 0).numpy(), torch.cat(labels, 0).numpy()

def subsample_per_id(x, y, samples_per_id, seed):
    if not samples_per_id:
        return x, y
    rng = np.random.RandomState(seed)
    xs = []
    ys = []
    for pid in np.unique(y):
        idx = np.where(y == pid)[0]
        if len(idx) > samples_per_id:
            idx = rng.choice(idx, samples_per_id, replace=False)
        xs.append(x[idx])
        ys.append(y[idx])
    return np.concatenate(xs, 0), np.concatenate(ys, 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NTU_ID')
    parser.add_argument('--case', type=int, default=0)
    parser.add_argument('--seg', type=int, default=20)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--ckpt', type=str, default='results/NTU_ID/SGN/0_best.pth')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=2000)
    parser.add_argument('--samples-per-id', type=int, default=100)
    parser.add_argument('--out', type=str, default='tsne_person.png')
    parser.add_argument('--motion-only', action='store_true')
    args = parser.parse_args()

    num_classes = get_num_classes(args.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SGN(num_classes, args.dataset, args.seg, args).to(device).eval()
    try:
        sd = torch.load(args.ckpt, weights_only=True)
    except TypeError:
        sd = torch.load(args.ckpt)
    state = sd['state_dict'] if 'state_dict' in sd else sd
    try:
        model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(state, strict=False)

    loader = get_loader(args, args.split)
    X, Y = collect_features(model, loader, device, args.max_samples)
    X, Y = subsample_per_id(X, Y, args.samples_per_id, args.seed)
    ids, counts = np.unique(Y, return_counts=True)
    for pid, cnt in zip(ids, counts):
        print(f'person {int(pid)}: {int(cnt)} samples')

    n = X.shape[0]
    max_p = max(5, (n - 1) // 3) if n > 3 else 5
    perplexity = min(args.perplexity, max_p)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=args.seed, init='random')
    Z = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    ids = np.unique(Y)
    cmap = plt.cm.get_cmap('tab20', len(ids))
    for i, pid in enumerate(ids):
        idx = np.where(Y == pid)[0]
        plt.scatter(Z[idx, 0], Z[idx, 1], s=6, alpha=0.7, color=cmap(i))
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)

if __name__ == '__main__':
    main()
