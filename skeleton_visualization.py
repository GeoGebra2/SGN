import argparse
import json
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from model import SGN
from data import NTUDataLoaders
from util import get_num_classes

def get_loader(args, split):
    try:
        loaders = NTUDataLoaders(dataset=args.dataset, case=args.case, seg=args.seg, args=args, return_meta=True)
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
    pids = []
    aids = []
    total = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 4:
            x, y, pid, aid = batch[0], batch[1], batch[2], batch[3]
            pids.append(pid.detach().cpu() if torch.is_tensor(pid) else torch.as_tensor(pid))
            aids.append(aid.detach().cpu() if torch.is_tensor(aid) else torch.as_tensor(aid))
        else:
            x, y = batch[0], batch[1]
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
    X = torch.cat(feats, 0).numpy()
    Y = torch.cat(labels, 0).numpy()
    PID = torch.cat(pids, 0).numpy() if pids else None
    AID = torch.cat(aids, 0).numpy() if aids else None
    return X, Y, PID, AID

def subsample_per_id(x, pid, aid, samples_per_id, seed):
    if (not samples_per_id) or (pid is None):
        return x, pid, aid
    rng = np.random.RandomState(seed)
    xs = []
    ps = []
    ac = []
    for _pid in np.unique(pid):
        idx = np.where(pid == _pid)[0]
        if len(idx) > samples_per_id:
            idx = rng.choice(idx, samples_per_id, replace=False)
        xs.append(x[idx])
        ps.append(pid[idx])
        if aid is not None:
            ac.append(aid[idx])
    x = np.concatenate(xs, 0)
    pid = np.concatenate(ps, 0)
    aid = np.concatenate(ac, 0) if ac else aid
    return x, pid, aid

def _rgba_to_hex(rgba):
    r, g, b = rgba[0], rgba[1], rgba[2]
    return "#{:02x}{:02x}{:02x}".format(int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))

def save_tsne_html(Z, pid, aid, out_html):
    pid_unique = np.unique(pid)
    cmap = plt.get_cmap('tab20', len(pid_unique))
    pid_to_color = {int(p): _rgba_to_hex(cmap(i)) for i, p in enumerate(pid_unique)}

    points = []
    for x, y, p, a in zip(Z[:, 0], Z[:, 1], pid, aid):
        points.append({
            "x": float(x),
            "y": float(y),
            "pid": int(p),
            "aid": int(a),
            "color": pid_to_color[int(p)],
        })

    xs = Z[:, 0]
    ys = Z[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
    pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>t-SNE</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
  <style>
    html, body {{ height: 100%; margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }}
    #wrap {{ position: relative; width: 100%; height: 100%; }}
    #tooltip {{
      position: absolute;
      pointer-events: none;
      background: rgba(0,0,0,0.75);
      color: #fff;
      padding: 6px 8px;
      border-radius: 6px;
      font-size: 12px;
      line-height: 1.2;
      display: none;
      white-space: nowrap;
    }}
    .dot {{ cursor: default; }}
  </style>
</head>
<body>
  <div id="wrap">
    <div id="tooltip"></div>
    <svg id="plot" width="100%" height="100%" viewBox="0 0 1000 750" preserveAspectRatio="xMidYMid meet"></svg>
  </div>
  <script>
    const points = {json.dumps(points, ensure_ascii=False)};
    const svg = d3.select("#plot");
    const tooltip = d3.select("#tooltip");

    const W = 1000, H = 750;
    const margin = {{left: 40, right: 20, top: 20, bottom: 40}};
    const innerW = W - margin.left - margin.right;
    const innerH = H - margin.top - margin.bottom;

    const x = d3.scaleLinear().domain([{x_min - pad_x}, {x_max + pad_x}]).range([0, innerW]);
    const y = d3.scaleLinear().domain([{y_min - pad_y}, {y_max + pad_y}]).range([innerH, 0]);

    const g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    g.append("g")
      .attr("transform", "translate(0," + innerH + ")")
      .call(d3.axisBottom(x).ticks(6))
      .call(g => g.selectAll("text").attr("font-size", 11));

    g.append("g")
      .call(d3.axisLeft(y).ticks(6))
      .call(g => g.selectAll("text").attr("font-size", 11));

    const dots = g.append("g")
      .selectAll("circle")
      .data(points)
      .enter()
      .append("circle")
      .attr("class", "dot")
      .attr("cx", d => x(d.x))
      .attr("cy", d => y(d.y))
      .attr("r", 2.5)
      .attr("fill", d => d.color)
      .attr("fill-opacity", 0.75);

    function onMove(event, d) {{
      const rect = document.getElementById("wrap").getBoundingClientRect();
      tooltip
        .style("display", "block")
        .style("left", (event.clientX - rect.left + 12) + "px")
        .style("top", (event.clientY - rect.top + 12) + "px")
        .html("pid: <b>" + d.pid + "</b><br/>aid: <b>" + d.aid + "</b>");
    }}

    dots
      .on("mousemove", onMove)
      .on("mouseenter", onMove)
      .on("mouseleave", () => tooltip.style("display", "none"));
  </script>
</body>
</html>
"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

def _pairwise_distances(x):
    x = np.asarray(x, dtype=np.float64)
    diffs = x[:, None, :] - x[None, :, :]
    d = np.sqrt(np.sum(diffs * diffs, axis=-1))
    return d

def analyze_action_clusters(Z, pid, aid, eps, min_action_samples, min_actions_per_cluster, min_samples):
    Z = np.asarray(Z)
    pid = np.asarray(pid)
    aid = np.asarray(aid)

    results = {}
    for p in np.unique(pid):
        mask_p = pid == p
        aids_p = aid[mask_p]
        Z_p = Z[mask_p]

        centroids = []
        action_ids = []
        action_counts = {}
        for a in np.unique(aids_p):
            mask_a = aids_p == a
            cnt = int(mask_a.sum())
            action_counts[int(a)] = cnt
            if cnt < min_action_samples:
                continue
            action_ids.append(int(a))
            centroids.append(Z_p[mask_a].mean(axis=0))

        if len(action_ids) < 2:
            continue

        centroids = np.asarray(centroids, dtype=np.float64)
        used_eps = eps
        if (used_eps is None) or (used_eps <= 0):
            d = _pairwise_distances(centroids)
            tri = d[np.triu_indices(d.shape[0], k=1)]
            if tri.size == 0:
                used_eps = 0.0
            else:
                med = float(np.median(tri))
                used_eps = med * 0.5
                if used_eps <= 0:
                    used_eps = float(np.mean(tri)) * 0.5

        if used_eps <= 0:
            continue

        db = DBSCAN(eps=used_eps, min_samples=min_samples)
        labels = db.fit_predict(centroids)

        clusters = {}
        for idx, lab in enumerate(labels):
            if lab == -1:
                continue
            clusters.setdefault(int(lab), []).append(int(action_ids[idx]))

        kept = {}
        for lab, acts in clusters.items():
            if len(acts) < min_actions_per_cluster:
                continue
            idxs = [action_ids.index(a) for a in acts]
            c = centroids[idxs]
            d = _pairwise_distances(c)
            tri = d[np.triu_indices(d.shape[0], k=1)]
            kept[int(lab)] = {
                "actions": sorted(acts),
                "action_counts": {int(a): action_counts[int(a)] for a in acts},
                "centroid_dist_mean": float(tri.mean()) if tri.size else 0.0,
                "centroid_dist_max": float(tri.max()) if tri.size else 0.0,
            }

        if kept:
            results[int(p)] = {
                "eps": float(used_eps),
                "min_action_samples": int(min_action_samples),
                "min_actions_per_cluster": int(min_actions_per_cluster),
                "clusters": kept,
            }

    return results

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
    parser.add_argument('--out-html', type=str, default='')
    parser.add_argument('--motion-only', action='store_true')
    parser.add_argument('--font-size', type=float, default=3.0)
    parser.add_argument('--no-cluster-stats', action='store_true')
    parser.add_argument('--cluster-eps', type=float, default=0.0)
    parser.add_argument('--cluster-min-action-samples', type=int, default=5)
    parser.add_argument('--cluster-min-actions', type=int, default=2)
    parser.add_argument('--cluster-min-samples', type=int, default=2)
    parser.add_argument('--cluster-out', type=str, default='')
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
    X, Y, PID, AID = collect_features(model, loader, device, args.max_samples)
    if PID is None:
        PID = Y
    if AID is None:
        AID = np.full_like(PID, -1)
    X, PID, AID = subsample_per_id(X, PID, AID, args.samples_per_id, args.seed)
    ids, counts = np.unique(PID, return_counts=True)
    for pid, cnt in zip(ids, counts):
        print(f'person {int(pid)}: {int(cnt)} samples')

    n = X.shape[0]
    max_p = max(5, (n - 1) // 3) if n > 3 else 5
    perplexity = min(args.perplexity, max_p)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=args.seed, init='random')
    Z = tsne.fit_transform(X)

    if not args.no_cluster_stats:
        stats = analyze_action_clusters(
            Z,
            PID,
            AID,
            eps=args.cluster_eps,
            min_action_samples=args.cluster_min_action_samples,
            min_actions_per_cluster=args.cluster_min_actions,
            min_samples=args.cluster_min_samples,
        )
        for p in sorted(stats.keys()):
            info = stats[p]
            print(f'person {p}: eps={info["eps"]:.4f} clusters={len(info["clusters"])}')
            for lab in sorted(info["clusters"].keys()):
                c = info["clusters"][lab]
                acts = ",".join(str(a) for a in c["actions"])
                print(
                    f'  cluster {lab}: actions=[{acts}] '
                    f'mean_d={c["centroid_dist_mean"]:.4f} max_d={c["centroid_dist_max"]:.4f} '
                    f'counts={c["action_counts"]}'
                )
        if args.cluster_out:
            with open(args.cluster_out, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

    if args.out_html:
        save_tsne_html(Z, PID, AID, args.out_html)
    else:
        plt.figure(figsize=(8, 6))
        ids = np.unique(PID)
        cmap = plt.get_cmap('tab20', len(ids))
        for i, pid in enumerate(ids):
            idx = np.where(PID == pid)[0]
            plt.scatter(Z[idx, 0], Z[idx, 1], s=6, alpha=0.7, color=cmap(i))
        for j in range(Z.shape[0]):
            plt.text(Z[j, 0], Z[j, 1], f'{int(PID[j])},{int(AID[j])}', fontsize=args.font_size, alpha=0.8)
        plt.tight_layout()
        plt.savefig(args.out, dpi=200)

if __name__ == '__main__':
    main()
