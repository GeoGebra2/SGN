import os
import os.path as osp
import argparse
import shutil
import re
def list_basenames(root):
    s = set()
    for dp, _, files in os.walk(root):
        for f in files:
            s.add(osp.splitext(f)[0])
    return sorted(s)
def parse_name(name):
    m = re.match(r'^S(\d+)C(\d+)P(\d+)R(\d+)A(\d+)$', name)
    if not m:
        return None
    s, c, p, r, a = map(int, m.groups())
    return s, c, p, a
def index_sources(src_dir):
    idx = {}
    for dp, _, files in os.walk(src_dir):
        for f in files:
            if not f.lower().endswith('.skeleton'):
                continue
            n = osp.splitext(f)[0]
            meta = parse_name(n)
            if meta is None:
                continue
            key = meta
            idx.setdefault(key, []).append(osp.join(dp, f))
    for k in idx:
        idx[k].sort()
    return idx
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_dir', type=str, default='./ntu-batch')
    parser.add_argument('--src_dir', type=str, default='./nturgb+d_skeletons')
    parser.add_argument('--out_dir', type=str, default='./ntu-batch-origin')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    names = list_basenames(args.batch_dir)
    src_index = index_sources(args.src_dir)
    used_srcs = set()
    os.makedirs(args.out_dir, exist_ok=True)
    copied = 0
    missing = 0
    skipped = 0
    fallback = 0
    for n in names:
        dst = osp.join(args.out_dir, n + '.skeleton')
        if osp.exists(dst) and not args.overwrite:
            skipped += 1
            continue
        src = osp.join(args.src_dir, n + '.skeleton')
        if osp.exists(src) and src not in used_srcs:
            shutil.copy2(src, dst)
            used_srcs.add(src)
            copied += 1
            continue
        meta = parse_name(n)
        if meta is None:
            missing += 1
            continue
        candidates = src_index.get(meta, [])
        chosen = None
        for cand in candidates:
            if cand not in used_srcs:
                chosen = cand
                break
        if chosen is None:
            missing += 1
            continue
        shutil.copy2(chosen, dst)
        used_srcs.add(chosen)
        copied += 1
        fallback += 1
    print(f'copied={copied} skipped={skipped} missing={missing} fallback={fallback}')
if __name__ == '__main__':
    main()
