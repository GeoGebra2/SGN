import os
import os.path as osp
import re
import argparse
import numpy as np
def parse_name(name):
    m = re.match(r'^S(\d+)C(\d+)P(\d+)R(\d+)A(\d+)$', name)
    if not m:
        return None
    s, c, p, r, a = map(int, m.groups())
    return s, c, p, r, a
def list_skeletons(root):
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith('.skeleton'):
                out.append(osp.join(dirpath, f))
    out.sort()
    return out
def has_valid_frame(lines):
    try:
        num_frames = int(lines[0].strip())
    except Exception:
        return False
    cur = 1
    valid = False
    for _ in range(num_frames):
        if cur >= len(lines):
            break
        try:
            nb = int(lines[cur].strip())
        except Exception:
            break
        cur += 1
        if nb > 0:
            valid = True
        for _ in range(nb):
            if cur >= len(lines):
                break
            cur += 1
            if cur >= len(lines):
                break
            try:
                nj = int(lines[cur].strip())
            except Exception:
                nj = 0
            cur += 1
            cur += nj
    return valid
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skes_dir', type=str, default='./ntu-batch')
    parser.add_argument('--out_dir', type=str, default='./statistics')
    args = parser.parse_args()
    files = list_skeletons(args.skes_dir)
    names = []
    setups = []
    cameras = []
    performers = []
    replications = []
    labels = []
    missing = []
    for fp in files:
        bn = osp.basename(fp)
        name = bn[:-9]
        meta = parse_name(name)
        if meta is None:
            continue
        try:
            with open(fp, 'r') as fr:
                lines = fr.readlines()
        except Exception:
            missing.append(name)
            continue
        if not has_valid_frame(lines):
            missing.append(name)
            continue
        s, c, p, r, a = meta
        names.append(name)
        setups.append(s)
        cameras.append(c)
        performers.append(p)
        replications.append(r)
        labels.append(a)
    os.makedirs(args.out_dir, exist_ok=True)
    np.savetxt(osp.join(args.out_dir, 'skes_available_name.txt'), np.array(names, dtype=str), fmt='%s')
    np.savetxt(osp.join(args.out_dir, 'setup.txt'), np.array(setups, dtype=int), fmt='%d')
    np.savetxt(osp.join(args.out_dir, 'camera.txt'), np.array(cameras, dtype=int), fmt='%d')
    np.savetxt(osp.join(args.out_dir, 'performer.txt'), np.array(performers, dtype=int), fmt='%d')
    np.savetxt(osp.join(args.out_dir, 'replication.txt'), np.array(replications, dtype=int), fmt='%d')
    np.savetxt(osp.join(args.out_dir, 'label.txt'), np.array(labels, dtype=int), fmt='%d')
    np.savetxt(osp.join(args.out_dir, 'samples_with_missing_skeletons.txt'), np.array(missing, dtype=str), fmt='%s')
if __name__ == '__main__':
    main()
