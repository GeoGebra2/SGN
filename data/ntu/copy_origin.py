import os
import os.path as osp
import argparse
import shutil
def list_basenames(root):
    s = set()
    for dp, _, files in os.walk(root):
        for f in files:
            s.add(osp.splitext(f)[0])
    return sorted(s)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_dir', type=str, default='./ntu-batch')
    parser.add_argument('--src_dir', type=str, default='./nturgb+d_skeletons')
    parser.add_argument('--out_dir', type=str, default='./ntu-batch-origin')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    names = list_basenames(args.batch_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    copied = 0
    missing = 0
    skipped = 0
    for n in names:
        src = osp.join(args.src_dir, n + '.skeleton')
        if not osp.exists(src):
            missing += 1
            continue
        dst = osp.join(args.out_dir, osp.basename(src))
        if osp.exists(dst) and not args.overwrite:
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    print(f'copied={copied} skipped={skipped} missing={missing}')
if __name__ == '__main__':
    main()
