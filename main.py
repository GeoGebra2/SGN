# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import time
import shutil
import os
import os.path as osp
import csv
import numpy as np
import h5py

np.random.seed(1337)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from model import SGN
from data import NTUDataLoaders, AverageMeter
import fit
from util import make_dir, get_num_classes

parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')
fit.add_fit_args(parser)
parser.set_defaults(
    network='SGN',
    dataset = 'NTU',
    case = 0,
    batch_size=64,
    max_epochs=120,
    monitor='val_acc',
    lr=0.001,
    weight_decay=0.0001,
    lr_factor=0.1,
    workers=16,
    print_freq = 20,
    train = 0,
    seg = 20,
    )
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GROUPS = [
    [3, 20],
    [0, 1, 2, 4, 8, 12, 16],
    [8, 9, 10, 11, 23, 24],
    [4, 5, 6, 7, 21, 22],
    [16, 17, 18, 19],
    [12, 13, 14, 15],
]

def _case_metric(case_id):
    if case_id == 0:
        return 'CS'
    if case_id == 1:
        return 'CV'
    if case_id == 2:
        return 'CR'
    return 'CA'

def _guess_id_h5(case_id):
    metric = _case_metric(case_id)
    return osp.join('./data/ntu', 'NTU_ID_' + metric + '.h5')

def _guess_prim_h5(case_id):
    metric = _case_metric(case_id)
    return osp.join('./data/ntu', 'NTU_PRIM_' + metric + '.h5')

def _guess_prim_ckpt(case_id):
    return osp.join('./results/NTU_PRIM/SGN', '%s_best.pth' % case_id)

def _to_label(y):
    if y.ndim == 2:
        return np.argmax(y, axis=-1).astype(np.int64)
    return y.astype(np.int64)

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

def _segment_primitives_with_meta(seq150, out_len, min_len, max_segments):
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
    metas = []
    for score, gid, s, e in selected:
        clip = seq[s:e + 1]
        clip = _resample(clip, out_len)
        flat = clip.reshape(out_len, 75).astype(np.float32)
        prims.append(flat)
        metas.append({
            'gid': gid,
            'score': float(score / max(1.0, float(t))),
            'length': float((e - s + 1) / max(1.0, float(t))),
        })
    return prims, metas

def _build_cluster_pid_map(prim_label_h5, num_clusters, num_pid):
    with h5py.File(prim_label_h5, 'r') as f:
        prim_y = _to_label(f['y'][:])
        src_pid = f['src_action'][:].astype(np.int64)
    src_pid = np.clip(src_pid, 0, num_pid - 1)
    counts = np.ones((num_clusters, num_pid), dtype=np.float64) * 1e-6
    for c, p in zip(prim_y, src_pid):
        if 0 <= c < num_clusters:
            counts[c, p] += 1.0
    counts = counts / counts.sum(axis=1, keepdims=True)
    return counts.astype(np.float32)

def _weight_value(meta, conf, mode):
    if mode == 'uniform':
        return 1.0
    if mode == 'length':
        return max(meta['length'], 1e-6)
    if mode == 'conf':
        return max(conf, 1e-6)
    return max(meta['length'], 1e-6) * max(meta['score'], 1e-6) * max(conf, 1e-6)

def run_primitive_fusion_id_test():
    source_h5 = args.prim_source_h5 if len(args.prim_source_h5) > 0 else _guess_id_h5(args.case)
    prim_label_h5 = args.prim_label_h5 if len(args.prim_label_h5) > 0 else _guess_prim_h5(args.case)
    checkpoint = args.prim_checkpoint if len(args.prim_checkpoint) > 0 else _guess_prim_ckpt(args.case)
    with h5py.File(source_h5, 'r') as f:
        x_test = f['test_x'][:]
        y_test = f['test_y'][:]
    if y_test.ndim == 2:
        y_test_label = np.argmax(y_test, axis=-1).astype(np.int64)
        num_pid = y_test.shape[1]
    else:
        y_test_label = y_test.astype(np.int64)
        num_pid = int(y_test_label.max()) + 1
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt['state_dict']
    num_model_classes = int(state_dict['fc.weight'].shape[0])
    prim_model = SGN(num_model_classes, 'NTU_PRIM', args.seg, args).to(device)
    prim_model.load_state_dict(state_dict)
    prim_model.eval()
    cluster_pid_map = None
    if num_model_classes != num_pid:
        cluster_pid_map = _build_cluster_pid_map(prim_label_h5, num_model_classes, num_pid)
    top1 = 0
    top5 = 0
    t_start = time.time()
    for i in range(x_test.shape[0]):
        prims, metas = _segment_primitives_with_meta(
            x_test[i],
            out_len=args.seg,
            min_len=args.prim_min_len,
            max_segments=args.prim_max_segments,
        )
        if len(prims) == 0:
            prims = [np.zeros((args.seg, 75), dtype=np.float32)]
            metas = [{'gid': 1, 'score': 1.0, 'length': 1.0}]
        batch = np.zeros((len(prims), args.seg, 75), dtype=np.float32)
        for j, p in enumerate(prims):
            batch[j] = p
        with torch.no_grad():
            logits = prim_model(torch.from_numpy(batch).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        confs = probs.max(axis=1)
        if cluster_pid_map is not None:
            probs_pid = probs @ cluster_pid_map
        else:
            probs_pid = probs
        weights = np.array([
            _weight_value(metas[k], float(confs[k]), args.prim_weight_mode)
            for k in range(len(metas))
        ], dtype=np.float32)
        wsum = float(weights.sum())
        if wsum <= 0:
            weights = np.ones_like(weights)
            wsum = float(weights.sum())
        fused = (weights[:, None] * probs_pid).sum(axis=0) / wsum
        pred = int(np.argmax(fused))
        gt = int(y_test_label[i])
        if pred == gt:
            top1 += 1
        topk = min(5, fused.shape[0])
        top_ids = np.argsort(-fused)[:topk]
        if gt in top_ids:
            top5 += 1
        if (i + 1) % 200 == 0 or (i + 1) == x_test.shape[0]:
            print('Fusion test progress {}/{}'.format(i + 1, x_test.shape[0]))
    top1_acc = 100.0 * top1 / max(1, x_test.shape[0])
    top5_acc = 100.0 * top5 / max(1, x_test.shape[0])
    print('Fusion ID Test: Top-1 accu {:.3f}, Top-5 accu {:.3f}, time: {:.2f}s'.format(
        top1_acc, top5_acc, time.time() - t_start
    ))

def main():
    if args.fuse_prim_id:
        run_primitive_fusion_id_test()
        return
    ntu_loaders = NTUDataLoaders(args.dataset, args.case, seg=args.seg, args=args)
    args.num_classes = getattr(ntu_loaders, 'num_classes', None) or get_num_classes(args.dataset)
    model = SGN(args.num_classes, args.dataset, args.seg, args)

    total = get_n_params(model)
    print(model)
    print('The number of parameters: ', total)
    print('The modes is:', args.network)

    if torch.cuda.is_available():
        print('It is using GPU!')
    model = model.to(device)
    criterion = LabelSmoothingLoss(args.num_classes, smoothing=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.monitor == 'val_acc':
        mode = 'max'
        monitor_op = np.greater
        best = -np.inf
        str_op = 'improve'
    elif args.monitor == 'val_loss':
        mode = 'min'
        monitor_op = np.less
        best = np.inf
        str_op = 'reduce'

    scheduler = MultiStepLR(optimizer, milestones=[60, 90, 110], gamma=0.1)
    # Data loading
    train_loader = ntu_loaders.get_train_loader(args.batch_size, args.workers)
    val_loader = ntu_loaders.get_val_loader(args.batch_size, args.workers)
    train_size = ntu_loaders.get_train_size()
    val_size = ntu_loaders.get_val_size()


    test_loader = ntu_loaders.get_test_loader(32, args.workers)

    print('Train on %d samples, validate on %d samples' % (train_size, val_size))

    best_epoch = 0
    output_dir = make_dir(args.dataset)

    save_path = os.path.join(output_dir, args.network)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    checkpoint = osp.join(save_path, '%s_best.pth' % args.case)
    earlystop_cnt = 0
    csv_file = osp.join(save_path, '%s_log.csv' % args.case)
    log_res = list()

    lable_path = osp.join(save_path, '%s_lable.txt'% args.case)
    pred_path = osp.join(save_path, '%s_pred.txt' % args.case)

    # Training
    if args.train ==1:
        for epoch in range(args.start_epoch, args.max_epochs):

            print(epoch, optimizer.param_groups[0]['lr'])

            t_start = time.time()
            train_loss, train_acc1, train_acc2, train_acc3, train_acc4, train_acc5 = train(train_loader, model, criterion, optimizer, epoch)
            val_loss, val_acc1, val_acc2, val_acc3, val_acc4, val_acc5 = validate(val_loader, model, criterion)
            log_res += [[train_loss, float(train_acc1), float(train_acc2), float(train_acc3), float(train_acc4), float(train_acc5),\
                         val_loss, float(val_acc1), float(val_acc2), float(val_acc3), float(val_acc4), float(val_acc5)]]

            print('Epoch-{:<3d} {:.1f}s\t'
                  'Train: loss {:.4f}\tTop-1 accu {:.4f}\tTop-2 accu {:.4f}\tTop-3 accu {:.4f}\tTop-4 accu {:.4f}\tTop-5 accu {:.4f}\t'
                  'Valid: loss {:.4f}\tTop-1 accu {:.4f}\tTop-2 accu {:.4f}\tTop-3 accu {:.4f}\tTop-4 accu {:.4f}\tTop-5 accu {:.4f}'
                  .format(epoch + 1, time.time() - t_start, train_loss, train_acc1, train_acc2, train_acc3, train_acc4, train_acc5,
                          val_loss, val_acc1, val_acc2, val_acc3, val_acc4, val_acc5))

            current = val_loss if mode == 'min' else val_acc1

            ####### store tensor in cpu
            current = float(current)

            if monitor_op(current, best):
                print('Epoch %d: %s %sd from %.4f to %.4f, '
                      'saving model to %s'
                      % (epoch + 1, args.monitor, str_op, best, current, checkpoint))
                best = current
                best_epoch = epoch + 1
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best': best,
                    'monitor': args.monitor,
                    'optimizer': optimizer.state_dict(),
                }, checkpoint)
                earlystop_cnt = 0
            else:
                print('Epoch %d: %s did not %s' % (epoch + 1, args.monitor, str_op))
                earlystop_cnt += 1

            scheduler.step()

        print('Best %s: %.4f from epoch-%d' % (args.monitor, best, best_epoch))
        with open(csv_file, 'w') as fw:
            cw = csv.writer(fw)
            cw.writerow(['loss', 'acc@1', 'acc@2', 'acc@3', 'acc@4', 'acc@5', 'val_loss', 'val_acc@1', 'val_acc@2', 'val_acc@3', 'val_acc@4', 'val_acc@5'])
            cw.writerows(log_res)
        print('Save train and validation log into into %s' % csv_file)

    ### Test
    args.train = 0
    model = SGN(args.num_classes, args.dataset, args.seg, args)
    model = model.to(device)
    test(test_loader, model, checkpoint, lable_path, pred_path)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    acc1 = AverageMeter()
    acc2 = AverageMeter()
    acc3 = AverageMeter()
    acc4 = AverageMeter()
    acc5 = AverageMeter()
    model.train()

    for i, (inputs, target) in enumerate(train_loader):

        inputs = inputs.to(device)
        feats = model.forward_features(inputs)
        output = model.fc(feats)
        target = target.to(device, non_blocking=True)
        loss = criterion(output, target)
        if args.metric_loss != 'none':
            metric = metric_loss(feats, target, args)
            loss = loss + args.metric_weight * metric

        # measure accuracy and record loss
        acc = accuracy(output.data, target, topk=(1, 2, 3, 4, 5))
        losses.update(loss.item(), inputs.size(0))
        acc1.update(acc[0], inputs.size(0))
        acc2.update(acc[1], inputs.size(0))
        acc3.update(acc[2], inputs.size(0))
        acc4.update(acc[3], inputs.size(0))
        acc5.update(acc[4], inputs.size(0))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            print('Epoch-{:<3d} {:3d} batches\t'
                  'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-1 accu {acc1.val:.3f} ({acc1.avg:.3f})\t'
                  'Top-2 accu {acc2.val:.3f} ({acc2.avg:.3f})\t'
                  'Top-3 accu {acc3.val:.3f} ({acc3.avg:.3f})\t'
                  'Top-4 accu {acc4.val:.3f} ({acc4.avg:.3f})\t'
                  'Top-5 accu {acc5.val:.3f} ({acc5.avg:.3f})'.format(
                   epoch + 1, i + 1, loss=losses, acc1=acc1, acc2=acc2, acc3=acc3, acc4=acc4, acc5=acc5))

    return losses.avg, acc1.avg, acc2.avg, acc3.avg, acc4.avg, acc5.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    acc1 = AverageMeter()
    acc2 = AverageMeter()
    acc3 = AverageMeter()
    acc4 = AverageMeter()
    acc5 = AverageMeter()
    model.eval()

    for i, (inputs, target) in enumerate(val_loader):
        with torch.no_grad():
            output = model(inputs.to(device))
        target = target.to(device, non_blocking=True)
        with torch.no_grad():
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc = accuracy(output.data, target, topk=(1, 2, 3, 4, 5))
        losses.update(loss.item(), inputs.size(0))
        acc1.update(acc[0], inputs.size(0))
        acc2.update(acc[1], inputs.size(0))
        acc3.update(acc[2], inputs.size(0))
        acc4.update(acc[3], inputs.size(0))
        acc5.update(acc[4], inputs.size(0))

    return losses.avg, acc1.avg, acc2.avg, acc3.avg, acc4.avg, acc5.avg


def test(test_loader, model, checkpoint, lable_path, pred_path):
    acc1 = AverageMeter()
    acc2 = AverageMeter()
    acc3 = AverageMeter()
    acc4 = AverageMeter()
    acc5 = AverageMeter()
    # load learnt model that obtained best performance on validation set
    model.load_state_dict(torch.load(checkpoint, map_location=device)['state_dict'])
    model.eval()

    label_output = list()
    pred_output = list()

    t_start = time.time()
    for i, (inputs, target) in enumerate(test_loader):
        with torch.no_grad():
            output = model(inputs.to(device))
            output = output.view((-1, inputs.size(0)//target.size(0), output.size(1)))
            output = output.mean(1)

        label_output.append(target.cpu().numpy())
        pred_output.append(output.cpu().numpy())

        acc = accuracy(output.data, target.to(device, non_blocking=True), topk=(1, 2, 3, 4, 5))
        acc1.update(acc[0], inputs.size(0))
        acc2.update(acc[1], inputs.size(0))
        acc3.update(acc[2], inputs.size(0))
        acc4.update(acc[3], inputs.size(0))
        acc5.update(acc[4], inputs.size(0))


    label_output = np.concatenate(label_output, axis=0)
    np.savetxt(lable_path, label_output, fmt='%d')
    pred_output = np.concatenate(pred_output, axis=0)
    np.savetxt(pred_path, pred_output, fmt='%f')

    print('Test: Top-1 accu {:.3f}, Top-2 accu {:.3f}, Top-3 accu {:.3f}, Top-4 accu {:.3f}, Top-5 accu {:.3f}, time: {:.2f}s'
          .format(acc1.avg, acc2.avg, acc3.avg, acc4.avg, acc5.avg, time.time() - t_start))


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res

def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def metric_loss(feats, labels, args):
    if args.metric_loss == 'supcon':
        return supcon_loss(feats, labels, args.supcon_temp)
    if args.metric_loss == 'triplet':
        return batch_hard_triplet(feats, labels, args.triplet_margin)
    return feats.sum() * 0.0

def supcon_loss(feats, labels, temperature):
    feats = F.normalize(feats, dim=1)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    logits = torch.matmul(feats, feats.T) / temperature
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
    loss = -mean_log_prob_pos
    loss = loss[torch.isfinite(loss)]
    if loss.numel() == 0:
        return feats.sum() * 0.0
    return loss.mean()

def batch_hard_triplet(feats, labels, margin):
    feats = F.normalize(feats, dim=1)
    dist = 1.0 - torch.matmul(feats, feats.T)
    labels = labels.view(-1, 1)
    mask_pos = torch.eq(labels, labels.T)
    mask_neg = ~mask_pos
    mask_pos = mask_pos.float()
    mask_neg = mask_neg.float()
    dist_ap = dist * mask_pos
    dist_an = dist + (1.0 - mask_neg) * 1e6
    hardest_pos, _ = dist_ap.max(dim=1)
    hardest_neg, _ = dist_an.min(dim=1)
    valid = mask_pos.sum(dim=1) > 1
    if valid.sum() == 0:
        return feats.sum() * 0.0
    loss = F.relu(hardest_pos[valid] - hardest_neg[valid] + margin)
    return loss.mean()

if __name__ == '__main__':
    main()
    
