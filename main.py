# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import time
import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import os.path as osp
import csv
import numpy as np

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
    monitor='val_auc',
    lr=0.001,
    weight_decay=0.0001,
    lr_factor=0.1,
    workers=16,
    print_freq = 20,
    train = 0,
    seg = 20,
    )
args = parser.parse_args()

def main():

    if args.dataset == 'NTU_ID' and args.metric_loss == 'none':
        args.metric_loss = 'supcon'
        args.metric_weight = 1.0

    args.num_classes = get_num_classes(args.dataset)
    model = SGN(args.num_classes, args.dataset, args.seg, args)

    total = get_n_params(model)
    print(model)
    print('The number of parameters: ', total)
    print('The modes is:', args.network)

    if torch.cuda.is_available():
        print('It is using GPU!')
        model = model.cuda()

    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.monitor == 'val_auc':
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
    ntu_loaders = NTUDataLoaders(args.dataset, args.case, seg=args.seg)
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
            train_loss, train_acc, train_auc = train(train_loader, model, criterion, optimizer, epoch)
            val_loss, val_acc, val_auc = validate(val_loader, model, criterion)
            log_res += [[train_loss, float(train_acc), float(train_auc), val_loss, float(val_acc), float(val_auc)]]

            print('Epoch-{:<3d} {:.1f}s\t'
                  'Train: loss {:.4f}\tPair-Acc {:.4f}\tPair-AUC {:.4f}\t'
                  'Valid: loss {:.4f}\tPair-Acc {:.4f}\tPair-AUC {:.4f}'
                  .format(epoch + 1, time.time() - t_start, train_loss, train_acc, train_auc,
                          val_loss, val_acc, val_auc))

            current = val_loss if mode == 'min' else val_auc

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
            cw.writerow(['loss', 'pair_acc', 'pair_auc', 'val_loss', 'val_pair_acc', 'val_pair_auc'])
            cw.writerows(log_res)
        print('Save train and validation log into into %s' % csv_file)

    ### Test
    args.train = 0
    model = SGN(args.num_classes, args.dataset, args.seg, args)
    model = model.cuda()
    test(test_loader, model, checkpoint, lable_path, pred_path)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    acc = AverageMeter()
    auc = AverageMeter()
    model.train()

    for i, (inputs, target) in enumerate(train_loader):

        inputs = inputs.cuda()
        feats = model(inputs)
        target = target.cuda(non_blocking=True)
        pair_logits, pair_target = build_pair_logits(feats, target, model.logit_scale, max_pairs=4096)
        if pair_logits is None:
            continue
        loss = criterion(pair_logits, pair_target)
        if args.metric_loss != 'none':
            metric = metric_loss(feats, target, args)
            loss = loss + args.metric_weight * metric

        batch_acc, batch_auc = binary_scores(pair_logits.detach(), pair_target.detach())
        losses.update(loss.item(), inputs.size(0))
        acc.update(batch_acc, pair_target.size(0))
        if np.isfinite(batch_auc):
            auc.update(batch_auc, pair_target.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            print('Epoch-{:<3d} {:3d} batches\t'
                  'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Pair-Acc {acc.val:.3f} ({acc.avg:.3f})\t'
                  'Pair-AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                   epoch + 1, i + 1, loss=losses, acc=acc, auc=auc))

    return losses.avg, acc.avg, auc.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    acc = AverageMeter()
    auc = AverageMeter()
    model.eval()

    for i, (inputs, target) in enumerate(val_loader):
        with torch.no_grad():
            feats = model(inputs.cuda())
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            pair_logits, pair_target = build_pair_logits(feats, target, model.logit_scale, max_pairs=4096)
            if pair_logits is None:
                continue
            loss = criterion(pair_logits, pair_target)

        batch_acc, batch_auc = binary_scores(pair_logits.detach(), pair_target.detach())
        losses.update(loss.item(), inputs.size(0))
        acc.update(batch_acc, pair_target.size(0))
        if np.isfinite(batch_auc):
            auc.update(batch_auc, pair_target.size(0))

    return losses.avg, acc.avg, auc.avg


def test(test_loader, model, checkpoint, lable_path, pred_path):
    acc = AverageMeter()
    auc = AverageMeter()
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    model.eval()

    label_output = []
    pred_output = []
    emb_output = []

    t_start = time.time()
    for i, (inputs, target) in enumerate(test_loader):
        with torch.no_grad():
            feats = model(inputs.cuda())
            feats = feats.view((-1, inputs.size(0)//target.size(0), feats.size(1)))
            feats = feats.mean(1)

        emb_output.append(feats.cpu())
        label_output.append(target.cpu())

    embeddings = torch.cat(emb_output, dim=0).cuda()
    labels = torch.cat(label_output, dim=0).cuda()
    pair_logits, pair_target = build_pair_logits(embeddings, labels, model.logit_scale, max_pairs=200000)
    if pair_logits is not None:
        batch_acc, batch_auc = binary_scores(pair_logits.detach(), pair_target.detach())
        acc.update(batch_acc, pair_target.size(0))
        if np.isfinite(batch_auc):
            auc.update(batch_auc, pair_target.size(0))
        pred_output = torch.sigmoid(pair_logits).detach().cpu().numpy()
        label_output = pair_target.detach().cpu().numpy()
    else:
        pred_output = np.array([])
        label_output = np.array([])

    np.savetxt(lable_path, label_output, fmt='%d')
    np.savetxt(pred_path, pred_output, fmt='%f')

    print('Test: Pair-Acc {:.3f}, Pair-AUC {:.3f}, time: {:.2f}s'
          .format(acc.avg, auc.avg, time.time() - t_start))


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


def build_pair_logits(feats, labels, logit_scale, max_pairs=4096):
    n = labels.size(0)
    if n < 2:
        return None, None
    idx = torch.triu_indices(n, n, offset=1, device=labels.device)
    if idx.size(1) == 0:
        return None, None
    pair_targets_all = (labels[idx[0]] == labels[idx[1]]).float()
    pos_idx = torch.where(pair_targets_all > 0.5)[0]
    neg_idx = torch.where(pair_targets_all < 0.5)[0]
    if pos_idx.numel() == 0 and neg_idx.numel() == 0:
        return None, None
    if pos_idx.numel() > 0 and neg_idx.numel() > 0:
        n_each = min(pos_idx.numel(), neg_idx.numel(), max_pairs // 2)
        pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=labels.device)[:n_each]]
        neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=labels.device)[:n_each]]
        select_idx = torch.cat([pos_idx, neg_idx], dim=0)
    else:
        select_pool = pos_idx if pos_idx.numel() > 0 else neg_idx
        keep = min(select_pool.numel(), max_pairs)
        select_idx = select_pool[torch.randperm(select_pool.numel(), device=labels.device)[:keep]]
    i = idx[0][select_idx]
    j = idx[1][select_idx]
    pair_targets = pair_targets_all[select_idx]
    sim = (feats[i] * feats[j]).sum(dim=1)
    scale = torch.clamp(logit_scale, min=1.0, max=50.0)
    pair_logits = sim * scale
    return pair_logits, pair_targets


def binary_scores(pair_logits, pair_targets):
    prob = torch.sigmoid(pair_logits)
    pred = (prob >= 0.5).float()
    acc = (pred == pair_targets).float().mean().item() * 100.0
    auc = compute_auc(pair_targets.detach().cpu().numpy(), prob.detach().cpu().numpy()) * 100.0
    return acc, auc


def compute_auc(labels, scores):
    labels = labels.astype(np.int32)
    pos = labels.sum()
    neg = labels.shape[0] - pos
    if pos == 0 or neg == 0:
        return float('nan')
    order = np.argsort(-scores, kind='mergesort')
    labels_sorted = labels[order]
    tps = np.cumsum(labels_sorted == 1)
    fps = np.cumsum(labels_sorted == 0)
    tpr = np.concatenate([[0.0], tps / float(pos)])
    fpr = np.concatenate([[0.0], fps / float(neg)])
    return np.trapz(tpr, fpr)

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
    
