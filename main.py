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

def main():

    args.num_classes = get_num_classes(args.dataset)
    model = SGN(args.num_classes, args.dataset, args.seg, args)

    total = get_n_params(model)
    print(model)
    print('The number of parameters: ', total)
    print('The modes is:', args.network)

    if torch.cuda.is_available():
        print('It is using GPU!')
        model = model.cuda()

    criterion = LabelSmoothingLoss(args.num_classes, smoothing=0.1).cuda()
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
            train_loss, train_acc1, train_acc2, train_acc3 = train(train_loader, model, criterion, optimizer, epoch)
            val_loss, val_acc1, val_acc2, val_acc3 = validate(val_loader, model, criterion)
            log_res += [[train_loss, float(train_acc1), float(train_acc2), float(train_acc3),\
                         val_loss, float(val_acc1), float(val_acc2), float(val_acc3)]]

            print('Epoch-{:<3d} {:.1f}s\t'
                  'Train: loss {:.4f}\taccu@1 {:.4f}\taccu@2 {:.4f}\taccu@3 {:.4f}\t'
                  'Valid: loss {:.4f}\taccu@1 {:.4f}\taccu@2 {:.4f}\taccu@3 {:.4f}'
                  .format(epoch + 1, time.time() - t_start, train_loss, train_acc1, train_acc2, train_acc3,
                          val_loss, val_acc1, val_acc2, val_acc3))

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
            cw.writerow(['loss', 'acc@1', 'acc@2', 'acc@3', 'val_loss', 'val_acc@1', 'val_acc@2', 'val_acc@3'])
            cw.writerows(log_res)
        print('Save train and validation log into into %s' % csv_file)

    ### Test
    args.train = 0
    model = SGN(args.num_classes, args.dataset, args.seg, args)
    model = model.cuda()
    test(test_loader, model, checkpoint, lable_path, pred_path)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    acc1 = AverageMeter()
    acc2 = AverageMeter()
    acc3 = AverageMeter()
    model.train()

    for i, (inputs, target) in enumerate(train_loader):

        inputs = inputs.cuda()
        feats = model.forward_features(inputs)
        output = model.fc(feats)
        target = target.cuda(non_blocking=True)
        loss = criterion(output, target)
        if args.metric_loss != 'none':
            metric = metric_loss(feats, target, args)
            loss = loss + args.metric_weight * metric

        # measure accuracy and record loss
        acc = accuracy(output.data, target, topk=(1, 2, 3))
        losses.update(loss.item(), inputs.size(0))
        acc1.update(acc[0], inputs.size(0))
        acc2.update(acc[1], inputs.size(0))
        acc3.update(acc[2], inputs.size(0))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            print('Epoch-{:<3d} {:3d} batches\t'
                  'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'accu@1 {acc1.val:.3f} ({acc1.avg:.3f})\t'
                  'accu@2 {acc2.val:.3f} ({acc2.avg:.3f})\t'
                  'accu@3 {acc3.val:.3f} ({acc3.avg:.3f})'.format(
                   epoch + 1, i + 1, loss=losses, acc1=acc1, acc2=acc2, acc3=acc3))

    return losses.avg, acc1.avg, acc2.avg, acc3.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    acc1 = AverageMeter()
    acc2 = AverageMeter()
    acc3 = AverageMeter()
    model.eval()

    for i, (inputs, target) in enumerate(val_loader):
        with torch.no_grad():
            output = model(inputs.cuda())
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc = accuracy(output.data, target, topk=(1, 2, 3))
        losses.update(loss.item(), inputs.size(0))
        acc1.update(acc[0], inputs.size(0))
        acc2.update(acc[1], inputs.size(0))
        acc3.update(acc[2], inputs.size(0))

    return losses.avg, acc1.avg, acc2.avg, acc3.avg


def test(test_loader, model, checkpoint, lable_path, pred_path):
    acc1 = AverageMeter()
    acc2 = AverageMeter()
    acc3 = AverageMeter()
    # load learnt model that obtained best performance on validation set
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    model.eval()

    label_output = list()
    pred_output = list()

    t_start = time.time()
    for i, (inputs, target) in enumerate(test_loader):
        with torch.no_grad():
            output = model(inputs.cuda())
            output = output.view((-1, inputs.size(0)//target.size(0), output.size(1)))
            output = output.mean(1)

        label_output.append(target.cpu().numpy())
        pred_output.append(output.cpu().numpy())

        acc = accuracy(output.data, target.cuda(non_blocking=True), topk=(1, 2, 3))
        acc1.update(acc[0], inputs.size(0))
        acc2.update(acc[1], inputs.size(0))
        acc3.update(acc[2], inputs.size(0))


    label_output = np.concatenate(label_output, axis=0)
    np.savetxt(lable_path, label_output, fmt='%d')
    pred_output = np.concatenate(pred_output, axis=0)
    np.savetxt(pred_path, pred_output, fmt='%f')

    print('Test: accuracy@1 {:.3f}, accuracy@2 {:.3f}, accuracy@3 {:.3f}, time: {:.2f}s'
          .format(acc1.avg, acc2.avg, acc3.avg, time.time() - t_start))


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
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
    
