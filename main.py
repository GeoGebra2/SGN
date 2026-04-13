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
PART_NAMES = ['head', 'left_arm', 'right_arm', 'left_leg', 'right_leg']

def main():
    if args.cscl and not args.proto_decompose:
        raise ValueError('CSCL requires --proto-decompose because it uses reconstructed topology features.')

    ntu_loaders = NTUDataLoaders(args.dataset, args.case, seg=args.seg, args=args)
    args.num_classes = getattr(ntu_loaders, 'num_classes', None) or get_num_classes(args.dataset)
    model = SGN(args.num_classes, args.dataset, args.seg, args)

    total = get_n_params(model)
    print(model)
    print('The number of parameters: ', total)
    print('The modes is:', args.network)

    if torch.cuda.is_available():
        print('It is using GPU!')
        model = model.cuda()

    criterion = LabelSmoothingLoss(args.num_classes, smoothing=0.1).cuda()
    cscl_criterion = None
    if args.cscl:
        cscl_criterion = ClassSpecificContrastiveLoss(
            n_class=args.num_classes,
            n_channel=25 * 25,
            h_channel=args.cscl_hidden,
            tmp=args.cscl_temp,
            mom=args.cscl_momentum,
            pred_threshold=args.cscl_threshold,
        ).cuda()
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
            stage_name = 'branch-only' if (args.part_decompose and args.part_two_stage and epoch < args.part_branch_epochs) else 'fusion'

            print(epoch, optimizer.param_groups[0]['lr'], stage_name)

            t_start = time.time()
            train_loss, train_acc1, train_acc2, train_acc3, train_acc4, train_acc5, train_branch_acc1, train_fusion_acc1, train_part_acc = train(
                train_loader, model, criterion, optimizer, epoch, cscl_criterion=cscl_criterion
            )
            val_loss, val_acc1, val_acc2, val_acc3, val_acc4, val_acc5, val_branch_acc1, val_fusion_acc1, val_part_acc = validate(
                val_loader, model, criterion, epoch
            )
            log_res += [[train_loss, float(train_acc1), float(train_acc2), float(train_acc3), float(train_acc4), float(train_acc5),\
                         train_branch_acc1, train_fusion_acc1, train_part_acc[0], train_part_acc[1], train_part_acc[2], train_part_acc[3], train_part_acc[4],
                         val_loss, float(val_acc1), float(val_acc2), float(val_acc3), float(val_acc4), float(val_acc5),
                         val_branch_acc1, val_fusion_acc1, val_part_acc[0], val_part_acc[1], val_part_acc[2], val_part_acc[3], val_part_acc[4]]]

            print('Epoch-{:<3d} {:.1f}s\t'
                  'Train: loss {:.4f}\tTop-1 accu {:.4f}\tTop-2 accu {:.4f}\tTop-3 accu {:.4f}\tTop-4 accu {:.4f}\tTop-5 accu {:.4f}\t'
                  'Branch-only Top-1 {:.4f}\tFusion Top-1 {:.4f}\t'
                  'Valid: loss {:.4f}\tTop-1 accu {:.4f}\tTop-2 accu {:.4f}\tTop-3 accu {:.4f}\tTop-4 accu {:.4f}\tTop-5 accu {:.4f}\t'
                  'Branch-only Top-1 {:.4f}\tFusion Top-1 {:.4f}'
                  .format(epoch + 1, time.time() - t_start, train_loss, train_acc1, train_acc2, train_acc3, train_acc4, train_acc5,
                          train_branch_acc1, train_fusion_acc1, val_loss, val_acc1, val_acc2, val_acc3, val_acc4, val_acc5,
                          val_branch_acc1, val_fusion_acc1))
            print(
                'Part Branch Top-1 | '
                'Train {} {:.4f}, {} {:.4f}, {} {:.4f}, {} {:.4f}, {} {:.4f} | '
                'Valid {} {:.4f}, {} {:.4f}, {} {:.4f}, {} {:.4f}, {} {:.4f}'.format(
                    PART_NAMES[0], train_part_acc[0], PART_NAMES[1], train_part_acc[1], PART_NAMES[2], train_part_acc[2],
                    PART_NAMES[3], train_part_acc[3], PART_NAMES[4], train_part_acc[4],
                    PART_NAMES[0], val_part_acc[0], PART_NAMES[1], val_part_acc[1], PART_NAMES[2], val_part_acc[2],
                    PART_NAMES[3], val_part_acc[3], PART_NAMES[4], val_part_acc[4],
                )
            )

            if mode == 'min':
                current = val_loss
            elif args.part_decompose and args.part_two_stage and epoch < args.part_branch_epochs:
                current = val_branch_acc1
            else:
                current = val_acc1

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
            cw.writerow([
                'loss', 'acc@1', 'acc@2', 'acc@3', 'acc@4', 'acc@5', 'branch_acc@1', 'fusion_acc@1',
                'part_head_acc@1', 'part_left_arm_acc@1', 'part_right_arm_acc@1', 'part_left_leg_acc@1', 'part_right_leg_acc@1',
                'val_loss', 'val_acc@1', 'val_acc@2', 'val_acc@3', 'val_acc@4', 'val_acc@5',
                'val_branch_acc@1', 'val_fusion_acc@1',
                'val_part_head_acc@1', 'val_part_left_arm_acc@1', 'val_part_right_arm_acc@1', 'val_part_left_leg_acc@1', 'val_part_right_leg_acc@1',
            ])
            cw.writerows(log_res)
        print('Save train and validation log into into %s' % csv_file)

    ### Test
    args.train = 0
    model = SGN(args.num_classes, args.dataset, args.seg, args)
    model = model.cuda()
    test(test_loader, model, checkpoint, lable_path, pred_path)


def train(train_loader, model, criterion, optimizer, epoch, cscl_criterion=None):
    losses = AverageMeter()
    acc1 = AverageMeter()
    acc2 = AverageMeter()
    acc3 = AverageMeter()
    acc4 = AverageMeter()
    acc5 = AverageMeter()
    branch_acc1 = AverageMeter()
    fusion_acc1 = AverageMeter()
    part_acc1 = [AverageMeter() for _ in PART_NAMES]
    model.train()
    stage1_branch_only = bool(args.part_decompose and args.part_two_stage and epoch < args.part_branch_epochs)

    for i, (inputs, target) in enumerate(train_loader):

        inputs = inputs.cuda()
        output, feats, aux = forward_model(inputs, model)
        target = target.cuda(non_blocking=True)
        if stage1_branch_only:
            loss = branch_only_loss(aux, target, criterion)
        else:
            loss = criterion(output, target)
            if args.metric_loss != 'none':
                metric = metric_loss(feats, target, args)
                loss = loss + args.metric_weight * metric
            if args.proto_decompose:
                proto_rec_loss, proto_entropy_loss = prototype_terms(aux)
                loss = loss + args.proto_weight * proto_rec_loss + args.proto_entropy_weight * proto_entropy_loss
            if args.cscl and cscl_criterion is not None:
                cscl_feature = aux["reconstructed_topology"].view(inputs.size(0), -1)
                cscl = cscl_criterion(cscl_feature, target, output)
                loss = loss + args.cscl_weight * cscl

        # measure accuracy and record loss
        acc = accuracy(output.data, target, topk=(1, 2, 3, 4, 5))
        losses.update(loss.item(), inputs.size(0))
        acc1.update(acc[0], inputs.size(0))
        acc2.update(acc[1], inputs.size(0))
        acc3.update(acc[2], inputs.size(0))
        acc4.update(acc[3], inputs.size(0))
        acc5.update(acc[4], inputs.size(0))
        if "branch_fused_logits" in aux:
            branch_top1 = accuracy(aux["branch_fused_logits"].data, target, topk=(1,))[0]
            branch_acc1.update(branch_top1, inputs.size(0))
        update_part_branch_meters(part_acc1, aux, target)
        fusion_acc1.update(acc[0], inputs.size(0))

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
                  'Top-5 accu {acc5.val:.3f} ({acc5.avg:.3f})\t'
                  'Branch-only Top-1 {branch.val:.3f} ({branch.avg:.3f})\t'
                  'Fusion Top-1 {fusion.val:.3f} ({fusion.avg:.3f})\t'
                  'Part-Top1[{p0}:{p0v:.3f}, {p1}:{p1v:.3f}, {p2}:{p2v:.3f}, {p3}:{p3v:.3f}, {p4}:{p4v:.3f}]'.format(
                   epoch + 1, i + 1, loss=losses, acc1=acc1, acc2=acc2, acc3=acc3, acc4=acc4, acc5=acc5,
                   branch=branch_acc1, fusion=fusion_acc1,
                   p0=PART_NAMES[0], p1=PART_NAMES[1], p2=PART_NAMES[2], p3=PART_NAMES[3], p4=PART_NAMES[4],
                   p0v=part_acc1[0].avg, p1v=part_acc1[1].avg, p2v=part_acc1[2].avg, p3v=part_acc1[3].avg, p4v=part_acc1[4].avg))

    return losses.avg, acc1.avg, acc2.avg, acc3.avg, acc4.avg, acc5.avg, branch_acc1.avg, fusion_acc1.avg, [m.avg for m in part_acc1]


def validate(val_loader, model, criterion, epoch=0):
    losses = AverageMeter()
    acc1 = AverageMeter()
    acc2 = AverageMeter()
    acc3 = AverageMeter()
    acc4 = AverageMeter()
    acc5 = AverageMeter()
    branch_acc1 = AverageMeter()
    fusion_acc1 = AverageMeter()
    part_acc1 = [AverageMeter() for _ in PART_NAMES]
    model.eval()
    stage1_branch_only = bool(args.part_decompose and args.part_two_stage and epoch < args.part_branch_epochs)

    for i, (inputs, target) in enumerate(val_loader):
        with torch.no_grad():
            output, _, aux = forward_model(inputs.cuda(), model)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            if stage1_branch_only:
                loss = branch_only_loss(aux, target, criterion)
            else:
                loss = criterion(output, target)

        # measure accuracy and record loss
        acc = accuracy(output.data, target, topk=(1, 2, 3, 4, 5))
        losses.update(loss.item(), inputs.size(0))
        acc1.update(acc[0], inputs.size(0))
        acc2.update(acc[1], inputs.size(0))
        acc3.update(acc[2], inputs.size(0))
        acc4.update(acc[3], inputs.size(0))
        acc5.update(acc[4], inputs.size(0))
        if "branch_fused_logits" in aux:
            branch_top1 = accuracy(aux["branch_fused_logits"].data, target, topk=(1,))[0]
            branch_acc1.update(branch_top1, inputs.size(0))
        update_part_branch_meters(part_acc1, aux, target)
        fusion_acc1.update(acc[0], inputs.size(0))

    return losses.avg, acc1.avg, acc2.avg, acc3.avg, acc4.avg, acc5.avg, branch_acc1.avg, fusion_acc1.avg, [m.avg for m in part_acc1]


def test(test_loader, model, checkpoint, lable_path, pred_path):
    acc1 = AverageMeter()
    acc2 = AverageMeter()
    acc3 = AverageMeter()
    acc4 = AverageMeter()
    acc5 = AverageMeter()
    branch_acc1 = AverageMeter()
    fusion_acc1 = AverageMeter()
    part_acc1 = [AverageMeter() for _ in PART_NAMES]
    # load learnt model that obtained best performance on validation set
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    model.eval()

    label_output = list()
    pred_output = list()

    t_start = time.time()
    for i, (inputs, target) in enumerate(test_loader):
        with torch.no_grad():
            output, _, aux = forward_model(inputs.cuda(), model)
            output = output.view((-1, inputs.size(0)//target.size(0), output.size(1)))
            output = output.mean(1)
            if "branch_fused_logits" in aux:
                branch_output = aux["branch_fused_logits"].view(
                    (-1, inputs.size(0)//target.size(0), aux["branch_fused_logits"].size(1))
                ).mean(1)
            else:
                branch_output = None

        label_output.append(target.cpu().numpy())
        pred_output.append(output.cpu().numpy())

        acc = accuracy(output.data, target.cuda(non_blocking=True), topk=(1, 2, 3, 4, 5))
        acc1.update(acc[0], inputs.size(0))
        acc2.update(acc[1], inputs.size(0))
        acc3.update(acc[2], inputs.size(0))
        acc4.update(acc[3], inputs.size(0))
        acc5.update(acc[4], inputs.size(0))
        if branch_output is not None:
            branch_top1 = accuracy(branch_output.data, target.cuda(non_blocking=True), topk=(1,))[0]
            branch_acc1.update(branch_top1, inputs.size(0))
        update_part_branch_meters(part_acc1, aux, target.cuda(non_blocking=True))
        fusion_acc1.update(acc[0], inputs.size(0))


    label_output = np.concatenate(label_output, axis=0)
    np.savetxt(lable_path, label_output, fmt='%d')
    pred_output = np.concatenate(pred_output, axis=0)
    np.savetxt(pred_path, pred_output, fmt='%f')

    print('Test: Top-1 accu {:.3f}, Top-2 accu {:.3f}, Top-3 accu {:.3f}, Top-4 accu {:.3f}, Top-5 accu {:.3f}, '
          'Branch-only Top-1 {:.3f}, Fusion Top-1 {:.3f}, time: {:.2f}s'
          .format(acc1.avg, acc2.avg, acc3.avg, acc4.avg, acc5.avg, branch_acc1.avg, fusion_acc1.avg, time.time() - t_start))
    print(
        'Test Part Branch Top-1: {} {:.3f}, {} {:.3f}, {} {:.3f}, {} {:.3f}, {} {:.3f}'.format(
            PART_NAMES[0], part_acc1[0].avg, PART_NAMES[1], part_acc1[1].avg, PART_NAMES[2], part_acc1[2].avg,
            PART_NAMES[3], part_acc1[3].avg, PART_NAMES[4], part_acc1[4].avg
        )
    )


def forward_model(inputs, model):
    if args.part_decompose or args.proto_decompose:
        output, feats, aux = model.forward_with_aux(inputs)
    else:
        feats = model.forward_features(inputs)
        output = model.fc(feats)
        aux = {}
    return output, feats, aux


def branch_only_loss(aux, target, criterion):
    if "part_logits" not in aux or "part_mask" not in aux:
        raise ValueError('Branch-only stage requires --part-decompose to produce part logits and masks.')
    part_logits = aux["part_logits"]
    part_mask = aux["part_mask"]
    loss_terms = []
    for idx in range(part_logits.size(1)):
        active = part_mask[:, idx]
        if active.any():
            loss_terms.append(criterion(part_logits[active, idx, :], target[active]))
    if not loss_terms:
        return criterion(aux["branch_fused_logits"], target)
    return torch.stack(loss_terms).mean()


def update_part_branch_meters(part_meters, aux, target):
    if "part_logits" not in aux or "part_mask" not in aux:
        return
    part_logits = aux["part_logits"]
    part_mask = aux["part_mask"]
    for idx in range(min(len(part_meters), part_logits.size(1))):
        active = part_mask[:, idx]
        if active.any():
            part_top1 = accuracy(part_logits[active, idx, :].data, target[active], topk=(1,))[0]
            part_meters[idx].update(part_top1, int(active.sum().item()))


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

class ClassSpecificContrastiveLoss(nn.Module):
    def __init__(self, n_class, n_channel=625, h_channel=256, tmp=0.125, mom=0.9, pred_threshold=0.0):
        super(ClassSpecificContrastiveLoss, self).__init__()
        self.n_channel = n_channel
        self.h_channel = h_channel
        self.n_class = n_class
        self.tmp = tmp
        self.mom = mom
        self.pred_threshold = pred_threshold
        self.register_buffer('avg_f', torch.randn(self.h_channel, self.n_class))
        self.cl_fc = nn.Linear(self.n_channel, self.h_channel)
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def local_average(self, feature, mask):
        feature = feature.permute(1, 0)
        mask_sum = mask.sum(0, keepdim=True)
        f_mask = torch.matmul(feature, mask) / (mask_sum + 1e-12)
        has_object = (mask_sum > 1e-8).float()
        momentum_mask = torch.where(
            has_object > 0.1,
            torch.full_like(has_object, self.mom),
            torch.ones_like(has_object),
        )
        f_mem = self.avg_f * momentum_mask + (1 - momentum_mask) * f_mask
        with torch.no_grad():
            self.avg_f.copy_(f_mem.detach())
        return f_mem

    def get_score(self, feature, f_mem):
        feature = F.normalize(feature, p=2, dim=1)
        f_mem = F.normalize(f_mem.permute(1, 0), p=2, dim=-1)
        score = torch.matmul(f_mem, feature.permute(1, 0))
        return score / self.tmp

    def forward(self, feature, lbl, logit):
        feature = self.cl_fc(feature)
        pred = logit.max(1)[1]
        pred_one = F.one_hot(pred, num_classes=self.n_class).float()
        lbl_one = F.one_hot(lbl, num_classes=self.n_class).float()
        logit = torch.softmax(logit, dim=1)
        mask = lbl_one * pred_one
        mask = mask * (logit > self.pred_threshold).float()
        f_mem = self.local_average(feature, mask)
        score_cl = self.get_score(feature, f_mem).permute(1, 0).contiguous()
        return self.loss(score_cl, lbl).mean()

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

def prototype_terms(aux):
    topology = aux["topology"]
    reconstructed_topology = aux["reconstructed_topology"]
    prototype_weights = aux["prototype_weights"]
    rec_loss = F.mse_loss(reconstructed_topology, topology.detach())
    entropy_loss = -(prototype_weights * torch.log(prototype_weights + 1e-12)).sum(dim=1).mean()
    return rec_loss, entropy_loss

if __name__ == '__main__':
    main()
    
