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
    use_position_stream=0,
    use_velocity_stream=1,
    use_acceleration_stream=1,
    use_angular_velocity_stream=1,
    bone_length_scale_aug=1,
    bone_length_scale_range=0.15,
    disentangle=1,
    lambda_sta_proxy=1.0,
    lambda_dyn_adv=0.5,
    grl_lambda=1.0,
    lambda_swap_consistency=0.1,
    lambda_swap_id=0.1,
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
    ntu_loaders = NTUDataLoaders(args.dataset, args.case, seg=args.seg, args=args)
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
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
            val_loss, val_acc = validate(val_loader, model, criterion)
            log_res += [[train_loss, float(train_acc),\
                         val_loss, float(val_acc)]]

            print('Epoch-{:<3d} {:.1f}s\t'
                  'Train: loss {:.4f}\taccu {:.4f}\tValid: loss {:.4f}\taccu {:.4f}'
                  .format(epoch + 1, time.time() - t_start, train_loss, train_acc, val_loss, val_acc))

            current = val_loss if mode == 'min' else val_acc

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
            cw.writerow(['loss', 'acc', 'val_loss', 'val_acc'])
            cw.writerows(log_res)
        print('Save train and validation log into into %s' % csv_file)

    ### Test
    args.train = 0
    model = SGN(args.num_classes, args.dataset, args.seg, args)
    model = model.cuda()
    test(test_loader, model, checkpoint, lable_path, pred_path)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    cls_losses = AverageMeter()
    sta_losses = AverageMeter()
    adv_losses = AverageMeter()
    swap_cons_losses = AverageMeter()
    swap_id_losses = AverageMeter()
    acces = AverageMeter()
    model.train()

    for i, (inputs, target) in enumerate(train_loader):

        output = model(inputs.cuda(), target.cuda(non_blocking=True) if args.disentangle else None)
        target = target.cuda(non_blocking=True)
        if isinstance(output, dict):
            logits = output['id_logits']
            cls_loss = criterion(logits, target)
            sta_loss = output['sta_proxy_loss']
            adv_loss = output['dyn_adv_loss']
            swap_consistency_loss = output['swap_consistency_loss']
            swap_id_loss = output['swap_id_loss']
            loss = cls_loss + \
                   args.lambda_sta_proxy * sta_loss + \
                   args.lambda_dyn_adv * adv_loss + \
                   args.lambda_swap_consistency * swap_consistency_loss + \
                   args.lambda_swap_id * swap_id_loss
            cls_losses.update(cls_loss.item(), inputs.size(0))
            sta_losses.update(sta_loss.item(), inputs.size(0))
            adv_losses.update(adv_loss.item(), inputs.size(0))
            swap_cons_losses.update(swap_consistency_loss.item(), inputs.size(0))
            swap_id_losses.update(swap_id_loss.item(), inputs.size(0))
        else:
            logits = output
            loss = criterion(logits, target)
            cls_losses.update(loss.item(), inputs.size(0))

        # measure accuracy and record loss
        acc = accuracy(logits.data, target)
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        # backward
        optimizer.zero_grad()  # clear gradients out before each mini-batch
        loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            if args.disentangle:
                print('Epoch-{:<3d} {:3d} batches\t'
                      'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'cls {cls.val:.4f} ({cls.avg:.4f})\t'
                      'sta {sta.val:.4f} ({sta.avg:.4f})\t'
                      'adv {adv.val:.4f} ({adv.avg:.4f})\t'
                      'swc {swc.val:.4f} ({swc.avg:.4f})\t'
                      'swi {swi.val:.4f} ({swi.avg:.4f})\t'
                      'accu {acc.val:.3f} ({acc.avg:.3f})'.format(
                       epoch + 1, i + 1, loss=losses, cls=cls_losses, sta=sta_losses,
                       adv=adv_losses, swc=swap_cons_losses, swi=swap_id_losses, acc=acces))
            else:
                print('Epoch-{:<3d} {:3d} batches\t'
                      'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'accu {acc.val:.3f} ({acc.avg:.3f})'.format(
                       epoch + 1, i + 1, loss=losses, acc=acces))

    return losses.avg, acces.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    acces = AverageMeter()
    model.eval()

    for i, (inputs, target) in enumerate(val_loader):
        with torch.no_grad():
            output = model(inputs.cuda(), target.cuda(non_blocking=True) if args.disentangle else None)
        target = target.cuda(non_blocking=True)
        if isinstance(output, dict):
            logits = output['id_logits']
            with torch.no_grad():
                cls_loss = criterion(logits, target)
                loss = cls_loss + \
                       args.lambda_sta_proxy * output['sta_proxy_loss'] + \
                       args.lambda_dyn_adv * output['dyn_adv_loss'] + \
                       args.lambda_swap_consistency * output['swap_consistency_loss'] + \
                       args.lambda_swap_id * output['swap_id_loss']
        else:
            logits = output
            with torch.no_grad():
                loss = criterion(logits, target)

        # measure accuracy and record loss
        acc = accuracy(logits.data, target)
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

    return losses.avg, acces.avg


def test(test_loader, model, checkpoint, lable_path, pred_path):
    acces = AverageMeter()
    # load learnt model that obtained best performance on validation set
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    model.eval()

    label_output = list()
    pred_output = list()

    t_start = time.time()
    for i, (inputs, target) in enumerate(test_loader):
        with torch.no_grad():
            output = model(inputs.cuda())
            if isinstance(output, dict):
                output = output['id_logits']
            output = output.view((-1, inputs.size(0)//target.size(0), output.size(1)))
            output = output.mean(1)

        label_output.append(target.cpu().numpy())
        pred_output.append(output.cpu().numpy())

        acc = accuracy(output.data, target.cuda(non_blocking=True))
        acces.update(acc[0], inputs.size(0))


    label_output = np.concatenate(label_output, axis=0)
    np.savetxt(lable_path, label_output, fmt='%d')
    pred_output = np.concatenate(pred_output, axis=0)
    np.savetxt(pred_path, pred_output, fmt='%f')

    print('Test: accuracy {:.3f}, time: {:.2f}s'
          .format(acces.avg, time.time() - t_start))


def accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.view(-1).float().sum(0, keepdim=True)

    return correct.mul_(100.0 / batch_size)

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

if __name__ == '__main__':
    main()
    
