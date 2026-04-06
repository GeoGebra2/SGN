# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
import h5py
import random
import os.path as osp
import sys
import math
import scipy.misc
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class NTUDataset(Dataset):
    def __init__(self, x, y, pid=None, aid=None, return_meta=False):
        self.x = x
        self.y = np.array(y, dtype='int')
        self.pid = None if pid is None else np.array(pid, dtype='int')
        self.aid = None if aid is None else np.array(aid, dtype='int')
        self.return_meta = return_meta

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if self.return_meta and (self.pid is not None) and (self.aid is not None):
            return [self.x[index], int(self.y[index]), int(self.pid[index]), int(self.aid[index])]
        return [self.x[index], int(self.y[index])]

class NTUDataLoaders(object):
    def __init__(self, dataset='NTU', case=0, aug=1, seg=30, args=None, return_meta=False):
        self.dataset = dataset
        self.case = case
        self.aug = aug
        self.seg = seg
        self.args = args
        self.return_meta = return_meta
        self.drop_two_person = bool(getattr(args, 'drop_two_person', False)) if args is not None else False
        self.tta_clips = max(1, int(getattr(args, 'tta_clips', 5))) if args is not None else 5
        self.create_datasets()
        if self.return_meta:
            self.train_set = NTUDataset(self.train_X, self.train_Y, pid=self.train_pid, aid=self.train_aid, return_meta=True)
            self.val_set = NTUDataset(self.val_X, self.val_Y, pid=self.val_pid, aid=self.val_aid, return_meta=True)
            self.test_set = NTUDataset(self.test_X, self.test_Y, pid=self.test_pid, aid=self.test_aid, return_meta=True)
        else:
            self.train_set = NTUDataset(self.train_X, self.train_Y)
            self.val_set = NTUDataset(self.val_X, self.val_Y)
            self.test_set = NTUDataset(self.test_X, self.test_Y)

    def get_train_loader(self, batch_size, num_workers):
        if self.aug == 0:
            return DataLoader(self.train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              collate_fn=self.collate_fn_fix_val, pin_memory=False, drop_last=True)
        elif self.aug ==1:
            return DataLoader(self.train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              collate_fn=self.collate_fn_fix_train, pin_memory=True, drop_last=True)

    def get_val_loader(self, batch_size, num_workers):
        if self.dataset == 'NTU' or self.dataset == 'NTU_ID' or self.dataset == 'kinetics' or self.dataset == 'NTU120':
            return DataLoader(self.val_set, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              collate_fn=self.collate_fn_fix_val, pin_memory=True, drop_last=False)
        else:
            return DataLoader(self.val_set, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              collate_fn=self.collate_fn_fix_val, pin_memory=True, drop_last=False)


    def get_test_loader(self, batch_size, num_workers):
        return DataLoader(self.test_set, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          collate_fn=self.collate_fn_fix_test, pin_memory=True, drop_last=False)

    def get_train_size(self):
        return len(self.train_Y)

    def get_val_size(self):
        return len(self.val_Y)

    def get_test_size(self):
        return len(self.test_Y)

    def create_datasets(self):
        if self.dataset == 'NTU':
            if self.case ==0:
                self.metric = 'CS'
            elif self.case == 1:
                self.metric = 'CV'
            elif self.case == 2:
                self.metric = 'CR'
            elif self.case == 3:
                self.metric = 'CA'
            path = osp.join('./data/ntu', 'NTU_' + self.metric + '.h5')
        elif self.dataset == 'NTU_ID':
            if self.case == 0:
                self.metric = 'CS'
            elif self.case == 1:
                self.metric = 'CV'
            elif self.case == 2:
                self.metric = 'CR'
            elif self.case == 3:
                self.metric = 'CA'
            path = osp.join('./data/ntu', 'NTU_ID_' + self.metric + '.h5')

        f = h5py.File(path , 'r')
        self.train_X = f['x'][:]
        self.train_Y = np.argmax(f['y'][:],-1)
        self.val_X = f['valid_x'][:]
        self.val_Y = np.argmax(f['valid_y'][:], -1)
        self.test_X = f['test_x'][:]
        self.test_Y = np.argmax(f['test_y'][:], -1)
        self.train_pid = f['pid'][:] if 'pid' in f else None
        self.train_aid = f['aid'][:] if 'aid' in f else None
        self.val_pid = f['valid_pid'][:] if 'valid_pid' in f else None
        self.val_aid = f['valid_aid'][:] if 'valid_aid' in f else None
        self.test_pid = f['test_pid'][:] if 'test_pid' in f else None
        self.test_aid = f['test_aid'][:] if 'test_aid' in f else None
        f.close()

        if self.metric == 'CS' or self.metric == 'CV' or self.metric == 'CR':
            self.train_X = np.concatenate([self.train_X, self.val_X], axis=0)
            self.train_Y = np.concatenate([self.train_Y, self.val_Y], axis=0)
            if self.train_pid is not None and self.val_pid is not None:
                self.train_pid = np.concatenate([self.train_pid, self.val_pid], axis=0)
            if self.train_aid is not None and self.val_aid is not None:
                self.train_aid = np.concatenate([self.train_aid, self.val_aid], axis=0)
            self.val_X = self.test_X
            self.val_Y = self.test_Y
            self.val_pid = self.test_pid
            self.val_aid = self.test_aid

        if self.drop_two_person and self.dataset == 'NTU':
            self.train_X, self.train_Y, self.train_pid, self.train_aid = self._drop_two_person_split(
                self.train_X, self.train_Y, self.train_pid, self.train_aid
            )
            self.val_X, self.val_Y, self.val_pid, self.val_aid = self._drop_two_person_split(
                self.val_X, self.val_Y, self.val_pid, self.val_aid
            )
            self.test_X, self.test_Y, self.test_pid, self.test_aid = self._drop_two_person_split(
                self.test_X, self.test_Y, self.test_pid, self.test_aid
            )

        all_y = np.unique(np.concatenate([self.train_Y, self.val_Y, self.test_Y], axis=0))
        self.class_ids = all_y.astype(int)
        self.label_map = {int(old): int(i) for i, old in enumerate(self.class_ids)}
        self.num_classes = int(len(self.class_ids))
        if self.num_classes > 0 and (self.class_ids.min() != 0 or self.class_ids.max() != self.num_classes - 1):
            remap = np.vectorize(lambda v: self.label_map[int(v)], otypes=[int])
            self.train_Y = remap(self.train_Y)
            self.val_Y = remap(self.val_Y)
            self.test_Y = remap(self.test_Y)
        print('Dataset split:', self.dataset, self.metric, 'train', len(self.train_Y), 'val', len(self.val_Y), 'test', len(self.test_Y))
        print('Label coverage:', 'train', len(np.unique(self.train_Y)), 'val', len(np.unique(self.val_Y)), 'test', len(np.unique(self.test_Y)))
        print('Num classes:', self.num_classes)

    def _drop_two_person_split(self, x, y, pid=None, aid=None):
        if len(y) == 0:
            return x, y, pid, aid
        y = np.array(y, dtype='int')
        if aid is not None and len(aid) == len(y):
            aid_arr = np.array(aid, dtype='int')
            keep = aid_arr < 50
        else:
            keep = y < 49
        x_new = x[keep]
        y_new = y[keep]
        pid_new = None if pid is None else np.array(pid)[keep]
        aid_new = None if aid is None else np.array(aid)[keep]
        return x_new, y_new, pid_new, aid_new

    def collate_fn_fix_train(self, batch):
        """Puts each data field into a tensor with outer dimension batch size
        """
        if len(batch[0]) == 2:
            x, y = zip(*batch)
            pid, aid = None, None
        else:
            x, y, pid, aid = zip(*batch)

        if self.dataset == 'kinetics' and self.machine == 'philly':
            x = np.array(x)
            x = x.reshape(x.shape[0], x.shape[1],-1)
            x = x.reshape(-1, x.shape[1] * x.shape[2], x.shape[3]*x.shape[4])
            x = list(x)

        x, y = self.Tolist_fix(x, y, train=1)
        lens = np.array([x_.shape[0] for x_ in x], dtype=int)
        idx = lens.argsort()[::-1]  # sort sequence by valid length in descending order
        y = np.array(y)[idx]
        x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        #x = _view_normalize(x)

        if self.dataset == 'NTU':
            if self.case == 0:
                theta = 0.3
            elif self.case == 1:
                theta = 0.5
            elif self.case == 2:
                theta = 0.5
            elif self.case == 3:
                theta = 0.5
        elif self.dataset == 'NTU_ID':
            if self.case == 0:
                theta = 0.3
            elif self.case == 1:
                theta = 0.5
            elif self.case == 2:
                theta = 0.5
            elif self.case == 3:
                theta = 0.5
        elif self.dataset == 'NTU120':
            theta = 0.3

        #### data augmentation
        x = _transform(x, theta)
        #### data augmentation
        y = torch.LongTensor(y)

        if pid is not None and aid is not None and self.return_meta:
            pid = torch.LongTensor(np.array(pid)[idx])
            aid = torch.LongTensor(np.array(aid)[idx])
            return [x, y, pid, aid]
        return [x, y]

    def collate_fn_fix_val(self, batch):
        """Puts each data field into a tensor with outer dimension batch size
        """
        if len(batch[0]) == 2:
            x, y = zip(*batch)
            pid, aid = None, None
        else:
            x, y, pid, aid = zip(*batch)
        x, y = self.Tolist_fix(x, y, train=1)
        idx = range(len(x))
        y = np.array(y)

        x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        #x = _view_normalize(x)
        y = torch.LongTensor(y)

        if pid is not None and aid is not None and self.return_meta:
            pid = torch.LongTensor(np.array(pid))
            aid = torch.LongTensor(np.array(aid))
            return [x, y, pid, aid]
        return [x, y]

    def collate_fn_fix_test(self, batch):
        """Puts each data field into a tensor with outer dimension batch size
        """
        if len(batch[0]) == 2:
            x, y = zip(*batch)
            pid, aid = None, None
        else:
            x, y, pid, aid = zip(*batch)
        x, labels = self.Tolist_fix(x, y ,train=2)
        idx = range(len(x))
        y = np.array(y)


        x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        #x = _view_normalize(x)
        y = torch.LongTensor(y)

        if pid is not None and aid is not None and self.return_meta:
            pid = torch.LongTensor(np.array(pid))
            aid = torch.LongTensor(np.array(aid))
            return [x, y, pid, aid]
        return [x, y]

    def Tolist_fix(self, joints, y, train = 1):
        seqs = []

        for idx, seq in enumerate(joints):
            zero_row = []
            for i in range(len(seq)):
                if (seq[i, :] == np.zeros((1, 150))).all():
                        zero_row.append(i)

            seq = np.delete(seq, zero_row, axis = 0)

            seq = turn_two_to_one(seq)
            seqs = self.sub_seq(seqs, seq, train=train)

        return seqs, y

    def sub_seq(self, seqs, seq , train = 1):
        group = self.seg

        if self.dataset == 'SYSU' or self.dataset == 'SYSU_same':
            seq = seq[::2, :]

        if seq.shape[0] < self.seg:
            pad = np.zeros((self.seg - seq.shape[0], seq.shape[1])).astype(np.float32)
            seq = np.concatenate([seq, pad], axis=0)

        ave_duration = seq.shape[0] // group

        if train == 1:
            offsets = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            seq = seq[offsets]
            seqs.append(seq)

        elif train == 2:
            for _ in range(self.tta_clips):
                offsets = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
                seqs.append(seq[offsets])

        return seqs

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def turn_two_to_one(seq):
    new_seq = list()
    for idx, ske in enumerate(seq):
        if (ske[0:75] == np.zeros((1, 75))).all():
            new_seq.append(ske[75:])
        elif (ske[75:] == np.zeros((1, 75))).all():
            new_seq.append(ske[0:75])
        else:
            new_seq.append(ske[0:75])
            new_seq.append(ske[75:])
    return np.array(new_seq)

def _rot(rot):
    cos_r, sin_r = rot.cos(), rot.sin()
    zeros = rot.new(rot.size()[:2] + (1,)).zero_()
    ones = rot.new(rot.size()[:2] + (1,)).fill_(1)

    r1 = torch.stack((ones, zeros, zeros),dim=-1)
    rx2 = torch.stack((zeros, cos_r[:,:,0:1], sin_r[:,:,0:1]), dim = -1)
    rx3 = torch.stack((zeros, -sin_r[:,:,0:1], cos_r[:,:,0:1]), dim = -1)
    rx = torch.cat((r1, rx2, rx3), dim = 2)

    ry1 = torch.stack((cos_r[:,:,1:2], zeros, -sin_r[:,:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,:,1:2], zeros, cos_r[:,:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 2)

    rz1 = torch.stack((cos_r[:,:,2:3], sin_r[:,:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,:,2:3], cos_r[:,:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 2)

    rot = rz.matmul(ry).matmul(rx)
    return rot

def _transform(x, theta):
    x = x.contiguous().view(x.size()[:2] + (-1, 3))
    rot = x.new(x.size()[0],3).uniform_(-theta, theta)
    rot = rot.repeat(1, x.size()[1])
    rot = rot.contiguous().view((-1, x.size()[1], 3))
    rot = _rot(rot)
    x = torch.transpose(x, 2, 3)
    x = torch.matmul(rot, x)
    x = torch.transpose(x, 2, 3)

    x = x.contiguous().view(x.size()[:2] + (-1,))
    return x

def _view_normalize(x):
    if x.size(-1) % 3 != 0:
        return x
    num_joints = x.size(-1) // 3
    if num_joints <= 16:
        return x
    x_ = x.contiguous().view(x.size()[:2] + (num_joints, 3))
    left = x_[:, :, 12, :]
    right = x_[:, :, 16, :]
    v = (left - right).mean(dim=1)
    angle = torch.atan2(v[:, 2], v[:, 0])
    cos = torch.cos(-angle)
    sin = torch.sin(-angle)
    rot = x_.new_zeros((x_.size(0), 3, 3))
    rot[:, 0, 0] = cos
    rot[:, 0, 2] = sin
    rot[:, 1, 1] = 1
    rot[:, 2, 0] = -sin
    rot[:, 2, 2] = cos
    x_flat = x_.reshape(x_.size(0), -1, 3)
    x_rot = torch.bmm(x_flat, rot.transpose(1, 2))
    x_rot = x_rot.reshape_as(x_)
    return x_rot.view(x.size())
