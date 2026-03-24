# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from torch import nn
import torch
import math

class SGN(nn.Module):
    def __init__(self, num_classes, dataset, seg, args, bias = True):
        super(SGN, self).__init__()

        self.dim1 = 256
        self.dataset = dataset
        self.seg = seg
        self.motion_only = getattr(args, "motion_only", False)
        num_joint = 25

        self.tem_embed = embed(self.seg, 64*4, norm=False, bias=bias)
        self.spa_embed = embed(num_joint, 64, norm=False, bias=bias)
        self.joint_embed = embed(3, 64, norm=True, bias=bias)
        self.dif_embed = embed(3, 64, norm=True, bias=bias)
        self.acc_embed = embed(3, 64, norm=True, bias=bias)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.cnn = local(self.dim1, self.dim1 * 2, bias=bias)
        self.temporal = temporal_stack(self.dim1 * 2, bias=bias)
        self.temporal_att = temporal_attention(self.dim1 * 2, bias=bias)
        self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn3 = gcn_spa(self.dim1, self.dim1, bias=bias)
        self.fc = nn.Linear(self.dim1 * 2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)


    def forward_features(self, input):
        
        # Dynamic Representation
        bs, step, dim = input.size()
        num_joints = dim //3
        input = input.view((bs, step, num_joints, 3))
        input = input.permute(0, 3, 2, 1).contiguous()
        dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]
        dif = torch.cat([dif.new(bs, dif.size(1), num_joints, 1).zero_(), dif], dim=-1)
        dif_raw = dif
        spa = self.one_hot(bs, num_joints, step).permute(0, 3, 2, 1).to(input.device)
        tem = self.one_hot(bs, step, num_joints).permute(0, 3, 1, 2).to(input.device)
        tem1 = self.tem_embed(tem)
        spa1 = self.spa_embed(spa)
        dif = self.dif_embed(dif)
        if self.motion_only:
            acc = dif_raw[:, :, :, 1:] - dif_raw[:, :, :, 0:-1]
            acc = torch.cat([acc.new(bs, acc.size(1), num_joints, 1).zero_(), acc], dim=-1)
            acc = self.acc_embed(acc)
            dy = dif + acc
        else:
            pos = self.joint_embed(input)
            dy = pos + dif
        # Joint-level Module
        input= torch.cat([dy, spa1], 1)
        g = self.compute_g1(input)
        input = self.gcn1(input, g)
        input = self.gcn2(input, g)
        input = self.gcn3(input, g)
        # Frame-level Module
        input = input + tem1
        input = self.cnn(input)
        input = self.temporal(input)
        input = self.temporal_att(input)
        # Classification
        output = self.maxpool(input)
        output = torch.flatten(output, 1)
        return output

    def forward(self, input):
        output = self.forward_features(input)
        output = self.fc(output)
        return output

    def one_hot(self, bs, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot

class norm_data(nn.Module):
    def __init__(self, dim= 64):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim* 25)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x

class embed(nn.Module):
    def __init__(self, dim = 3, dim1 = 128, norm = True, bias = False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x

class cnn1x1(nn.Module):
    def __init__(self, dim1 = 3, dim2 =3, bias = True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x

class local(nn.Module):
    def __init__(self, dim1 = 3, dim2 = 3, bias = False):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 20))
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1):
        x1 = self.maxpool(x1)
        x = self.cnn1(x1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class temporal_block(nn.Module):
    def __init__(self, dim, dilation=1, bias=False):
        super(temporal_block, self).__init__()
        self.cnn = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, dilation), dilation=(1, dilation), bias=bias)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.cnn(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class temporal_stack(nn.Module):
    def __init__(self, dim, bias=False):
        super(temporal_stack, self).__init__()
        self.blocks = nn.ModuleList([
            temporal_block(dim, dilation=1, bias=bias),
            temporal_block(dim, dilation=2, bias=bias),
            temporal_block(dim, dilation=4, bias=bias),
            temporal_block(dim, dilation=8, bias=bias),
        ])

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x

class temporal_attention(nn.Module):
    def __init__(self, dim, reduction=4, bias=False):
        super(temporal_attention, self).__init__()
        hidden = max(1, dim // reduction)
        self.fc1 = nn.Conv1d(dim, hidden, kernel_size=1, bias=bias)
        self.fc2 = nn.Conv1d(hidden, dim, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_t = x.mean(dim=2)
        att = self.fc2(self.relu(self.fc1(x_t)))
        att = torch.softmax(att, dim=-1)
        x = x * att.unsqueeze(2)
        return x

class gcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias = False):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)


    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()
        x = g.matmul(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.w(x) + self.w1(x1)
        x = self.relu(self.bn(x))
        return x

class compute_g_spa(nn.Module):
    def __init__(self, dim1 = 64 *3, dim2 = 64*3, bias = False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):

        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()
        g2 = self.g2(x1).permute(0, 3, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g
    
