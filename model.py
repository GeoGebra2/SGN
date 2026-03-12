# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from torch import nn
import torch
import math

NTU_BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19)
]

class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coeff * grad_output, None

def grad_reverse(x, coeff=1.0):
    return GradientReverseFunction.apply(x, coeff)

class SGN(nn.Module):
    def __init__(self, num_classes, dataset, seg, args, bias = True):
        super(SGN, self).__init__()

        self.dim1 = 256
        self.dataset = dataset
        self.seg = seg
        self.use_position_stream = bool(getattr(args, 'use_position_stream', 0))
        self.use_velocity_stream = bool(getattr(args, 'use_velocity_stream', 1))
        self.use_acceleration_stream = bool(getattr(args, 'use_acceleration_stream', 1))
        self.use_angular_velocity_stream = bool(getattr(args, 'use_angular_velocity_stream', 1))
        self.disentangle = bool(getattr(args, 'disentangle', 1))
        self.grl_lambda = float(getattr(args, 'grl_lambda', 1.0))
        num_joint = 25

        self.tem_embed = embed(self.seg, 64*4, norm=False, bias=bias)
        self.spa_embed = embed(num_joint, 64, norm=False, bias=bias)
        self.joint_embed = embed(3, 64, norm=True, bias=bias)
        self.sta_joint_embed = embed(3, 64, norm=True, bias=bias)
        self.dif_embed = embed(3, 64, norm=True, bias=bias)
        self.acc_embed = embed(3, 64, norm=True, bias=bias)
        self.ang_embed = embed(3, 64, norm=True, bias=bias)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.cnn_dyn = local(self.dim1, self.dim1 * 2, bias=bias)
        self.compute_g_dyn = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn_dyn1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        self.gcn_dyn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn_dyn3 = gcn_spa(self.dim1, self.dim1, bias=bias)
        self.cnn_sta = local(self.dim1, self.dim1 * 2, bias=bias)
        self.compute_g_sta = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn_sta1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        self.gcn_sta2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn_sta3 = gcn_spa(self.dim1, self.dim1, bias=bias)
        self.fc = nn.Linear(self.dim1 * 2, num_classes)
        self.sta_proxy_head = nn.Sequential(
            nn.Linear(self.dim1 * 2, self.dim1),
            nn.ReLU(),
            nn.Linear(self.dim1, len(NTU_BONES))
        )
        self.dyn_adv_head = nn.Sequential(
            nn.Linear(self.dim1 * 2, self.dim1),
            nn.ReLU(),
            nn.Linear(self.dim1, len(NTU_BONES))
        )
        self.swap_id_head = nn.Sequential(
            nn.Linear(self.dim1 * 4, self.dim1 * 2),
            nn.ReLU(),
            nn.Linear(self.dim1 * 2, num_classes)
        )
        self.mse = nn.MSELoss()
        self.kld = nn.KLDivLoss(reduction='batchmean')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn_dyn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn_dyn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn_dyn3.w.cnn.weight, 0)
        nn.init.constant_(self.gcn_sta1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn_sta2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn_sta3.w.cnn.weight, 0)


    def forward(self, input, target=None):
        
        # Dynamic Representation
        bs, step, dim = input.size()
        num_joints = dim //3
        input = input.view((bs, step, num_joints, 3))
        input = input.permute(0, 3, 2, 1).contiguous()
        vel = input[:, :, :, 1:] - input[:, :, :, 0:-1]
        vel = torch.cat([vel.new(bs, vel.size(1), num_joints, 1).zero_(), vel], dim=-1)
        acc = vel[:, :, :, 1:] - vel[:, :, :, 0:-1]
        acc = torch.cat([acc.new(bs, acc.size(1), num_joints, 1).zero_(), acc], dim=-1)
        rel = input - input[:, :, 1:2, :]
        rel_norm = torch.linalg.norm(rel, dim=1, keepdim=True).clamp_min(1e-6)
        rel_unit = rel / rel_norm
        ang = torch.cross(rel_unit[:, :, :, 0:-1], rel_unit[:, :, :, 1:], dim=1)
        ang = torch.cat([ang.new(bs, ang.size(1), num_joints, 1).zero_(), ang], dim=-1)
        spa = self.one_hot(bs, num_joints, step).permute(0, 3, 2, 1).to(input.device)
        tem = self.one_hot(bs, step, num_joints).permute(0, 3, 1, 2).to(input.device)
        tem1 = self.tem_embed(tem)
        spa1 = self.spa_embed(spa)
        streams = []
        if self.use_position_stream:
            streams.append(self.joint_embed(input))
        if self.use_velocity_stream:
            streams.append(self.dif_embed(vel))
        if self.use_acceleration_stream:
            streams.append(self.acc_embed(acc))
        if self.use_angular_velocity_stream:
            streams.append(self.ang_embed(ang))
        if len(streams) == 0:
            streams.append(self.dif_embed(vel))
        dy = sum(streams) / len(streams)
        sta = input.mean(dim=-1, keepdim=True).expand(-1, -1, -1, step).contiguous()
        sta = self.sta_joint_embed(sta)
        dyn_feat = self._encode_branch(dy, spa1, tem1, self.compute_g_dyn, self.gcn_dyn1, self.gcn_dyn2, self.gcn_dyn3, self.cnn_dyn)
        sta_feat = self._encode_branch(sta, spa1, tem1, self.compute_g_sta, self.gcn_sta1, self.gcn_sta2, self.gcn_sta3, self.cnn_sta)
        id_logits = self.fc(dyn_feat)

        if (not self.disentangle) or (target is None):
            return id_logits

        bone_target = self._extract_bone_length_target(input)
        sta_pred = self.sta_proxy_head(sta_feat)
        sta_proxy_loss = self.mse(sta_pred, bone_target)
        dyn_adv_in = grad_reverse(dyn_feat, self.grl_lambda)
        dyn_adv_pred = self.dyn_adv_head(dyn_adv_in)
        dyn_adv_loss = self.mse(dyn_adv_pred, bone_target)

        perm = torch.randperm(bs, device=input.device)
        swap_sta_feat = sta_feat[perm]
        swap_input = torch.cat([dyn_feat, swap_sta_feat], dim=1)
        swap_logits = self.swap_id_head(swap_input)
        swap_id_loss = nn.functional.cross_entropy(swap_logits, target)
        swap_consistency_loss = self.kld(
            nn.functional.log_softmax(swap_logits, dim=1),
            nn.functional.softmax(id_logits.detach(), dim=1)
        )

        return {
            'id_logits': id_logits,
            'sta_proxy_loss': sta_proxy_loss,
            'dyn_adv_loss': dyn_adv_loss,
            'swap_consistency_loss': swap_consistency_loss,
            'swap_id_loss': swap_id_loss,
            'swap_logits': swap_logits,
        }

    def _encode_branch(self, feat, spa1, tem1, compute_g, gcn1, gcn2, gcn3, cnn):
        feat = torch.cat([feat, spa1], 1)
        g = compute_g(feat)
        feat = gcn1(feat, g)
        feat = gcn2(feat, g)
        feat = gcn3(feat, g)
        feat = feat + tem1
        feat = cnn(feat)
        feat = self.maxpool(feat)
        feat = torch.flatten(feat, 1)
        return feat

    def _extract_bone_length_target(self, input):
        lengths = []
        for parent, child in NTU_BONES:
            vec = input[:, :, child, :] - input[:, :, parent, :]
            lengths.append(torch.linalg.norm(vec, dim=1).mean(dim=-1))
        lengths = torch.stack(lengths, dim=1)
        return lengths

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
    
