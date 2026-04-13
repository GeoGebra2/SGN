# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from torch import nn
import torch
import math

class PrototypeDecomposer(nn.Module):
    def __init__(self, num_joints, n_prototype=100, dropout=0.1):
        super(PrototypeDecomposer, self).__init__()
        self.num_joints = num_joints
        dim = num_joints * num_joints
        self.query = nn.Linear(dim, n_prototype, bias=False)
        self.memory = nn.Linear(n_prototype, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, topology):
        bs = topology.size(0)
        flat = topology.view(bs, -1)
        weights = torch.softmax(self.query(flat), dim=-1)
        reconstructed = self.memory(weights)
        reconstructed = self.dropout(reconstructed).view(bs, self.num_joints, self.num_joints)
        return reconstructed, weights

class SGN(nn.Module):
    def __init__(self, num_classes, dataset, seg, args, bias = True):
        super(SGN, self).__init__()

        self.dim1 = 256
        self.dataset = dataset
        self.seg = seg
        self.motion_only = getattr(args, "motion_only", False)
        self.proto_decompose = getattr(args, "proto_decompose", False)
        self.part_decompose = getattr(args, "part_decompose", False)
        self.part_sig_threshold = getattr(args, "part_sig_threshold", 0.5)
        self.part_fuse_weight = getattr(args, "part_fuse_weight", 1.0)
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
        if self.part_decompose:
            self.part_groups = self._build_ntu_five_parts()
            self.part_branches = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.dim1, self.dim1),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(self.dim1, self.dim1 * 2),
                    nn.ReLU(),
                )
                for _ in self.part_groups
            ])
            self.part_classifiers = nn.ModuleList([
                nn.Linear(self.dim1 * 2, num_classes) for _ in self.part_groups
            ])
        else:
            self.part_groups = []
            self.part_branches = None
            self.part_classifiers = None
        if self.proto_decompose:
            proto_num = getattr(args, "proto_num", 100)
            proto_dropout = getattr(args, "proto_dropout", 0.1)
            self.prototype_decomposer = PrototypeDecomposer(num_joint, n_prototype=proto_num, dropout=proto_dropout)
        else:
            self.prototype_decomposer = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)


    def forward_features(self, input, return_aux=False):
        
        # Dynamic Representation
        bs, step, dim = input.size()
        raw_input = input
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
        joint_level_feat = input
        input = input + tem1
        input = self.cnn(input)
        input = self.temporal(input)
        input = self.temporal_att(input)
        # Classification
        output = self.maxpool(input)
        output = torch.flatten(output, 1)
        if self.part_decompose and self.part_branches is not None:
            part_scores = self._compute_part_scores(input_raw=raw_input, num_joints=num_joints)
            part_mask = self._build_significant_mask(part_scores)
            fused_part_feat, part_logits, branch_fused_logits = self._encode_part_features(
                joint_level_feat, part_scores, part_mask
            )
            output = output + self.part_fuse_weight * fused_part_feat
        if return_aux:
            aux = {}
            if self.part_decompose and self.part_branches is not None:
                aux.update({
                    "part_scores": part_scores,
                    "part_mask": part_mask,
                    "part_logits": part_logits,
                    "branch_fused_logits": branch_fused_logits,
                })
            if self.proto_decompose and self.prototype_decomposer is not None:
                topology = g.mean(dim=1)
                reconstructed_topology, prototype_weights = self.prototype_decomposer(topology)
                aux.update({
                    "topology": topology,
                    "reconstructed_topology": reconstructed_topology,
                    "prototype_weights": prototype_weights,
                })
                return output, aux
            return output, aux
        return output

    def forward(self, input):
        output = self.forward_features(input)
        output = self.fc(output)
        return output

    def forward_with_aux(self, input):
        feats, aux = self.forward_features(input, return_aux=True)
        output = self.fc(feats)
        return output, feats, aux

    def one_hot(self, bs, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot

    def _build_ntu_five_parts(self):
        # NTU-25 indices in zero-based format:
        # head/trunk, left arm, right arm, left leg, right leg
        return [
            [0, 1, 2, 3, 20],
            [4, 5, 6, 7, 21, 22],
            [8, 9, 10, 11, 23, 24],
            [12, 13, 14, 15],
            [16, 17, 18, 19],
        ]

    def _compute_part_scores(self, input_raw, num_joints):
        # FineTec-style significance:
        # D_avg(i) = mean_t ||S_{t+1,i} - S_{t,i}||_2
        bs, step, _ = input_raw.size()
        skeleton = input_raw.view(bs, step, num_joints, 3)
        disp = skeleton[:, 1:, :, :] - skeleton[:, :-1, :, :]
        joint_motion = torch.norm(disp, dim=-1).mean(dim=1)
        part_scores = []
        for group in self.part_groups:
            part_scores.append(joint_motion[:, group].mean(dim=1))
        return torch.stack(part_scores, dim=1)

    def _build_significant_mask(self, part_scores):
        max_score = part_scores.max(dim=1, keepdim=True)[0]
        normalized_scores = part_scores / (max_score + 1e-6)
        mask = normalized_scores >= self.part_sig_threshold
        # Guarantee at least one active local branch for each sample.
        empty_mask = mask.sum(dim=1) == 0
        if empty_mask.any():
            top_idx = normalized_scores.argmax(dim=1, keepdim=True)
            mask[empty_mask] = False
            mask.scatter_(1, top_idx, True)
        return mask

    def _encode_part_features(self, joint_level_feat, part_scores, part_mask):
        # joint_level_feat: [bs, C, V, T]
        bs = joint_level_feat.size(0)
        out_dim = self.dim1 * 2
        fused_part_feat = joint_level_feat.new_zeros(bs, out_dim)
        part_logits = joint_level_feat.new_zeros(bs, len(self.part_groups), self.fc.out_features)
        weighted_scores = part_scores * part_mask.float()
        weighted_scores = weighted_scores / (weighted_scores.sum(dim=1, keepdim=True) + 1e-6)

        # True sparse routing: only compute branch features for active samples.
        for idx, group in enumerate(self.part_groups):
            active_idx = torch.nonzero(part_mask[:, idx], as_tuple=False).squeeze(1)
            if active_idx.numel() == 0:
                continue
            part_joint_feat = joint_level_feat[active_idx][:, :, group, :]
            pooled = part_joint_feat.mean(dim=(2, 3))
            part_feat = self.part_branches[idx](pooled)
            part_logits[active_idx, idx, :] = self.part_classifiers[idx](part_feat)
            fused_part_feat[active_idx] += weighted_scores[active_idx, idx:idx + 1] * part_feat
        branch_fused_logits = (part_logits * weighted_scores.unsqueeze(-1)).sum(dim=1)
        return fused_part_feat, part_logits, branch_fused_logits

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
    
