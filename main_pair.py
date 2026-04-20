import argparse
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from data import NTUDataLoaders, turn_two_to_one
from model import SGN


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_split(
    x: np.ndarray,
    y: np.ndarray,
    aid: Optional[np.ndarray],
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    valid_idx: List[int] = []
    invalid_nan_inf = 0
    invalid_empty = 0

    for i in range(len(y)):
        seq = np.asarray(x[i])
        if seq.ndim != 2:
            invalid_empty += 1
            continue
        if not np.isfinite(seq).all():
            invalid_nan_inf += 1
            continue
        # 至少有一个非零帧，避免纯空样本进入训练
        if np.all(seq == 0):
            invalid_empty += 1
            continue
        valid_idx.append(i)

    if len(valid_idx) == 0:
        raise ValueError(f"{split_name} split has no valid samples after sanitization.")

    x_clean = x[valid_idx]
    y_clean = y[valid_idx]
    aid_clean = aid[valid_idx] if aid is not None else None

    print(
        f"[Sanitize:{split_name}] total={len(y)} valid={len(valid_idx)} "
        f"drop_nan_inf={invalid_nan_inf} drop_empty={invalid_empty}"
    )
    return x_clean, y_clean, aid_clean


class SGNPairDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        aid: Optional[np.ndarray],
        seg: int,
        pairs_per_epoch: int,
        train: bool = True,
        seed: int = 1337,
        hard_negative_ratio: float = 0.0,
        neg_pos_ratio: float = 1.0,
    ) -> None:
        self.x = x
        self.y = y.astype(np.int64)
        self.aid = None if aid is None else aid.astype(np.int64)
        self.seg = seg
        self.pairs_per_epoch = pairs_per_epoch
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.hard_negative_ratio = float(np.clip(hard_negative_ratio, 0.0, 1.0))
        self.neg_pos_ratio = max(0.0, float(neg_pos_ratio))

        self.label_to_indices: Dict[int, np.ndarray] = {}
        for label in np.unique(self.y):
            self.label_to_indices[int(label)] = np.where(self.y == label)[0]

        self.labels = np.array(sorted(self.label_to_indices.keys()), dtype=np.int64)
        self.same_capable_labels = np.array(
            [k for k, v in self.label_to_indices.items() if len(v) >= 2], dtype=np.int64
        )
        self.action_to_label_indices: Dict[int, Dict[int, np.ndarray]] = {}
        if self.aid is not None:
            action_tmp: Dict[int, Dict[int, List[int]]] = {}
            for idx in range(len(self.y)):
                action = int(self.aid[idx])
                label = int(self.y[idx])
                if action not in action_tmp:
                    action_tmp[action] = {}
                if label not in action_tmp[action]:
                    action_tmp[action][label] = []
                action_tmp[action][label].append(idx)
            for action, label_dict in action_tmp.items():
                self.action_to_label_indices[action] = {
                    label: np.array(indices, dtype=np.int64)
                    for label, indices in label_dict.items()
                }

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def __getitem__(self, idx: int):
        del idx
        anchor_idx = int(self.rng.integers(0, len(self.y)))
        anchor_label = int(self.y[anchor_idx])
        pos_prob = 1.0 / (1.0 + self.neg_pos_ratio)
        same_pair = int(float(self.rng.random()) < pos_prob)

        if same_pair == 1 and anchor_label in self.same_capable_labels:
            candidates = self.label_to_indices[anchor_label]
            pair_idx = int(anchor_idx)
            while pair_idx == anchor_idx:
                pair_idx = int(candidates[self.rng.integers(0, len(candidates))])
        else:
            same_pair = 0
            pair_idx = self._sample_negative(anchor_idx, anchor_label)

        x1 = self._preprocess_single(self.x[anchor_idx])
        x2 = self._preprocess_single(self.x[pair_idx])
        y = torch.tensor(float(same_pair), dtype=torch.float32)
        return x1, x2, y

    def _preprocess_single(self, seq: np.ndarray) -> torch.Tensor:
        seq = np.asarray(seq, dtype=np.float32)
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        if seq.ndim != 2:
            seq = np.zeros((self.seg, 150), dtype=np.float32)

        keep = ~(np.all(seq == 0, axis=1))
        seq = seq[keep]
        if seq.shape[0] == 0:
            seq = np.zeros((1, 150), dtype=np.float32)

        seq = turn_two_to_one(seq)
        if seq.shape[0] == 0:
            seq = np.zeros((1, 75), dtype=np.float32)

        if seq.shape[0] < self.seg:
            pad_num = self.seg - seq.shape[0]
            last = seq[-1:, :]
            seq = np.concatenate([seq, np.repeat(last, pad_num, axis=0)], axis=0)

        frames = seq.shape[0]
        avg_duration = max(1, frames // self.seg)
        if self.train:
            offsets = np.arange(self.seg) * avg_duration + self.rng.integers(
                0, avg_duration, size=self.seg
            )
        else:
            offsets = np.arange(self.seg) * avg_duration + (avg_duration // 2)
        offsets = np.clip(offsets, 0, frames - 1)
        seq = seq[offsets].astype(np.float32)
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(seq)

    def _sample_negative(self, anchor_idx: int, anchor_label: int) -> int:
        if self.aid is not None and self.hard_negative_ratio > 0.0:
            if float(self.rng.random()) < self.hard_negative_ratio:
                action = int(self.aid[anchor_idx])
                label_dict = self.action_to_label_indices.get(action, {})
                hard_neg_labels = [k for k in label_dict.keys() if k != anchor_label]
                if hard_neg_labels:
                    neg_label = int(hard_neg_labels[self.rng.integers(0, len(hard_neg_labels))])
                    candidates = label_dict[neg_label]
                    return int(candidates[self.rng.integers(0, len(candidates))])

        neg_label = int(anchor_label)
        while neg_label == anchor_label:
            neg_label = int(self.labels[self.rng.integers(0, len(self.labels))])
        candidates = self.label_to_indices[neg_label]
        return int(candidates[self.rng.integers(0, len(candidates))])


class SGNPairMatcher(nn.Module):
    def __init__(
        self,
        encoder_q: SGN,
        encoder_r: Optional[SGN] = None,
        hidden_dim: int = 512,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder_q = encoder_q
        self.encoder_r = encoder_q if encoder_r is None else encoder_r
        self.shared = self.encoder_q is self.encoder_r
        feat_dim = int(self.encoder_q.fc.in_features)
        in_dim = feat_dim * 4
        hidden2 = max(128, hidden_dim // 2)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        f1 = self.encoder_q.forward_features(x1)
        f2 = self.encoder_r.forward_features(x2)
        fused = torch.cat([f1, f2, torch.abs(f1 - f2), f1 * f2], dim=1)
        logits = self.head(fused).squeeze(1)
        return logits


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_weight = alpha_t * torch.pow((1.0 - pt).clamp(min=1e-8), self.gamma)
        loss = focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > float(threshold)).float()
    labels = labels.float()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    total = tp + tn + fp + fn

    acc = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    beta = 0.5
    beta2 = beta * beta
    f05 = (1.0 + beta2) * precision * recall / max(1e-12, beta2 * precision + recall)
    pred_pos_rate = preds.mean().item() if preds.numel() > 0 else 0.0
    auc, eer = compute_auc_eer(labels, probs)
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f0.5": f05,
        "pred_pos_rate": pred_pos_rate,
        "auc": auc,
        "eer": eer,
    }


def tune_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    objective: str = "f0.5",
    min_thr: float = 0.5,
    max_thr: float = 0.99,
    steps: int = 81,
    target_precision: float = 0.7,
) -> Tuple[float, Dict[str, float]]:
    steps = max(2, int(steps))
    best_thr = float(min_thr)
    best_m = compute_metrics(logits, labels, threshold=best_thr)
    best_score = _threshold_score(best_m, objective=objective, target_precision=target_precision)
    for thr in np.linspace(min_thr, max_thr, steps):
        m = compute_metrics(logits, labels, threshold=float(thr))
        score = _threshold_score(m, objective=objective, target_precision=target_precision)
        if score > best_score:
            best_m = m
            best_thr = float(thr)
            best_score = score
    return best_thr, best_m


def _threshold_score(metrics: Dict[str, float], objective: str, target_precision: float) -> Tuple[float, float, float, float]:
    if objective == "precision_priority":
        feasible = 1.0 if metrics["precision"] >= float(target_precision) else 0.0
        return (feasible, metrics["f0.5"] if feasible > 0 else metrics["precision"], metrics["precision"], metrics["f1"])
    if objective == "f1":
        return (metrics["f1"], metrics["precision"], -metrics["recall"], -metrics["pred_pos_rate"])
    return (metrics["f0.5"], metrics["precision"], -metrics["recall"], -metrics["pred_pos_rate"])


def compute_auc_eer(labels: torch.Tensor, probs: torch.Tensor) -> Tuple[float, float]:
    y = labels.detach().cpu().numpy().astype(np.int64)
    s = probs.detach().cpu().numpy().astype(np.float64)
    if y.size == 0:
        return 0.0, 1.0
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return 0.0, 1.0

    order = np.argsort(-s)
    y_sorted = y[order]
    s_sorted = s[order]

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    distinct = np.where(np.diff(s_sorted))[0]
    thresh_idx = np.r_[distinct, y_sorted.size - 1]

    tpr = tps[thresh_idx] / max(1, pos)
    fpr = fps[thresh_idx] / max(1, neg)

    tpr = np.r_[0.0, tpr, 1.0]
    fpr = np.r_[0.0, fpr, 1.0]

    auc = float(np.trapezoid(tpr, fpr))
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fnr - fpr)))
    eer = float((fnr[idx] + fpr[idx]) * 0.5)
    return auc, eer


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_seen = 0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    skipped_batches = 0

    for x1, x2, labels in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if (not torch.isfinite(x1).all()) or (not torch.isfinite(x2).all()) or (not torch.isfinite(labels).all()):
            skipped_batches += 1
            continue

        logits = model(x1, x2)
        if not torch.isfinite(logits).all():
            skipped_batches += 1
            continue
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            skipped_batches += 1
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_seen += labels.size(0)
        all_logits.append(logits.detach())
        all_labels.append(labels.detach())

    if skipped_batches > 0:
        print(f"[Warn] skipped {skipped_batches} non-finite batches during training")
    if len(all_logits) == 0:
        return float("nan"), {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "f0.5": 0.0, "pred_pos_rate": 0.0, "auc": 0.0, "eer": 1.0}

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits, labels, threshold=threshold)
    return total_loss / max(1, total_seen), metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    return_outputs: bool = False,
) -> Tuple[float, Dict[str, float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    skipped_batches = 0

    for x1, x2, labels in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if (not torch.isfinite(x1).all()) or (not torch.isfinite(x2).all()) or (not torch.isfinite(labels).all()):
            skipped_batches += 1
            continue

        logits = model(x1, x2)
        if not torch.isfinite(logits).all():
            skipped_batches += 1
            continue
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            skipped_batches += 1
            continue

        total_loss += loss.item() * labels.size(0)
        total_seen += labels.size(0)
        all_logits.append(logits)
        all_labels.append(labels)

    if skipped_batches > 0:
        print(f"[Warn] skipped {skipped_batches} non-finite batches during eval")
    if len(all_logits) == 0:
        empty_m = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "f0.5": 0.0, "pred_pos_rate": 0.0, "auc": 0.0, "eer": 1.0}
        if return_outputs:
            return float("nan"), empty_m, None, None
        return float("nan"), empty_m, None, None

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits, labels, threshold=threshold)
    if return_outputs:
        return total_loss / max(1, total_seen), metrics, logits, labels
    return total_loss / max(1, total_seen), metrics, None, None


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_score: float,
    best_threshold: float,
    best_metric: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_score": best_score,
            "best_threshold": best_threshold,
            "best_metric": best_metric,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> Tuple[int, float, float, str]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    best_score = float(ckpt.get("best_score", ckpt.get("best_f1", 0.0)))
    best_metric = str(ckpt.get("best_metric", "f1"))
    return int(ckpt.get("epoch", 0)), best_score, float(ckpt.get("best_threshold", 0.5)), best_metric


def build_pair_loaders(args) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    ntu = NTUDataLoaders(dataset="NTU_ID", case=args.case, seg=args.seg, aug=0, args=args)
    train_x, train_y, train_aid = sanitize_split(ntu.train_X, ntu.train_Y, ntu.train_aid, "train")
    val_x, val_y, val_aid = sanitize_split(ntu.val_X, ntu.val_Y, ntu.val_aid, "val")
    test_x, test_y, test_aid = sanitize_split(ntu.test_X, ntu.test_Y, ntu.test_aid, "test")
    train_set = SGNPairDataset(
        train_x,
        train_y,
        aid=train_aid,
        seg=args.seg,
        pairs_per_epoch=args.train_pairs,
        train=True,
        seed=args.seed,
        hard_negative_ratio=args.hard_negative_ratio,
        neg_pos_ratio=args.neg_pos_ratio,
    )
    val_set = SGNPairDataset(
        val_x,
        val_y,
        aid=val_aid,
        seg=args.seg,
        pairs_per_epoch=args.val_pairs,
        train=False,
        seed=args.seed + 1,
        hard_negative_ratio=args.hard_negative_ratio,
        neg_pos_ratio=args.neg_pos_ratio,
    )
    test_set = SGNPairDataset(
        test_x,
        test_y,
        aid=test_aid,
        seg=args.seg,
        pairs_per_epoch=args.test_pairs,
        train=False,
        seed=args.seed + 2,
        hard_negative_ratio=args.hard_negative_ratio,
        neg_pos_ratio=args.neg_pos_ratio,
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False
    )
    return train_loader, val_loader, test_loader, int(ntu.num_classes)


def parse_args():
    parser = argparse.ArgumentParser(description="SGN 共享编码器配对识别 (方案A)")
    parser.add_argument("--case", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--seg", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-type", type=str, default="bce", choices=["bce", "bce_pos", "focal"])
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=1.0,
        help="用于 bce_pos；>1 提高正类权重，<1 降低正类权重（可抑制过度报正）",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25,
        help="用于 focal；正类权重系数，过度报正可尝试更小值如 0.15",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="用于 focal；难样本聚焦强度")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train", type=int, default=1, choices=[0, 1])
    parser.add_argument("--train-pairs", type=int, default=32000)
    parser.add_argument("--val-pairs", type=int, default=8000)
    parser.add_argument("--test-pairs", type=int, default=12000)
    parser.add_argument("--save-dir", type=str, default="./results/NTU_ID/SGN_PAIR_A")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--motion-only", action="store_true")
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--thresh-objective", type=str, default="f0.5", choices=["f1", "f0.5", "precision_priority"])
    parser.add_argument("--target-precision", type=float, default=0.7)
    parser.add_argument("--thresh-min", type=float, default=0.5)
    parser.add_argument("--thresh-max", type=float, default=0.99)
    parser.add_argument("--thresh-steps", type=int, default=81)
    parser.add_argument(
        "--pair-arch",
        type=str,
        default="B",
        choices=["A", "B"],
        help="A=共享双分支, B=双塔不共享（LAN风格）",
    )
    parser.add_argument(
        "--hard-negative-ratio",
        type=float,
        default=0.5,
        help="负样本中采用 hard negative（同动作不同身份）的比例，范围[0,1]",
    )
    parser.add_argument("--neg-pos-ratio", type=float, default=3.0, help="负正样本比例 K，表示 1:K")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, num_classes = build_pair_loaders(args)
    encoder_q = SGN(num_classes=num_classes, dataset="NTU_ID", seg=args.seg, args=args)
    if args.pair_arch == "B":
        encoder_r = SGN(num_classes=num_classes, dataset="NTU_ID", seg=args.seg, args=args)
    else:
        encoder_r = None
    model = SGNPairMatcher(encoder_q=encoder_q, encoder_r=encoder_r).to(device)
    print(f"Pair architecture: {args.pair_arch} ({'non-shared' if args.pair_arch == 'B' else 'shared'})")

    if args.loss_type == "bce_pos":
        pos_weight = torch.tensor([float(args.pos_weight)], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    elif args.loss_type == "focal":
        criterion = BinaryFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, reduction="mean").to(device)
    else:
        criterion = nn.BCEWithLogitsLoss().to(device)
    print(f"Loss: {args.loss_type}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    best_score = 0.0
    best_metric = str(args.thresh_objective)
    best_threshold = float(args.decision_threshold)
    best_path = os.path.join(args.save_dir, f"case{args.case}_best.pth")

    if args.resume:
        start_epoch, best_score, best_threshold, best_metric = load_checkpoint(args.resume, model, optimizer, device)
        print(
            f"Resume from {args.resume}, epoch={start_epoch}, best_{best_metric}={best_score:.4f}, best_thr={best_threshold:.3f}"
        )

    if args.train == 1:
        for epoch in range(start_epoch, args.epochs):
            tr_loss, tr_m = run_epoch(
                model, train_loader, criterion, optimizer, device, threshold=args.decision_threshold
            )
            va_loss, va_m, va_logits, va_labels = evaluate(
                model, val_loader, criterion, device, threshold=args.decision_threshold, return_outputs=True
            )
            if va_logits is not None and va_labels is not None:
                tuned_thr, va_m_tuned = tune_threshold(
                    va_logits,
                    va_labels,
                    objective=args.thresh_objective,
                    min_thr=args.thresh_min,
                    max_thr=args.thresh_max,
                    steps=args.thresh_steps,
                    target_precision=args.target_precision,
                )
            else:
                tuned_thr, va_m_tuned = float(args.decision_threshold), va_m
            current_score = float(va_m_tuned[args.thresh_objective])
            print(
                f"Epoch {epoch + 1:03d} | "
                f"Train loss {tr_loss:.4f} acc {tr_m['acc']:.4f} precision {tr_m['precision']:.4f} recall {tr_m['recall']:.4f} "
                f"f1 {tr_m['f1']:.4f} f0.5 {tr_m['f0.5']:.4f} pos {tr_m['pred_pos_rate']:.4f} auc {tr_m['auc']:.4f} eer {tr_m['eer']:.4f} | "
                f"Val loss {va_loss:.4f} acc {va_m_tuned['acc']:.4f} precision {va_m_tuned['precision']:.4f} recall {va_m_tuned['recall']:.4f} "
                f"f1 {va_m_tuned['f1']:.4f} f0.5 {va_m_tuned['f0.5']:.4f} pos {va_m_tuned['pred_pos_rate']:.4f} "
                f"auc {va_m_tuned['auc']:.4f} eer {va_m_tuned['eer']:.4f} thr {tuned_thr:.3f} sel {args.thresh_objective}={current_score:.4f}"
            )
            if current_score >= best_score:
                best_score = current_score
                best_metric = str(args.thresh_objective)
                best_threshold = tuned_thr
                save_checkpoint(best_path, model, optimizer, epoch + 1, best_score, best_threshold, best_metric)
                print(
                    f"Saved best checkpoint to {best_path} ({best_metric}={best_score:.4f}, thr={best_threshold:.3f})"
                )

    if os.path.exists(best_path):
        _, best_score, best_threshold, best_metric = load_checkpoint(best_path, model, optimizer=None, device=device)
        print(f"Loaded best checkpoint: {best_path} ({best_metric}={best_score:.4f}, thr={best_threshold:.3f})")

    if args.train == 0:
        _, va_m, va_logits, va_labels = evaluate(
            model, val_loader, criterion, device, threshold=args.decision_threshold, return_outputs=True
        )
        if va_logits is not None and va_labels is not None:
            best_threshold, _ = tune_threshold(
                va_logits,
                va_labels,
                objective=args.thresh_objective,
                min_thr=args.thresh_min,
                max_thr=args.thresh_max,
                steps=args.thresh_steps,
                target_precision=args.target_precision,
            )

    te_loss, te_m, _, _ = evaluate(model, test_loader, criterion, device, threshold=best_threshold, return_outputs=False)
    print(
        f"Test  | loss {te_loss:.4f} acc {te_m['acc']:.4f} "
        f"precision {te_m['precision']:.4f} recall {te_m['recall']:.4f} "
        f"f1 {te_m['f1']:.4f} auc {te_m['auc']:.4f} eer {te_m['eer']:.4f} thr {best_threshold:.3f}"
    )


if __name__ == "__main__":
    main()
