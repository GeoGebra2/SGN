import argparse
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
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
    ) -> None:
        self.x = x
        self.y = y.astype(np.int64)
        self.aid = None if aid is None else aid.astype(np.int64)
        self.seg = seg
        self.pairs_per_epoch = pairs_per_epoch
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.hard_negative_ratio = float(np.clip(hard_negative_ratio, 0.0, 1.0))

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
        same_pair = int(self.rng.integers(0, 2))

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
    def __init__(self, encoder: SGN, hidden_dim: int = 512, dropout: float = 0.5) -> None:
        super().__init__()
        self.encoder = encoder
        feat_dim = int(self.encoder.fc.in_features)
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
        f1 = self.encoder.forward_features(x1)
        f2 = self.encoder.forward_features(x2)
        fused = torch.cat([f1, f2, torch.abs(f1 - f2), f1 * f2], dim=1)
        logits = self.head(fused).squeeze(1)
        return logits


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
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
    auc, eer = compute_auc_eer(labels, probs)
    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc, "eer": eer}


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

    auc = float(np.trapz(tpr, fpr))
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
        return float("nan"), {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0, "eer": 1.0}

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits, labels)
    return total_loss / max(1, total_seen), metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
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
        return float("nan"), {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0, "eer": 1.0}

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits, labels)
    return total_loss / max(1, total_seen), metrics


def save_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer, epoch: int, best_f1: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_f1": best_f1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_f1", 0.0))


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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train", type=int, default=1, choices=[0, 1])
    parser.add_argument("--train-pairs", type=int, default=32000)
    parser.add_argument("--val-pairs", type=int, default=8000)
    parser.add_argument("--test-pairs", type=int, default=12000)
    parser.add_argument("--save-dir", type=str, default="./results/NTU_ID/SGN_PAIR_A")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--motion-only", action="store_true")
    parser.add_argument(
        "--hard-negative-ratio",
        type=float,
        default=0.5,
        help="负样本中采用 hard negative（同动作不同身份）的比例，范围[0,1]",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, num_classes = build_pair_loaders(args)
    encoder = SGN(num_classes=num_classes, dataset="NTU_ID", seg=args.seg, args=args)
    model = SGNPairMatcher(encoder=encoder).to(device)

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    best_f1 = 0.0
    best_path = os.path.join(args.save_dir, f"case{args.case}_best.pth")

    if args.resume:
        start_epoch, best_f1 = load_checkpoint(args.resume, model, optimizer, device)
        print(f"Resume from {args.resume}, epoch={start_epoch}, best_f1={best_f1:.4f}")

    if args.train == 1:
        for epoch in range(start_epoch, args.epochs):
            tr_loss, tr_m = run_epoch(model, train_loader, criterion, optimizer, device)
            va_loss, va_m = evaluate(model, val_loader, criterion, device)
            print(
                f"Epoch {epoch + 1:03d} | "
                f"Train loss {tr_loss:.4f} acc {tr_m['acc']:.4f} f1 {tr_m['f1']:.4f} auc {tr_m['auc']:.4f} eer {tr_m['eer']:.4f} | "
                f"Val loss {va_loss:.4f} acc {va_m['acc']:.4f} f1 {va_m['f1']:.4f} auc {va_m['auc']:.4f} eer {va_m['eer']:.4f}"
            )
            if va_m["f1"] >= best_f1:
                best_f1 = va_m["f1"]
                save_checkpoint(best_path, model, optimizer, epoch + 1, best_f1)
                print(f"Saved best checkpoint to {best_path} (f1={best_f1:.4f})")

    if os.path.exists(best_path):
        load_checkpoint(best_path, model, optimizer=None, device=device)
        print(f"Loaded best checkpoint: {best_path}")

    te_loss, te_m = evaluate(model, test_loader, criterion, device)
    print(
        f"Test  | loss {te_loss:.4f} acc {te_m['acc']:.4f} "
        f"precision {te_m['precision']:.4f} recall {te_m['recall']:.4f} "
        f"f1 {te_m['f1']:.4f} auc {te_m['auc']:.4f} eer {te_m['eer']:.4f}"
    )


if __name__ == "__main__":
    main()
