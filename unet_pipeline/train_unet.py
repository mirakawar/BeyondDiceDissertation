#!/usr/bin/env python3
"""
train_unet.py
-------------
Trains the custom U-Net.

This script trains on nnU-Net preprocessed `.b2nd` tensors so the U-Net sees
the same z-score normalised train/val inputs as nnU-Net. Test inference is run
separately through `predict_unet.py`, because the held-out test split is kept as
raw TIFF inputs in the evaluation pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

WORKER_BASE_SEED = 0
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DEFAULT = REPO_ROOT / "config.json"

def _progress(iterable, **kwargs):
    return iterable

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_b2nd(case_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load nnU-Net .b2nd file pair → (image H×W float32, mask H×W float32).
    Files are already z-score normalised by nnU-Net preprocessing.
    Shape on disk: (1, 1, H, W).
    """
    import blosc2
    img  = np.array(blosc2.open(str(case_path))[0, 0], dtype=np.float32)
    seg  = case_path.parent / (case_path.stem + "_seg.b2nd")
    mask = (np.array(blosc2.open(str(seg))[0, 0]) > 0).astype(np.float32)
    return img, mask

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)
    base_dir = config_path.resolve().parent
    for key, value in cfg.get("paths", {}).items():
        if isinstance(value, str):
            path = Path(value)
            if not path.is_absolute():
                cfg["paths"][key] = str(base_dir / path)
    return cfg

def load_fold0_split(split_path: Path) -> Tuple[List[str], List[str]]:
    if split_path.suffix == ".json":
        with open(split_path) as f:
            splits = json.load(f)
    elif split_path.suffix == ".pkl":
        with open(split_path, "rb") as f:
            splits = pickle.load(f)
    else:
        raise RuntimeError(f"Unsupported split file format: {split_path}")

    if not isinstance(splits, list) or not splits:
        raise RuntimeError(f"Invalid split file: {split_path}")

    fold0 = splits[0]
    if "train" not in fold0 or "val" not in fold0:
        raise RuntimeError(f"Fold 0 in {split_path} is missing train/val keys")

    return list(fold0["train"]), list(fold0["val"])

def load_splits(config_path: Path) -> Tuple[List[str], List[str], List[str], Path, Path]:
    """Return (train_ids, val_ids, test_ids, preproc_dir, split_source).

    Train/val: prefer the same splits_final.json that nnU-Net v2 trains with.
    Test:      nnunet_id_map.json entries where split=='test'.
    """
    cfg = load_config(config_path)

    preproc_dataset_dir = Path(cfg["paths"]["nnunet_preproc"]) / "Dataset001_Mitochondria"
    raw_dataset_dir = Path(cfg["paths"]["nnunet_raw"]) / "Dataset001_Mitochondria"
    preproc_dir = (Path(cfg["paths"]["nnunet_preproc"])
                   / "Dataset001_Mitochondria" / "nnUNetPlans_2d")

    candidate_paths = [
        preproc_dataset_dir / "splits_final.json",
        raw_dataset_dir / "splits_final.json",
        preproc_dataset_dir / "splits_final.pkl",
        raw_dataset_dir / "splits_final.pkl",
    ]
    split_source = next((p for p in candidate_paths if p.exists()), None)
    if split_source is None:
        raise FileNotFoundError(
            "No splits_final.json or splits_final.pkl found in nnUNet raw/preprocessed folders."
        )

    train_all, val_all = load_fold0_split(split_source)

    raw_json = raw_dataset_dir / "splits_final.json"
    preproc_json = preproc_dataset_dir / "splits_final.json"
    if split_source == preproc_json and raw_json.exists():
        raw_train, raw_val = load_fold0_split(raw_json)
        if raw_train != train_all or raw_val != val_all:
            raise RuntimeError(
                f"Split mismatch between {preproc_json} and {raw_json}. "
                "Re-run nnunet_preprocessing/extract_config.py so U-Net and nnU-Net stay aligned."
            )

    train_ids = [c for c in train_all if (preproc_dir / f"{c}.b2nd").exists()]
    val_ids   = [c for c in val_all   if (preproc_dir / f"{c}.b2nd").exists()]

    id_map_path = Path(cfg["paths"]["nnunet_id_map"])
    with open(id_map_path) as f:
        id_map = json.load(f)

    test_ids = [
        cid for cid, meta in id_map.items()
        if meta.get("split") == "test"
        and (preproc_dir / f"{cid}.b2nd").exists()
    ]

    return train_ids, val_ids, test_ids, preproc_dir, split_source

def get_checkpoint_paths(config_path: Path) -> Tuple[Path, Path]:
    cfg = load_config(config_path)
    checkpoint_dir = Path(cfg["paths"]["unet_checkpoint"])
    return checkpoint_dir / "best_model.pt", checkpoint_dir / "last_checkpoint.pt"

def compute_train_pos_weight(config_path: Path) -> float:
    cfg = load_config(config_path)
    splits_path = Path(cfg["paths"]["splits_json"])
    stats_path = Path(cfg["paths"]["dataset_stats"])

    splits = json.loads(splits_path.read_text())
    train_keys = {(entry["sample"], entry["filename"]) for entry in splits["train"]}

    fg_pixels = 0
    bg_pixels = 0
    with stats_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["sample"], row["filename"]) not in train_keys:
                continue
            fg_pixels += int(row["fg_pixels"])
            bg_pixels += int(row["bg_pixels"])

    if fg_pixels <= 0:
        raise RuntimeError("Computed zero foreground pixels for the active train split.")
    return bg_pixels / fg_pixels

class B2ndDataset(Dataset):
    """Loads nnU-Net preprocessed .b2nd files.
    Images are already z-score normalised — no further normalisation applied.
    Augmentation is applied only on the training split.
    """

    def __init__(
        self,
        case_ids: List[str],
        preproc_dir: Path,
        pad_factor: int = 32,
        augment: bool = False,
    ) -> None:
        self.case_ids   = case_ids
        self.preproc_dir = Path(preproc_dir)
        self.pad_factor  = pad_factor
        self.augment     = augment

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict:
        cid = self.case_ids[index]
        img, mask = load_b2nd(self.preproc_dir / f"{cid}.b2nd")

        image_tensor = torch.from_numpy(img).unsqueeze(0)
        mask_tensor  = torch.from_numpy(mask).unsqueeze(0)

        if self.augment:

            if random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[2])
                mask_tensor  = torch.flip(mask_tensor,  dims=[2])

            if random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[1])
                mask_tensor  = torch.flip(mask_tensor,  dims=[1])

            if random.random() < 0.5:
                k = random.randint(1, 3)
                image_tensor = torch.rot90(image_tensor, k, dims=[1, 2])
                mask_tensor  = torch.rot90(mask_tensor,  k, dims=[1, 2])

            if random.random() < 0.3:
                image_tensor = image_tensor + random.uniform(-0.2, 0.2)

            if random.random() < 0.3:
                mean = image_tensor.mean()
                image_tensor = (image_tensor - mean) * random.uniform(0.8, 1.2) + mean

            if random.random() < 0.2:
                image_tensor = image_tensor + torch.randn_like(image_tensor) * random.uniform(0.0, 0.08)

        original_h, original_w = image_tensor.shape[-2:]
        pad_h = (self.pad_factor - original_h % self.pad_factor) % self.pad_factor
        pad_w = (self.pad_factor - original_w % self.pad_factor) % self.pad_factor
        if pad_h or pad_w:
            image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h))
            mask_tensor  = F.pad(mask_tensor,  (0, pad_w, 0, pad_h))

        return {
            "image":          image_tensor.float(),
            "mask":           mask_tensor.float(),
            "original_shape": torch.tensor([original_h, original_w], dtype=torch.int64),
            "path":           str(self.preproc_dir / f"{cid}.b2nd"),
        }

def pad_collate_fn(batch: List[dict]) -> dict:
    max_h = max(s["image"].shape[-2] for s in batch)
    max_w = max(s["image"].shape[-1] for s in batch)

    def _pad(t: Tensor) -> Tensor:
        ph = max_h - t.shape[-2]
        pw = max_w - t.shape[-1]
        return F.pad(t, (0, pw, 0, ph)) if (ph or pw) else t

    return {
        "image":          torch.stack([_pad(s["image"]) for s in batch]),
        "mask":           torch.stack([_pad(s["mask"])  for s in batch]),
        "original_shape": torch.stack([s["original_shape"] for s in batch]),
        "path":           [s["path"] for s in batch],
    }

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        self.bilinear = bilinear

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bilinear   = bilinear
        factor = 2 if bilinear else 1

        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1   = Up(1024, 512 // factor, bilinear)
        self.up2   = Up(512,  256 // factor, bilinear)
        self.up3   = Up(256,  128 // factor, bilinear)
        self.up4   = Up(128,  64,            bilinear)
        self.outc  = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)

def targets_to_class_indices(targets: Tensor) -> Tensor:
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets[:, 0]
    return targets.long()

def nnunet_softmax_dice_loss(logits: Tensor, targets: Tensor, smooth: float = 1e-5) -> Tensor:
    """Match nnU-Net's CE path: softmax + batch Dice, excluding background."""
    probs = torch.softmax(logits, dim=1)
    target_idx = targets_to_class_indices(targets)
    target_onehot = F.one_hot(target_idx, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()

    probs_fg = probs[:, 1:]
    target_fg = target_onehot[:, 1:]

    intersect = probs_fg.mul(target_fg).sum(dim=(0, 2, 3), dtype=torch.float32)
    sum_pred = probs_fg.sum(dim=(0, 2, 3), dtype=torch.float32)
    sum_gt = target_fg.sum(dim=(0, 2, 3), dtype=torch.float32)

    dice = (2.0 * intersect + smooth) / (sum_gt + sum_pred + smooth).clamp_min(1e-8)
    return -dice.mean()

class CombinedCEDiceLoss(nn.Module):
    """Cross-entropy + nnU-Net-style softmax Dice for label-map training."""
    def __init__(self, ce_weight=1.0, dice_weight=1.0, smooth=1e-5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        target_idx = targets_to_class_indices(targets)
        ce = self.ce(logits, target_idx)
        dice = nnunet_softmax_dice_loss(logits, targets, self.smooth)
        return self.ce_weight * ce + self.dice_weight * dice

class CombinedBCEDiceLoss(nn.Module):
    """BCE + nnU-Net-style batch soft Dice."""
    def __init__(self, bce_weight=1.0, dice_weight=1.0,
                 pos_weight=17.12810816704046, smooth=1e-5):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.smooth      = smooth
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def soft_dice_loss(self, logits, targets):

        probs = torch.sigmoid(logits)
        targets = targets.float()
        intersect = (probs * targets).sum(dtype=torch.float32)
        sum_pred = probs.sum(dtype=torch.float32)
        sum_gt = targets.sum(dtype=torch.float32)
        dice = (2.0 * intersect + self.smooth) / (sum_gt + sum_pred + self.smooth).clamp_min(1e-8)
        return -dice

    def forward(self, logits, targets):
        bce  = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="mean", pos_weight=self.pos_weight)
        dice = self.soft_dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice

class FocalDiceLoss(nn.Module):
    """Focal loss + Dice for severe class imbalance."""
    def __init__(self, focal_weight=1.0, dice_weight=1.0,
                 gamma=2.0, alpha=0.25, smooth=1e-5):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight  = dice_weight
        self.gamma = gamma; self.alpha = alpha; self.smooth = smooth

    def focal_loss(self, logits, targets):
        probs     = torch.sigmoid(logits)
        p_t       = probs * targets + (1 - probs) * (1 - targets)
        fw        = (1 - p_t) ** self.gamma
        alpha_t   = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * fw * F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")).mean()

    def soft_dice_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        intersect = (probs * targets).sum(dtype=torch.float32)
        sum_pred = probs.sum(dtype=torch.float32)
        sum_gt = targets.sum(dtype=torch.float32)
        dice = (2.0 * intersect + self.smooth) / (sum_gt + sum_pred + self.smooth).clamp_min(1e-8)
        return -dice

    def forward(self, logits, targets):
        return (self.focal_weight * self.focal_loss(logits, targets)
                + self.dice_weight * self.soft_dice_loss(logits, targets))

def compute_hd95(pred, target, spacing=(1.0, 1.0)):
    from scipy.ndimage import distance_transform_edt
    pb = (pred > 0.5).astype(np.uint8)
    tb = (target > 0.5).astype(np.uint8)
    if pb.sum() == 0 or tb.sum() == 0:
        return float("inf")
    pred_border   = pb - np.array(Image.fromarray(pb).filter(ImageFilter.MinFilter(3)))
    target_border = tb - np.array(Image.fromarray(tb).filter(ImageFilter.MinFilter(3)))
    if pred_border.sum() == 0 or target_border.sum() == 0:
        return float("inf")
    dt_pred   = distance_transform_edt(~pred_border.astype(bool),   sampling=spacing)
    dt_target = distance_transform_edt(~target_border.astype(bool), sampling=spacing)
    return float(np.percentile(
        np.concatenate([dt_target[pred_border > 0], dt_pred[target_border > 0]]), 95))

def hard_metric_sums(logits, masks, original_shapes, threshold=0.5):
    """Accumulate hard TP/FP/FN/TN like nnU-Net online evaluation.

    Dice is computed from thresholded predictions with global TP/FP/FN sums
    across the whole epoch, matching nnU-Net's reported pseudo Dice.
    """
    if logits.shape[1] == 1:
        preds = (torch.sigmoid(logits) > threshold).float()[:, 0]
    else:
        preds = logits.argmax(dim=1).float()
    shapes = original_shapes.cpu().tolist()
    totals = dict(tp=0.0, fp=0.0, fn=0.0, tn=0.0, hd95=0.0, hd_n=0)

    for i, (h, w) in enumerate(shapes):
        pred   = preds[i, :h, :w]
        target = masks[i, 0, :h, :w]
        try:
            hd = compute_hd95(pred.cpu().numpy(), target.cpu().numpy())
            if not np.isinf(hd):
                totals["hd95"] += float(hd)
                totals["hd_n"] += 1
        except Exception:
            pass
        pf = pred.reshape(-1); tf = target.reshape(-1)
        tp = (pf * tf).sum(); fp = (pf * (1-tf)).sum()
        fn = ((1-pf) * tf).sum(); tn = ((1-pf) * (1-tf)).sum()
        totals["tp"] += float(tp.item())
        totals["fp"] += float(fp.item())
        totals["fn"] += float(fn.item())
        totals["tn"] += float(tn.item())

    return totals

def summarise_hard_metrics(totals):
    eps = 1e-6
    tp = totals["tp"]
    fp = totals["fp"]
    fn = totals["fn"]
    tn = totals["tn"]
    dice = float((2 * tp + eps) / (2 * tp + fp + fn + eps))
    precision = float((tp + eps) / (tp + fp + eps))
    recall = float((tp + eps) / (tp + fn + eps))
    f1 = float((2 * tp + eps) / (2 * tp + fp + fn + eps))
    acc = float((tp + tn) / (tp + tn + fp + fn + eps))
    hd95 = float(totals["hd95"] / totals["hd_n"]) if totals["hd_n"] else float("inf")
    return dice, precision, recall, f1, acc, hd95

def _compose_sample_image(image, target, pred):
    image  = np.clip(image, 0.0, 1.0)
    target = (target > 0.5).astype(np.float32)
    pred   = (pred   > 0.5).astype(np.float32)
    base   = np.repeat(image[..., None], 3, axis=2)
    tgt    = np.zeros_like(base); tgt[..., 1] = target
    prd    = np.zeros_like(base); prd[..., 0] = pred
    blend  = base.copy()
    blend[..., 0] = np.clip(blend[..., 0] + pred   * 0.5, 0, 1)
    blend[..., 1] = np.clip(blend[..., 1] + target * 0.5, 0, 1)
    return Image.fromarray((np.concatenate([base, tgt, prd, blend], axis=1)*255).astype(np.uint8))

@torch.no_grad()
def save_sample_predictions(model, loader, device, output_dir, epoch, max_samples, threshold=0.5):
    if max_samples <= 0:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for batch in loader:
        images = batch["image"].to(device=device, non_blocking=True)
        masks  = batch["mask"].to(device=device, non_blocking=True)
        logits = model(images)
        if logits.shape[1] == 1:
            preds = (torch.sigmoid(logits) > threshold).float()[:, 0]
        else:
            preds = logits.argmax(dim=1).float()
        for i in range(images.size(0)):
            if saved >= max_samples:
                break
            h, w = batch["original_shape"][i]
            img  = images[i, 0, :h, :w].cpu().numpy()

            lo, hi = img.min(), img.max()
            img_disp = (img - lo) / (hi - lo + 1e-8)
            tgt  = masks[i,  0, :h, :w].cpu().numpy()
            pred = preds[i, :h, :w].cpu().numpy()
            composite = _compose_sample_image(img_disp, tgt, pred)
            stem = Path(batch["path"][i]).stem
            composite.save(output_dir / f"epoch{epoch:03d}_{stem}.png")
            saved += 1
        break

def plot_loss_curves(train_losses, val_losses, val_epochs, output_path):
    if not train_losses:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_losses, label="train_loss", marker="o", linewidth=1.5)
    if val_losses and val_epochs:
        plt.plot(val_epochs, val_losses, label="val_loss", marker="s",
                 linestyle="--", linewidth=1.5)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(); plt.tight_layout()
    plt.savefig(output_path); plt.close()

def train_one_epoch(model, loader, optimizer, criterion, device, progress_desc=None):
    model.train()
    totals = dict(loss=0, tp=0, fp=0, fn=0, tn=0, hd95=0, hd_n=0, n=0)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device,  non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss   = criterion(logits, masks)
        loss.backward(); optimizer.step()
        bs = images.size(0)
        totals["loss"] += loss.item() * bs
        totals["n"]    += bs
        with torch.no_grad():
            hard_totals = hard_metric_sums(logits, masks, batch["original_shape"])
            for key in ("tp", "fp", "fn", "tn", "hd95", "hd_n"):
                totals[key] += hard_totals[key]
    n = max(totals["n"], 1)
    dice, prec, rec, f1, acc, hd95 = summarise_hard_metrics(totals)
    return totals["loss"] / n, dice, prec, rec, f1, acc, hd95

@torch.no_grad()
def evaluate(model, loader, criterion, device, progress_desc=None):
    model.eval()
    totals = dict(loss=0, tp=0, fp=0, fn=0, tn=0, hd95=0, hd_n=0, n=0)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device,  non_blocking=True)
        logits = model(images)
        loss   = criterion(logits, masks)
        bs = images.size(0)
        totals["loss"] += loss.item() * bs; totals["n"] += bs
        hard_totals = hard_metric_sums(logits, masks, batch["original_shape"])
        for key in ("tp", "fp", "fn", "tn", "hd95", "hd_n"):
            totals[key] += hard_totals[key]
    n = max(totals["n"], 1)
    dice, prec, rec, f1, acc, hd95 = summarise_hard_metrics(totals)
    return totals["loss"] / n, dice, prec, rec, f1, acc, hd95

def dataloader_worker_init(worker_id):
    seed = WORKER_BASE_SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the canonical resplit U-Net on nnU-Net preprocessed .b2nd files.")
    parser.add_argument("--config", type=Path,
                        default=CONFIG_DEFAULT)
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last_checkpoint.pt when available.")
    parser.add_argument("--epochs",       type=int,   default=1000)
    parser.add_argument("--batch-size",   type=int,   default=4)
    parser.add_argument("--learning-rate",type=float, default=1e-3)
    parser.add_argument("--num-workers",  type=int,   default=8)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--pad-factor",   type=int,   default=32)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Random flips, rotations, brightness (default: on).")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--loss-type", type=str, default="ce_dice",
                        choices=["ce_dice", "ce_only", "bce_dice", "focal_dice", "bce_only"])
    parser.add_argument("--bce-weight",   type=float, default=1.0)
    parser.add_argument("--pos-weight",   type=float, default=None,
                        help="Optional manual BCE positive-class weight for legacy BCE modes only.")
    parser.add_argument("--dice-weight",  type=float, default=1.0)
    parser.add_argument("--focal-gamma",  type=float, default=2.0)
    parser.add_argument("--focal-alpha",  type=float, default=0.25)
    parser.add_argument("--sample-output-dir", type=str, default="sample_predictions")
    parser.add_argument("--sample-count",      type=int,   default=3)
    parser.add_argument("--sample-threshold",  type=float, default=0.5)
    parser.add_argument("--loss-plot-path",    type=str,
                        default="loss_curve.png")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    train_ids, val_ids, test_ids, preproc_dir, split_source = load_splits(args.config)
    best_checkpoint_path, last_checkpoint_path = get_checkpoint_paths(args.config)
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  U-Net Training  (b2nd pipeline)")
    print(f"  Config  : {args.config}")
    print(f"  Augment : {args.augment}")
    print(f"  Resume  : {args.resume}")
    print("=" * 70)
    print(f"Split : {split_source}")
    print(f"Train : {len(train_ids)}  ({split_source.name} fold 0)")
    print(f"Val   : {len(val_ids)}   ({split_source.name} fold 0)")
    print(f"Test  : {len(test_ids)}  (preprocessed test cases available for in-script eval)")
    if not test_ids:
        print("Note  : no test .b2nd files are available; final test eval will be run via predict/evaluate scripts.")

    train_dataset = B2ndDataset(train_ids, preproc_dir,
                                pad_factor=args.pad_factor, augment=args.augment)
    val_dataset   = B2ndDataset(val_ids,   preproc_dir,
                                pad_factor=args.pad_factor, augment=False)
    test_dataset = None
    if test_ids:
        test_dataset = B2ndDataset(test_ids, preproc_dir,
                                   pad_factor=args.pad_factor, augment=False)

    global WORKER_BASE_SEED
    WORKER_BASE_SEED = args.seed

    def make_loader(ds, shuffle):
        if ds is None:
            return None
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=args.num_workers > 0,
                          worker_init_fn=dataloader_worker_init if args.num_workers > 0 else None,
                          collate_fn=pad_collate_fn)

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader   = make_loader(val_dataset,   shuffle=False)
    test_loader  = make_loader(test_dataset,  shuffle=False)

    device = torch.device(args.device)
    model  = UNet(n_channels=1, n_classes=2).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device    : {device}")

    effective_pos_weight = None
    if args.loss_type in {"bce_dice", "focal_dice", "bce_only"}:
        effective_pos_weight = args.pos_weight
        if effective_pos_weight is None:
            effective_pos_weight = compute_train_pos_weight(args.config)
            print(f"Pos weight: {effective_pos_weight:.6f} (computed from active train split)")
        else:
            print(f"Pos weight: {effective_pos_weight:.6f} (CLI override)")

    if args.loss_type == "ce_dice":
        criterion = CombinedCEDiceLoss(args.bce_weight, args.dice_weight)
        print("Loss: CE+Dice (nnU-Net-style objective)")
    elif args.loss_type == "ce_only":
        criterion = nn.CrossEntropyLoss()
        print("Loss: CE only")
    elif args.loss_type == "bce_dice":
        criterion = CombinedBCEDiceLoss(args.bce_weight, args.dice_weight, effective_pos_weight)
        print(f"Loss: BCE+Dice (pos_weight={effective_pos_weight:.6f})")
    elif args.loss_type == "focal_dice":
        criterion = FocalDiceLoss(args.bce_weight, args.dice_weight,
                                  args.focal_gamma, args.focal_alpha)
        print(f"Loss: Focal+Dice (gamma={args.focal_gamma})")
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(effective_pos_weight, device=device))
        print(f"Loss: BCE only (pos_weight={effective_pos_weight:.6f})")
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5)

    start_epoch    = 1
    best_val_loss  = math.inf
    patience_counter = 0
    train_losses: List[float] = []
    val_losses:   List[float] = []
    val_epochs:   List[int]   = []

    resume_path = None
    if args.resume:
        if last_checkpoint_path.is_file():
            resume_path = last_checkpoint_path
        elif best_checkpoint_path.is_file():
            resume_path = best_checkpoint_path

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", math.inf))
        patience_counter = ckpt.get("patience_counter", 0)
        train_losses = list(ckpt.get("train_losses", []))
        val_losses = list(ckpt.get("val_losses", []))
        val_epochs = list(ckpt.get("val_epochs", []))
        print(
            f"\nResumed from {resume_path.name} at epoch {ckpt['epoch']} "
            f"(best_val_loss={best_val_loss:.4f}, patience={patience_counter})"
        )
    elif args.resume:
        print(f"\nWARNING: --resume set but no checkpoint found at {last_checkpoint_path} "
              f"or {best_checkpoint_path}. "
              f"Starting from scratch.")

    sample_output_dir = Path(args.sample_output_dir) if args.sample_output_dir else None
    if sample_output_dir:
        sample_output_dir.mkdir(parents=True, exist_ok=True)
    loss_plot_path = Path(args.loss_plot_path) if args.loss_plot_path else None

    early_stop_patience = 50
    early_stop_delta     = 0.001

    for epoch in range(start_epoch, args.epochs + 1):
        do_val = (epoch % 2 == 0)

        train_loss, train_dice, train_prec, train_rec, train_f1, train_acc, train_hd95 =\
            train_one_epoch(model, train_loader, optimizer, criterion, device,
                            progress_desc=f"Epoch {epoch:03d} train")
        train_losses.append(train_loss)

        if do_val:
            val_loss, val_dice, val_prec, val_rec, val_f1, val_acc, val_hd95 =\
                evaluate(model, val_loader, criterion, device,
                         progress_desc=f"Epoch {epoch:03d} val")
            scheduler.step(val_loss)
            val_losses.append(val_loss); val_epochs.append(epoch)
            if sample_output_dir:
                save_sample_predictions(model, val_loader, device,
                                        sample_output_dir, epoch,
                                        args.sample_count, args.sample_threshold)
        else:
            val_loss = val_dice = val_prec = val_rec = val_f1 = val_acc = math.nan
            val_hd95 = float("inf")

        val_msg = (
            f"| val_loss={val_loss:.4f} val_pseudo_dice={val_dice:.4f} "
            f"val_precision={val_prec:.4f} val_recall={val_rec:.4f} "
            f"val_hd95={val_hd95:.2f}"
            if do_val else
            "| val_loss=---- val_pseudo_dice=---- val_precision=---- val_recall=---- val_hd95=----"
        )
        print(f"Epoch {epoch:03d} complete "
              f"| train_loss={train_loss:.4f} train_pseudo_dice={train_dice:.4f} "
              f"train_precision={train_prec:.4f} train_recall={train_rec:.4f} "
              f"train_hd95={train_hd95:.2f} {val_msg}")

        improved = do_val and val_loss < (best_val_loss - early_stop_delta)
        if improved:
            best_val_loss    = val_loss
            patience_counter = 0
        elif do_val:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} val checks "
                  f"(patience={early_stop_patience}, delta={early_stop_delta})")
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_epochs": val_epochs,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1,
            "val_acc": val_acc,
            "val_hd95": val_hd95,
            "augment": args.augment,
            "loss_type": args.loss_type,
            "pos_weight": effective_pos_weight,
        }
        torch.save(checkpoint_state, last_checkpoint_path)
        if improved:
            torch.save(checkpoint_state, best_checkpoint_path)
            print(f"  Saved new best model (val_loss={val_loss:.4f})")
        if do_val and patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch}. "
                  f"Best val_loss={best_val_loss:.4f}")
            break

    final_checkpoint_path = best_checkpoint_path if best_checkpoint_path.is_file() else last_checkpoint_path
    if final_checkpoint_path.is_file():
        ckpt = torch.load(final_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from epoch {ckpt['epoch']} ({final_checkpoint_path.name}).")

    if loss_plot_path:
        plot_loss_curves(train_losses, val_losses, val_epochs, loss_plot_path)

    if test_loader is None:
        print("\nSkipped in-script test evaluation because no test .b2nd files are available.")
        print("Run unet_pipeline/predict_unet.py followed by evaluation_pipeline/evaluate.py for held-out test results.\n")
        return

    test_loss, test_dice, test_prec, test_rec, test_f1, test_acc, test_hd95 =\
        evaluate(model, test_loader, criterion, device, progress_desc="Test")

    print(f"\n{'='*70}")
    print("FINAL TEST RESULTS:")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Dice:      {test_dice:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1:        {test_f1:.4f}")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  HD95:      {test_hd95:.2f}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
