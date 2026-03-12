#!/usr/bin/env python3
"""
Day 4 - predict_unet.py
=======================
Run inference with the trained U-Net checkpoint on a split from nnunet_id_map.json
and save outputs in a layout compatible with the evaluation pipeline:

  predictions/unet/
    ├── binary/
    │   ├── mito_XXXX.nii.gz
    │   └── ...
    └── probs/
        ├── mito_XXXX.npz            (key: probabilities, shape: [2, 1, H, W])
        └── ...

Binary NIfTI files follow the same orientation as Dataset001_Mitochondria labels
created in Day 1 (transposed from raw TIFF to [W, H, 1]).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def load_b2nd_image(case_path: Path) -> Tensor:
    import blosc2

    array = np.array(blosc2.open(str(case_path))[0, 0], dtype=np.float32)
    return torch.from_numpy(array).unsqueeze(0)


def to_model_tensor(
    image: Image.Image,
    clip_percentiles: Tuple[float, float] | None = None,
    normalize: str = "zscore",
) -> Tensor:
    array = np.array(image, dtype=np.float32)
    if array.size == 0:
        return torch.from_numpy(array).unsqueeze(0)

    if clip_percentiles is not None:
        low, high = np.percentile(array, clip_percentiles)
        if high > low:
            array = np.clip(array, low, high)

    if normalize == "minmax":
        min_val = float(array.min())
        max_val = float(array.max())
        if max_val > min_val:
            array = (array - min_val) / (max_val - min_val)
        else:
            array = np.zeros_like(array)
    elif normalize == "zscore":
        mean_val = float(array.mean())
        std_val = float(array.std())
        if std_val > 0:
            array = (array - mean_val) / std_val
        else:
            array = array - mean_val
    elif normalize != "none":
        raise ValueError(f"Unknown normalization mode '{normalize}'")

    if array.ndim == 2:
        return torch.from_numpy(array).unsqueeze(0)
    return torch.from_numpy(array).permute(2, 0, 1)


class InferenceDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, str]],
        pad_factor: int = 32,
        clip_percentiles: Tuple[float, float] | None = None,
        normalize: str = "zscore",
    ) -> None:
        self.records = records
        self.pad_factor = pad_factor
        self.clip_percentiles = clip_percentiles
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        source_type = record["source_type"]
        source_path = Path(record["source_path"])

        if source_type == "b2nd":
            image_tensor = load_b2nd_image(source_path)
        elif source_type == "raw":
            image = Image.open(source_path)
            if image.mode not in ("I;16", "I", "F"):
                image = image.convert("L")

            image_tensor = to_model_tensor(
                image,
                clip_percentiles=self.clip_percentiles,
                normalize=self.normalize,
            )
        else:
            raise ValueError(f"Unknown inference source_type '{source_type}'")

        original_h, original_w = image_tensor.shape[-2:]
        pad_h = (self.pad_factor - original_h % self.pad_factor) % self.pad_factor
        pad_w = (self.pad_factor - original_w % self.pad_factor) % self.pad_factor
        if pad_h or pad_w:
            image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h))

        return {
            "image": image_tensor.float(),
            "original_shape": torch.tensor([original_h, original_w], dtype=torch.int64),
            "case_id": record["case_id"],
            "source_type": source_type,
        }


def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    max_h = max(sample["image"].shape[-2] for sample in batch)
    max_w = max(sample["image"].shape[-1] for sample in batch)

    def _pad(tensor: Tensor) -> Tensor:
        pad_h = max_h - tensor.shape[-2]
        pad_w = max_w - tensor.shape[-1]
        if pad_h == 0 and pad_w == 0:
            return tensor
        return F.pad(tensor, (0, pad_w, 0, pad_h))

    images = torch.stack([_pad(sample["image"]) for sample in batch], dim=0)
    shapes = torch.stack([sample["original_shape"] for sample in batch], dim=0)
    case_ids = [sample["case_id"] for sample in batch]
    source_types = [sample["source_type"] for sample in batch]
    return {"image": images, "original_shape": shapes, "case_id": case_ids, "source_type": source_types}


def load_case_records(
    id_map_path: Path,
    data_root: Path,
    preproc_dir: Path,
    split: str,
    input_source: str,
) -> List[Dict[str, str]]:
    id_map = json.loads(id_map_path.read_text())
    records: List[Dict[str, str]] = []
    missing = 0
    b2nd_used = 0
    raw_used = 0

    for case_id, meta in id_map.items():
        if meta.get("split") != split:
            continue
        b2nd_path = preproc_dir / f"{case_id}.b2nd"
        raw_path = data_root / meta["sample"] / "images" / meta["filename"]

        if input_source == "b2nd":
            if not b2nd_path.exists():
                missing += 1
                continue
            records.append({"case_id": case_id, "source_type": "b2nd", "source_path": str(b2nd_path)})
            b2nd_used += 1
            continue

        if input_source == "auto" and b2nd_path.exists():
            records.append({"case_id": case_id, "source_type": "b2nd", "source_path": str(b2nd_path)})
            b2nd_used += 1
            continue

        if not raw_path.exists():
            missing += 1
            continue
        records.append({"case_id": case_id, "source_type": "raw", "source_path": str(raw_path)})
        raw_used += 1

    def _case_key(rec: Dict[str, str]) -> int:
        return int(rec["case_id"].split("_")[1])

    records.sort(key=_case_key)
    if missing:
        print(f"Warning: skipped {missing} {split} cases with missing image files.")
    print(
        f"Input source summary for split '{split}': "
        f"b2nd={b2nd_used}, raw={raw_used}, mode={input_source}"
    )
    if input_source == "auto" and raw_used > 0:
        print(
            f"Warning: {raw_used} {split} cases are using raw TIFF fallback because "
            "no preprocessed .b2nd file exists for them."
        )
    return records


def load_unet_model(checkpoint_path: Path, device: torch.device):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from day4.train_unet import UNet  # import the exact model definition used in resplit training

    model = UNet(n_channels=1, n_classes=1).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint at {checkpoint_path} missing 'model_state_dict'.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint.get("epoch")


def save_outputs(
    case_id: str,
    prob_fg: np.ndarray,
    threshold: float,
    binary_dir: Path,
    probs_dir: Path,
    compress_probs: bool,
) -> None:
    prob_fg = prob_fg.astype(np.float32, copy=False)
    binary_hw = (prob_fg >= threshold).astype(np.uint8)

    # Match Day 1 NIfTI orientation: [W, H, 1]
    binary_nii = binary_hw.T[:, :, np.newaxis]
    nib.save(nib.Nifti1Image(binary_nii, np.eye(4)), binary_dir / f"{case_id}.nii.gz")

    # Match nnU-Net .npz convention: [C, Z, H, W]
    probabilities = np.stack([1.0 - prob_fg, prob_fg], axis=0)[:, np.newaxis, :, :]
    if compress_probs:
        np.savez_compressed(probs_dir / f"{case_id}.npz", probabilities=probabilities)
    else:
        np.savez(probs_dir / f"{case_id}.npz", probabilities=probabilities)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run U-Net inference and save binary + probability outputs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/home/s2292949/dissertation_resplit/config.json"),
        help="Path to config.json",
    )
    parser.add_argument(
        "--split",
        choices=["test", "val", "train"],
        default="test",
        help="Which split from nnunet_id_map.json to run inference on.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override model checkpoint path (default: <unet_checkpoint>/best_model.pt)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output root (default: config['paths']['predictions_unet'])",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--pad-factor", type=int, default=32, help="Pad H/W to multiples of this value.")
    parser.add_argument("--threshold", type=float, default=None, help="Binary threshold (default from config).")
    parser.add_argument(
        "--input-source",
        choices=["auto", "raw", "b2nd"],
        default="auto",
        help="Use nnU-Net preprocessed .b2nd when available, raw TIFF, or require b2nd only.",
    )
    parser.add_argument(
        "--normalize",
        choices=["zscore", "minmax", "none"],
        default="zscore",
        help="Input normalisation for raw TIFF inference. Default is z-score to match training.",
    )
    parser.add_argument(
        "--clip-low",
        type=float,
        default=None,
        help="Optional lower percentile for clipping before normalisation.",
    )
    parser.add_argument(
        "--clip-high",
        type=float,
        default=None,
        help="Optional upper percentile for clipping before normalisation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit for smoke testing.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (default: skip existing case files).",
    )
    parser.add_argument(
        "--no-compress-probs",
        action="store_true",
        help="Write .npz probabilities without compression for faster IO.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text())

    data_root = Path(cfg["paths"]["data_root"])
    id_map_path = Path(cfg["paths"]["nnunet_id_map"])
    preproc_dir = Path(cfg["paths"]["nnunet_preproc"]) / "Dataset001_Mitochondria" / "nnUNetPlans_2d"
    checkpoint = args.checkpoint or (Path(cfg["paths"]["unet_checkpoint"]) / "best_model.pt")
    output_root = args.output_root or Path(cfg["paths"]["predictions_unet"])
    threshold = args.threshold
    if threshold is None:
        threshold = float(cfg["shared"].get("prediction_threshold", 0.5))

    seed = args.seed if args.seed is not None else int(cfg["shared"].get("random_seed", 42))
    set_seed(seed)

    batch_size = args.batch_size if args.batch_size is not None else int(cfg["unet"]["batch_size"])
    device = torch.device(args.device)
    pin_memory = device.type == "cuda"

    binary_dir = output_root / "binary"
    probs_dir = output_root / "probs"
    binary_dir.mkdir(parents=True, exist_ok=True)
    probs_dir.mkdir(parents=True, exist_ok=True)

    records = load_case_records(
        id_map_path=id_map_path,
        data_root=data_root,
        preproc_dir=preproc_dir,
        split=args.split,
        input_source=args.input_source,
    )
    if args.limit is not None:
        records = records[: args.limit]

    if not records:
        raise RuntimeError(f"No cases found for split '{args.split}'.")

    if (args.clip_low is None) != (args.clip_high is None):
        raise ValueError("Specify both --clip-low and --clip-high together, or neither.")
    clip_percentiles = None
    if args.clip_low is not None and args.clip_high is not None:
        if not (0.0 <= args.clip_low < args.clip_high <= 100.0):
            raise ValueError("Clipping percentiles must satisfy 0 <= low < high <= 100.")
        clip_percentiles = (args.clip_low, args.clip_high)

    dataset = InferenceDataset(
        records=records,
        pad_factor=args.pad_factor,
        clip_percentiles=clip_percentiles,
        normalize=args.normalize,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    model, ckpt_epoch = load_unet_model(checkpoint, device)

    print("=" * 72)
    print("U-Net Inference")
    print(f"  config      : {args.config}")
    print(f"  checkpoint  : {checkpoint}")
    print(f"  split       : {args.split} ({len(records)} cases)")
    print(f"  input src   : {args.input_source}")
    print(f"  batch size  : {batch_size}")
    print(f"  threshold   : {threshold}")
    print(f"  normalize   : {args.normalize}")
    print(f"  clip pct    : {clip_percentiles if clip_percentiles is not None else 'none'}")
    print(f"  device      : {device}")
    if ckpt_epoch is not None:
        print(f"  ckpt epoch  : {ckpt_epoch}")
    print(f"  binary out  : {binary_dir}")
    print(f"  probs out   : {probs_dir}")
    print("=" * 72)

    total = 0
    skipped = 0
    with torch.no_grad():
        for batch in _progress(loader, desc="Predicting", leave=False):
            images = batch["image"].to(device=device, non_blocking=True)
            logits = model(images)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            shapes = batch["original_shape"].cpu().numpy()
            case_ids = batch["case_id"]

            for idx, case_id in enumerate(case_ids):
                out_bin = binary_dir / f"{case_id}.nii.gz"
                out_prob = probs_dir / f"{case_id}.npz"
                if not args.overwrite and out_bin.exists() and out_prob.exists():
                    skipped += 1
                    continue

                height, width = int(shapes[idx, 0]), int(shapes[idx, 1])
                prob_fg = prob[idx, 0, :height, :width]
                save_outputs(
                    case_id=case_id,
                    prob_fg=prob_fg,
                    threshold=threshold,
                    binary_dir=binary_dir,
                    probs_dir=probs_dir,
                    compress_probs=not args.no_compress_probs,
                )
                total += 1

    print("\nInference complete.")
    print(f"  saved   : {total}")
    print(f"  skipped : {skipped}")
    print(f"  total   : {len(records)}")


if __name__ == "__main__":
    main()
