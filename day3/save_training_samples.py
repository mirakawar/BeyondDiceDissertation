#!/usr/bin/env python3
"""
save_training_samples.py
Produces 4-panel training sample visualisations styled like the MLP
validation panels (white bg, bold monospace headers above each image).

Panels: Original EM | Ground Truth (green) | Binary Mask | Statistics
No GPU needed — runs on the login node.

Usage:
    python save_training_samples.py
    python save_training_samples.py --n 8 --split train
"""

import argparse, json, random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_b2nd(case_path: Path):
    import blosc2
    img  = np.array(blosc2.open(str(case_path))[0, 0], dtype=np.float32)
    seg  = case_path.parent / (case_path.stem + "_seg.b2nd")
    mask = np.array(blosc2.open(str(seg))[0, 0]) > 0
    return img, mask

def norm_display(img):
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)

def green_overlay(img_disp, mask, alpha=0.5):
    rgb = np.stack([img_disp]*3, axis=-1).copy()
    if mask.any():
        rgb[mask] = (1-alpha)*rgb[mask] + alpha*np.array([0., 0.85, 0.3])
    return rgb

def save_panel(idx, case_id, img, mask, meta, out_dir):
    d        = norm_display(img)
    h, w     = img.shape
    fg       = float(mask.mean())
    n_inst   = meta.get("n_instances", "?")
    filename = meta.get("filename", "")
    split    = meta.get("split", "?").upper()

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.78,
                        bottom=0.02, wspace=0.04)

    fig.suptitle(f"CEM-MitoLab Training Sample  —  {case_id}",
                 fontsize=13, fontweight="bold", fontfamily="monospace",
                 color="black", y=0.97)
    fig.text(0.5, 0.875,
             f"{filename}     split={split}     "
             f"FG={fg*100:.2f}%     instances={n_inst}     {h}×{w}px",
             ha="center", fontsize=8, fontfamily="monospace", color="#555")

    ax = axes[0]
    ax.imshow(d, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Original EM Image", fontsize=10, fontweight="bold",
                 fontfamily="monospace", pad=7)
    ax.axis("off")

    ax = axes[1]
    ax.imshow(green_overlay(d, mask))
    ax.set_title("Ground Truth  (Green)", fontsize=10, fontweight="bold",
                 fontfamily="monospace", pad=7)
    ax.axis("off")
    ax.legend(handles=[mpatches.Patch(color=(0,.85,.3), label="Mitochondria")],
              loc="lower right", fontsize=7, framealpha=0.6, edgecolor="none")

    ax = axes[2]
    ax.imshow(mask.astype(np.float32), cmap="gray", vmin=0, vmax=1)
    ax.set_title("Binary Mask", fontsize=10, fontweight="bold",
                 fontfamily="monospace", pad=7)
    ax.axis("off")

    ax = axes[3]
    ax.set_facecolor("#f7f7f7")
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.set_title("Sample Statistics", fontsize=10, fontweight="bold",
                 fontfamily="monospace", pad=7)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor("#ccc"); sp.set_linewidth(0.8)

    rows = [
        ("Case ID",    case_id),
        ("Split",      split),
        ("Shape",      f"{h} × {w} px"),
        ("FG ratio",   f"{fg:.5f}"),
        ("FG %",       f"{fg*100:.2f}%"),
        ("Instances",  str(n_inst)),
        ("Empty",      str(meta.get("is_empty", False))),
        ("z  μ",       f"{img.mean():.4f}"),
        ("z  σ",       f"{img.std():.4f}"),
        ("Range",      f"{img.min():.2f} → {img.max():.2f}"),
    ]
    y = 0.92
    for label, val in rows:
        ax.text(0.06, y, label, fontsize=8.5, fontfamily="monospace",
                color="#666", va="top", transform=ax.transAxes)
        ax.text(0.94, y, val, fontsize=8.5, fontfamily="monospace",
                color="#111", va="top", ha="right", fontweight="bold",
                transform=ax.transAxes)
        ax.plot([0.04, 0.96], [y-0.028, y-0.028], color="#ddd",
                lw=0.5, transform=ax.transAxes)
        y -= 0.09

    out_path = out_dir / f"{idx:03d}_{case_id}_panel.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [{idx:03d}] {case_id:15s}  FG={fg:.4f}  "
          f"inst={str(n_inst):>3}  {h}×{w}  → {out_path.name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path("/home/s2292949/dissertation_resplit/config.json"))
    ap.add_argument("--n",     type=int, default=6)
    ap.add_argument("--out",   type=Path,
                    default=Path("/home/s2292949/dissertation_resplit/samples/training"))
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--split", choices=["train","val","all"], default="train")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    id_map_path = Path(cfg["paths"]["nnunet_id_map"])
    preproc_dir = (Path(cfg["paths"]["nnunet_preproc"])
                   / "Dataset001_Mitochondria" / "nnUNetPlans_2d")
    id_map = json.loads(id_map_path.read_text())

    case_ids = [
        cid for cid, m in id_map.items()
        if (args.split == "all" or m.get("split") == args.split)
        and (preproc_dir / f"{cid}.b2nd").exists()
    ]
    print(f"Found {len(case_ids)} '{args.split}' cases with .b2nd files")
    if not case_ids:
        print("ERROR: no files found — check preproc_dir in config.json"); return

    srt   = sorted(case_ids, key=lambda c: id_map[c].get("fg_ratio", 0))
    n     = min(args.n, len(case_ids))
    n_ext = max(1, n // 3)
    rng   = random.Random(args.seed)
    mid   = rng.sample(srt[n_ext:-n_ext],
                       min(n - 2*n_ext, len(srt) - 2*n_ext))
    sel   = (srt[:n_ext] + mid + srt[-n_ext:])[:n]
    rng.shuffle(sel)

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(sel)} panels → {args.out}\n")
    for i, cid in enumerate(sel):
        img, mask = load_b2nd(preproc_dir / f"{cid}.b2nd")
        save_panel(i, cid, img, mask, id_map[cid], args.out)
    print(f"\nDone. {len(sel)} panels in {args.out}/")

if __name__ == "__main__":
    main()
