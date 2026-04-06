#!/usr/bin/env python3
"""
Generate count error distribution figure for the appendix.
Outputs: figures/count_error_dist/count_error_distribution.png
Run from repo root or any directory.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "evaluation" / "results"
OUT_DIR = REPO / "figures" / "count_error_dist"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":        "monospace",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.labelsize":     17,
    "legend.fontsize":    15,
    "xtick.labelsize":    14,
    "ytick.labelsize":    14,
})

UNET_COLOUR   = "#2980b9"
NNUNET_COLOUR = "#e74c3c"
CUTOFF        = 4
CLIP_AT       = 30   # values above this are grouped into the last bin

unet   = pd.read_csv(RESULTS / "unet_per_image.csv")
nnunet = pd.read_csv(RESULTS / "nnunet_per_image.csv")

# absolute count error, GT-positive slices only, clip tail
u_err  = unet.loc[unet["gt_objects"] > 0, "obj_count_error"].clip(upper=CLIP_AT).dropna()
nn_err = nnunet.loc[nnunet["gt_objects"] > 0, "obj_count_error"].clip(upper=CLIP_AT).dropna()

bins = np.arange(0, CLIP_AT + 2) - 0.5   # integer-centred bins 0..CLIP_AT

fig, ax = plt.subplots(figsize=(8, 4))

ax.hist(u_err,  bins=bins, density=True, alpha=0.55, color=UNET_COLOUR,
        label="U-Net",   edgecolor="white", linewidth=0.4)
ax.hist(nn_err, bins=bins, density=True, alpha=0.55, color=NNUNET_COLOUR,
        label="nnU-Net", edgecolor="white", linewidth=0.4)

ax.axvline(CUTOFF, color="black", linestyle="--", linewidth=1.4,
           label=f"Corner-case cutoff (>{CUTOFF})")

ax.set_xlabel("Absolute object count error (clipped at 30)")
ax.set_ylabel("Density")
ax.set_xlim(-0.5, CLIP_AT + 0.5)
ax.set_xticks([0, 4, 10, 15, 20, 25, 30])
ax.legend(frameon=False)

# annotate IQR upper bounds
u_q75  = int(np.percentile(u_err,  75))
nn_q75 = int(np.percentile(nn_err, 75))
ax.text(CUTOFF + 0.3, ax.get_ylim()[1] * 0.88,
        f"U-Net Q3 = {u_q75}\nnnU-Net Q3 = {nn_q75}",
        fontsize=11, va="top", color="black")

fig.tight_layout()
out_path = OUT_DIR / "count_error_distribution.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved: {out_path}")
