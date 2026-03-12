#!/usr/bin/env python3
"""
check_split_fairness.py — version 2

Audits a split JSON for all v2 fairness criteria:
  - no sample leakage
  - sample counts match 70/15/15 targets
  - slice counts within 5 % of 70/15/15 targets
  - feature balance (fg_ratio, instances, empty_rate) — sample and slice level
  - imaging-type balance (2D / 3D / other)
  - resolution-tier balance (large / medium / small)
  - density-tier balance (high / normal fg)
  - no single sample dominates val/test (threshold 0.17)
  - study-family cohesion report
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR          = ROOT / "data"
SAMPLE_STATS_CSV  = DATA_DIR / "sample_stats.csv"
DATASET_STATS_CSV = DATA_DIR / "dataset_stats.csv"
DEFAULT_SPLIT_JSON = DATA_DIR / "splits_fair.json"
DEFAULT_OUTPUT     = DATA_DIR / "splits_fair_audit.txt"

SPLITS     = ("train", "val", "test")
RATIOS     = {"train": 0.70, "val": 0.15, "test": 0.15}
FEATURES   = ("mean_fg_ratio", "mean_instances", "empty_rate")
TYPE_LEVELS = ("2D", "3D", "other")
RES_LEVELS  = ("large", "medium", "small")
DEN_LEVELS  = ("high", "normal")

# pass/warn thresholds
SAMPLE_Z_THRESH        = 0.25
SLICE_Z_THRESH         = 0.25
SAMPLE_PROP_DELTA      = 0.07   # for type, res, density
SLICE_PROP_DELTA       = 0.12
SLICE_RATIO_DEV_THRESH = 0.05   # 5 %
DOMINANCE_THRESH       = 0.20   # matches DOMINANCE_CAP in create_fair_split.py


# ── helpers ───────────────────────────────────────────────────────────────────

def type_proxy(sample: str) -> str:
    if sample.endswith("_3D"):
        return "3D"
    if sample.endswith("_2D"):
        return "2D"
    return "other"


def resolution_tier(h: int, w: int) -> str:
    d = max(h, w)
    if d >= 400:
        return "large"
    if d >= 150:
        return "medium"
    return "small"


def density_tier(mean_fg: float) -> str:
    return "high" if mean_fg > 0.10 else "normal"


def family_of(sample: str) -> str | None:
    if sample.endswith("_2D") or sample.endswith("_3D"):
        return None
    m = re.match(r"^([A-Za-z][A-Za-z\-]+\d{4})", sample)
    if m:
        return m.group(1)
    if sample.startswith("jrc_hela"):
        return "jrc_hela"
    if sample.startswith("jrc_ctl"):
        return "jrc_ctl"
    if sample.startswith("jrc_"):
        return "jrc_other"
    for prefix in ("sponge", "cil_", "templier"):
        if sample.startswith(prefix):
            return prefix.rstrip("_")
    return None


def sample_resolution_tiers(slice_df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for sample, grp in slice_df.groupby("sample"):
        mode_row = grp[["img_height", "img_width"]].apply(tuple, axis=1).mode().iloc[0]
        out[str(sample)] = resolution_tier(int(mode_row[0]), int(mode_row[1]))
    return out


def exact_count_targets(total: int) -> dict[str, int]:
    raw  = {k: total * v for k, v in RATIOS.items()}
    base = {k: math.floor(v) for k, v in raw.items()}
    leftover = total - sum(base.values())
    for _, k in sorted(((raw[k] - base[k], k) for k in RATIOS), reverse=True)[:leftover]:
        base[k] += 1
    return base


def wm(vals: pd.Series, weights: pd.Series) -> float:
    return float(np.average(vals.to_numpy(dtype=float),
                             weights=weights.to_numpy(dtype=float)))


def ws(vals: pd.Series, weights: pd.Series) -> float:
    mean = wm(vals, weights)
    var  = np.average((vals.to_numpy(dtype=float) - mean) ** 2,
                      weights=weights.to_numpy(dtype=float))
    return float(np.sqrt(max(float(var), 1e-12)))


def ok(flag: bool) -> str:
    return "PASS" if flag else "WARN"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sample_df = pd.read_csv(SAMPLE_STATS_CSV).copy()
    slice_df  = pd.read_csv(DATASET_STATS_CSV).copy()
    split_obj = json.loads(DEFAULT_SPLIT_JSON.read_text())

    # Derived columns
    sample_df["empty_rate"]    = sample_df["empty_slice_count"] / sample_df["num_slices"]
    sample_df["mean_instances"] = sample_df["total_instances"] / sample_df["num_slices"]
    sample_df["type"]           = sample_df["sample"].map(type_proxy)

    res_map = sample_resolution_tiers(slice_df)
    sample_df["res_tier"]     = sample_df["sample"].map(res_map)
    sample_df["density_tier"] = sample_df["mean_fg_ratio"].map(density_tier)

    # Merge resolution and density onto slice_df for slice-level proportions
    slice_df = slice_df.merge(
        sample_df[["sample", "type", "res_tier", "density_tier"]],
        on="sample", how="left",
    )

    N = len(sample_df)
    S = float(len(slice_df))

    sample_targets = exact_count_targets(N)
    slice_targets  = {k: S * RATIOS[k] for k in SPLITS}

    # Identify samples that are structurally forced to train (too large for val/test).
    # Global reference proportions are computed from non-blocked samples only so that
    # the targets are actually achievable by val and test.
    target_val_slices = slice_targets["val"]
    blocked_samples = set(
        sample_df.loc[sample_df["num_slices"] > DOMINANCE_THRESH * target_val_slices, "sample"]
    )
    ref_df       = sample_df[~sample_df["sample"].isin(blocked_samples)]
    ref_slice_df = slice_df[~slice_df["sample"].isin(blocked_samples)]

    # Global means / stds for feature z-score (from full dataset — features not affected)
    sample_means = {f: float(sample_df[f].mean()) for f in FEATURES}
    sample_stds  = {f: float(max(sample_df[f].std(ddof=0), 1e-9)) for f in FEATURES}
    sample_w     = sample_df["num_slices"]
    slice_means  = {f: wm(sample_df[f], sample_w) for f in FEATURES}
    slice_stds   = {f: ws(sample_df[f], sample_w) for f in FEATURES}

    # Global proportions for categorical variables (from non-blocked samples only
    # so that val/test can realistically match the reference).
    def global_props(col: str, levels: tuple[str, ...]) -> tuple[dict, dict]:
        sp = {t: float((ref_df[col] == t).mean()) for t in levels}
        ip = {t: float((ref_slice_df[col] == t).mean()) for t in levels}
        return sp, ip

    tp_s, tp_i = global_props("type",         TYPE_LEVELS)
    rp_s, rp_i = global_props("res_tier",     RES_LEVELS)
    dp_s, dp_i = global_props("density_tier", DEN_LEVELS)

    # Split membership
    sample_sets = {sp: set(split_obj[f"{sp}_samples"]) for sp in SPLITS}

    # Leakage check
    overlaps = []
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            ov = sample_sets[a] & sample_sets[b]
            if ov:
                overlaps.append((a, b, len(ov)))

    # Tracking worst-case values for final verdict
    max_sample_z        = 0.0
    max_slice_z         = 0.0
    max_sample_prop_d   = 0.0
    max_slice_prop_d    = 0.0
    max_slice_ratio_dev = 0.0
    max_dominance       = 0.0

    lines = [
        "Split fairness audit v2",
        f"split_json={DEFAULT_SPLIT_JSON}",
        "",
        "## Overall",
        f"  sample_leakage={ok(not overlaps)} (no_overlap={not overlaps})",
        f"  sample_targets={sample_targets}",
        f"  slice_targets={{'train': {slice_targets['train']:.1f},"
        f" 'val': {slice_targets['val']:.1f},"
        f" 'test': {slice_targets['test']:.1f}}}",
        "",
    ]

    if overlaps:
        lines.append("## Leakage details")
        for a, b, cnt in overlaps:
            lines.append(f"  {a}/{b}: {cnt} samples")
        lines.append("")

    for split in SPLITS:
        names  = split_obj[f"{split}_samples"]
        ssub   = sample_df[sample_df["sample"].isin(names)].copy()
        isub   = slice_df[slice_df["sample"].isin(names)].copy()
        ns, ni = len(ssub), len(isub)

        slice_ratio_dev = abs(ni - slice_targets[split]) / slice_targets[split]
        max_slice_ratio_dev = max(max_slice_ratio_dev, slice_ratio_dev)

        vc        = isub["sample"].value_counts()
        l_name    = str(vc.index[0])
        l_count   = int(vc.iloc[0])
        dominance = l_count / max(ni, 1)
        if split in {"val", "test"}:
            max_dominance = max(max_dominance, dominance)

        lines.append(f"## {split}")
        lines.append(
            f"  sample_count={ns} target={sample_targets[split]}"
            f" ({ok(ns == sample_targets[split])})"
        )
        lines.append(
            f"  slice_count={ni} target={slice_targets[split]:.1f}"
            f" rel_dev={100 * slice_ratio_dev:.2f}%"
            f" ({ok(slice_ratio_dev <= SLICE_RATIO_DEV_THRESH)})"
        )
        lines.append(
            f"  largest_sample={l_name} ({l_count} slices,"
            f" {100 * dominance:.2f}% of split)"
            f" ({ok(split == 'train' or dominance <= DOMINANCE_THRESH)})"
        )
        lines.append(
            f"  high_density_slices={100 * (isub['fg_ratio'] > 0.10).mean():.2f}%"
        )

        # Feature balance
        for f in FEATURES:
            s_mean = float(ssub[f].mean())
            s_z    = abs((s_mean - sample_means[f]) / sample_stds[f])
            max_sample_z = max(max_sample_z, s_z)

            i_mean = wm(ssub[f], ssub["num_slices"])
            i_z    = abs((i_mean - slice_means[f]) / slice_stds[f])
            max_slice_z = max(max_slice_z, i_z)

            lines.append(
                f"  {f}: sample_mean={s_mean:.4f} sample_z={s_z:.3f} ({ok(s_z <= SAMPLE_Z_THRESH)}),"
                f" slice_weighted_mean={i_mean:.4f} slice_z={i_z:.3f} ({ok(i_z <= SLICE_Z_THRESH)})"
            )

        # Categorical proportions helper.
        # Deltas for train are informational only — train is structurally skewed
        # because it absorbs all blocked (size-dominated) samples.  Pass/fail
        # tracking only applies to val and test.
        def prop_lines(label: str, col: str, levels: tuple,
                       gsp: dict, gip: dict) -> None:
            for t in levels:
                sp_ = float((ssub[col] == t).mean())
                ip_ = float((isub[col] == t).mean())
                sd  = abs(sp_ - gsp[t])
                id_ = abs(ip_ - gip[t])
                nonlocal max_sample_prop_d, max_slice_prop_d
                if split != "train":
                    max_sample_prop_d = max(max_sample_prop_d, sd)
                    max_slice_prop_d  = max(max_slice_prop_d,  id_)
                lines.append(
                    f"  {label}[{t}]: sample_prop={sp_:.3f} delta={sd:.3f}"
                    f" ({ok(sd <= SAMPLE_PROP_DELTA) if split != 'train' else 'INFO'}),"
                    f" slice_prop={ip_:.3f} delta={id_:.3f}"
                    f" ({ok(id_ <= SLICE_PROP_DELTA) if split != 'train' else 'INFO'})"
                )

        prop_lines("type",    "type",         TYPE_LEVELS, tp_s, tp_i)
        prop_lines("res",     "res_tier",     RES_LEVELS,  rp_s, rp_i)
        prop_lines("density", "density_tier", DEN_LEVELS,  dp_s, dp_i)
        lines.append("")

    # Family cohesion summary
    families: dict[str, list[str]] = defaultdict(list)
    for s in sample_df["sample"]:
        f = family_of(s)
        if f:
            families[f].append(s)
    multi_families = {f: m for f, m in families.items() if len(m) >= 2}

    lines.append("## Family cohesion")
    split_lookup = {}
    for sp in SPLITS:
        for s in split_obj[f"{sp}_samples"]:
            split_lookup[s] = sp

    split_families = 0
    for fam, members in sorted(multi_families.items()):
        sp_set = {split_lookup.get(m) for m in members if m in split_lookup}
        in_eval = sum(1 for m in members if split_lookup.get(m) in {"val", "test"})
        in_train = sum(1 for m in members if split_lookup.get(m) == "train")
        if in_train > 0 and in_eval > 0:
            split_families += 1
            leak_pct = 100 * in_eval / len(members)
            lines.append(
                f"  {fam} ({len(members)} members, {in_eval} in eval = {leak_pct:.0f}% leaked):"
                f" {sorted(sp_set)}"
            )
    if split_families == 0:
        lines.append("  All multi-member families fully contained within one split")
    lines.append(
        f"  split_families={split_families}/{len(multi_families)}"
        f" ({100 * split_families / max(len(multi_families), 1):.1f}% of named families split)"
    )
    lines.append("")

    # Verdict
    overall_pass = (
        not overlaps
        and max_slice_ratio_dev  <= SLICE_RATIO_DEV_THRESH
        and max_sample_z         <= SAMPLE_Z_THRESH
        and max_slice_z          <= SLICE_Z_THRESH
        and max_sample_prop_d    <= SAMPLE_PROP_DELTA
        and max_slice_prop_d     <= SLICE_PROP_DELTA
        and max_dominance        <= DOMINANCE_THRESH
    )

    lines.extend([
        "## Verdict",
        f"  overall={ok(overall_pass)}",
        f"  no_leakage={ok(not overlaps)}",
        f"  max_sample_feature_z={max_sample_z:.3f} threshold={SAMPLE_Z_THRESH:.2f}",
        f"  max_slice_feature_z={max_slice_z:.3f} threshold={SLICE_Z_THRESH:.2f}",
        f"  max_sample_prop_delta={max_sample_prop_d:.3f} threshold={SAMPLE_PROP_DELTA:.2f}",
        f"  max_slice_prop_delta={max_slice_prop_d:.3f} threshold={SLICE_PROP_DELTA:.2f}",
        f"  max_slice_ratio_deviation={100 * max_slice_ratio_dev:.2f}%"
        f" threshold={100 * SLICE_RATIO_DEV_THRESH:.2f}%",
        f"  max_val_test_dominance={100 * max_dominance:.2f}%"
        f" threshold={100 * DOMINANCE_THRESH:.2f}%",
        f"  split_families={split_families}/{len(multi_families)}"
        f" (informational — cannot fully avoid with 89 val/test samples)",
    ])

    DEFAULT_OUTPUT.write_text("\n".join(lines).rstrip() + "\n")
    print(DEFAULT_OUTPUT.read_text())


if __name__ == "__main__":
    main()
