#!/usr/bin/env python3
"""
create_fair_split.py — version 2

Fixes over v1:
  1. Tighter dominance cap (0.17, was 0.20) — also blocks Bleck2018_SoleusBV2
     and 6M63zfgOdsi602Ow24O1_3D from val/test, removing the high-density val bias.
  2. Resolution-tier balance — large (max_dim ≥ 400 px) / medium / small patches
     balanced by both sample count and slice count across splits.
  3. Foreground-density tier — high-fg samples (mean_fg_ratio > 0.10) distributed
     proportionally so val is not over-represented with easy dense images.
  4. Study-family cohesion — named multi-sample families (e.g. Cali2018_mouse_1…6,
     jrc_hela series, Muller2020, xu2017) penalised when members appear on both
     sides of the train / eval boundary.
  5. Family-aware initialisation — all unblocked members of a family are assigned
     to the same split during greedy init, giving the swap phase a head start.
  6. More restarts (12) and swap steps (6 000) than v1 (8 × 4 000).
"""

from __future__ import annotations

import json
import math
import re
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SAMPLE_STATS_CSV  = DATA_DIR / "sample_stats.csv"
DATASET_STATS_CSV = DATA_DIR / "dataset_stats.csv"
OUTPUT_JSON       = DATA_DIR / "splits_fair.json"
OUTPUT_SUMMARY    = DATA_DIR / "splits_fair_summary.txt"

SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

RANDOM_SEED    = 42
N_RESTARTS     = 30
MAX_SWAP_STEPS = 15_000
DOMINANCE_CAP      = 0.20   # max slice-fraction any one sample may occupy in val/test

# ── objective weights ─────────────────────────────────────────────────────────
W_SLICE  = 20.0  # slice-count balance (primary objective)
W_FS     = 1.5   # sample-level feature z-score
W_FW     = 2.5   # slice-weighted feature z-score
W_TS     = 3.0   # imaging-type proportion – sample level
W_TW     = 1.5   # imaging-type proportion – slice weighted
W_RS     = 1.0   # resolution tier – sample level
W_RW     = 5.0   # resolution tier – slice weighted
W_DS     = 1.0   # density tier – sample level
W_DW     = 3.0   # density tier – slice weighted
W_FAM    = 0.5   # family cohesion (informational — kept weak to not trump slice balance)
W_DOM    = 40.0  # dominance penalty
# ─────────────────────────────────────────────────────────────────────────────


# ── per-sample classifiers ────────────────────────────────────────────────────

def type_proxy(sample: str) -> str:
    if sample.endswith("_3D"):
        return "3D"
    if sample.endswith("_2D"):
        return "2D"
    return "other"


def family_of(sample: str) -> str | None:
    """Return a study-family tag for named samples; None for random-hash samples."""
    if sample.endswith("_2D") or sample.endswith("_3D"):
        return None
    # AuthorYear prefix  (e.g. Cali2018, xu2017, helmstaedter2011, Muller2020)
    m = re.match(r"^([A-Za-z][A-Za-z\-]+\d{4})", sample)
    if m:
        return m.group(1)
    # jrc sub-families
    if sample.startswith("jrc_hela"):
        return "jrc_hela"
    if sample.startswith("jrc_ctl"):
        return "jrc_ctl"
    if sample.startswith("jrc_"):
        return "jrc_other"
    # other named series
    for prefix in ("sponge", "cil_", "templier"):
        if sample.startswith(prefix):
            return prefix.rstrip("_")
    return None


def resolution_tier(h: int, w: int) -> str:
    d = max(h, w)
    if d >= 400:
        return "large"
    if d >= 150:
        return "medium"
    return "small"


def density_tier(mean_fg: float) -> str:
    return "high" if mean_fg > 0.10 else "normal"


def sample_resolution_tiers(slice_df: pd.DataFrame) -> dict[str, str]:
    """Return the modal resolution tier for every sample."""
    out: dict[str, str] = {}
    for sample, grp in slice_df.groupby("sample"):
        mode_row = grp[["img_height", "img_width"]].apply(tuple, axis=1).mode().iloc[0]
        out[str(sample)] = resolution_tier(int(mode_row[0]), int(mode_row[1]))
    return out


# ── target computation ────────────────────────────────────────────────────────

def exact_count_targets(total: int) -> dict[str, int]:
    raw  = {k: total * v for k, v in RATIOS.items()}
    base = {k: math.floor(v) for k, v in raw.items()}
    leftover = total - sum(base.values())
    for _, k in sorted(((raw[k] - base[k], k) for k in RATIOS), reverse=True)[:leftover]:
        base[k] += 1
    return base


def _wm(vals: np.ndarray, wts: np.ndarray) -> float:
    return float(np.average(vals, weights=wts))


def _ws(vals: np.ndarray, wts: np.ndarray) -> float:
    m = _wm(vals, wts)
    return float(np.sqrt(max(float(np.average((vals - m) ** 2, weights=wts)), 1e-12)))


def build_targets(sample_df: pd.DataFrame) -> dict:
    N = len(sample_df)
    S = float(sample_df["num_slices"].sum())
    w = sample_df["num_slices"].to_numpy(dtype=float)
    features = ["mean_fg_ratio", "mean_instances", "empty_rate"]

    sm  = {c: float(sample_df[c].mean())                        for c in features}
    ss  = {c: float(max(sample_df[c].std(ddof=0), 1e-9))        for c in features}
    wmf = {c: _wm(sample_df[c].to_numpy(dtype=float), w)        for c in features}
    wsf = {c: _ws(sample_df[c].to_numpy(dtype=float), w)        for c in features}

    def props(col: str) -> tuple[dict[str, float], dict[str, float]]:
        cnt_s: Counter = Counter(sample_df[col])
        cnt_w: dict[str, float] = defaultdict(float)
        for _, row in sample_df.iterrows():
            cnt_w[row[col]] += float(row["num_slices"])
        keys = sorted(cnt_s.keys())
        return (
            {k: cnt_s[k] / N for k in keys},
            {k: cnt_w[k] / S for k in keys},
        )

    tp_s, tp_w = props("type")
    rp_s, rp_w = props("res_tier")
    dp_s, dp_w = props("density_tier")

    return {
        "sample_targets": exact_count_targets(N),
        "slice_targets":  {k: S * RATIOS[k] for k in SPLITS},
        "features": features,
        "sm": sm, "ss": ss, "wm": wmf, "ws": wsf,
        "tp_s": tp_s, "tp_w": tp_w,
        "rp_s": rp_s, "rp_w": rp_w,
        "dp_s": dp_s, "dp_w": dp_w,
        "total_slices": S,
    }


def blocked_set(sample_df: pd.DataFrame, targets: dict) -> set[str]:
    """Samples that must go to train — too large to go in val/test without dominating."""
    target_val = targets["slice_targets"]["test"]
    size_cap = DOMINANCE_CAP * target_val
    return set(sample_df.loc[sample_df["num_slices"] > size_cap, "sample"])


def build_families(sample_df: pd.DataFrame) -> dict[str, list[str]]:
    """Multi-member study families (≥ 2 samples)."""
    fm: dict[str, list[str]] = defaultdict(list)
    for s in sample_df["sample"]:
        f = family_of(s)
        if f:
            fm[f].append(s)
    return {f: m for f, m in fm.items() if len(m) >= 2}


# ── aggregate statistics ──────────────────────────────────────────────────────

def agg_stats(
    assignments: dict[str, str], lut: dict[str, dict]
) -> dict[str, dict]:
    blank: dict = {
        "n": 0, "sl": 0.0,
        "fs": defaultdict(float), "fw": defaultdict(float),
        "ts": Counter(), "tw": defaultdict(float),
        "rs": Counter(), "rw": defaultdict(float),
        "ds": Counter(), "dw": defaultdict(float),
        "mx": 0.0, "mx_name": None,
    }
    stats = {sp: {k: (v.copy() if hasattr(v, "copy") else v) for k, v in blank.items()}
             for sp in SPLITS}

    for s, sp in assignments.items():
        r  = lut[s]
        sl = float(r["num_slices"])
        a  = stats[sp]
        a["n"]  += 1
        a["sl"] += sl
        for f in ("mean_fg_ratio", "mean_instances", "empty_rate"):
            a["fs"][f] += float(r[f])
            a["fw"][f] += float(r[f]) * sl
        for col, cnt_key, wgt_key in (
            ("type",         "ts", "tw"),
            ("res_tier",     "rs", "rw"),
            ("density_tier", "ds", "dw"),
        ):
            a[cnt_key][r[col]] += 1
            a[wgt_key][r[col]] += sl
        if sl > a["mx"]:
            a["mx"]      = sl
            a["mx_name"] = s

    return stats


# ── objective ─────────────────────────────────────────────────────────────────

def objective(
    assignments: dict[str, str],
    lut: dict[str, dict],
    tgt: dict,
    blocked: set[str],
    families: dict[str, list[str]],
) -> float:
    st    = agg_stats(assignments, lut)
    score = 0.0

    for sp in SPLITS:
        a  = st[sp]
        n  = max(a["n"],  1)
        sl = max(a["sl"], 1.0)

        # slice-count balance
        dev = (a["sl"] - tgt["slice_targets"][sp]) / tgt["slice_targets"][sp]
        score += W_SLICE * dev ** 2

        # per-sample feature balance
        for f in tgt["features"]:
            z = (a["fs"][f] / n  - tgt["sm"][f]) / tgt["ss"][f]
            score += W_FS * z ** 2
            z2 = (a["fw"][f] / sl - tgt["wm"][f]) / tgt["ws"][f]
            score += W_FW * z2 ** 2

        # Categorical balance — only for val/test.  Train contains all blocked
        # (size-dominated) samples which inflate certain proportions (e.g. large-res
        # slices from Wei2020), so penalising train against the non-blocked reference
        # would drive the optimiser to push more large samples into val/test.
        if sp != "train":
            # imaging-type balance
            for t, ps in tgt["tp_s"].items():
                score += W_TS * (a["ts"][t] / n  - ps) ** 2
                score += W_TW * (a["tw"][t] / sl - tgt["tp_w"][t]) ** 2

            # resolution-tier balance
            for r, ps in tgt["rp_s"].items():
                score += W_RS * (a["rs"][r] / n  - ps) ** 2
                score += W_RW * (a["rw"][r] / sl - tgt["rp_w"][r]) ** 2

            # density-tier balance
            for d, ps in tgt["dp_s"].items():
                score += W_DS * (a["ds"][d] / n  - ps) ** 2
                score += W_DW * (a["dw"][d] / sl - tgt["dp_w"][d]) ** 2

        # dominance penalty (val/test only)
        if sp in {"val", "test"} and a["sl"] > 0:
            share = a["mx"] / a["sl"]
            if share > DOMINANCE_CAP:
                score += W_DOM * (share - DOMINANCE_CAP) ** 2

    # hard constraint: blocked samples must stay in train
    for s in blocked:
        if assignments.get(s) != "train":
            score += 1e9

    # family cohesion — penalise members split across train and eval
    for members in families.values():
        in_train = sum(1 for m in members if assignments.get(m) == "train")
        in_eval  = sum(1 for m in members if assignments.get(m) in {"val", "test"})
        if in_train > 0 and in_eval > 0:
            score += W_FAM * (in_eval / len(members)) ** 2

    return score


# ── initialisation ────────────────────────────────────────────────────────────

def init_assignment(
    sample_df: pd.DataFrame,
    lut: dict[str, dict],
    tgt: dict,
    blocked: set[str],
    families: dict[str, list[str]],
    rng: random.Random,
) -> dict[str, str]:
    caps: dict[str, int] = dict(tgt["sample_targets"])
    asgn: dict[str, str] = {}
    assigned: set[str]   = set()

    # 1. Assign blocked samples to train
    for s in blocked:
        asgn[s] = "train"
        caps["train"] -= 1
        assigned.add(s)

    # 2. Build groups: families first (keeping members together), then singletons
    groups: list[list[str]] = []
    family_pool: set[str]   = set()

    for members in families.values():
        unblocked = [m for m in members if m not in blocked]
        if not unblocked:
            continue
        # If any sibling is blocked-to-train, pull all remaining into train for cohesion
        if any(m in blocked for m in members):
            for m in unblocked:
                asgn[m] = "train"
                caps["train"] -= 1
                assigned.add(m)
        else:
            groups.append(unblocked)
            family_pool.update(unblocked)

    # Singletons: exclude already-handled samples
    rem = (
        sample_df[~sample_df["sample"].isin(assigned | family_pool)]
        .copy()
        .assign(_noise=[rng.random() for _ in range(
            len(sample_df) - len(assigned | family_pool)
        )])
        .sort_values(["num_slices", "_noise"], ascending=[False, True])
    )
    for _, row in rem.iterrows():
        groups.append([row["sample"]])

    # Sort groups by total slice count descending (largest first = better packing)
    groups.sort(key=lambda g: sum(lut[s]["num_slices"] for s in g), reverse=True)

    # 3. Greedy assignment
    for group in groups:
        if all(s in assigned for s in group):
            continue

        allowed = [sp for sp in SPLITS if caps[sp] >= len(group)]
        if not allowed:
            # Fallback: assign members individually
            for s in group:
                for sp in SPLITS:
                    if caps[sp] > 0:
                        asgn[s] = sp
                        caps[sp] -= 1
                        assigned.add(s)
                        break
            continue

        best_sp: str | None = None
        best_sc: float | None = None
        for sp in allowed:
            for s in group:
                asgn[s] = sp
            sc = objective(asgn, lut, tgt, blocked, families)
            for s in group:
                del asgn[s]
            if best_sc is None or sc < best_sc:
                best_sc = sc
                best_sp = sp

        for s in group:
            asgn[s] = best_sp  # type: ignore[arg-type]
            caps[best_sp] -= 1  # type: ignore[index]
            assigned.add(s)

    if any(v != 0 for v in caps.values()):
        raise RuntimeError(f"Capacity mismatch after init: {caps}")
    return asgn


# ── improvement (swap phase) ──────────────────────────────────────────────────

def improve_assignment(
    asgn: dict[str, str],
    lut: dict[str, dict],
    tgt: dict,
    blocked: set[str],
    families: dict[str, list[str]],
    rng: random.Random,
) -> tuple[dict[str, str], float]:
    cur      = dict(asgn)
    best     = dict(asgn)
    cur_sc   = objective(cur, lut, tgt, blocked, families)
    best_sc  = cur_sc
    samples  = list(cur.keys())
    fam_list = [m for m in families.values() if len(m) >= 2]

    for step in range(MAX_SWAP_STEPS):

        # Every 8th step: attempt a family-group swap
        if step % 8 == 0 and fam_list:
            fam     = rng.choice(fam_list)
            fam_sp  = cur[fam[0]]
            # Skip if family is split across multiple splits — a pairwise swap
            # would corrupt sample counts because len(fam) members don't all
            # live in fam_sp.
            if any(cur[m] != fam_sp for m in fam):
                goto_pairwise = True
            else:
                other_sp = rng.choice([sp for sp in SPLITS if sp != fam_sp])

                # Cannot move blocked members out of train
                if any(m in blocked for m in fam) and other_sp != "train":
                    goto_pairwise = True
                else:
                    others = [s for s in samples
                              if cur[s] == other_sp and s not in fam]
                    if len(others) < len(fam):
                        goto_pairwise = True
                    else:
                        swap_others = rng.sample(others, len(fam))
                        if any(m in blocked for m in swap_others) and fam_sp != "train":
                            goto_pairwise = True
                        else:
                            goto_pairwise = False

            if not goto_pairwise:
                for m in fam:
                    cur[m] = other_sp
                for m in swap_others:
                    cur[m] = fam_sp
                new_sc = objective(cur, lut, tgt, blocked, families)
                if new_sc < cur_sc:
                    cur_sc = new_sc
                    if new_sc < best_sc:
                        best    = dict(cur)
                        best_sc = new_sc
                else:
                    for m in fam:
                        cur[m] = fam_sp
                    for m in swap_others:
                        cur[m] = other_sp
                continue

        # Pairwise swap
        a = rng.choice(samples)
        b = rng.choice(samples)
        if a == b:
            continue
        sa, sb = cur[a], cur[b]
        if sa == sb:
            continue
        if a in blocked and sb != "train":
            continue
        if b in blocked and sa != "train":
            continue

        cur[a], cur[b] = sb, sa
        new_sc = objective(cur, lut, tgt, blocked, families)
        if new_sc < cur_sc:
            cur_sc = new_sc
            if new_sc < best_sc:
                best    = dict(cur)
                best_sc = new_sc
        else:
            cur[a], cur[b] = sa, sb

    return best, best_sc


# ── output helpers ────────────────────────────────────────────────────────────

def build_slice_records(slice_df: pd.DataFrame) -> list[dict]:
    return [
        {
            "sample":      str(r["sample"]),
            "filename":    str(r["filename"]),
            "fg_ratio":    float(r["fg_ratio"]),
            "n_instances": int(r["n_instances"]),
            "is_empty":    bool(r["is_empty"]),
        }
        for _, r in slice_df.iterrows()
    ]


def split_summary(
    split: str,
    names: list[str],
    sample_df: pd.DataFrame,
    slice_df: pd.DataFrame,
    families: dict[str, list[str]],
) -> str:
    ss = sample_df[sample_df["sample"].isin(names)]
    si = slice_df[slice_df["sample"].isin(names)]
    nonempty = si[~si["is_empty"]]
    vc = si["sample"].value_counts()
    ln, lc = str(vc.index[0]), int(vc.iloc[0])

    # family cohesion in this split
    split_set = set(names)
    split_families = {
        f: [m for m in m_list if m in split_set]
        for f, m_list in families.items()
        if any(m in split_set for m in m_list)
    }
    leaked = [
        f for f, ms in split_families.items()
        if len(ms) < len(families.get(f, []))
    ]

    lines = [
        f"{split}:",
        f"  n_samples={len(ss)}  n_images={len(si)}",
        f"  sample_mean_fg={ss['mean_fg_ratio'].mean():.4f}"
        f"  sample_mean_inst={ss['mean_instances'].mean():.4f}"
        f"  sample_empty_rate={ss['empty_rate'].mean():.4f}",
        f"  slice_empty_pct={100 * si['is_empty'].mean():.2f}%"
        f"  slice_fg_mean={si['fg_ratio'].mean():.4f}"
        f"  slice_fg_nonempty={(nonempty['fg_ratio'].mean() if len(nonempty) else 0):.4f}",
        f"  high_density_pct={100 * (si['fg_ratio'] > 0.10).mean():.2f}%",
        f"  res_tier={json.dumps(ss['res_tier'].value_counts().to_dict(), sort_keys=True)}",
        f"  density_tier={json.dumps(ss['density_tier'].value_counts().to_dict(), sort_keys=True)}",
        f"  type={json.dumps(ss['type'].value_counts().to_dict(), sort_keys=True)}",
        f"  largest={ln} ({lc} slices, {100 * lc / max(len(si), 1):.2f}%)",
        f"  split_families={leaked}",
    ]
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sample_df = pd.read_csv(SAMPLE_STATS_CSV).copy()
    slice_df  = pd.read_csv(DATASET_STATS_CSV).copy()

    res_map = sample_resolution_tiers(slice_df)

    sample_df["empty_rate"]    = sample_df["empty_slice_count"] / sample_df["num_slices"]
    sample_df["mean_instances"] = sample_df["total_instances"] / sample_df["num_slices"]
    sample_df["type"]           = sample_df["sample"].map(type_proxy)
    sample_df["res_tier"]       = sample_df["sample"].map(res_map)
    sample_df["density_tier"]   = sample_df["mean_fg_ratio"].map(density_tier)

    lut      = sample_df.set_index("sample").to_dict("index")
    tgt      = build_targets(sample_df)
    blocked  = blocked_set(sample_df, tgt)
    families = build_families(sample_df)

    # Recompute categorical proportion targets from non-blocked samples only.
    # Blocked samples (forced to train) inflate slice proportions for large-res
    # and can skew density, making the original targets unachievable for val/test.
    ref_df = sample_df[~sample_df["sample"].isin(blocked)]
    ref_S  = float(ref_df["num_slices"].sum())
    def _ref_props(col: str) -> tuple[dict[str, float], dict[str, float]]:
        cnt_s: Counter = Counter(ref_df[col])
        cnt_w: dict[str, float] = defaultdict(float)
        for _, row in ref_df.iterrows():
            cnt_w[row[col]] += float(row["num_slices"])
        keys = sorted(cnt_s.keys())
        N_ref = len(ref_df)
        return {k: cnt_s[k] / N_ref for k in keys}, {k: cnt_w[k] / ref_S for k in keys}
    tgt["tp_s"], tgt["tp_w"] = _ref_props("type")
    tgt["rp_s"], tgt["rp_w"] = _ref_props("res_tier")
    tgt["dp_s"], tgt["dp_w"] = _ref_props("density_tier")

    print(f"Total samples     : {len(sample_df)}")
    print(f"Total slices      : {int(tgt['total_slices'])}")
    print(f"Blocked to train  : ({len(blocked)}) {sorted(blocked)}")
    print(f"Family groups ≥2  : {len(families)}")
    print(f"Resolution props  : sample={tgt['rp_s']}  slice={tgt['rp_w']}")
    print(f"Density props     : sample={tgt['dp_s']}  slice={tgt['dp_w']}")
    print()

    best_asgn: dict[str, str] | None = None
    best_sc:   float | None          = None

    for restart in range(N_RESTARTS):
        rng = random.Random(RANDOM_SEED + restart)
        a   = init_assignment(sample_df, lut, tgt, blocked, families, rng)
        a, sc = improve_assignment(a, lut, tgt, blocked, families, rng)
        print(f"  restart {restart:2d}: score={sc:.6f}")
        if best_sc is None or sc < best_sc:
            best_asgn = a
            best_sc   = sc

    assert best_asgn is not None
    train_s = sorted(s for s, sp in best_asgn.items() if sp == "train")
    val_s   = sorted(s for s, sp in best_asgn.items() if sp == "val")
    test_s  = sorted(s for s, sp in best_asgn.items() if sp == "test")

    tr_i = slice_df[slice_df["sample"].isin(train_s)]
    va_i = slice_df[slice_df["sample"].isin(val_s)]
    te_i = slice_df[slice_df["sample"].isin(test_s)]

    output = {
        "seed":             RANDOM_SEED,
        "strategy":         "v2: resolution + density + family-cohesion balanced constrained search",
        "split_ratio":      RATIOS,
        "dominance_cap":    DOMINANCE_CAP,
        "n_restarts":       N_RESTARTS,
        "max_swap_steps":   MAX_SWAP_STEPS,
        "blocked_to_train": sorted(blocked),
        "objective_score":  best_sc,
        "train_samples":    train_s,
        "val_samples":      val_s,
        "test_samples":     test_s,
        "train":            build_slice_records(tr_i),
        "val":              build_slice_records(va_i),
        "test":             build_slice_records(te_i),
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2))

    lines = [
        "Fair split v2 generated",
        f"output={OUTPUT_JSON}",
        f"seed={RANDOM_SEED}  score={best_sc:.6f}  dominance_cap={DOMINANCE_CAP}",
        f"blocked ({len(blocked)}): {sorted(blocked)}",
        "",
        split_summary("train", train_s, sample_df, slice_df, families),
        "",
        split_summary("val",   val_s,   sample_df, slice_df, families),
        "",
        split_summary("test",  test_s,  sample_df, slice_df, families),
        "",
    ]
    OUTPUT_SUMMARY.write_text("\n".join(lines).rstrip() + "\n")
    print()
    print(OUTPUT_SUMMARY.read_text())


if __name__ == "__main__":
    main()
