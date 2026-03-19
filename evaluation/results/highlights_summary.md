# Resplit Statistics Highlights

Generated from `evaluation/results/` - 3,312 paired test cases, threshold = 0.5, `min_object_size_px = 10`

## Overall Read

- Both models were evaluated on the same resplit test set, so this is a direct comparison.
- nnU-Net is better on the main voxel and object metrics.
- U-Net is slightly better only in a few narrow respects:
  - less negative signed count error (`-1.07` vs `-1.72`), so it under-counts a bit less on average
  - fewer completely missed positive slices (`278` vs `331`)
- The main trade-off is that U-Net recovers a few more positive cases, but does so with more false positives on empty slices and weaker overlap and object quality overall.

## Per-Image Pixel Metrics (Mean +- SD, t = 0.5)

### All 3,312 test cases

| Metric | U-Net mean +- sd | nnU-Net mean +- sd | Better |
|---|---:|---:|---|
| Foreground Dice | 0.6362 +- 0.3568 | 0.6857 +- 0.3519 | nnU-Net |
| IoU | 0.5566 +- 0.3503 | 0.6141 +- 0.3503 | nnU-Net |
| Precision | 0.7164 +- 0.3658 | 0.7730 +- 0.3503 | nnU-Net |
| Recall | 0.6189 +- 0.3692 | 0.6590 +- 0.3646 | nnU-Net |
| Boundary F1 (2 px) | 0.6259 +- 0.3551 | 0.6844 +- 0.3537 | nnU-Net |
| Boundary F1 (5 px) | 0.6808 +- 0.3617 | 0.7291 +- 0.3538 | nnU-Net |
| ECE | 0.0197 +- 0.0333 | 0.0181 +- 0.0376 | nnU-Net (slightly) |
| HD95 (finite cases only, px) | 47.19 +- 68.84 | 40.35 +- 65.71 | nnU-Net |

### GT-positive slices only (n = 2,694)

| Metric | U-Net mean +- sd | nnU-Net mean +- sd | Better |
|---|---:|---:|---|
| Foreground Dice | 0.6122 +- 0.3307 | 0.6426 +- 0.3420 | nnU-Net |
| IoU | 0.5143 +- 0.3118 | 0.5545 +- 0.3264 | nnU-Net |
| Precision | 0.7107 +- 0.3469 | 0.7499 +- 0.3503 | nnU-Net |
| Recall | 0.5909 +- 0.3455 | 0.6098 +- 0.3537 | nnU-Net |
| Boundary F1 (2 px) | 0.5995 +- 0.3275 | 0.6409 +- 0.3441 | nnU-Net |
| Boundary F1 (5 px) | 0.6670 +- 0.3403 | 0.6959 +- 0.3502 | nnU-Net |
| ECE | 0.0230 +- 0.0347 | 0.0218 +- 0.0404 | nnU-Net (slightly) |
| HD95 (finite cases only, px) | 56.14 +- 71.66 | 49.56 +- 69.63 | nnU-Net |

## Per-Image Object Metrics (Mean +- SD, thresholded at default t = 0.5)

### All test cases

| Metric | U-Net mean +- sd | nnU-Net mean +- sd | Better |
|---|---:|---:|---|
| Count error \|pred-GT\| | 2.8684 +- 4.0174 | 2.6229 +- 3.7442 | nnU-Net |
| Count error signed (pred-GT) | -1.0676 +- 4.8196 | -1.7159 +- 4.2374 | U-Net (less under-counting) |
| Merge rate | 0.0410 +- 0.0991 | 0.0354 +- 0.0903 | nnU-Net |
| Split rate | 0.0364 +- 0.1312 | 0.0226 +- 0.1555 | nnU-Net |
| Connectivity error | 0.0774 +- 0.1765 | 0.0580 +- 0.1814 | nnU-Net |
| Panoptic Quality (PQ) | 0.4798 +- 0.3399 | 0.5508 +- 0.3403 | nnU-Net |
| Matched IoU mean | 0.4901 +- 0.3578 | 0.5194 +- 0.3752 | nnU-Net |
| Obj precision @0.5 | 0.6016 +- 0.3821 | 0.6946 +- 0.3767 | nnU-Net |
| Obj recall @0.5 | 0.5608 +- 0.3785 | 0.6062 +- 0.3717 | nnU-Net |

### GT-positive slices only

| Metric | U-Net mean +- sd | nnU-Net mean +- sd | Better |
|---|---:|---:|---|
| Count error \|pred-GT\| | 3.3563 +- 4.1634 | 3.1529 +- 3.9018 | nnU-Net |
| Count error signed (pred-GT) | -1.4826 +- 5.1385 | -2.1811 +- 4.5177 | U-Net (less under-counting) |
| Merge rate | 0.0504 +- 0.1077 | 0.0435 +- 0.0984 | nnU-Net |
| Split rate | 0.0448 +- 0.1442 | 0.0278 +- 0.1720 | nnU-Net |
| Connectivity error | 0.0952 +- 0.1913 | 0.0714 +- 0.1988 | nnU-Net |
| Panoptic Quality (PQ) | 0.4165 +- 0.2796 | 0.4760 +- 0.2961 | nnU-Net |
| Matched IoU mean | 0.6026 +- 0.2994 | 0.6385 +- 0.3114 | nnU-Net |
| Obj precision @0.5 | 0.5663 +- 0.3612 | 0.6527 +- 0.3746 | nnU-Net |
| Obj recall @0.5 | 0.5161 +- 0.3508 | 0.5440 +- 0.3528 | nnU-Net |

## Error Pattern Snapshot

| Pattern | U-Net | nnU-Net | Better |
|---|---:|---:|---|
| Zero-Dice slices | 551 | 493 | nnU-Net |
| Dice < 0.5 | 929 | 803 | nnU-Net |
| Dice > 0.9 | 1005 | 1295 | nnU-Net |
| Perfect Dice = 1.0 | 458 | 540 | nnU-Net |
| Predicted-empty slices | 736 | 871 | mixed |
| Both-empty correct slices | 458 | 540 | nnU-Net |
| Missed positive slices | 278 | 331 | U-Net |
| False positives on empty slices | 160 | 78 | nnU-Net |

Interpretation:

- U-Net misses fewer positive slices entirely, which is one genuine strength.
- nnU-Net handles empty/background-only slices much better, with many fewer false positives.
- The better nnU-Net Dice and IoU is not just a few easy wins - it also has fewer total failures and more very strong cases.

## Threshold Sensitivity (Foreground Dice Mean)

### All cases

- `t=0.30`  U-Net 0.6394  |  nnU-Net 0.6926
- `t=0.40`  U-Net 0.6379  |  nnU-Net 0.6885
- `t=0.50`  U-Net 0.6362  |  nnU-Net 0.6857
- `t=0.60`  U-Net 0.6340  |  nnU-Net 0.6806
- `t=0.70`  U-Net 0.6304  |  nnU-Net 0.6755

### GT-positive only

- `t=0.30`  U-Net 0.6209  |  nnU-Net 0.6566
- `t=0.40`  U-Net 0.6165  |  nnU-Net 0.6494
- `t=0.50`  U-Net 0.6122  |  nnU-Net 0.6426
- `t=0.60`  U-Net 0.6075  |  nnU-Net 0.6355
- `t=0.70`  U-Net 0.6021  |  nnU-Net 0.6277

nnU-Net stays ahead across the full threshold sweep, so the comparison is not being driven by a lucky default threshold.

## Key Deltas

- Foreground Dice: nnU-Net is higher by **0.0495** overall and **0.0304** on GT-positive slices.
- IoU: nnU-Net is higher by **0.0575** overall and **0.0402** on GT-positive slices.
- Panoptic Quality: nnU-Net is higher by **0.0710** overall and **0.0595** on GT-positive slices.
- Obj precision @0.5: nnU-Net is higher by **0.0929** overall.
- Obj recall @0.5: nnU-Net is higher by **0.0454** overall.
- HD95: nnU-Net is lower by **6.85 px** overall and **6.58 px** on GT-positive slices.

## Bottom Line

- On the corrected resplit test set, nnU-Net is the stronger model overall.
- The gap is consistent across voxel overlap, boundary accuracy, and object-level quality.
- The U-Net is still a strong improvement over the old dissertation U-Net, and it is competitive enough to make the comparison meaningful.
- The clearest remaining U-Net weakness is more false positives on empty slices and weaker object-level quality, even though it misses slightly fewer positive slices outright.
