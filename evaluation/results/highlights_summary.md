# Evaluation Highlights

This summary documents the current unsuffixed evaluation outputs.

## Scope

- Test cases evaluated: `3312 / 3312` for both models
- Ground-truth-positive cases: `2694`

## Output Files

- `unet_aggregate.csv`
- `nnunet_aggregate.csv`
- `unet_aggregate_gt_positive.csv`
- `nnunet_aggregate_gt_positive.csv`
- `unet_summary.json`
- `nnunet_summary.json`
- `unet_corner_case_summary.json`
- `nnunet_corner_case_summary.json`
- `corner_cases_combined.csv`
- `stats_tests.csv`
- `stats_tests_summary.json`
- `stratified_fg_quartile.csv`
- `stratified_obj_count_quartile.csv`
- `stratified_sample_type.csv`
- `stratified_empty_vs_nonempty.csv`
- `stratified_summary.json`

## Headline Result

- Metrics tested: `27`
- Significant after BH correction: `27 / 27`
- Metrics favouring nnU-Net: `27 / 27`
- Metrics favouring U-Net: `0 / 27`

## Main Pixel Metrics at Threshold 0.5

| Metric | U-Net | nnU-Net |
|---|---|---|
| Foreground Dice | 0.5778 | 0.6245 |
| IoU | 0.4854 | 0.5389 |
| Precision | 0.6709 | 0.7288 |
| Recall | 0.5578 | 0.5926 |
| HD95 | 47.19 | 40.35 |
| Boundary F1 (2 px) | 0.6259 | 0.6844 |
| Boundary F1 (5 px) | 0.6808 | 0.7291 |
| ECE | 0.0197 | 0.0181 |

## Main Object Metrics

| Metric | U-Net | nnU-Net |
|---|---|---|
| Predicted objects per slice | 6.6078 | 5.1274 |
| Ground-truth objects per slice | 6.5704 | 6.5704 |
| Object count error | 3.0658 | 2.5589 |
| Signed count error | 0.0374 | -1.4429 |
| Merge rate | 0.0410 | 0.0354 |
| Split rate | 0.0910 | 0.0406 |
| Connectivity error | 0.1320 | 0.0760 |
| Matched IoU mean | 0.4818 | 0.5140 |
| PQ (IoU > 0.5) | 0.4594 | 0.5436 |
| Object precision @0.5 | 0.5570 | 0.6756 |
| Object recall @0.5 | 0.5581 | 0.6056 |
| Object precision @0.9 | 0.2468 | 0.3291 |
| Object recall @0.9 | 0.2455 | 0.3080 |

## GT-Positive Object Metrics

| Metric | U-Net | nnU-Net |
|---|---|---|
| Predicted objects per slice | 7.8938 | 6.2264 |
| Ground-truth objects per slice | 8.0776 | 8.0776 |
| Object count error | 3.5393 | 3.0687 |
| Merge rate | 0.0504 | 0.0435 |
| Split rate | 0.1119 | 0.0499 |
| Connectivity error | 0.1623 | 0.0935 |
| Matched IoU mean | 0.5923 | 0.6319 |
| PQ (IoU > 0.5) | 0.3948 | 0.4678 |
| Object precision @0.5 | 0.5148 | 0.6302 |
| Object recall @0.5 | 0.5161 | 0.5440 |
| Object precision @0.9 | 0.1334 | 0.2042 |
| Object recall @0.9 | 0.1319 | 0.1782 |

## High-Dice Corner Cases

- Rule: `Dice > 0.85` and (`obj_count_error > 4` or `merge_rate > 0` or `split_rate > 0`)
- U-Net usable cases: `2854`
- U-Net high-Dice cases: `882`
- U-Net corner cases among usable cases: `373 / 2854 = 13.07%`
- U-Net corner cases among high-Dice cases: `373 / 882 = 42.3%`
- nnU-Net usable cases: `2772`
- nnU-Net high-Dice cases: `1144`
- nnU-Net corner cases among usable cases: `436 / 2772 = 15.73%`
- nnU-Net corner cases among high-Dice cases: `436 / 1144 = 38.1%`

## Stratified Analysis Highlights

| Quartile | Dice U-Net | Dice nnU-Net | PQ U-Net | PQ nnU-Net | Split U-Net | Split nnU-Net |
|---|---|---|---|---|---|---|
| Q1 (low) | 0.398 | 0.435 | 0.290 | 0.325 | 0.039 | 0.020 |
| Q2 | 0.570 | 0.605 | 0.385 | 0.445 | 0.067 | 0.051 |
| Q3 | 0.722 | 0.767 | 0.486 | 0.591 | 0.109 | 0.046 |
| Q4 (high) | 0.759 | 0.763 | 0.419 | 0.510 | 0.231 | 0.083 |

- In dense slices (`Q4`), Dice is close: `0.759` vs `0.763`.
- The object-level gap remains large in the same slices: PQ `0.419` vs `0.510`, split rate `0.231` vs `0.083`.

## Bottom Line

- The current connected-component evaluation preserves the ranking: nnU-Net remains better across the tested metrics.
- The stratified results show that high voxel overlap in dense slices still understates the object-level gap.
