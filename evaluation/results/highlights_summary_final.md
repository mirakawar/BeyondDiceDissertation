# No-Filter Evaluation Highlights (`_final`)

This summary documents the no-filter object-level evaluation produced by the `_final` pipeline.

## Scope

- Evaluation setting: `min-object-size = 1`
- Test cases evaluated: `3312 / 3312` for both models
- Ground-truth-positive cases: `2694`

## Output Files

- `corner_cases_combined_final.csv`
- `nnunet_aggregate_final.csv`
- `nnunet_aggregate_gt_empty_final.csv`
- `nnunet_aggregate_gt_positive_final.csv`
- `nnunet_corner_cases_final.csv`
- `nnunet_per_image_final.csv`
- `stats_tests_final.csv`
- `stratified_empty_vs_nonempty_final.csv`
- `stratified_fg_quartile_final.csv`
- `stratified_obj_count_quartile_final.csv`
- `stratified_sample_type_final.csv`
- `unet_aggregate_final.csv`
- `unet_aggregate_gt_empty_final.csv`
- `unet_aggregate_gt_positive_final.csv`
- `unet_corner_cases_final.csv`
- `unet_per_image_final.csv`
- `nnunet_corner_case_summary_final.json`
- `nnunet_summary_final.json`
- `stats_tests_summary_final.json`
- `stratified_summary_final.json`
- `unet_corner_case_summary_final.json`
- `unet_summary_final.json`

## Headline Result

- Metrics tested: `27`
- Significant after BH correction: `27 / 27`
- Metrics favouring nnU-Net: `27 / 27`
- Metrics favouring U-Net: `0 / 27`

## Main Pixel Metrics at Threshold 0.5

| Metric | U-Net | nnU-Net |
|---|---|---|
| Foreground Dice | 0.6362 | 0.6857 |
| IoU | 0.5566 | 0.6141 |
| Precision | 0.7164 | 0.7730 |
| Recall | 0.6189 | 0.6590 |
| HD95 | 47.19 | 40.35 |
| Boundary F1 (2 px) | 0.6259 | 0.6844 |
| Boundary F1 (5 px) | 0.6808 | 0.7291 |
| ECE | 0.0197 | 0.0181 |

## Main Object Metrics

| Metric | U-Net | nnU-Net |
|---|---|---|
| Predicted objects per slice | 5.5027 | 4.8545 |
| Ground-truth objects per slice | 6.5704 | 6.5704 |
| Object count error | 2.8684 | 2.6229 |
| Signed count error | -1.0676 | -1.7159 |
| Merge rate | 0.0410 | 0.0354 |
| Split rate | 0.0364 | 0.0226 |
| Connectivity error | 0.0774 | 0.0580 |
| Matched IoU mean | 0.4901 | 0.5194 |
| PQ (IoU > 0.5) | 0.4798 | 0.5508 |
| Object precision @0.5 | 0.6016 | 0.6946 |
| Object recall @0.5 | 0.5608 | 0.6062 |
| Object precision @0.9 | 0.2589 | 0.3349 |
| Object recall @0.9 | 0.2483 | 0.3086 |

## GT-Positive Object Metrics

| Metric | U-Net | nnU-Net |
|---|---|---|
| Predicted objects per slice | 6.5950 | 5.8964 |
| Ground-truth objects per slice | 8.0776 | 8.0776 |
| Object count error | 3.3563 | 3.1529 |
| Merge rate | 0.0504 | 0.0435 |
| Split rate | 0.0448 | 0.0278 |
| Connectivity error | 0.0952 | 0.0714 |
| Matched IoU mean | 0.6026 | 0.6385 |
| PQ (IoU > 0.5) | 0.4165 | 0.4760 |
| Object precision @0.5 | 0.5663 | 0.6527 |
| Object recall @0.5 | 0.5161 | 0.5440 |
| Object precision @0.9 | 0.1449 | 0.2106 |
| Object recall @0.9 | 0.1319 | 0.1782 |

## High-Dice Corner Cases

- Rule: `Dice > 0.85` and (`obj_count_error > 4` or `merge_rate > 0` or `split_rate > 0`)
- U-Net high-Dice cases: `1340`
- U-Net corner cases among high-Dice cases: `373 / 1340 = 27.8%`
- nnU-Net high-Dice cases: `1684`
- nnU-Net corner cases among high-Dice cases: `436 / 1684 = 25.9%`
- All-case corner rate: U-Net `11.26%`, nnU-Net `13.16%`

## Stratified Analysis Highlights

| Quartile | Dice U-Net | Dice nnU-Net | PQ U-Net | PQ nnU-Net | Split U-Net | Split nnU-Net |
|---|---|---|---|---|---|---|
| Q1 (low) | 0.398 | 0.435 | 0.290 | 0.325 | 0.039 | 0.020 |
| Q2 | 0.570 | 0.605 | 0.385 | 0.445 | 0.067 | 0.051 |
| Q3 | 0.722 | 0.767 | 0.486 | 0.591 | 0.109 | 0.046 |
| Q4 (high) | 0.759 | 0.763 | 0.419 | 0.510 | 0.231 | 0.083 |

- In dense slices (`Q4`), Dice is close: `0.759` vs `0.763`.
- The object-level gap remains large in the same slices: PQ `0.419` vs `0.510`, split rate `0.231` vs `0.083`.

## Sensitivity Relative to the Original Filtered Evaluation

### U-Net

- Predicted objects: `5.5027 -> 5.5027`
- Object count error: `2.8684 -> 2.8684`
- Split rate: `0.0364 -> 0.0364`
- Connectivity error: `0.0774 -> 0.0774`
- PQ: `0.4798 -> 0.4798`

### nnU-Net

- Predicted objects: `4.8545 -> 4.8545`
- Object count error: `2.6229 -> 2.6229`
- Split rate: `0.0226 -> 0.0226`
- Connectivity error: `0.0580 -> 0.0580`
- PQ: `0.5508 -> 0.5508`

## Bottom Line

- Removing the small-object filter affects U-Net much more strongly than nnU-Net.
- The ranking does not reverse.
- The no-filter evaluation is stronger and more defensible for the thesis.
