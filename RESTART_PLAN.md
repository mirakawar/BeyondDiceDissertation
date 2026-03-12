# Resplit Rerun Plan

Current goal: rerun the experiment in a clean workspace using a revised split,
without touching the original `dissertation` folder.

## Workspace

- New workspace: `/home/s2292949/dissertation_resplit`
- Original workspace kept unchanged: `/home/s2292949/dissertation`
- Heavy static assets are shared by symlink:
  - `data/cem_mitolab`
  - `nnUNet`
  - `nnunet/nnUNet_raw`
  - `nnunet/nnUNet_preprocessed`
- New outputs stay local to `dissertation_resplit`:
  - `logs/`
  - `models/`
  - `predictions/`
  - `nnunet/nnUNet_results/`

## Proposed split change

- Start from the current split.
- Move `5j54Gst6j5069xCUpF5M_2D` from `test` to `train`.
- This is the cleaner one-sample move based on the slice-level checks:
  - it reduces test-set skew
  - it lowers test foreground density slightly
  - it distorts the remaining test set less than moving `Wei2020_MitoEM-R`

## Files prepared

- Split revision script:
  - `data/create_supervisor_resplit.py`
- Summary output after running the script:
  - `data/resplit_summary.txt`

## Rerun checklist

1. Apply the revised split.
2. Regenerate nnU-Net dataset split artefacts:
   - `nnunet_id_map.json` is updated by the resplit script
   - rerun the nnU-Net format script so `splits_final.pkl` matches the new split
3. Sanity-check the new split stats:
   - train/val/test image counts
   - empty-image rate
   - foreground density
   - instances per image
   - large-sample dominance
4. Retrain nnU-Net on the new split.
5. Retrain U-Net on the new split.
6. Generate test predictions for both models.
7. Rerun evaluation:
   - `evaluate.py`
   - `corner_case_analysis.py`
   - `stats_tests.py`
   - `stratified_analysis.py`
   - `make_highlights.py`
8. Regenerate figures.
9. Compare old vs new results.
10. Update the dissertation only after the rerun is complete.

## Important

- Do not overwrite results in the original `dissertation` folder.
- Treat this rerun as a sensitivity analysis on the split, not as a replacement
  for the original experiment until the new results are in.
