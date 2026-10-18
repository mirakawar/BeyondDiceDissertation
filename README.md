# Beyond Dice: Evaluating U-Net and nnU-Net for Mitochondria Segmentation in Electron Microscopy Using Biologically Motivated Metrics

Final-year dissertation, BEng Software Engineering, University of Edinburgh (2026). Supervisor: Dr. Eleonora D'Arnese.

The full dissertation is in [`Kawar_2026_Beyond_Dice.pdf`](Kawar_2026_Beyond_Dice.pdf).

## Summary

Segmentation models are usually compared with pixel-overlap metrics such as Dice. These metrics miss object-level errors (merged, split or miscounted mitochondria) that distort downstream biological analysis. This project trains a custom U-Net and nnU-Net on the CEM-MitoLab dataset under a shared biological source-level split and evaluates both with one pipeline spanning pixel, boundary, calibration and object-level metrics.

## Key results (test set, 3,312 slices)

| Metric | U-Net | nnU-Net |
|---|---|---|
| Foreground Dice | 0.578 | 0.624 |
| Panoptic Quality (IoU > 0.5) | 0.459 | 0.544 |
| Split rate | 0.091 | 0.041 |
| Merge rate | 0.041 | 0.035 |

- nnU-Net was better on all 27 paired per-image metrics tested (two-sided Wilcoxon signed-rank, all significant after Benjamini-Hochberg correction).
- The object-level gap is larger than the pixel-level gap. In the densest slices, Dice is nearly equal (0.759 vs 0.763) while Panoptic Quality is not (0.419 vs 0.510).
- Among high-Dice predictions (Dice > 0.85), 42.3% of U-Net slices and 38.1% of nnU-Net slices still had object-level errors.
- U-Net failures are mainly fragmentation; nnU-Net failures are mainly merging.

**Conclusion:** standard overlap metrics alone cannot distinguish these failure modes, so object-level evaluation is needed to judge models for biological use.

# CEM-MitoLab Mitochondria Segmentation: Code

This repo contains the code I wrote for my dissertation comparing a custom U-Net against nnU-Net on the CEM-MitoLab mitochondria segmentation dataset (EMPIAR-11037).

The point of this README is to explain where the data came from, how I processed it, and how to reproduce the experiments and figures from the dissertation.

## Folder structure

- `data_preparation/`: inspecting the raw dataset, computing summary statistics, and converting everything into nnU-Net's raw format
- `nnunet_preprocessing/`: running nnU-Net's planning/preprocessing step and extracting the shared `config.json`
- `nnunet_pipeline/`: nnU-Net training and prediction scripts
- `unet_pipeline/`: custom U-Net training and prediction scripts
- `evaluation_pipeline/`: per-image evaluation, corner-case analysis, statistical tests, and summary generation
- `figures/`: scripts that generate the dissertation figures
- `data/`: split files and dataset metadata
- `evaluation/results/`: per-image results, aggregate metrics, corner cases, statistical tests and stratified analysis

The `data/` folder also holds the split scripts (`create_fair_split.py`, `check_split_fairness.py`, `sync_split_metadata.py`), which are the versions used for the final split. It also contains older copies of three data preparation scripts from early in the project (`create_nnunet_format.py`, `dataset_stats.py`, `inspect_dataset.py`). For those three, use the versions in `data_preparation/`.

## Data

The dataset is CEM-MitoLab from EMPIAR-11037:

- Entry page: <https://www.ebi.ac.uk/empiar/EMPIAR-11037/>
- Dataset title: `CEM-MitoLab: a dataset of ~22K cellular EM 2D images with label maps of ~135K mitochondrial instances, for deep learning`

Files used:

- Entry XML: `https://ftp.ebi.ac.uk/empiar/world_availability/11037/11037.xml`
- Dataset ZIP: `https://ftp.ebi.ac.uk/empiar/world_availability/11037/data/cem_mitolab.zip`
- Metadata spreadsheet: `https://ftp.ebi.ac.uk/empiar/world_availability/11037/data/metadata/cem_mitolab_metadata.xlsx`

EMPIAR download documentation:

- <https://www.ebi.ac.uk/training/online/courses/empiar-quick-tour/downloading-data-from-empiar/>
- <https://www.ebi.ac.uk/empiar/faq>

## Expected dataset layout

After extracting the archive, the code expects:

```text
data/cem_mitolab/
  <sample_name>/
    images/
      *.tiff
    masks/
      *.tiff
```

Example:

```text
data/cem_mitolab/000r10530ND0Jf1lZ9gQ_3D/images/
data/cem_mitolab/000r10530ND0Jf1lZ9gQ_3D/masks/
```

## Path handling

The Python scripts find the repo root from their own location, and `config.json` stores all paths relative to the repo root. That means you can run scripts from any directory without editing paths.

Two things worth knowing:

- the SLURM `#SBATCH --output` and `#SBATCH --error` lines use relative paths, so it's best to submit jobs from the repo root
- old log files and archived metadata in the repo still contain the original cluster path, but the live scripts don't depend on it

## Dependencies

- Python 3
- PyTorch
- NumPy
- Pandas
- SciPy
- Matplotlib
- nibabel
- tifffile
- Pillow
- blosc2
- tqdm
- nnU-Net v2 on `PATH`
- a CUDA GPU for training and prediction
- SLURM if you want to use the cluster job scripts

For the dissertation runs I used PyTorch 2.8.0 (CUDA 12.8) and nnU-Net 2.6.2 on a single NVIDIA GeForce RTX 2080 Ti (11 GB) with 32 GB RAM, under SLURM.

The shell scripts can also be run with `bash` directly if you have the same software available locally.

## Installing nnU-Net v2

- Repository: <https://github.com/MIC-DKFZ/nnUNet>
- Installation guide: <https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md>

I used `nnunetv2==2.6.2`. Example setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip

# Install the right PyTorch build for your machine first.
pip install nnunetv2==2.6.2
```

nnU-Net needs its data directories set:

```bash
export nnUNet_raw="$PWD/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnunet/nnUNet_preprocessed"
export nnUNet_results="$PWD/nnunet/nnUNet_results"
```

Quick check that the install worked:

```bash
nnUNetv2_plan_and_preprocess -h
nnUNetv2_train -h
nnUNetv2_predict -h
```

## Downloading the dataset

From the repo root:

```bash
mkdir -p data/downloads data/metadata

curl -L -C - \
  -o data/downloads/cem_mitolab.zip \
  https://ftp.ebi.ac.uk/empiar/world_availability/11037/data/cem_mitolab.zip

curl -L -C - \
  -o data/metadata/cem_mitolab_metadata.xlsx \
  https://ftp.ebi.ac.uk/empiar/world_availability/11037/data/metadata/cem_mitolab_metadata.xlsx

curl -L -C - \
  -o data/metadata/11037.xml \
  https://ftp.ebi.ac.uk/empiar/world_availability/11037/11037.xml

unzip data/downloads/cem_mitolab.zip -d data
```

`wget`, Globus, or Aspera all work too, and EMPIAR documents those options at the links above. The only requirement is that the extracted folder ends up at `data/cem_mitolab/`.

## Reproducing the experiments

Run everything from the repo root.

### 1. Inspect the raw dataset

```bash
python3 data_preparation/inspect_dataset.py
```

Outputs:

- `data/inspection_report.csv`
- `data/inspection_summary.txt`

### 2. Compute dataset statistics

```bash
python3 data_preparation/dataset_stats.py
```

Outputs:

- `data/dataset_stats.csv`
- `data/sample_stats.csv`

### 3. Source-level split

The split used in the dissertation is already committed as `data/splits.json` (415 / 89 / 89 sources, 15,312 / 3,247 / 3,312 slices). To rebuild it:

```bash
python3 data/create_fair_split.py
python3 data/check_split_fairness.py
cp data/splits_fair.json data/splits.json
```

`data/sync_split_metadata.py` updates `data/nnunet_id_map.json` to match `splits.json` once the nnU-Net conversion below has been run.

### 4. Convert to nnU-Net raw format

```bash
python3 data_preparation/create_nnunet_format.py
```

Outputs:

- `nnunet/nnUNet_raw/Dataset001_Mitochondria/`
- `nnunet/nnUNet_raw/Dataset001_Mitochondria/splits_final.json`
- `nnunet/nnUNet_raw/Dataset001_Mitochondria/splits_final.pkl`
- `data/nnunet_id_map.json`

### 5. nnU-Net preprocessing

Interactive:

```bash
bash nnunet_preprocessing/preprocess_nnunet.sh
```

SLURM:

```bash
sbatch nnunet_preprocessing/preprocess_nnunet.sh
```

This runs `nnUNetv2_plan_and_preprocess` and then writes the shared `config.json`.

Key outputs:

- `nnunet/nnUNet_preprocessed/Dataset001_Mitochondria/`
- `config.json`

### 6. Train nnU-Net (fold 0)

Interactive:

```bash
bash nnunet_pipeline/train_nnunet_fold0.sh
```

SLURM:

```bash
sbatch nnunet_pipeline/train_nnunet_fold0.sh
```

Output:

- `nnunet/nnUNet_results/Dataset001_Mitochondria/nnUNetTrainer__nnUNetPlans__2d/fold_0/`

### 7. Train the custom U-Net

Interactive:

```bash
bash unet_pipeline/submit_unet_new.sh
```

SLURM:

```bash
sbatch unet_pipeline/submit_unet_new.sh
```

Or run the trainer directly:

```bash
python3 -u unet_pipeline/train_unet.py \
  --config config.json \
  --batch-size 4 \
  --epochs 1000 \
  --learning-rate 1e-3 \
  --num-workers 2 \
  --augment \
  --loss-type ce_dice \
  --sample-output-dir samples/unet_val \
  --loss-plot-path logs/unet_b2nd_loss.png
```

Key outputs:

- `models/unet/checkpoints/best_model.pt`
- `models/unet/checkpoints/last_checkpoint.pt`
- `logs/unet_b2nd_loss.png`
- `samples/unet_val/`

There is also `unet_pipeline/submit_unet.sh` in the repo. I kept it as a record of an earlier run, but `submit_unet_new.sh` is what I actually used for the final training.

### 8. nnU-Net test predictions

Interactive:

```bash
bash nnunet_pipeline/predict_nnunet.sh
```

SLURM:

```bash
sbatch nnunet_pipeline/predict_nnunet.sh
```

Outputs:

- `predictions/nnunet/binary/`
- `predictions/nnunet/probs/`

### 9. U-Net test predictions

```bash
python3 unet_pipeline/predict_unet.py \
  --config config.json \
  --split test \
  --overwrite \
  --batch-size 8 \
  --num-workers 2 \
  --device cuda \
  --no-compress-probs
```

Outputs:

- `predictions/unet/binary/`
- `predictions/unet/probs/`

### 10. Evaluate both models

nnU-Net:

```bash
python3 evaluation_pipeline/evaluate.py \
  --model nnunet \
  --config config.json \
  --strict
```

U-Net:

```bash
python3 evaluation_pipeline/evaluate.py \
  --model unet \
  --config config.json \
  --strict
```

Main outputs:

- `evaluation/results/nnunet_per_image.csv`
- `evaluation/results/unet_per_image.csv`
- `evaluation/results/nnunet_aggregate.csv`
- `evaluation/results/unet_aggregate.csv`
- `evaluation/results/nnunet_summary.json`
- `evaluation/results/unet_summary.json`

### 11. Corner-case analysis

```bash
python3 evaluation_pipeline/corner_case_analysis.py \
  --config config.json \
  --models unet nnunet \
  --dice-threshold 0.85
```

Outputs:

- `evaluation/results/unet_corner_cases.csv`
- `evaluation/results/nnunet_corner_cases.csv`
- `evaluation/results/corner_cases_combined.csv`
- `evaluation/results/unet_corner_case_summary.json`
- `evaluation/results/nnunet_corner_case_summary.json`

### 12. Statistical tests

```bash
python3 evaluation_pipeline/stats_tests.py
```

Outputs:

- `evaluation/results/stats_tests.csv`
- `evaluation/results/stats_tests_summary.json`

### 13. Stratified analysis

```bash
python3 evaluation_pipeline/stratified_analysis.py
```

Outputs:

- `evaluation/results/stratified_fg_quartile.csv`
- `evaluation/results/stratified_obj_count_quartile.csv`
- `evaluation/results/stratified_sample_type.csv`
- `evaluation/results/stratified_empty_vs_nonempty.csv`
- `evaluation/results/stratified_summary.json`

### 14. Written highlights summary

```bash
python3 evaluation_pipeline/make_highlights.py
```

Outputs:

- `evaluation/results/highlights_summary.txt`
- `evaluation/results/highlights_summary.md`

### 15. Generate figures

```bash
python3 figures/visualise.py --config config.json --mode all --model both --n 6
```

Figures are written under:

- `figures/dice_distribution.png`
- `figures/dice_distribution_overlay.png`
- `figures/convergence/`
- `figures/training_panels/`
- `figures/eval_instances/`
- `figures/contour_overlap/`
- `figures/corner_cases/`
- `figures/scatter/`
- `figures/threshold_curves/`
- `figures/stratified/`

There is also `figures/count_error_dist.py` which generates the count error distribution figure used in the appendix:

```bash
python3 figures/count_error_dist.py
```

Output: `figures/count_error_dist/count_error_distribution.png`

## Regenerating figures without rerunning training

If you just want to regenerate the summaries and figures from the result files already in the repo:

```bash
python3 evaluation_pipeline/stats_tests.py
python3 evaluation_pipeline/stratified_analysis.py
python3 evaluation_pipeline/make_highlights.py
python3 figures/visualise.py --config config.json --mode all --model both --n 6
python3 figures/count_error_dist.py
```

This assumes the contents of `evaluation/results/`, `logs/`, and `figures/` are already present.

## Where outputs end up

Data preparation:

- `data/inspection_report.csv`
- `data/inspection_summary.txt`
- `data/dataset_stats.csv`
- `data/sample_stats.csv`
- `data/nnunet_id_map.json`
- `data/splits.json`

Model files:

- `nnunet/nnUNet_raw/`
- `nnunet/nnUNet_preprocessed/`
- `nnunet/nnUNet_results/`
- `models/unet/checkpoints/`
- `predictions/unet/`
- `predictions/nnunet/`

Evaluation:

- `evaluation/results/`

Figures:

- `figures/`
