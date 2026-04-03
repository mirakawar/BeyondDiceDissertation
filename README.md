# CEM-MitoLab Mitochondria Segmentation Dissertation Code

This repository contains the code and report source for a dissertation comparing a custom U-Net and nnU-Net on the CEM-MitoLab mitochondria segmentation dataset from EMPIAR-11037.

This README is intended to satisfy the project-materials requirement:
- where the data came from
- how it was downloaded
- how it was processed
- how the reported outputs can be regenerated from the code in this repository

## What In This Repository Was Written For The Project

The project-authored code is primarily in:
- `day1/`
- `day2/`
- `day3/`
- `day4/`
- `day5/`
- `figures/`
- `data/`

The external nnU-Net package itself is not included here as authored code. The scripts in this repository call the installed `nnUNetv2_*` command-line tools.

## Data Provenance

Primary dataset:
- EMPIAR-11037
- Entry page: <https://www.ebi.ac.uk/empiar/EMPIAR-11037/>
- Title on EMPIAR: `CEM-MitoLab: a dataset of ~22K cellular EM 2D images with label maps of ~135K mitochondrial instances, for deep learning`

Official EMPIAR download guidance:
- <https://www.ebi.ac.uk/training/online/courses/empiar-quick-tour/downloading-data-from-empiar/>
- EMPIAR FAQ: <https://www.ebi.ac.uk/empiar/faq>

Public download locations used here:
- Entry XML: `https://ftp.ebi.ac.uk/empiar/world_availability/11037/11037.xml`
- Dataset ZIP: `https://ftp.ebi.ac.uk/empiar/world_availability/11037/data/cem_mitolab.zip`
- Metadata spreadsheet: `https://ftp.ebi.ac.uk/empiar/world_availability/11037/data/metadata/cem_mitolab_metadata.xlsx`

As of April 2, 2026, the public directory listing for the entry is:
- `https://ftp.ebi.ac.uk/empiar/world_availability/11037/`
- `https://ftp.ebi.ac.uk/empiar/world_availability/11037/data/`

The main archive provided by EMPIAR for this project is `cem_mitolab.zip` and is listed there as approximately `2.8G`.

## Expected Local Layout

The code expects the extracted dataset at:

```text
data/cem_mitolab/
  <sample_name>/
    images/
      *.tiff
    masks/
      *.tiff
```

Examples from the extracted archive:

```text
data/cem_mitolab/000r10530ND0Jf1lZ9gQ_3D/images/
data/cem_mitolab/000r10530ND0Jf1lZ9gQ_3D/masks/
data/cem_mitolab/005231Mi4tR936D758lh_2D/images/
data/cem_mitolab/005231Mi4tR936D758lh_2D/masks/
```

## Important Path Assumption

Most of the training and preprocessing scripts were written and run on the School of Informatics teaching cluster and contain hard-coded absolute paths under:

```text
/home/s2292949/dissertation_resplit
```

The commands below therefore assume one of the following:
- the repository is run from that same absolute path
- or those absolute paths are edited before the full pipeline is run

For submission review, this is important because the source code is complete, but full re-execution is easiest if the repository is placed at the original path used during development.

## Environment Requirements

The full pipeline requires:
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
- nnU-Net v2 installed and available on `PATH`
- CUDA GPU for the training and prediction commands used in the dissertation
- SLURM for the provided job-submission shell scripts

The wrapper shell scripts are SLURM-ready, but can also be run with `bash` if the same environment is available interactively.

## Installing nnU-Net v2

Official upstream repository:
- <https://github.com/MIC-DKFZ/nnUNet>
- Official installation instructions: <https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md>

According to the official nnU-Net v2 installation instructions, PyTorch should be installed first, and then nnU-Net v2 can be installed in one of two ways:
- standard package install: `pip install nnunetv2`
- editable source install:

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

Recommended setup for this project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip

# Install PyTorch first, choosing the correct CUDA or CPU build for the target machine.
# Official instructions: https://pytorch.org/get-started/locally/

pip install nnunetv2==2.6.2
```

For a local editable copy of the upstream source:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip

# Install PyTorch first.

git clone https://github.com/MIC-DKFZ/nnUNet.git external/nnUNet
cd external/nnUNet
git checkout v2.6.2
pip install -e .
cd ../..
```

Version pin used for reproducibility in this repository:
- Verified locally on April 2, 2026: `nnunetv2` installed version was `2.6.2`
- Upstream tag matching that release: `v2.6.2`

An installation performed later without version pinning may obtain newer code and slightly different behaviour. For that reason, this repository should be treated as depending specifically on `nnunetv2==2.6.2` unless stated otherwise.

After installation, nnU-Net v2 also requires environment variables telling it where to place raw data, preprocessed data, and trained results. For this repository those are:

```bash
export nnUNet_raw="$PWD/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnunet/nnUNet_preprocessed"
export nnUNet_results="$PWD/nnunet/nnUNet_results"
```

Installation can be verified with:

```bash
nnUNetv2_plan_and_preprocess -h
nnUNetv2_train -h
nnUNetv2_predict -h
```

This project used nnU-Net v2 as an unmodified external dependency. The version used was nnunetv2==2.6.2 (upstream tag v2.6.2). The code written for the project is contained in this repository; nnU-Net itself was installed separately from its official source.

## Downloading EMPIAR-11037

The following commands assume execution from the repository root.

### Option 1: HTTPS download with `curl`

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
```

Extract the main archive:

```bash
unzip data/downloads/cem_mitolab.zip -d data
```

After extraction, the following directory should exist:

```bash
test -d data/cem_mitolab && echo "Dataset extracted correctly"
```

### Option 2: HTTPS download with `wget`

```bash
mkdir -p data/downloads data/metadata

wget -c \
  -O data/downloads/cem_mitolab.zip \
  https://ftp.ebi.ac.uk/empiar/world_availability/11037/data/cem_mitolab.zip

wget -c \
  -O data/metadata/cem_mitolab_metadata.xlsx \
  https://ftp.ebi.ac.uk/empiar/world_availability/11037/data/metadata/cem_mitolab_metadata.xlsx

wget -c \
  -O data/metadata/11037.xml \
  https://ftp.ebi.ac.uk/empiar/world_availability/11037/11037.xml

unzip data/downloads/cem_mitolab.zip -d data
```

### Option 3: Globus

EMPIAR's FAQ states that Globus downloads should use:
- Collection: `EMBL-EBI Public Data`
- Directory: `/gridftp/empiar/world_availability/`

For this entry, the relevant Globus paths are:
- `/gridftp/empiar/world_availability/11037/data/cem_mitolab.zip`
- `/gridftp/empiar/world_availability/11037/data/metadata/cem_mitolab_metadata.xlsx`
- `/gridftp/empiar/world_availability/11037/11037.xml`

### Option 4: Aspera

EMPIAR also recommends Aspera for large downloads. See the EMPIAR FAQ for current firewall and client instructions:
- <https://www.ebi.ac.uk/empiar/faq>

This repository does not rely on Aspera-specific shell scripts; the simplest documented route for this project is the direct HTTPS ZIP download above.

## Canonical Command Sequence

All commands below are intended to be run from the repository root.

### 1. Inspect the raw extracted dataset

```bash
python3 day1/inspect_dataset.py
```

Outputs:
- `data/inspection_report.csv`
- `data/inspection_summary.txt`

### 2. Compute per-slice and per-sample dataset statistics

```bash
python3 day1/dataset_stats.py
```

Outputs:
- `data/dataset_stats.csv`
- `data/sample_stats.csv`

### 3. Convert the TIFF dataset into the nnU-Net raw format

Canonical script used for the main `day1` pipeline:

```bash
python3 day1/create_nnunet_format.py
```

This writes:
- `nnunet/nnUNet_raw/Dataset001_Mitochondria/`
- `data/nnunet_id_map.json`
- `splits_final.json`
- `splits_final.pkl`

### 4. Run nnU-Net preprocessing and extract the shared config

Interactive:

```bash
bash day2/preprocess_nnunet.sh
```

SLURM:

```bash
sbatch day2/preprocess_nnunet.sh
```

This runs:
- `nnUNetv2_plan_and_preprocess -d 001 -npfp 2 -np 2 --verify_dataset_integrity --verbose --clean`
- `python day2/extract_config.py`

Key outputs:
- `nnunet/nnUNet_preprocessed/Dataset001_Mitochondria/`
- `config.json`

### 5. Train nnU-Net fold 0

Interactive:

```bash
bash day3/train_nnunet_fold0.sh
```

SLURM:

```bash
sbatch day3/train_nnunet_fold0.sh
```

Key output location:
- `nnunet/nnUNet_results/Dataset001_Mitochondria/nnUNetTrainer__nnUNetPlans__2d/fold_0/`

### 6. Train the custom U-Net

Recommended wrapper used in this repository:

Interactive:

```bash
bash day4/submit_unet_new.sh
```

SLURM:

```bash
sbatch day4/submit_unet_new.sh
```

Equivalent direct command:

```bash
python3 -u day4/train_unet.py \
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

Key output locations:
- `models/unet/checkpoints/best_model.pt`
- `models/unet/checkpoints/last_checkpoint.pt`
- `logs/unet_b2nd_loss.png`
- `samples/unet_val/`

### 7. Generate nnU-Net test predictions

Interactive:

```bash
bash day3/predict_nnunet.sh
```

SLURM:

```bash
sbatch day3/predict_nnunet.sh
```

Outputs:
- `predictions/nnunet/binary/`
- `predictions/nnunet/probs/`

### 8. Generate U-Net test predictions

```bash
python3 day4/predict_unet.py \
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

### 9. Run per-image and aggregate evaluation for both models

nnU-Net:

```bash
python3 day5/evaluate.py \
  --model nnunet \
  --config config.json \
  --strict
```

U-Net:

```bash
python3 day5/evaluate.py \
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

### 10. Corner-case analysis

```bash
python3 day5/corner_case_analysis.py \
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

### 11. Statistical tests

```bash
python3 day5/stats_tests.py
```

Outputs:
- `evaluation/results/stats_tests.csv`
- `evaluation/results/stats_tests_summary.json`

### 12. Stratified analysis

```bash
python3 day5/stratified_analysis.py
```

Outputs:
- `evaluation/results/stratified_fg_quartile.csv`
- `evaluation/results/stratified_obj_count_quartile.csv`
- `evaluation/results/stratified_sample_type.csv`
- `evaluation/results/stratified_empty_vs_nonempty.csv`
- `evaluation/results/stratified_summary.json`

### 13. Generate written highlights summaries

```bash
python3 day5/make_highlights.py
```

Outputs:
- `evaluation/results/highlights_summary.txt`
- `evaluation/results/highlights_summary.md`

### 14. Generate figures used in the dissertation

Generate all figure groups:

```bash
python3 figures/visualise.py --config config.json --mode all --model both --n 6
```

This writes figures under:
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


## Minimal Regeneration Path

If the large training artefacts and raw data are not available, the smallest useful subset to regenerate the report text and figures from already-produced evaluation outputs is:

```bash
python3 day5/stats_tests.py
python3 day5/stratified_analysis.py
python3 day5/make_highlights.py
python3 figures/visualise.py --config config.json --mode all --model both --n 6

```

This assumes the existing contents of:
- `evaluation/results/`
- `logs/`
- `figures/`

are already present.

## Main Output Locations

Data preparation:
- `data/inspection_report.csv`
- `data/inspection_summary.txt`
- `data/dataset_stats.csv`
- `data/sample_stats.csv`
- `data/nnunet_id_map.json`
- `data/splits.json`

Model artefacts:
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
