# U-Net - Evaluation Highlights

> **Model** Custom U-Net · 2D · `best_model.pt` (epoch 86)  
> **Data**  CEM-MitoLab (EMPIAR-11037) · family-cohesion split  
> **Date**  2026-03-19 · `min_object_size_px = 10`  
> **Inference** Test-time inputs used raw TIFF fallback with z-score normalisation (`.b2nd` test files were not available)

---

## Test-Set Breakdown

- **Total slices:** 3,312
- **GT-positive** (mito present): 2,694  (81.3 %)
- **GT-empty** (background only): 618  (18.7 %)
- **HD95 finite** (FG on both sides): 2,874  (86.8 %)
- **HD95 undefined** (one side empty): 438  (13.2 %)

---

## 1. Voxel-Level Metrics  *(threshold = 0.50)*

Each entry: `mean ± std` and `median [Q25 - Q75]` - **all** 3,312 cases then **GT-positive** only (n = 2,694).

### 1a. Dice

- **Dice FG**
  - all:        mean 0.6362 ± 0.3568,  median 0.7756  [0.4173 - 0.9184]
  - GT-pos:     mean 0.6122 ± 0.3307,  median 0.7355  [0.4234 - 0.8852]

- **Dice BG**
  - all:        mean 0.9854 ± 0.0240,  median 0.9928  [0.9814 - 0.9985]
  - GT-pos:     mean 0.9828 ± 0.0251,  median 0.9901  [0.9778 - 0.9963]

- **Dice Mean (macro avg)**
  - all:        mean 0.8108 ± 0.1809,  median 0.8781  [0.6930 - 0.9544]
  - GT-pos:     mean 0.7975 ± 0.1673,  median 0.8581  [0.6962 - 0.9364]

### 1b. IoU · Precision · Recall

- **IoU (foreground)**
  - all:        mean 0.5566 ± 0.3503,  median 0.6335  [0.2636 - 0.8491]
  - GT-pos:     mean 0.5143 ± 0.3118,  median 0.5816  [0.2686 - 0.7940]

- **Precision**
  - all:        mean 0.7164 ± 0.3658,  median 0.8990  [0.6073 - 0.9688]
  - GT-pos:     mean 0.7107 ± 0.3469,  median 0.8854  [0.6137 - 0.9405]

- **Recall**
  - all:        mean 0.6189 ± 0.3692,  median 0.7500  [0.3247 - 0.9485]
  - GT-pos:     mean 0.5909 ± 0.3455,  median 0.6865  [0.3285 - 0.8988]

### 1c. Hausdorff Distance 95th Percentile

> HD95 is undefined when one side has no foreground (438 cases, 13.2 %).  
> Means below are computed over **finite cases only**.

- **All cases** (n = 2,874)
  - mean 47.19 px  ± 68.84,  median 18.49,  Q25 2.24,  Q75 63.07,  max 607.87
- **GT-positive** (n = 2,416)
  - mean 56.14 px  ± 71.66,  median 29.85,  Q25 6.71,  Q75 78.57,  max 607.87

### 1d. Boundary F1

- **Boundary F1  (2 px tolerance)**
  - all:        mean 0.6259 ± 0.3551,  median 0.7484  [0.3861 - 0.9153]
  - GT-pos:     mean 0.5995 ± 0.3275,  median 0.7118  [0.3917 - 0.8594]

- **Boundary F1  (5 px tolerance)**
  - all:        mean 0.6808 ± 0.3617,  median 0.8393  [0.4908 - 0.9731]
  - GT-pos:     mean 0.6670 ± 0.3403,  median 0.8044  [0.4959 - 0.9285]

### 1e. Calibration - ECE

- **Expected Calibration Error (ECE)**
  - all:        mean 0.0197 ± 0.0333,  median 0.0094  [0.0023 - 0.0246]
  - GT-pos:     mean 0.0230 ± 0.0347,  median 0.0126  [0.0051 - 0.0285]

---

## 2. Threshold Sensitivity  *(means over all 3,312 cases)*

HD95 uses **median** to avoid inf inflation. `*` = default threshold.

- **Dice FG:** 0.30: 0.6394  |  0.40: 0.6379  |  0.50 *: 0.6362  |  0.60: 0.6340  |  0.70: 0.6304
- **Dice BG:** 0.30: 0.9855  |  0.40: 0.9854  |  0.50 *: 0.9854  |  0.60: 0.9852  |  0.70: 0.9851
- **Dice Mean:** 0.30: 0.8124  |  0.40: 0.8117  |  0.50 *: 0.8108  |  0.60: 0.8096  |  0.70: 0.8078
- **IoU:** 0.30: 0.5600  |  0.40: 0.5584  |  0.50 *: 0.5566  |  0.60: 0.5541  |  0.70: 0.5503
- **Precision:** 0.30: 0.7025  |  0.40: 0.7100  |  0.50 *: 0.7164  |  0.60: 0.7207  |  0.70: 0.7263
- **Recall:** 0.30: 0.6336  |  0.40: 0.6259  |  0.50 *: 0.6189  |  0.60: 0.6116  |  0.70: 0.6029
- **HD95 (median px):** 0.30: 16.12  |  0.40: 17.04  |  0.50 *: 18.49  |  0.60: 19.42  |  0.70: 20.29
- **BF1 2 px:** 0.30: 0.6268  |  0.40: 0.6267  |  0.50 *: 0.6259  |  0.60: 0.6245  |  0.70: 0.6217
- **BF1 5 px:** 0.30: 0.6815  |  0.40: 0.6814  |  0.50 *: 0.6808  |  0.60: 0.6794  |  0.70: 0.6772

**Dice FG - GT-positive cases only:**

- **mean:** 0.30: 0.6209  |  0.40: 0.6165  |  0.50 *: 0.6122  |  0.60: 0.6075  |  0.70: 0.6021
- **median:** 0.30: 0.7507  |  0.40: 0.7419  |  0.50 *: 0.7355  |  0.60: 0.7292  |  0.70: 0.7201

---

## 3. Object-Level Metrics

> GT uses **original non-binary instance masks** (labels 0-255).  
> Matching: Hungarian algorithm. Predictions: objects < 10 px removed before matching.

### 3a. Object Counts

- **GT objects per image**
  - all:        mean 6.5704 ± 7.1431,  median 3.0000  [1.0000 - 11.0000]
  - GT-pos:     mean 8.0776 ± 7.1101,  median 5.0000  [2.0000 - 13.0000]

- **Predicted objects per image**
  - all:        mean 5.5027 ± 5.9101,  median 3.0000  [1.0000 - 9.0000]
  - GT-pos:     mean 6.5950 ± 5.9427,  median 5.0000  [2.0000 - 11.0000]

- **Count error  |pred - gt|**
  - all:        mean 2.8684 ± 4.0174,  median 1.0000  [0.0000 - 4.0000]
  - GT-pos:     mean 3.3563 ± 4.1634,  median 2.0000  [1.0000 - 5.0000]

- **Count error signed  (pred - gt)**
  - all:        mean -1.0676 ± 4.8196,  median 0.0000  [-3.0000 - 0.0000]
  - GT-pos:     mean -1.4826 ± 5.1385,  median -1.0000  [-4.0000 - 0.0000]

### 3b. Topology Errors

- **Merge rate**
  - all:        mean 0.0410 ± 0.0991,  median 0.0000  [0.0000 - 0.0000]
  - GT-pos:     mean 0.0504 ± 0.1077,  median 0.0000  [0.0000 - 0.0413]

- **Split rate**
  - all:        mean 0.0364 ± 0.1312,  median 0.0000  [0.0000 - 0.0000]
  - GT-pos:     mean 0.0448 ± 0.1442,  median 0.0000  [0.0000 - 0.0000]

- **Connectivity error**
  - all:        mean 0.0774 ± 0.1765,  median 0.0000  [0.0000 - 0.0769]
  - GT-pos:     mean 0.0952 ± 0.1913,  median 0.0000  [0.0000 - 0.1429]

- **Mean matched IoU**
  - all:        mean 0.4901 ± 0.3578,  median 0.6446  [0.0000 - 0.7965]
  - GT-pos:     mean 0.6026 ± 0.2994,  median 0.7067  [0.5245 - 0.8171]

### 3c. Object Precision & Recall

Format: `prec / recall` - `mean ± std`, all cases then GT-positive.

- **IoU > 0.3**
  - all:    prec 0.6448 ± 0.3827  /  rec 0.5961 ± 0.3736
  - GT-pos: prec 0.6193 ± 0.3664  /  rec 0.5596 ± 0.3493
- **IoU > 0.5**
  - all:    prec 0.6016 ± 0.3821  /  rec 0.5608 ± 0.3785
  - GT-pos: prec 0.5663 ± 0.3612  /  rec 0.5161 ± 0.3508
- **IoU > 0.7**
  - all:    prec 0.5285 ± 0.3830  /  rec 0.4972 ± 0.3838
  - GT-pos: prec 0.4764 ± 0.3513  /  rec 0.4379 ± 0.3462
- **IoU > 0.9**
  - all:    prec 0.2589 ± 0.3691  /  rec 0.2483 ± 0.3687
  - GT-pos: prec 0.1449 ± 0.2356  /  rec 0.1319 ± 0.2282

### 3d. Panoptic Quality

- **PQ  (IoU > 0.5)**
  - all:        mean 0.4798 ± 0.3399,  median 0.4825  [0.1957 - 0.7415]
  - GT-pos:     mean 0.4165 ± 0.2796,  median 0.4398  [0.1937 - 0.6293]

---

## 4. Key Numbers at a Glance

- **Dice FG - mean (all)** ·············· 0.6362
- **Dice FG - median (all)** ············ 0.7756
- **Dice FG - mean (GT-positive)** ······ 0.6122
- **Dice FG - median (GT-positive)** ···· 0.7355
- **Dice Mean - mean (all)** ············ 0.8108
- **IoU - mean (all)** ·················· 0.5566
- **Precision - mean (all)** ············ 0.7164
- **Recall - mean (all)** ··············· 0.6189
- **HD95 - median, finite cases** ······· 18.49 px
- **HD95 - mean, finite cases** ········· 47.19 px
- **Boundary F1 2 px - mean** ··········· 0.6259
- **Boundary F1 5 px - mean** ··········· 0.6808
- **ECE - mean** ························ 0.0197
- **Obj Precision @ IoU 0.5 (all)** ····· 0.6016
- **Obj Recall @ IoU 0.5 (all)** ········ 0.5608
- **PQ @ IoU > 0.5 - mean** ············· 0.4798
- **Merge rate - mean** ················· 0.0410
- **Split rate - mean** ················· 0.0364
- **Mean matched IoU - mean (all)** ····· 0.4901
- **Mean matched IoU - mean (GT-pos)** ·· 0.6026
