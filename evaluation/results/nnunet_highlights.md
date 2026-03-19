# nnU-Net — Evaluation Highlights

> **Model** nnUNetTrainer · 2D · Fold 0 · `checkpoint_best`  
> **Data**  CEM-MitoLab (EMPIAR-11037) · family-cohesion split  
> **Date**  2026-03-15 · `min_object_size_px = 10`

---

## Test-Set Breakdown

- **Total slices:** 3,312
- **GT-positive** (mito present): 2,694  (81.3 %)
- **GT-empty** (background only): 618  (18.7 %)
- **HD95 finite** (FG on both sides): 2,903  (87.7 %)
- **HD95 undefined** (one side empty): 409  (12.3 %)

---

## 1. Voxel-Level Metrics  *(threshold = 0.50)*

Each entry: `mean ± std` and `median [Q25 – Q75]` — **all** 3,312 cases then **GT-positive** only (n = 2,694).

### 1a. Dice

- **Dice FG**
  - all:        mean 0.6857 ± 0.3519,  median 0.8535  [0.5201 – 0.9430]
  - GT-pos:     mean 0.6426 ± 0.3420,  median 0.8043  [0.4415 – 0.9089]

- **Dice BG**
  - all:        mean 0.9875 ± 0.0251,  median 0.9948  [0.9863 – 0.9989]
  - GT-pos:     mean 0.9849 ± 0.0270,  median 0.9927  [0.9836 – 0.9971]

- **Dice Mean (macro avg)**
  - all:        mean 0.8366 ± 0.1794,  median 0.9219  [0.7467 – 0.9680]
  - GT-pos:     mean 0.8137 ± 0.1742,  median 0.8941  [0.7062 – 0.9499]

### 1b. IoU · Precision · Recall

- **IoU (foreground)**
  - all:        mean 0.6141 ± 0.3503,  median 0.7445  [0.3514 – 0.8921]
  - GT-pos:     mean 0.5545 ± 0.3264,  median 0.6727  [0.2833 – 0.8329]

- **Precision**
  - all:        mean 0.7730 ± 0.3503,  median 0.9319  [0.8100 – 0.9903]
  - GT-pos:     mean 0.7499 ± 0.3503,  median 0.9166  [0.7624 – 0.9629]

- **Recall**
  - all:        mean 0.6590 ± 0.3646,  median 0.8232  [0.4007 – 0.9605]
  - GT-pos:     mean 0.6098 ± 0.3537,  median 0.7562  [0.3275 – 0.9129]

### 1c. Hausdorff Distance 95th Percentile

> HD95 is undefined when one side has no foreground (409 cases, 12.3 %).  
> Means below are computed over **finite cases only**.

- **All cases** (n = 2,903)
  - mean 40.35 px  ± 65.71,  median 8.94,  Q25 1.41,  Q75 55.39,  max 497.10
- **GT-positive** (n = 2,363)
  - mean 49.56 px  ± 69.63,  median 21.00,  Q25 3.61,  Q75 66.90,  max 497.10

### 1d. Boundary F1

- **Boundary F1  (2 px tolerance)**
  - all:        mean 0.6844 ± 0.3537,  median 0.8369  [0.5156 – 0.9670]
  - GT-pos:     mean 0.6409 ± 0.3441,  median 0.7754  [0.4346 – 0.9088]

- **Boundary F1  (5 px tolerance)**
  - all:        mean 0.7291 ± 0.3538,  median 0.9006  [0.6271 – 1.0000]
  - GT-pos:     mean 0.6959 ± 0.3502,  median 0.8529  [0.5664 – 0.9542]

### 1e. Calibration — ECE

- **Expected Calibration Error (ECE)**
  - all:        mean 0.0181 ± 0.0376,  median 0.0071  [0.0014 – 0.0178]
  - GT-pos:     mean 0.0218 ± 0.0404,  median 0.0103  [0.0038 – 0.0218]

---

## 2. Threshold Sensitivity  *(means over all 3,312 cases)*

HD95 uses **median** to avoid inf inflation. ★ = default threshold.

- **Dice FG:** 0.30: 0.6926  |  0.40: 0.6885  |  0.50 ★: 0.6857  |  0.60: 0.6806  |  0.70: 0.6755
- **Dice BG:** 0.30: 0.9877  |  0.40: 0.9876  |  0.50 ★: 0.9875  |  0.60: 0.9873  |  0.70: 0.9871
- **Dice Mean:** 0.30: 0.8402  |  0.40: 0.8381  |  0.50 ★: 0.8366  |  0.60: 0.8339  |  0.70: 0.8313
- **IoU:** 0.30: 0.6209  |  0.40: 0.6169  |  0.50 ★: 0.6141  |  0.60: 0.6088  |  0.70: 0.6034
- **Precision:** 0.30: 0.7640  |  0.40: 0.7685  |  0.50 ★: 0.7730  |  0.60: 0.7755  |  0.70: 0.7787
- **Recall:** 0.30: 0.6754  |  0.40: 0.6663  |  0.50 ★: 0.6590  |  0.60: 0.6497  |  0.70: 0.6402
- **HD95 (median px):** 0.30: 9.01  |  0.40: 9.01  |  0.50 ★: 8.94  |  0.60: 9.06  |  0.70: 10.00
- **BF1 2 px:** 0.30: 0.6899  |  0.40: 0.6866  |  0.50 ★: 0.6844  |  0.60: 0.6799  |  0.70: 0.6753
- **BF1 5 px:** 0.30: 0.7336  |  0.40: 0.7309  |  0.50 ★: 0.7291  |  0.60: 0.7247  |  0.70: 0.7204

**Dice FG — GT-positive cases only:**

- **mean:** 0.30: 0.6566  |  0.40: 0.6494  |  0.50 ★: 0.6426  |  0.60: 0.6355  |  0.70: 0.6277
- **median:** 0.30: 0.8131  |  0.40: 0.8077  |  0.50 ★: 0.8043  |  0.60: 0.7972  |  0.70: 0.7871

---

## 3. Object-Level Metrics

> GT uses **original non-binary instance masks** (labels 0–255).  
> Matching: Hungarian algorithm.  Predictions: objects < 10 px removed before matching.
> `matched_iou_mean` counts only overlapping pairs (IoU > 0).

### 3a. Object Counts

- **GT objects per image**
  - all:        mean 6.5704 ± 7.1431,  median 3.0000  [1.0000 – 11.0000]
  - GT-pos:     mean 8.0776 ± 7.1101,  median 5.0000  [2.0000 – 13.0000]

- **Predicted objects per image**
  - all:        mean 4.8545 ± 5.4170,  median 3.0000  [0.0000 – 8.0000]
  - GT-pos:     mean 5.8964 ± 5.4545,  median 4.0000  [1.0000 – 10.0000]

- **Count error  |pred − gt|**
  - all:        mean 2.6229 ± 3.7442,  median 1.0000  [0.0000 – 4.0000]
  - GT-pos:     mean 3.1529 ± 3.9018,  median 2.0000  [1.0000 – 4.0000]

- **Count error signed  (pred − gt)**
  - all:        mean -1.7159 ± 4.2374,  median -1.0000  [-3.0000 – 0.0000]
  - GT-pos:     mean -2.1811 ± 4.5177,  median -1.0000  [-4.0000 – 0.0000]

### 3b. Topology Errors

- **Merge rate**
  - all:        mean 0.0354 ± 0.0903,  median 0.0000  [0.0000 – 0.0000]
  - GT-pos:     mean 0.0435 ± 0.0984,  median 0.0000  [0.0000 – 0.0000]

- **Split rate**
  - all:        mean 0.0226 ± 0.1555,  median 0.0000  [0.0000 – 0.0000]
  - GT-pos:     mean 0.0278 ± 0.1720,  median 0.0000  [0.0000 – 0.0000]

- **Connectivity error**
  - all:        mean 0.0580 ± 0.1814,  median 0.0000  [0.0000 – 0.0000]
  - GT-pos:     mean 0.0714 ± 0.1988,  median 0.0000  [0.0000 – 0.0762]

- **Mean matched IoU  (IoU > 0 pairs only)**
  - all:        mean 0.5194 ± 0.3752,  median 0.7097  [0.0000 – 0.8316]
  - GT-pos:     mean 0.6385 ± 0.3114,  median 0.7609  [0.5979 – 0.8482]

### 3c. Object Precision & Recall

Format: `prec / recall` — `mean ± std`, all cases then GT-positive.

- **IoU > 0.3**
  - all:    prec 0.7304 ± 0.3748  /  rec 0.6335 ± 0.3663
  - GT-pos: prec 0.6967 ± 0.3767  /  rec 0.5776 ± 0.3514
- **IoU > 0.5**
  - all:    prec 0.6946 ± 0.3767  /  rec 0.6062 ± 0.3717
  - GT-pos: prec 0.6527 ± 0.3746  /  rec 0.5440 ± 0.3528
- **IoU > 0.7**
  - all:    prec 0.6271 ± 0.3823  /  rec 0.5563 ± 0.3784
  - GT-pos: prec 0.5698 ± 0.3705  /  rec 0.4827 ± 0.3496
- **IoU > 0.9**
  - all:    prec 0.3349 ± 0.3918  /  rec 0.3086 ± 0.3875
  - GT-pos: prec 0.2106 ± 0.2847  /  rec 0.1782 ± 0.2621

### 3d. Panoptic Quality

> PQ uses IoU **> 0.5** (strict, per the original PQ paper). `obj_precision_t50` uses **≥ 0.5** (detection convention) — these are intentionally different.

- **PQ  (IoU > 0.5)**
  - all:        mean 0.5508 ± 0.3403,  median 0.5890  [0.3138 – 0.8369]
  - GT-pos:     mean 0.4760 ± 0.2961,  median 0.5340  [0.2593 – 0.7105]

---

## 4. Key Numbers at a Glance

- **Dice FG — mean (all)** ·············· 0.6857
- **Dice FG — median (all)** ············ 0.8535
- **Dice FG — mean (GT-positive)** ······ 0.6426
- **Dice FG — median (GT-positive)** ···· 0.8043
- **Dice Mean — mean (all)** ············ 0.8366
- **IoU — mean (all)** ·················· 0.6141
- **Precision — mean (all)** ············ 0.7730
- **Recall — mean (all)** ··············· 0.6590
- **HD95 — median, finite cases** ······· 8.94 px
- **HD95 — mean, finite cases** ········· 40.35 px
- **Boundary F1 2 px — mean** ··········· 0.6844
- **Boundary F1 5 px — mean** ··········· 0.7291
- **ECE — mean** ························ 0.0181
- **Obj Precision @ IoU 0.5 (all)** ····· 0.6946
- **Obj Recall @ IoU 0.5 (all)** ········ 0.6062
- **PQ @ IoU > 0.5 — mean** ············· 0.5508
- **Merge rate — mean** ················· 0.0354
- **Split rate — mean** ················· 0.0226
- **Mean matched IoU — mean (all)** ····· 0.5194
- **Mean matched IoU — mean (GT-pos)** ·· 0.6385