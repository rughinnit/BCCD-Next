# Blood-cell detection results — BCCD, trained on the RX 570

Hardware: AMD RX 570 (gfx803) via ROCm 5.4.3 + PyTorch 1.13.1, 4 GB VRAM.
Dataset: BCCD (364 images) + BCD Kaggle (866 usable, 8 near-duplicates removed).
Training: 1,071 images, 640 px, batch 4–8, fp32, 100 epochs, seed 0.
Test: BCCD official test split — 72 images, 945 boxes (RBC 805, WBC 71, Platelets 69).

## Executive summary

- **Best counting configuration:** `yolo26n`, 960 px export, class-aware NMS, IoU 0.50,
  confidence 0.50 (tuned on val). Test all-class MAE **3.17 cells/image** — RBC 2.79,
  WBC 0.10, platelets 0.28.
- **Aggregate counts over 72 fields:** RBC 808/805 (+0.4 %), WBC 78/71 (+9.9 %),
  platelets 67/69 (−2.9 %).
- **Best detection:** `yolo26n` mAP50 **0.9287** (TTA); `yolov9t` mAP50-95 **0.6407**.
- **Ensemble (WBF `v9t+v11n+v12n`):** 0.6295 mAP50-95 — below the best single model, so it was
  not adopted.
- **Nine counting remedies failed** (NMS sweep, 1280 export, TTA/WBF, multi-scale, tiling,
  per-class configs, segmentation). The residual RBC error is structural to box counting of
  touching cells; better labels or real masks are the only remaining fixes.
- **Deployment:** `yolo11n` — 7.85 ms (MIGraphX), 12.33 ms (ncnn), 13.26 ms (PyTorch).

## Family comparison (test split)

| Model | mAP50 | mAP50 TTA | mAP50-95 | mAP50-95 TTA | Train time | img/s |
|---|---|---|---|---|---|---|
| yolov8n | 0.9027 | 0.9089 | 0.6201 | 0.6266 | 44 min | 40.4 |
| **yolov9t** | 0.9136 | 0.9137 | **0.6407** | 0.6395 | 61 min | 29.4 |
| yolov10n | 0.9176 | 0.9176 | 0.6356 | 0.6356 | 58 min | 30.9 |
| **yolo11n** | 0.9069 | 0.9191 | 0.6301 | **0.6398** | 51 min | 35.0 |
| yolo11s | 0.9165 | 0.9203 | 0.6263 | 0.6341 | 3.5 h (resumed) | — |
| yolo12n | 0.9221 | 0.9232 | 0.6355 | 0.6368 | 87 min | 20.6 |
| **yolo26n** | 0.9162 | **0.9287** | 0.6073 | 0.6186 | 56 min | 32.0 |

Best mAP50: **yolo26n 0.9287** (finds the most cells).
Best mAP50-95: **yolov9t 0.6407 plain / yolo11n 0.6398 TTA** (tightest boxes).

## Per-class, TTA (`AP50 / AP50-95`)

| Model | RBC | WBC | Platelets |
|---|---|---|---|
| yolov9t | 0.872 / 0.630 | 0.971 / 0.798 | 0.898 / **0.490** |
| yolo11n | 0.872 / 0.633 | **0.990 / 0.813** | 0.895 / 0.474 |
| yolo12n | **0.881 / 0.637** | 0.974 / 0.788 | **0.914 / 0.485** |
| yolo26n | 0.883 / 0.629 | 0.993 / 0.759 | 0.910 / 0.468 |

Platelets: detection is fine (AP50 ≈ 0.90), localisation is not (AP50-95 ≈ 0.48) —
a 15 px object loses most of its IoU to a 2 px box error.

## Ensemble (weighted box fusion, identical COCO harness)

| Configuration | mAP50-95 |
|---|---|
| yolo12n alone | 0.6228 |
| v9t + v11n | 0.6251 |
| **v9t + v11n + v12n** | **0.6295** |
| v9t + v11n + v12n + v26n | 0.6280 |

Ensemble gains **+0.7 mAP50-95** over the best single model, at 3× inference cost.

## Deployment card — yolo11n (winner, single model)

| Runtime | Raw forward | Notes |
|---|---|---|
| PyTorch + ROCm 5.4.3 | 13.26 ms | needs the 3 GB ROCm stack |
| ncnn + Vulkan (RADV) | 12.33 ms | ~100 MB runtime, no ROCm |
| **MIGraphX 2.4** | **7.85 ms** | graph fusion; ~127 img/s |
| ncnn full pipeline (letterbox+decode+NMS) | 15.08 ms | 66 FPS |

Accuracy of this model: **0.9191 mAP50 / 0.6398 mAP50-95** (TTA).

## Methodology notes (read before quoting these numbers)

1. **CBC was excluded.** 183 of its 420 images are the same photographs as BCCD val/test
   (71/72 test, 84/87 val). Training on it would have leaked the test set.
2. **BCD near-duplicates removed** with dHash ≤ 4 (8 of 874 matched a BCCD val/test image).
3. **BCCD annotations are incomplete** — unlabelled cells count as false positives, which
   depresses precision for every model. Papers reporting 99 % on BCCD are not comparable.
4. **Extra data did not help**: yolov8n 0.616 → 0.620 mAP50-95 going 364 → 1,071 images;
   yolo11n 0.627 → 0.630. The BCD domain (416×416 crops) differs from BCCD (640×480 smears).
5. **Capacity saturated**: yolo11s (9.4 M params, 3.5 h) scored below yolo11n (2.6 M, 51 min).
6. **TTA is not universally beneficial**: +1.2 mAP50 for 11n, +1.3 for 26n, ~0 for 9t/10n/12n.
7. **ROCm instability**: a GPU page fault occurred during multi-model batch inference;
   the same job on CPU was fine.

## Reproduction

```bash
# dataset (VOC -> YOLO, merged with BCD, filtered)
python scripts/build_yolo.py            # BCCD
python scripts/make_merged.py           # create datasets/merged + configs/merged.yaml

# training (ROCm environment, see docs/SETUP.md)
python scripts/train_merged.py          # 11n, 26n, 11s
python scripts/sweep_nano.py            # 8n, 9t, 10n, 12n

# evaluation / ensemble
python scripts/ensemble.py              # COCO-harness single vs WBF
```

## Counting accuracy (the clinically meaningful metric)

Protocol: class-aware NMS at IoU 0.70 (matches the evaluation path), confidence tuned on the
**val** split (87 images), reported on the **test** split (72 images, GT: RBC 805, WBC 71,
Platelets 69). Detections from the ncnn deployment models; square 640 letterbox.

| Model | val-tuned conf | RBC MAE | WBC MAE | Platelets MAE | All-class MAE |
|---|---|---|---|---|---|
| yolov9t | 0.60 | 3.49 (31 %) | 0.08 (8 %) | 0.33 (35 %) | 3.90 |
| yolo11n | 0.50 | 3.22 (29 %) | 0.08 (8 %) | 0.33 (35 %) | 3.64 |
| **yolo26n** | **0.50** | **3.00** (27 %) | **0.07** (7 %) | **0.28** (29 %) | **3.36** |

Detailed breakdown for yolo26n @ conf 0.50:

| Class | MAE | Median | Within ±1 cell | Exact | Bias |
|---|---|---|---|---|---|
| RBC | 3.00 | 2 | 34.7 % | 12.5 % | +0.19 |
| WBC | 0.07 | 0 | **100 %** | 93.1 % | +0.07 |
| Platelets | 0.28 | 0 | 98.6 % | 73.6 % | +0.03 |
| All classes | — | — | — | 8.3 % of images | — |

Aggregate counts over the whole test split (v26n): **RBC 820 vs 805 (+1.9 %)**,
**WBC 76 vs 71 (+7.0 %)**, **Platelets 71 vs 69 (+2.9 %)**. Per-image error is ±3 RBC, but
the bias is small, so errors cancel over many fields.

**Counting flips the model ranking**: yolo26n, which lost on mAP50-95 (0.619 vs 0.640),
is the best counter (3.36 vs 3.64). Its higher mAP50 (0.929) was the right signal.

Letterbox matters: with rectangular (val-style) inference the same model scores 4.04 all-class
MAE on 24 test images versus 3.75 with square padding — square is better for counting here, and
torch-square and ncnn-square agree exactly (3.75 both), confirming the deployment path matches
the reference once NMS and letterbox are aligned.

### Fixing the counting errors

Four remedies were tried for the RBC duplicate / platelet-precision problem, all tuned on val
and reported on test (yolo26n):

| Attempt | All-class MAE | RBC MAE | Verdict |
|---|---|---|---|
| baseline 640, NMS iou 0.7 | 3.36 | 3.00 | reference |
| NMS IoU sweep (0.50–0.80) | 3.36 | 3.00 | no effect — duplicates don't overlap enough |
| **export at 960, iou 0.50, conf 0.50** | **3.17** | **2.79** | **best** |
| export at 1280 | 3.74 | 3.28 | worse (over-splitting) |
| flip TTA + WBF | 3.33 | 2.92 | undercounts (fusion merges neighbours) |
| multi-scale 640+960 WBF | 3.31 | 2.92 | no better than 960 alone |

Final counting accuracy (yolo26n @ 960, val-tuned):

| Class | MAE | Within ±1 | Bias | Total pred/gt |
|---|---|---|---|---|
| RBC | 2.79 | 45.8 % | +0.04 | 808 / 805 (+0.4 %) |
| WBC | 0.10 | 98.6 % | +0.10 | 78 / 71 (+9.9 %) |
| Platelets | 0.28 | 98.6 % | −0.03 | 67 / 69 (−2.9 %) |
| All | 3.17 | — | — | — |

Cost: 960 inference is 39.5 ms/frame (25 FPS) on the RX 570 versus 15 ms at 640 — irrelevant for
still images, and the gain is a 7 % reduction in RBC counting error.

### Exhaustive fix attempts (all val-tuned, test-reported)

| Attempt | All-class MAE | Notes |
|---|---|---|
| baseline 640 | 3.36 | reference |
| NMS IoU sweep | 3.36 | duplicates don't overlap enough |
| **960 export** | **3.17** | **best** (RBC 3.00 → 2.79) |
| 1280 export | 3.74 | over-splits |
| flip TTA + WBF | 3.33 | merges neighbours |
| multi-scale 640+960 | 3.31 | no gain |
| 2×2 tiling (640 / 960) | not selected on val | tiles upscale but LOSE small boxes in fusion |
| per-class config selection | 3.24 | val overfits (WBC chose conf 0.05) |

Conclusion: post-processing, resolution and fusion are exhausted. The residual RBC error
(2.79 cells/field) is structural to box-based counting of touching cells — two cells either
merge into one box or split into two, and no NMS/fusion setting can tell which is correct.
The remaining honest fixes are label quality (BCCD misses cells) or instance segmentation.

### Segmentation experiment (the last structural fix) — failed

BCCD ships boxes only, so a seg dataset was synthesised by turning every box into a 4-point
rectangle polygon (14,475 train / 1,137 val / 945 test). `yolo26n-seg` trained 100 epochs
(2.05 h). Counting was evaluated with both box-NMS and **mask-IoU NMS** (which should keep two
touching cells apart when their masks are disjoint), thresholds tuned on val.

| Method | RBC MAE | WBC MAE | Platelets MAE | All-class MAE |
|---|---|---|---|---|
| detection @960 (best) | **2.79** | 0.10 | **0.28** | **3.17** |
| yolo26n-seg | 3.97 | 0.06 | 0.57 | 4.60 |

Mask-IoU NMS was **not selected on val** — it never beat plain box NMS. Reasons:

1. Synthetic rectangle masks carry no boundary information, so mask IoU between two touching
   cells is nearly the same as box IoU — there is nothing for mask-NMS to exploit.
2. The mask head costs capacity: platelets (10–20 px) collapsed from 0.28 to 0.57 MAE and the
   model predicted 30 of 69 platelets.
3. Segmentation is only worth it with **real polygon annotations**, which BCCD does not have.

**Conclusion: box detection at 960 px remains the best counting configuration on this dataset
(3.17 all-class MAE). The residual RBC error is a label/boundary problem, not an architecture one.**

## 7-class Roboflow BCCD (`models-u4dak`) — separate domain

Dataset: `datasets/roboflow/models-u4dak__bccd-blw1i__v1`
Classes: Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil, Platelets, RBC
Splits: 762 train / 163 val / 165 test (stratified 70/15/15, seed 0)
Model: `yolo26n`, 640 px, batch 8, 100 epochs, fp32, 0.79 h
Test: 165 images, 1,290 instances

| Metric | Plain | TTA |
|---|---|---|
| mAP50 | 0.9583 | **0.9619** |
| mAP50-95 | 0.8046 | **0.8368** |
| Precision | 0.8958 | 0.9046 |
| Recall | 0.9368 | 0.9429 |

Per-class TTA:

| Class | AP50 | AP50-95 |
|---|---|---|
| Basophil | 0.9758 | 0.9023 |
| Eosinophil | 0.9796 | 0.8891 |
| Lymphocyte | 0.9683 | 0.8726 |
| Monocyte | 0.9201 | **0.7364** |
| Neutrophil | 0.9423 | 0.8790 |
| Platelets | **0.9778** | **0.7176** |
| RBC | 0.9696 | 0.8605 |

Caveats:

- **Separate domain.** Do not compare these with the 3-class BCCD numbers above. The images are
  640x640, produced by vertically stretching 640x480 sources, and the labels include five WBC
  subtypes that the 3-class data does not have.
- The archive shipped only `train/`; `valid/` and `test/` are a local stratified re-split
  (`scripts/split_yolo_dataset.py`, seed 0). No stems are shared between splits.
- Monocyte (0.736) and Platelets (0.718) have the weakest AP50-95. Both are small or
  morphologically ambiguous; the 2.8 % platelet share does not hurt AP50 (0.978) but does hurt
  localisation.
- First run on this dataset; no hyperparameter sweep was done.

## Common-space benchmark (plain inference, both domains)

To compare the 3-class BCCD models with the 7-class Roboflow model, all three were run on
**both test sets in one shared 3-class space** (RBC, WBC, Platelets), with identical settings:
640 px, conf 0.001, IoU 0.7, max_det 300, no TTA, COCOeval. The 7-class model's five WBC
subtypes are collapsed to WBC, and duplicate subtype boxes are merged with per-class NMS.
Inference ran on CPU because loading three models in one process triggers the known gfx803
multi-model GPU fault in ROCm 5.4.3.

| Model | Test set | mAP50 | mAP50-95 | RBC | WBC | Platelets |
|---|---|---|---|---|---|---|
| yolo11n (3-class) | BCCD official | 0.8842 | **0.6184** | 0.5926 | 0.8067 | 0.4559 |
| yolo26n (3-class) | BCCD official | 0.8828 | 0.5968 | 0.5763 | 0.7621 | 0.4521 |
| yolo26n (7-class) | BCCD official | 0.8181 | 0.5577 | 0.4798 | 0.7343 | 0.4592 |
| yolo11n (3-class) | u4dak | 0.1875 | 0.1369 | 0.1440 | 0.2643 | 0.0023 |
| yolo26n (3-class) | u4dak | 0.0589 | 0.0380 | 0.0044 | 0.1093 | 0.0003 |
| yolo26n (7-class) | u4dak | **0.9747** | **0.8255** | 0.8662 | 0.8941 | 0.7163 |

Reading it:

- **Each model wins its own domain.** On BCCD, the 3-class models score 0.597-0.618
  mAP50-95; on u4dak, the 7-class model scores 0.826 in the collapsed 3-class space.
- **Domain transfer is asymmetric.** The 7-class model on BCCD reaches 0.558; the 3-class
  models on u4dak collapse to 0.137 and 0.038. u4dak is a much harder transfer target for a
  model trained on BCCD/BCD, and platelets nearly vanish (AP50-95 0.002).
- **The earlier gap was task and domain, not model quality.** Comparing the 7-class 0.8368
  (native 7-class mAP, TTA) with the 3-class 0.6186 (3-class mAP, TTA) was misleading. On a
  common protocol and common class space, the 7-class model is not universally better; it is
  better on its own corpus.
- This benchmark uses plain inference, so its numbers are lower than the TTA numbers in the
  tables above (for example yolo26n 3-class 0.5968 here vs 0.6186 TTA). It is internally
  consistent; do not mix the two protocols.

Raw: `results/benchmark_common.json`.
