# BCCD-Next

End-to-end blood-cell detection and counting for **RBC**, **WBC**, and **platelets** in
microscope blood-smear images, trained on a single AMD RX 570 (gfx803) with ROCm 5.4.3 +
PyTorch 1.13.1.

The pipeline is complete: datasets → VOC→YOLO conversion and merge → training → evaluation →
counting → ncnn/MIGraphX export. Seven YOLO families were compared, and a class-aware counting
protocol was tuned on val and reported on the held-out test split.

## Contents

```
.
├── datasets/
│   ├── bccd/        BCCD — 364 images, VOC XML annotations, official splits
│   ├── bcd/         Kaggle BCD — YOLO format (train/valid/test), data.yaml included
│   └── (cbc)        CBC lives in third_party/CBC (VOC layout)
├── scripts/         dataset builders, training, evaluation, counting, export
├── configs/         bccd.yaml, merged.yaml, merged_seg.yaml
├── results/         RESULTS.md + machine-readable JSON (mAP, counting, sweeps)
├── docs/            SETUP.md (ROCm recipe), HARDWARE.md (stability notes)
├── models/          deployable weights + ncnn export
├── media/           demo video and detection grid
├── third_party/
│   ├── BCCD_Dataset/   upstream BCCD repo (MIT)
│   ├── CBC/            Complete Blood Cell Count dataset repo (MIT)
│   └── CST-YOLO/       YOLOv7 + CNN-Swin Transformer reference implementation (GPL-3.0)
├── LICENSE              GPL-3.0 (because CST-YOLO is vendored)
└── THIRD_PARTY_NOTICES.md
```

## Datasets

| Dataset | Images | Classes | Annotation format | Source | License |
|---|---|---|---|---|---|
| **BCCD** | 364 (640×480) | RBC, WBC, Platelets | VOC XML | [Shenggan/BCCD_Dataset](https://github.com/Shenggan/BCCD_Dataset) | MIT |
| **BCD** | see `datasets/bcd` | Platelets, RBC, WBC | YOLO txt + `data.yaml` | [Kaggle: adhoppin/blood-cell-detection-datatset](https://www.kaggle.com/datasets/adhoppin/blood-cell-detection-datatset) | MIT (Kaggle terms apply) |
| **CBC** | see `third_party/CBC` | RBC, WBC, Platelets | VOC XML (Annotations/) | [MahmudulAlam/Complete-Blood-Cell-Count-Dataset](https://github.com/MahmudulAlam/Complete-Blood-Cell-Count-Dataset) | MIT |

Roboflow Universe copies (YOLO format, plus a 7-class WBC-subtype export) live in
`datasets/roboflow/`; provenance, split caveats and verification numbers are in
[`datasets/roboflow/README.md`](datasets/roboflow/README.md).

### BCCD class distribution (measured from the annotations)

| Class | Objects | Share |
|---|---|---|
| RBC | 4 155 | 85.0 % |
| WBC | 372 | 7.6 % |
| Platelets | 361 | 7.4 % |
| **Total** | **4 888** (13.4 per image) | |

The 11:1 RBC-to-WBC imbalance is the main reason per-class AP matters more than aggregate mAP
on this dataset.

### BCCD splits (as shipped, `datasets/bccd/ImageSets/Main/`)

| Split | Images |
|---|---|
| train | 205 |
| val | 87 |
| test | 72 |

These are the upstream splits. Many papers re-split randomly; results are only comparable if the
same split is used, which is why the shipped one is kept here.

> **Known issue:** the BCCD annotations are incomplete — not every visible cell is boxed. This
> caps achievable precision and makes very high reported mAP values suspect.

## Reference implementation

`third_party/CST-YOLO` is the official code for *CST-YOLO: A Novel Method for Blood Cell
Detection Based on Improved YOLOv7 and CNN-Swin Transformer* (ICIP 2024). It is YOLOv7-based,
expects PyTorch ≤ 1.7.1 + CUDA, and is vendored unmodified as reference only.


## Results

Trained and evaluated on the RX 570 (ROCm 5.4.3 + PyTorch 1.13.1), then exported to ncnn/MIGraphX.
Full tables, per-class breakdown, ensemble, counting protocol, failed experiments and methodology
caveats: **[results/RESULTS.md](results/RESULTS.md)**.

### Detection (BCCD test split — 72 images, 945 boxes)

| Model | mAP50 | mAP50 TTA | mAP50-95 | mAP50-95 TTA |
|---|---|---|---|---|
| yolov8n | 0.9027 | 0.9089 | 0.6201 | 0.6266 |
| **yolov9t** | 0.9136 | 0.9137 | **0.6407** | 0.6395 |
| yolov10n | 0.9176 | 0.9176 | 0.6356 | 0.6356 |
| **yolo11n** | 0.9069 | 0.9191 | 0.6301 | **0.6398** |
| yolo11s | 0.9165 | 0.9203 | 0.6263 | 0.6341 |
| yolo12n | 0.9221 | 0.9232 | 0.6355 | 0.6368 |
| **yolo26n** | 0.9162 | **0.9287** | 0.6073 | 0.6186 |

- **Best at finding cells (mAP50):** `yolo26n` — 0.9287 with TTA.
- **Best tight boxes (mAP50-95):** `yolov9t` 0.6407 plain / `yolo11n` 0.6398 TTA.
- `yolo11s` (9.4 M params, 3.5 h) scored below `yolo11n` (2.6 M, 51 min): capacity saturated.
- **Ensemble (WBF):** best combo `v9t+v11n+v12n` = **0.6295** mAP50-95. That is below
  `yolov9t` alone (0.6407) in the main comparison — it only beat the ensemble harness's own
  single-model baseline (0.6228) — so **the ensemble was not adopted**.

### Counting accuracy (val-tuned, test-reported)

Protocol: class-aware NMS, confidence tuned on **val (87)**, reported on **test (72)**.
Best config: **`yolo26n` @ 960 px export, IoU 0.50, conf 0.50**.

| Class | MAE (cells/image) | Within ±1 | Bias | Total pred / GT |
|---|---|---|---|---|
| **RBC** | **2.79** | 45.8 % | +0.04 | 808 / 805 (**+0.4 %**) |
| **WBC** | **0.10** | 98.6 % | +0.10 | 78 / 71 (+9.9 %) |
| **Platelets** | **0.28** | 98.6 % | −0.03 | 67 / 69 (−2.9 %) |
| **All classes** | **3.17** | — | — | — |

Counting flips the detection ranking: `yolo26n` loses on mAP50-95 but wins on counting because
mAP50 (finding every cell) matters more than tight boxes.

### Deployment (RX 570)

| Runtime | Speed | Notes |
|---|---|---|
| PyTorch + ROCm | 13.26 ms | needs the 3 GB ROCm stack |
| ncnn + Vulkan | 12.33 ms | ~100 MB runtime, no ROCm |
| **MIGraphX 2.4** | **7.85 ms** | ~127 img/s |
| ncnn full pipeline | 15.08 ms | 66 FPS |

Counting at 960 px runs at **39.5 ms/frame (25 FPS)** — irrelevant for still images.

### What did not work

Nine counting remedies were measured; none beat 960 detection (3.17 all-class MAE): NMS IoU
sweep, 1280 export, flip TTA + WBF, multi-scale, 2×2 tiling, per-class configs, and
`yolo26n-seg` with mask-IoU NMS (4.60 — worse, because BCCD has no real masks). **The residual
RBC error is structural to box counting of touching cells**, not a model or post-processing
choice. The only remaining honest fixes are better labels or real polygon annotations.

- **CBC was excluded** — 183/420 of its images duplicate BCCD val/test.
- Platelets are localisation-limited: AP50 ≈ 0.90, AP50-95 ≈ 0.48.

Environment recipe: **[docs/SETUP.md](docs/SETUP.md)**. Hardware/stability notes:
**[docs/HARDWARE.md](docs/HARDWARE.md)**. Demo: `media/bccd_final_demo.mp4`,
weights: `models/bccd_v11n_best.pt` (+ ncnn export).

## Quick start

```bash
# 1. BCCD VOC -> YOLO, then merge BCCD + filtered BCD into datasets/merged
python scripts/build_yolo.py
python scripts/make_merged.py

# 2. train (ROCm environment — see docs/SETUP.md)
python scripts/train_merged.py      # yolo11n / yolo26n / yolo11s
python scripts/sweep_nano.py        # yolov8n / v9t / v10n / v12n

# 3. evaluate + ensemble (COCO harness, weighted box fusion)
python scripts/ensemble.py

# 4. count — confidence tuned on val, reported on test
python scripts/count_tune.py        # per-model 640 baseline
python scripts/count_tta.py         # 960 / TTA variants
python scripts/count_fix_all.py     # tiling / per-class selection
python scripts/count_seg.py         # segmentation comparison
```

Derived datasets (`datasets/merged/`, `datasets/merged_seg/`) and run outputs are gitignored;
regenerate them with the scripts above.

## Licensing

The combined repository is **GPL-3.0** because it contains CST-YOLO code. BCCD and CBC are MIT
and require their copyright notices to be retained (see `THIRD_PARTY_NOTICES.md`). The Kaggle
BCD dataset is distributed under Kaggle's dataset terms; it is included here in a private
repository for research convenience and must not be redistributed publicly.

## Provenance

| Upstream | Commit |
|---|---|
| Shenggan/BCCD_Dataset | `d272fb14cdff` |
| MahmudulAlam/Complete-Blood-Cell-Count-Dataset | `ac6f9ff39760` |
| mkang315/CST-YOLO | `30af001eed65` |
