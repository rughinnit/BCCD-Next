# BCCD-Next

Blood cell detection workspace: datasets, reference implementations, and (next) training /
evaluation tooling for detecting **RBC**, **WBC**, and **platelets** in microscope blood-smear
images.

This repository currently vendors the three public datasets and the CST-YOLO reference code,
together with provenance and licensing. Tooling (VOC→YOLO conversion, training configs,
evaluation, ncnn export) is the next step.

## Contents

```
.
├── datasets/
│   ├── bccd/        BCCD — 364 images, VOC XML annotations, official splits
│   ├── bcd/         Kaggle BCD — YOLO format (train/valid/test), data.yaml included
│   └── (cbc)        CBC lives in third_party/CBC (VOC layout)
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

## Quick start (pending tooling)

Planned, not yet in this repo:

```bash
# 1. convert BCCD VOC XML -> YOLO txt + data.yaml
python tools/voc2yolo.py --src datasets/bccd --dst datasets/bccd_yolo

# 2. train a baseline
yolo detect train model=yolov8n.pt data=datasets/bccd_yolo/data.yaml imgsz=640 epochs=100

# 3. evaluate per class
yolo detect val model=runs/detect/train/weights/best.pt data=datasets/bccd_yolo/data.yaml
```

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
