# Third-party notices

This repository vendors the following upstream works. Each was cloned at the commit listed,
with its `.git` directory removed and, where noted, copied under `datasets/`.

## 1. BCCD Dataset

- **Source:** https://github.com/Shenggan/BCCD_Dataset
- **Commit:** `d272fb14cdff`
- **License:** MIT (see `third_party/BCCD_Dataset/LICENSE`)
- **Used as:** `third_party/BCCD_Dataset/` and `datasets/bccd/`
- **Copyright:** original annotations from cosmicad and akshaylamba; re-organised into VOC
  format by Shenggan. Retained for attribution.

## 2. Complete Blood Cell Count (CBC) Dataset

- **Source:** https://github.com/MahmudulAlam/Complete-Blood-Cell-Count-Dataset
- **Commit:** `ac6f9ff39760`
- **License:** MIT (see `third_party/CBC/LICENSE`)
- **Used as:** `third_party/CBC/` (Training / Validation / Testing, VOC layout)

## 3. CST-YOLO

- **Source:** https://github.com/mkang315/CST-YOLO
- **Commit:** `30af001eed65`
- **License:** **GPL-3.0** (see `third_party/CST-YOLO/LICENSE`)
- **Reference:** Kang, M., Ting, C.-M., Ting, F. F., & Phan, R. C.-W. (2024). *CST-YOLO: A Novel
  Method for Blood Cell Detection Based on Improved YOLOv7 and CNN-Swin Transformer.* IEEE ICIP
  2024, pp. 3024–3029. doi:10.1109/ICIP51287.2024.10647618
- **Used as:** `third_party/CST-YOLO/`, unmodified, reference only.

Because GPL-3.0 code is included, the combined work in this repository is licensed GPL-3.0.

## 4. Kaggle Blood Cell Detection (BCD)

- **Source:** https://www.kaggle.com/datasets/adhoppin/blood-cell-detection-datatset
- **License:** MIT, as stated in the dataset's `data.yaml`; **Kaggle dataset terms also apply.**
- **Used as:** `datasets/bcd/` (already in YOLO format, `data.yaml` included)
- **Note:** this data is redistributed here only inside a private repository for research
  convenience. Do not make this repository public while this directory is present.

## Papers referenced (not vendored)

| Paper | Venue | Link |
|---|---|---|
| Small object detection: A comprehensive survey… | Intelligent Systems with Applications, 2025 | doi:10.1016/j.iswa.2025.200870 |
| Dyna-HemoNet: A lightweight and adaptive framework… | Pattern Recognition Letters 207, 2026 | doi:10.1016/j.patrec.2026.06.002 |
| Complete Blood Cell Detection and Counting… | Applied Sciences 12(16):8140, 2022 | doi:10.3390/app12168140 |
| SDE-YOLO: A Novel Method for Blood Cell Detection | Biomimetics 8(5):404, 2023 | doi:10.3390/biomimetics8050404 |
| Small object detection leveraging density-aware scale adaptation | The Photogrammetric Record 38(182), 2023 | doi:10.1111/phor.12446 |
| CST-YOLO (see above) | IEEE ICIP 2024 | doi:10.1109/ICIP51287.2024.10647618 |
