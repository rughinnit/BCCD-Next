# Dataset Verification: roboflow-100__bccd-ouzjz__v2

- Root: `E:\Documents\Electrical Engineering\Semester 7\Rapal\BCCD-Next\datasets\roboflow\roboflow-100__bccd-ouzjz__v2`
- Format: YOLO (Roboflow 100 export, BCCD blood cell detection)
- `data.yaml`: `nc: 3`, `names: ['Platelets', 'RBC', 'WBC']`, license CC BY 4.0
- Verified: 2026-11-05, Python 3.12 stdlib only (manual JPEG SOF / PNG IHDR parsing, no PIL)

## Verdict

Structurally sound but **not fully clean**: file pairing is complete, class ids and coordinate ranges are valid, and there are no duplicates or empty labels. Two label rows in `train` contain degenerate boxes (`w = h = 0`), which are annotation errors and were excluded from object counts.

## Split summary

| Split | Images | Labels | Raw label rows | Valid objects | Objects/image |
|-------|--------|--------|----------------|---------------|---------------|
| train | 255 | 255 | 3450 | 3448 | 13.52 |
| valid | 73 | 73 | 967 | 967 | 13.25 |
| test | 36 | 36 | 471 | 471 | 13.08 |
| **total** | **364** | **364** | **4888** | **4886** | **13.42** |

Every split folder (`train/`, `valid/`, `test/`) is present with `images/` and `labels/` subfolders. All 364 images are `.jpg`; all 364 labels are `.txt`.

## Per-class object counts

| id | Class | train | valid | test | total | Share |
|----|-------|-------|-------|------|-------|-------|
| 0 | Platelets | 249 | 76 | 36 | 361 | 7.4% |
| 1 | RBC | 2936 | 819 | 398 | 4153 | 85.0% |
| 2 | WBC | 263 | 72 | 37 | 372 | 7.6% |

Class ratios are consistent across splits (RBC roughly 85%, Platelets and WBC roughly 7-8% each), so the split is stratified correctly.

## Validation results

Checks run against all 4888 label rows:

- Class ids: all integers within `[0, 2]`. Pass.
- Field count: all rows have exactly 5 whitespace-separated fields. Pass.
- Coordinate range: all center-x, center-y, width, height values within `[0, 1]`. Pass.
- Positive size: **2 failures** (degenerate zero-area boxes).
- Image/label pairing: 0 images without a label, 0 labels without an image, 0 stem collisions.

### Problems found (2)

| File | Line | Content | Issue |
|------|------|---------|-------|
| `train/labels/BloodImage_00338_jpg.rf.3caf9de9e3846e233f900377b5271a11.txt` | 13 | `1 0.7875 0.7020833333333333 0 0` | class 1 (RBC), width = height = 0 |
| `train/labels/BloodImage_00343_jpg.rf.d8c56063ce5e40c50efb00a7e0c83c3b.txt` | 4 | `1 0.2828125 0.6854166666666667 0 0` | class 1 (RBC), width = height = 0 |

Impact: negligible in volume (2 of 4888 rows, 0.04%), but these rows will train badly or divide by zero in loss/area computations. Recommended fix: delete both rows, or repair by giving each box a minimal size (for example 0.01 x 0.01 around the stated center). The centers look plausible, so the boxes may have been collapsed during export.

## Empty label files

None. All 364 label files contain at least one object row. Minimum objects per image is at least 1.

## Duplicates

**0 duplicates.** Checked across all splits combined using file size plus SHA-256 of the first 2048 bytes: 364 unique (size, head-hash) keys for 364 images, so no image appears twice, including no cross-split leakage by this measure.

## Image sizes

Parsed directly from headers (JPEG SOF0/SOF1/SOF2 markers):

- All 364 images are **640 x 480**.
- 1 distinct size across the whole dataset; **yes, all images are the same size.**
- Header parse succeeded for 364/364 images.

Sampled 20 images (evenly spaced over the sorted file list, spread across train/valid/test): all reported 640 x 480.

## Warnings

- Class imbalance: RBC (class 1) accounts for 85.0% of all objects. Expected for blood smear microscopy and acceptable for detection training, but mAP on Platelets and WBC will be the limiting metric; consider per-class metrics when evaluating.
- No class is empty and no class exceeds the 90% dominance threshold.

## Reproduction

Script: `C:\Users\Acer\AppData\Local\Temp\opencode\verify_bccd.py` (read-only; no dataset files were modified).
