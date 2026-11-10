# Dataset verification: models-u4dak__bccd-blw1i__v1

- Root: `BCCD-Next/datasets/roboflow/models-u4dak__bccd-blw1i__v1`
- Format: YOLO (Roboflow export, 2026-11-04)
- Checked: 2026-11-05, Python 3.12 stdlib only, dataset read-only
- Verification script: `C:\Users\Acer\AppData\Local\Temp\opencode\verify_bccd_yolo.py`

## Summary

| Metric | Value |
| --- | --- |
| Splits present | 1 (`train`) |
| Total images | 1090 |
| Total label files | 1090 |
| Total objects | 8615 |
| Objects per image | 7.90 mean / 7 median / 1-26 range |
| Classes in `data.yaml` | 7 |
| Validation problems | 0 |
| Exact duplicate images | 0 |
| Image dimensions | uniform 640x640 |

## Splits: declared vs present

`data.yaml` declares three splits, but the archive contains only `train/`:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
```

- Present on disk: `train/` only (`train/images`, `train/labels`).
- `valid/` and `test/` do not exist anywhere in the extracted dataset.
- The zip `_zips/models-u4dak__bccd-blw1i__v1.zip` was listed read-only and also contains only the `train/` top-level folder, so this is missing from the archive itself, not from extraction.
- No `valid` or `test` images/labels exist, so there is no validation or test split to evaluate against.

Path quirk: the declared values `../train/images` etc. resolve to `datasets/roboflow/train/images`, which does not exist (one level above the dataset root). The actual directory is `train/images`. Ultralytics has a fallback that strips a leading `../`, but `train: train/images` would be unambiguous.

## Split table

| Split | Images | Labels | Objects |
| --- | --- | --- | --- |
| train | 1090 | 1090 | 8615 |
| valid | 0 (missing) | 0 (missing) | 0 |
| test | 0 (missing) | 0 (missing) | 0 |
| **total** | **1090** | **1090** | **8615** |

## Class distribution (whole dataset, single split)

| id | name | objects | share |
| --- | --- | --- | --- |
| 0 | Basophil | 713 | 8.3% |
| 1 | Eosinophil | 722 | 8.4% |
| 2 | Lymphocyte | 725 | 8.4% |
| 3 | Monocyte | 745 | 8.6% |
| 4 | Neutrophil | 696 | 8.1% |
| 5 | Platelets | 245 | 2.8% |
| 6 | RBC | 4769 | 55.4% |

Per-split counts are identical to the totals because only `train` exists. RBC dominates at 55.4%; Platelets are the rarest class at 2.8%.

## Validation results

Parsed `data.yaml`: `nc: 7`, `names: ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelets', 'RBC']`.

All 1090 label files were checked line by line (8615 lines):

| Check | Result |
| --- | --- |
| Image with no matching label | 0 |
| Label with no matching image | 0 |
| Lines with exactly 5 fields | 8615 / 8615 |
| Class id is an integer in [0, 6] | 8615 / 8615 |
| All four coordinates in [0, 1] | 8615 / 8615 |
| Width > 0 and height > 0 | 8615 / 8615 |
| Box corners inside the frame | 8615 / 8615 |
| Empty label files (0 objects) | 0 |
| Unreadable image headers | 0 |

Observed coordinate ranges: x 0.0086-0.9906, y 0.0078-0.9969, w 0.0172-0.6406, h 0.0063-0.6250. Smallest box area 0.0011, largest 0.3386.

**Validation passed cleanly: no problems found.**

## Duplicates and augmentation check

- Exact duplicates: 0. All 1090 images have a distinct full-file SHA-256.
- Near-duplicates by file size + SHA-256 of the first 2 KB: 0 duplicate groups.
- Repeated stems: 0. All 1090 stems are unique; 805 unique timestamp prefixes (up to 8 frame suffixes per prefix, e.g. `_0` to `_7`), which is normal source-camera naming.
- Same byte size but different content: 47 pairs out of 1090 (4.3%). All differ in content; this is ordinary JPEG encoder size variance, not copies.
- `README.roboflow.txt` states only "Resize to 640x640 (Stretch)" and "No image augmentation techniques were applied".

Assessment: this is **not a byte-level duplicate or augmented copy** of the local 364-image `datasets/bccd` set. The count 1090 is roughly 3 x 364, but filenames use capture timestamps (`20190526_163519_0`) rather than `BloodImage_00000`, all stems and payloads are unique, and Roboflow augmentation would have produced repeated stems with different `.rf.` hashes. It looks like an independently imported, larger BCCD compilation. Pixel-level identity against the 364-image set was not tested because no image decoder was available outside the stdlib; only file-level evidence is reported.

## Image size summary

All 1090 images are **640x640** (100%), read from the JPEG SOF markers (no PIL). The sample of 20 images is uniform:

| # | stem | dimensions |
| --- | --- | --- |
| 1 | 20190526_163519_0 | 640x640 |
| 2 | 20190526_163608_1 | 640x640 |
| 3 | 20190526_163729_0 | 640x640 |
| 4 | 20190526_163753_0 | 640x640 |
| 5 | 20190526_164053_0 | 640x640 |
| 6 | 20190526_164136_0 | 640x640 |
| 7 | 20190526_164157_0 | 640x640 |
| 8 | 20190526_164453_0 | 640x640 |
| 9 | 20190526_164646_0 | 640x640 |
| 10 | 20190526_165148_0 | 640x640 |
| 11 | 20190526_165257_0 | 640x640 |
| 12 | 20190526_165317_0 | 640x640 |
| 13 | 20190526_165338_0 | 640x640 |
| 14 | 20190526_165411_0 | 640x640 |
| 15 | 20190526_165958_0 | 640x640 |
| 16 | 20190526_170021_0 | 640x640 |
| 17 | 20190526_170125_0 | 640x640 |
| 18 | 20190526_170157_0 | 640x640 |
| 19 | 20190526_170233_0 | 640x640 |
| 20 | 20190526_170409_0 | 640x640 |

File sizes: 19,528 to 43,221 bytes, median 28,671 bytes, 31.5 MB total.

Note: the source `datasets/bccd` images are 640x480, so the 640x640 export is a non-uniform **stretch** (vertical scale 1.333) rather than letterbox padding. Relative YOLO coordinates remain valid, but object aspect ratios are distorted relative to the original images; this matters if results are compared with models trained on 640x480 BCCD.

## Issues to fix before training

1. No validation split. Ultralytics requires a valid `val` path; either create a split from `train` or repoint `val` to `train` (which would make validation meaningless).
2. No test split, so held-out evaluation is impossible with this archive as-is.
3. `data.yaml` has no `path:` key and uses `../train/images`; use `train: train/images` (and a real `val:` directory) to avoid resolution ambiguity.
4. Class imbalance (RBC 55.4%, Platelets 2.8%) should be considered for sampling or loss weighting.
