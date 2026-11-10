# Roboflow Universe BCCD copies

Datasets pulled from the Roboflow Universe search for `bccd`
(https://universe.roboflow.com/search?q=bccd), exported as **YOLOv8** format
(`txt` labels + `data.yaml`). Nothing here is trained. The exports are kept as downloaded,
with one exception: `models-u4dak` arrived as a single `train/` split and was re-split
locally (documented below).

Selected because they are the three that add something the vendored sets do not:

| Folder | Source | License | Images | Splits | Classes |
|---|---|---|---|---|---|
| `roboflow-100__bccd-ouzjz__v2` | [roboflow-100/bccd-ouzjz v2](https://universe.roboflow.com/roboflow-100/bccd-ouzjz/dataset/2) | CC BY 4.0 | 364 | 255 / 73 / 36 | Platelets, RBC, WBC |
| `models-u4dak__bccd-blw1i__v1` | [models-u4dak/bccd-blw1i v1](https://universe.roboflow.com/models-u4dak/bccd-blw1i/dataset/1) | CC BY 4.0 | 1,090 | 762 / 163 / 165 (local split) | 5 WBC subtypes + Platelets + RBC |
| `quan-c2r6k__bccd-ccnkh__v1` (local only, git-ignored) | [quan-c2r6k/bccd-ccnkh v1](https://universe.roboflow.com/quan-c2r6k/bccd-ccnkh/dataset/1) | Public Domain | 8,599 | 6,025 / 1,759 / 815 | BAS, EOS, LYM, MON, NEU, PLT, RBC |

Per-dataset verification output: [`_reports/`](_reports).

## Why these three

- `roboflow-100` is the RF100 benchmark copy. Same 364 source images and the same boxes as
  `datasets/bccd` (verified: 4,886/4,888 VOC boxes match at IoU >= 0.5), but in YOLO format
  and with RF100's own random 70/20/10 re-split. That split is **not** the upstream BCCD
  split: 117 of its 255 train images are official val or test images. Use it for RF100
  comparability, never mixed with `datasets/bccd`.
- `models-u4dak` is the only BCCD-derived copy with **fine-grained WBC subtypes**
  (Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil) alongside RBC and Platelets.
- `quan-c2r6k` is the largest (8,599 images) and is public domain. It is **not** a clean
  BCCD superset and is not committed to git - see caveats below.

Not fetched on purpose: `joseph-nelson/bccd` (3.5k views, MIT) is the same 364 images as
`datasets/bccd` with no extra annotations, and the `de-jgbno`, `str`, `school-*` copies are
plain re-uploads. Dozens of the 300 results are 364-image duplicates of each other.

## 1. roboflow-100__bccd-ouzjz__v2

- 364 images, all 640x480 JPEG; nc=3, `names: ['Platelets', 'RBC', 'WBC']`
- Splits: train 255 / valid 73 / test 36, every image has a label file
- Objects: 4,886 total (train 3,448 / valid 967 / test 471), 13.4 per image
- Per class: Platelets 361 (7.4%), RBC 4,153 (85.0%), WBC 372 (7.6%)
- Clean: no missing pairs, no empty labels, 0 duplicate images
- **Split caveat:** RF100 uses a random 70/20/10 re-split, not the upstream
  `ImageSets/Main` split shipped in `datasets/bccd`. 60 of its train images are official val
  images and 57 are official test images; only 4 of its 36 test images are official test.
  Numbers from this copy are not comparable with `results/RESULTS.md`.
- Known defect: 2 degenerate boxes with `w = h = 0` in train
  (`BloodImage_00338...txt` line 13, `BloodImage_00343...txt` line 4, both class 1 = RBC).
  Drop or clamp them before training.

```bibtex
@misc{ bccd-ouzjz_dataset,
  title = { bccd Dataset },
  type = { Open Source Dataset },
  author = { Roboflow 100 },
  howpublished = { \url{https://universe.roboflow.com/roboflow-100/bccd-ouzjz } },
  url = { https://universe.roboflow.com/roboflow-100/bccd-ouzjz },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2022 },
}
```

## 2. models-u4dak__bccd-blw1i__v1

- 1,090 images, all 640x640; nc=7
- `names: ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelets', 'RBC']`
- Objects: 8,615 total, 7.9 per image (median 7, range 1-26)
- Per class: Basophil 713, Eosinophil 722, Lymphocyte 725, Monocyte 745, Neutrophil 696,
  Platelets 245, RBC 4,769
- Labels validate cleanly: 5 fields, ids in range, coords in [0,1], w/h > 0, no out-of-frame
  boxes, 0 duplicates
- **Re-split locally.** The export ships only `train/`. `scripts/split_yolo_dataset.py`
  moved the 1,090 pairs into a stratified 70/15/15 split (seed 0): train 762 / valid 163 /
  test 165, every class present in all three, zero stems shared between splits. The object
  total is unchanged (8,615). `data.yaml` keeps the `../<split>/images` convention used by
  the other Roboflow exports.
- 640x640 comes from a vertical stretch of 640x480 sources, not letterboxing. Objects look
  vertically stretched relative to `datasets/bccd`.
- Not an augmented copy of the 364-image set: perceptual hash distance to every BCCD image
  is >= 10 (no near-duplicates), nor does it overlap the CBC set. Timestamp-based names,
  subtype labels are genuinely new annotation work.

## 3. quan-c2r6k__bccd-ccnkh__v1

- 8,599 images, all 640x640; nc=7 (`BAS, EOS, LYM, MON, NEU, PLT, RBC`)
- Splits: train 6,025 / valid 1,759 / test 815
- Objects: 28,397 total, 3.3 per image (median 1, max 31); RBC 19,380 (68.2%)
- Per class: BAS 1,321, EOS 1,361, LYM 1,284, MON 1,388, NEU 1,269, PLT 2,394, RBC 19,380

Caveats, in order of importance:

1. **Labels are segmentation polygons, not boxes.** 8,113 of 28,397 lines carry 11-67
   fields (`cls x1 y1 x2 y2 ...`) even though the export was requested as YOLOv8. No single
   file mixes both styles, but the split is by class: all five WBC subtypes exist *only* as
   polygons (BAS 1,321, EOS 1,361, LYM 1,284, MON 1,388, NEU 1,269), Platelets appear both
   ways (904 boxes, 1,490 polygons) and RBC only as boxes (19,380). A detection loader will
   reject or silently read zero objects from the polygon files; convert polygons to boxes
   first (`x_min,y_min,x_max,y_max` of the point list) or use it for instance segmentation.
2. **Split leakage.** 981 of 5,622 base stems (17.5%) appear in more than one split,
   including 147 BCCD-derived stems (for example `BloodImage_00072` in both train and test).
   Any model trained on this set without de-leaking has seen part of its test data.
3. **Only ~10% is BCCD.** 365 stems trace back to the 364 BCCD stems; the other 5,257 come
   from MO, EO, BA, LY, PLATELET, SNE, BNE, NEUTROPHIL and `image-*` single-class sources.
   Of the 364 reference stems, 363 appear by name, but their annotations do not match the
   originals (denser RBC boxes, missing WBC boxes), so this is a different label set.
4. **1.53x stem repetition.** 5,622 base stems produce 8,599 files; 41.3% of stems repeat,
   up to 3 copies, mostly re-encoded with re-drawn annotations. The bundled README claims
   "no image augmentation" - the stems contradict that.
5. 13 exact duplicate groups (26 images), 0.3%.
6. All images were resized from 640x480 to square 640x640.

Conclusion: usable as a separate 7-class instance-segmentation corpus after polygon-to-box
conversion and de-leaking; not usable as a BCCD detection baseline. It is git-ignored
because it is 231 MB of 269 MB and re-downloadable with the fetch script.

## Re-downloading

Everything here came from the Roboflow export API with a free Roboflow account key:

```bash
export ROBOFLOW_API_KEY=...          # app.roboflow.com -> Settings -> API Keys
python scripts/fetch_roboflow_datasets.py            # fetch + extract
python scripts/fetch_roboflow_datasets.py --verify   # recheck counts only
```

The script is stdlib-only, refuses to overwrite an existing directory unless `--force` is
passed, and never writes the key to disk. The three source zips are not kept in the repo
(~333 MB); `_zips/` is recreated on demand and is git-ignored.

Re-creating the local split of `models-u4dak` after a fresh fetch (the committed copy is
already split, so this is only needed if you re-download it):

```bash
python scripts/split_yolo_dataset.py datasets/roboflow/models-u4dak__bccd-blw1i__v1
```

Committed size: about 38 MB (364 + 1,090 images). The 8,599-image `quan-c2r6k` set adds
231 MB locally and is git-ignored.
