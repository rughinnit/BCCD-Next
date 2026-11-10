# Dataset verification: quan-c2r6k / bccd-ccnkh / v1

Root: `E:\Documents\Electrical Engineering\Semester 7\Rapal\BCCD-Next\datasets\roboflow\quan-c2r6k__bccd-ccnkh__v1`

Method: self-contained Python 3.12 stdlib script. Images counted by image extension; labels counted as `*.txt` only (`README*.txt` excluded). Dimensions parsed from JPEG SOF markers. No dataset files were modified.

## Headline

- 8599 images / 8599 labels, fully paired, in 3 splits.
- `nc: 7` with 7 names. 20284 bounding-box objects and 8113 polygon objects (28.6% of lines). **Polygon lines are not valid YOLOv8 detection rows** despite the README claim of YOLOv8 export.
- Only 363 of the 364 original BCCD images are represented by name (99.7%), and their labels are RBC/PLT only. The remaining images come from other single-class blood-smear sources.
- 8599 images collapse to 5622 distinct base stems (1.53x repeat factor), while the README states no augmentation was applied.

## 1. Splits: images and labels

| Split | Images | Label .txt | Missing label | Orphan label | Other files in labels/ |
|---|---|---|---|---|---|
| train | 6025 | 6025 | 0 | 0 | - |
| valid | 1759 | 1759 | 0 | 0 | - |
| test | 815 | 815 | 0 | 0 | - |
| **total** | **8599** | **8599** | **0** | **0** | |

Image count matches the 8599 stated in `README.roboflow.txt`.

No unlabeled images and no orphan labels. Every image has exactly one `.txt` label file.

Split ratios: train 70.1%, valid 20.5%, test 9.5%.

## 2. data.yaml and label validation

- `nc` = 7
- `names` = ['BAS', 'EOS', 'LYM', 'MON', 'NEU', 'PLT', 'RBC']
- name count equals nc: **True**
- label files inspected: **8599**, non-empty object lines: **28397**

### Line format distribution

| Split | 5-field bbox rows | polygon rows (>=7, odd field count) | malformed rows | files mixing formats |
|---|---|---|---|---|
| train | 13236 | 5745 | 0 | 0 |
| valid | 5049 | 1592 | 0 | 0 |
| test | 1999 | 776 | 0 | 0 |
| **total** | **20284** | **8113** | **0** | **0** |

Format is consistent per file: no file mixes bbox and polygon rows. The format is determined by the source subset, not by split.

### Violations against the declared YOLOv8 detection format

| Check | Violating rows | Example files |
|---|---|---|
| line does not have exactly 5 fields | 8113 | `BA_100102_jpg.rf.1262a14b01e60d0698d85a644e241de6.jpg`, `BA_100102_jpg.rf.62d48987731cfb545513eb714405895c.jpg`, `BA_101381_jpg.rf.ad5c20a9e070d2d4fd04979e9e1fc3d2.jpg`, `BA_102750_jpg.rf.6b3f0bbd5b235a62ae964e75f016f53b.jpg`, `BA_102750_jpg.rf.fb5845830e581ea0a5f8517df395d52a.jpg` |
| class id is not an integer | 0 | - |
| class id outside [0, nc-1] | 0 | - |
| any coordinate outside [0,1] | 0 | - |
| bbox width or height <= 0 | 0 | - |
| bbox width or height > 1 | 0 | - |
| polygon has fewer than 3 points | 0 | - |
| polygon has zero area | 0 | - |

### Validation under the polygon interpretation (for completeness)

If the odd-field rows are read as YOLO segmentation polygons (`cls x1 y1 x2 y2 ...`), they parse cleanly: all coordinates fall in [0,1], every polygon has at least 3 points, and every polygon has non-zero area. None of the structural errors above occur once the polygon rows are excluded.

Images with an empty label file: **0**. Objects per label file: min **1**, max **31**.

Practical impact: a standard YOLOv8/YOLO11 detector loader expects exactly 5 fields and will either reject or silently mis-parse the 8113 polygon rows (28.6% of the corpus). They must be converted to bounding boxes (min/max of each polygon) or dropped before detection training. The dataset is usable as-is only for instance segmentation.

## 3. Object statistics

| Split | BAS | EOS | LYM | MON | NEU | PLT | RBC | Total |
|---|---|---|---|---|---|---|---|---|
| train | 931 | 987 | 908 | 984 | 890 | 1637 | 12644 | 18981 |
| valid | 269 | 268 | 255 | 275 | 234 | 521 | 4819 | 6641 |
| test | 121 | 106 | 121 | 129 | 145 | 236 | 1917 | 2775 |
| **total** | **1321** | **1361** | **1284** | **1388** | **1269** | **2394** | **19380** | **28397** |

| Class id | Name | bbox rows | polygon rows | Total | Share |
|---|---|---|---|---|---|
| 0 | BAS | 0 | 1321 | 1321 | 4.7% |
| 1 | EOS | 0 | 1361 | 1361 | 4.8% |
| 2 | LYM | 0 | 1284 | 1284 | 4.5% |
| 3 | MON | 0 | 1388 | 1388 | 4.9% |
| 4 | NEU | 0 | 1269 | 1269 | 4.5% |
| 5 | PLT | 904 | 1490 | 2394 | 8.4% |
| 6 | RBC | 19380 | 0 | 19380 | 68.2% |

- Total images: **8599**
- Total objects: **28397** (bbox 20284 + polygon 8113)
- Objects per image: mean **3.30**, median **1**, min **1**, max **31**
- Images with zero objects: **0**

Per-split object density:

- train: 18981 objects over 6025 images = 3.15 objects/image (min 1, max 31)
- valid: 6641 objects over 1759 images = 3.78 objects/image (min 1, max 29)
- test: 2775 objects over 815 images = 3.40 objects/image (min 1, max 31)

Class balance is heavily skewed: RBC alone is 68.2% of all objects, and the five rarer classes (BAS, EOS, LYM, MON, NEU) together are 23.3%. Any training split must be stratified or the minority classes will be effectively ignored.

### Objects by source subset (filename prefix)

| Source prefix | Images | Base stems | Repeat factor | Objects | Class ids present |
|---|---|---|---|---|---|
| MO | 1380 | 937 | 1.47x | 1388 | 3:1388 |
| EO | 1333 | 917 | 1.45x | 1361 | 1:1360, 4:1 |
| BA | 1241 | 851 | 1.46x | 1353 | 0:1321, 4:32 |
| LY | 1214 | 820 | 1.48x | 1284 | 2:1284 |
| PLATELET | 1126 | 807 | 1.40x | 1487 | 1:1, 5:1486 |
| BloodImage | 859 | 365 | 2.35x | 16667 | 5:903, 6:15764 |
| SNE | 600 | 417 | 1.44x | 612 | 4:612 |
| BNE | 599 | 398 | 1.51x | 614 | 4:610, 5:4 |
| image-* | 233 | 100 | 2.33x | 3617 | 5:1, 6:3616 |
| NEUTROPHIL | 14 | 10 | 1.40x | 14 | 4:14 |

Every non-BCCD subset is effectively single-class: MO->MON, EO->EOS, BA->BAS, LY->LYM, PLATELET->PLT, BNE/SNE/NEUTROPHIL->NEU, image-*->RBC. The 7-class problem is assembled by concatenating per-class source datasets, not by re-annotating one corpus consistently. Cross-source label noise and image-domain shift should be expected.

## 4. Duplicates and augmentation families

### Exact duplicates (identical file size + identical MD5 of first 4 KB)

- Duplicate groups: **13**
- Images involved: **26** (0.3%)
- Redundant copies beyond one per group: **13**

- 2 copies, 24787 bytes, e.g. `BloodImage_00125_jpg.rf.ac54caeab1bad90e1d592e7dcc04a1be.jpg`, `BloodImage_00125_jpg.rf.b783067225eb5cd43c9de24bfc71e5a9.jpg`
- 2 copies, 23873 bytes, e.g. `BloodImage_00072_jpg.rf.b6953940efd0482a20e8e17838e7767e.jpg`, `BloodImage_00072_jpg.rf.c8db38d7ae74bf271c4fa6bb9c357895.jpg`
- 2 copies, 24016 bytes, e.g. `BloodImage_00144_jpg.rf.2293306a05c0914d46b16e1d506c6c9f.jpg`, `BloodImage_00144_jpg.rf.e7126c60dbebff7f071978883c399811.jpg`
- 2 copies, 25232 bytes, e.g. `BloodImage_00191_jpg.rf.3a91e672133def47b545d61ef579d7a6.jpg`, `BloodImage_00191_jpg.rf.ef29012395455c1f7905e7d1e9eac00a.jpg`
- 2 copies, 26583 bytes, e.g. `BloodImage_00265_jpg.rf.09a72b0438ed1a3f0ef695d9320d7cc5.jpg`, `BloodImage_00265_jpg.rf.8ae99ab855a1522a0be3de6425eca054.jpg`
- 2 copies, 26187 bytes, e.g. `BloodImage_00287_jpg.rf.4ba763e3f641c9263625e95e5a32e8a8.jpg`, `BloodImage_00287_jpg.rf.04f339938e25fb74694acefd6dd5bb5b.jpg`

Byte-level duplication is negligible. The inflation is not from repeated identical files.

### Repeat families (Roboflow suffix stripped)

Stripping rule: remove `.rf.<hex>` / `_jpg.rf.<hex>` export suffixes, a trailing `_jpg`, and common augmentation tokens. Roboflow writes a fresh `<32 hex>` content hash into every exported filename, so distinct filenames with the same base stem are re-encodes or variants of one source image.

- Image files: **8599**
- Distinct base stems: **5622**
- Mean images per base stem: **1.53x**
- Base stems with more than one image: **2322** (41.3% of stems)
- Extra images beyond one per stem: **2977** (34.6%)

Multiplicity histogram (images per stem -> stems):

| Images per stem | Stems |
|---|---|
| 1 | 3300 |
| 2 | 1667 |
| 3 | 655 |

Duplicate multiplicity within one stem is the export pattern of repeated uploads/variants, not of a clean 1:1 export. The nominal 8599 images correspond to only 5622 unique source images. Of those, only 365 stems are BCCD-derived; the other 5257 stems belong to the other donor datasets.

### BCCD-derived subset multiplicity

- `BloodImage_*` base stems: **365** covering **859** images (10.0% of the dataset)
- Mean images per BloodImage stem: **2.35x**
- Most repeated stems: `BloodImage_00409` (3), `BloodImage_00405` (3), `BloodImage_00404` (3), `BloodImage_00403` (3), `BloodImage_00402` (3), `BloodImage_00398` (3)

So even the genuine BCCD portion is duplicated roughly 2-3 times. If a downstream evaluation wants the original 364 images exactly once, these stems must be deduplicated, otherwise the same blood smear appears in both train and test and inflates accuracy.

Copies of the same stem are not byte-identical (see above) and usually do not carry identical labels: only **325 of 2322** multi-copy stems have byte-identical label files. The rest are re-encoded images with re-drawn annotations, which is a further source of label noise if they straddle splits.

## 5. Cross-check against the original 364-image BCCD set

- Reference images in `JPEGImages`: **364**; annotation `.xml`: **364**
- Reference stems found as a prefix of a raw Roboflow filename: **363 / 364** (99.7%)
- Reference stems found after stripping the Roboflow export suffix and `_jpg`: **363 / 364** (99.7%)
- `BloodImage_*` base stems in the Roboflow set: **365**, of which **363** match the reference and **2** do not (BloodImage_00096, BloodImage_00329)
- Reference stems with no Roboflow counterpart: `BloodImage_00134`

This is a strong, filename-level confirmation that the `BloodImage_*` subset is derived from the original BCCD set. It does not prove pixel equality: all Roboflow images were resized and re-encoded, so byte-level or 4 KB prefix hashes cannot match across the two sets without a decoder (unavailable without PIL).

Reference image dimensions (`JPEGImages`, SOF markers):

| Dimensions | Count |
|---|---|
| 640 x 480 | 364 |

Roboflow image dimensions, full scan of all images:

| Dimensions | Count |
|---|---|
| 640 x 640 | 8599 |

All 8599 Roboflow images are 640 x 640, i.e. square. The original BCCD images are mostly 640 x 480. The export therefore resized or padded/cropped non-square images to 640 x 640, which the README does not mention (it only claims auto-orientation). Coordinate frames still map 1:1 to the exported pixels, so labels are consistent with the images as shipped, but absolute cell shapes are distorted relative to the original set.

Sample of 30 Roboflow images (seed 42):

| File | W | H |
|---|---|---|
| `BNE_982036_jpg.rf.3f39df7de02bfd752cf7691292f4622d.jpg` | 640 | 640 |
| `BA_515577_jpg.rf.2ed0abdf6414254a7d74b3258be0b567.jpg` | 640 | 640 |
| `MO_718520_jpg.rf.71d23d3fdadff9e0a9409eee1c6723ad.jpg` | 640 | 640 |
| `MO_25276_jpg.rf.69eed66443224548ef9f90eaa58c78ae.jpg` | 640 | 640 |
| `LY_845737_jpg.rf.e7551f36db98724338c8235a2861cb1b.jpg` | 640 | 640 |
| `EO_499434_jpg.rf.d27e3e513cd60e2742149ac37a8d2180.jpg` | 640 | 640 |
| `BNE_647348_jpg.rf.547f85512df8be77e6df005b7aef24a2.jpg` | 640 | 640 |
| `BNE_106934_jpg.rf.614ee4bc5872c12f03ef0bfff4462397.jpg` | 640 | 640 |
| `image-48_png.rf.e984cb6e97ee06cb0533c16e0de5aba2.jpg` | 640 | 640 |
| `BA_634407_jpg.rf.213fbd1df40f40720c16f9f68db8217b.jpg` | 640 | 640 |
| `BA_604921_jpg.rf.6d4e43e0d3b27f8c3a50ba6eddb7cb32.jpg` | 640 | 640 |
| `BNE_331697_jpg.rf.0e5ade4426be46e0a0b2a44652e1c355.jpg` | 640 | 640 |
| `LY_781116_jpg.rf.fed65140c32082f9abc31d74ac3ccdf7.jpg` | 640 | 640 |
| `LY_996704_jpg.rf.b8304a4aa5f5b947673f2b49f0a4e15b.jpg` | 640 | 640 |
| `LY_91407_jpg.rf.7e261efc9ca53c02059ed2ae012c0787.jpg` | 640 | 640 |
| `BA_537842_jpg.rf.d3ddfdb41b1fe00c3dff3860baea5a8b.jpg` | 640 | 640 |
| `LY_434773_jpg.rf.d6eb9527e92d7f3777f04769d15178c4.jpg` | 640 | 640 |
| `EO_952649_jpg.rf.9d325057b729566349d195342e2eb0eb.jpg` | 640 | 640 |
| `LY_815802_jpg.rf.df41c36b7343dd2cac6f22080ec299a5.jpg` | 640 | 640 |
| `MO_657432_jpg.rf.274986e028e1a31d63200d36f740cedc.jpg` | 640 | 640 |
| `MO_767982_jpg.rf.48e1a4ef7fb6c217f18a500f608ee2f5.jpg` | 640 | 640 |
| `BA_220744_jpg.rf.a2976a67c251ef0e103b984363948f8e.jpg` | 640 | 640 |
| `EO_830203_jpg.rf.dc57d85aee2f9655394de37ad924db01.jpg` | 640 | 640 |
| `image-69_png.rf.1c221a5b0e9d2f93ab8f1a11e66b4861.jpg` | 640 | 640 |
| `PLATELET_963383_jpg.rf.797d5c1b0a27f807719b45c903d02215.jpg` | 640 | 640 |
| `MO_765056_jpg.rf.87bda1d084129e3cde240c1ebad56ad7.jpg` | 640 | 640 |
| `EO_75419_jpg.rf.17615043c45ab8e218959e7e73b0cd09.jpg` | 640 | 640 |
| `LY_720543_jpg.rf.c89d7fb323ee17ccfc1cdba2add387b4.jpg` | 640 | 640 |
| `PLATELET_902144_jpg.rf.29e6fa86645cbdc7e5094ab18e3c43d0.jpg` | 640 | 640 |
| `BNE_638211_jpg.rf.2c30893b58eff71eb3669005a9d8faf1.jpg` | 640 | 640 |

Reference annotation class names: RBC=4155, WBC=372, Platelets=361 (4888 objects over 364 files).

The original BCCD annotates 3 labels (RBC, WBC, Platelets). The Roboflow `BloodImage_*` subset only uses class 5 (PLT, 903 objects) and class 6 (RBC, 15764 objects); no WBC/neutrophil objects survive for those images. The WBC subtypes in the 7-class scheme come from other donor datasets, so the class definitions are not consistent across sources.

Annotation counts do not reproduce the original either. For the 109 `BloodImage_*` stems that have exactly one Roboflow copy, the export holds 2098 objects versus 1471 objects in the reference XML (mean 19.2 vs 13.5 per image), and only 4 of 109 images have the same object count. The annotations were re-drawn or re-filtered, not copied from `Annotations/`. In particular the reference WBC boxes are gone and RBC boxes are denser, so this cannot be used as a label-identical superset of the original.

## 6. README source statements

`README.dataset.txt` (full):

```
# BCCD > 2025-10-18 2:17am
https://universe.roboflow.com/quan-c2r6k/bccd-ccnkh

Provided by a Roboflow user
License: Public Domain
```

`README.roboflow.txt` (relevant excerpt):

```
BCCD - v1 2025-10-18 2:17am
This dataset was exported via roboflow.com on October 18, 2025 at 6:37 AM GMT
The dataset includes 8599 images.
Objects are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)

No image augmentation techniques were applied.
```

Stated source: only the Roboflow Universe project `quan-c2r6k/bccd-ccnkh`, exported 2025-10-18, Public Domain, "Provided by a Roboflow user". No upstream paper, DOI, or original BCCD URL is cited. The word "BCCD" in the dataset name is the only explicit provenance claim, and it is not supported by the image inventory, which is dominated by non-BCCD source prefixes.

## 7. Is this a superset of the 364-image BCCD set?

Partly, and only by filename. 363 of the 364 original stems (`BloodImage_*`) are present, so the original corpus is contained in name terms. But:

1. Each original stem appears about 2-3 times, so it is not a clean one-copy superset.
2. The original 3-class annotation (RBC, WBC, Platelets) is not preserved: `BloodImage_*` images carry only RBC and PLT objects in this export. WBC-class objects were dropped or re-sourced.
3. 90.0% of the images (7740) are unrelated single-class blood-smear datasets (MO, EO, BA, LY, PLATELET, SNE, BNE, NEUTROPHIL, image-*). This is a concatenated multi-source corpus, not an augmented BCCD.
4. All images were resized to 640 x 640, and 28.6% of annotation rows are polygons rather than the declared YOLOv8 bbox rows.

Verdict: **not a usable drop-in superset**. It is a separate multi-source dataset that reuses the BCCD images as one of ten inputs. For a BCCD baseline, use the original 364-image set; for a 7-class blood-cell benchmark, treat this export as a new dataset and document the polygon/bbox mix, the 640 x 640 resize, and the cross-source class definitions.

