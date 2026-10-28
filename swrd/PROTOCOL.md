# SWRD split and tiles (v1): protocol and reproducibility log

Source: the official SWRD release (`SWXD_Data`: `Raw_data` and `crop_weld_data`), downloaded from the URL given in the
SWRD paper. The release has no official training–validation split. The dataset used for training is the output of
`build_swrd_tiles.py` **v4.1** (folder `swrd_tiles`). Rendering environment: Python 3.12 (Windows),
opencv-python-headless 5.0.0 (CLAHE), Pillow, NumPy 2.2.6.

The scripts in this folder are the ones that were run; their comments and printed messages are in Indonesian.
Every script expects to be placed in the `SWXD_Data` folder (next to `Raw_data` and `crop_weld_data`) unless its
docstring says otherwise.

## Order of the scripts

| Step | Script | Output |
|---|---|---|
| 1 | `swrd_audit.py SWXD_Data` | per-film inventory of the raw labels (`swrd_audit_raw.csv` = `inputs/in_labels.csv`) |
| 2 | `swrd_audit2.py` | file inventory, MD5 duplicate groups, crop inventory (`inputs/in_inventory.csv`, `inputs/in_dup.csv`, `swrd_audit2_crop_inventory.csv`) |
| 3 | `make_split_swrd.py` (reads `in_*.csv` from the working folder) | `swrd_split_v1.csv`, `swrd_excluded_v1.csv` |
| 4 | `swrd_audit3.py`, `swrd_audit4.py` | leakage audit across the split (label-geometry fingerprints, high-pass NCC, full-resolution patch NCC) |
| 5 | `verify_crops.py`, `diag_crops.py` | checks that the official crops and their labels are reproduced from the raw radiographs |
| 6 | `build_swrd_tiles.py` (`--plan-only` first, then without it) | `swrd_tiles/` (PNG tiles, YOLO labels, `data.yaml`, `manifest.csv`, `summary.txt`) |
| 7 | `audit_labels.py`, `check_tiles.py` | audit of every label against the original annotations; final integrity check |
| 8 | `swrd_geometry_diagnostic.py` | Supplementary S2: share of boxes the detection head can match, share of boxes touching a tile border |

`make_split_swrd.py` run on the three files in `inputs/` reproduces `swrd_split_v1.csv` and `swrd_excluded_v1.csv`
byte for byte. `swrd_geometry_diagnostic.py` was run under the name `diagnostik_geometri_swrd.py`, which its usage
line still shows.

The column `alasan` (reason) of `swrd_excluded_v1.csv` takes the values `8-bit (domain pindaian lain)` (8-bit film,
a different scanning domain), `salinan identik (md5) dari batch 20230321` (byte-identical copy of batch 20230321),
`sumber non-DJ-RT (RRT/bam)` (outside the DJ-RT series), `duplikat identik dengan label konflik` (identical duplicate
with conflicting labels), `berkas tidak terbaca` (unreadable file) and `bukan citra (Thumbs.db)` (not an image).

## Outputs of the checks (`outputs/`)

Written by the scripts above on the real data, after the last change to each script; in Indonesian.

| File | Script | Supports |
|---|---|---|
| `verify_crops_summary.txt` | `verify_crops.py` | pairing of the 4,610 official crops, crop origins (4,261 of 4,263), 37 sampled crops pixel-identical |
| `diag_crops_summary.txt` | `diag_crops.py` | label mismatches (1,209 official labels paired with a deviation above 1.5 px), 38 further crops pixel-identical |
| `swrd_audit4_console.txt` | `swrd_audit4.py` | full-resolution patch NCC of the highest-scoring pairs across the split against random pairs |
| `audit_labels_v3_summary.txt` | `audit_labels.py` on the v3 tiles | the defects of v3 (1 box without a defect, 89 boxes less than half filled, 249 loose) |
| `audit_labels_v4.1_summary.txt` | `audit_labels.py` on the v4.1 tiles | v4.1 passes every check (4,038 boxes) |
| `geometry_diagnostic/` | `swrd_geometry_diagnostic.py` | Supplementary S2: per-box geometry, share of boxes the detection head can match, boxes touching a tile edge |

## Cleaning and split

- Films kept: 3,358 from 76 production batches (DJ-RT). Excluded: 322, namely 258 films stored at 8-bit depth, 22 films
  outside the DJ-RT acquisition series (RRT-nnR, bam5), 38 byte-identical copies (batch 20230909 duplicates batch
  20230321), 2 duplicates with conflicting labels, 1 unreadable film, and a `Thumbs.db` file.
- Split by production batch, about 90/10 (`make_split_swrd.py`, seed 7): training 3,026 films from 63 batches,
  validation 332 films from 13 batches.
- Leakage audit: MD5 hashes, label-geometry fingerprints (0 matches), high-pass normalized cross-correlation at ×16
  downsampling and at full resolution on patches (the 24 highest-scoring pairs are at the level of random pairs,
  about 0.02). No evidence of duplicates across the split; radiographs of the same weld taken in different batches
  cannot be excluded.

## Verification of the official crops (`verify_crops.py`, `diag_crops.py`)

- All 4,610 official crops pair with a weld-seam polygon box; official sizes are inclusive (x1 − x0 + 1).
- The crop origin is the corner of the weld-seam polygon (integer coordinates) for 4,261 of the 4,263 crops whose
  origin could be estimated; 75 sampled crops are pixel-identical to the raw radiograph.
- Official crop labels are the raw labels clipped to the crop (1,208 of the 1,209 boxes that do not match exactly
  are smaller, median area 45%); degenerate fragments exist. Decision: use the official crop labels (polygons) as
  they are.

## Builder versions

- v1, v2, v2.1: never rendered on the real data (assumed horizontal welds; labels from the raw JSON).
- v3: label = official object box clipped to the tile. `audit_labels.py` found all label origins correct, but one
  box without any defect and 89 boxes in which the defect fills less than 50% of the box (lack of penetration 73 of
  240), plus 249 loose boxes (50–90%). Abandoned.
- v4: polygon clipped to the tile, but short-axis visibility used the global box height, so curved
  lack-of-penetration defects were wrongly treated as truncated. Never run.
- **v4.1 (final)**: box = bounding box of the part of the defect polygon inside the tile; visibility = [extent in tile
  / min(object extent, tile)] along the crop's long axis × [polygon area in tile / polygon area in the tile's band]
  along the short axis; kept if ≥ 0.5, ambiguous if 0.1–0.5 (tile discarded). Geometry functions tested against
  shapely on about 36,000 random polygons (error 0 / 5e-12).

## Tiles v4.1 (seed 2026)

- Unit = official crop; pixels from the official TIFF (for crops without an image, taken from the raw radiograph at
  the crop origin); every tile checked pixel-identical with the raw radiograph (0 discarded).
- Windows of 640 px without overlap along the long axis, a random 640 px band on short axes longer than 640 px;
  tiles ≤ 640 × 640, no scaling, no rotation.
- A tile is discarded if ambiguous, if it contains a label outside the six classes, or if a raw defect is visible
  without an official label. Pseudo-defects are background. Negative tiles touch no defect box.
- Positive:negative 1:1, at most 2 + 2 tiles per film, square-root quota with an 8% floor. Per-tile preprocessing:
  0.5–99.5 percentile stretch → 8 bit → CLAHE (clip limit 2.0, 8 × 8 grid).
- Result: training 2,700 tiles (1,350 with defects), validation 700 (350 with defects); 4,038 boxes
  (3,154 training, 884 validation).
- Objects / tiles / films, training: porosity 2,204/840/835, inclusion 254/179/179, crack 244/203/203,
  undercut 124/108/91, lack of fusion 130/121/120, lack of penetration 198/193/193.
- Objects / tiles / films, validation: porosity 619/209/198, inclusion 78/54/53, crack 85/75/74, undercut 14/13/9,
  lack of fusion 47/37/29, lack of penetration 41/40/38. Undercut, lack of fusion and lack of penetration have fewer
  than 40 validation films and are reported but not interpreted in the paper.
- `audit_labels.py` (v4.1): passed. `check_tiles.py`: passed.

## Differences from the SWRD authors' protocol

Split by production batch (not 9:1 over an unspecified unit), non-overlapping 640 px tiles (not overlapping
windows), boxes from polygons clipped to the tile, our own contrast enhancement. Scores on these tiles are therefore
not comparable with those reported by the SWRD authors.
