# Deformable scanning in Mamba-YOLO for radiographic weld defect detection: code, configurations, and per-seed results

Material released with the paper

> Beny, Hendry, I. K. Timotius, and B. Kristianto, "A Capacity-Matched Experimental Evaluation of Deformable Scanning
> in Mamba-YOLO for Radiographic Weld Defect Detection," submitted to *IEEE Access*.

The paper compares DeformSS2D, a deformable-scanning replacement for the SS2D module of Mamba-YOLO-T, with the
unmodified fixed-path SS2D baseline at a parameter count matched to within 1%, on the steel-pipe weld dataset and on
SWRD, with five seeds per configuration. This repository contains what is needed to check the reported numbers: the
module code in every version that produced a result, the patch that integrates it into Mamba-YOLO, the model
configurations, the notebooks that ran the experiments, the final metrics and training logs of the runs, the analysis
scripts, and the scripts that build our SWRD split and tiles.

## Quick start: reproduce the tables

No GPU, dataset or checkpoint is needed.

```bash
pip install -r requirements-analysis.txt
python analysis/reproduce_tables.py            # Tables 1-6, statistics quoted in the text, Supplementary S2-S4
python analysis/power_simulation.py            # design resolution of the tests (Section IV-D), several minutes
python analysis/make_figures.py                # Figures 3-6 (written to ./figures)
```

`reproduce_tables.py` also prints the training outcome of every five-seed run under the collapse criteria of
Section IV-D.

## Repository layout

| Path | Content |
|---|---|
| `code/deform_ss2d.py` | DeformSS2D as described in Section III (grid-fixed version, SHA-256 `675445871a2e...`). Used for every deformable result except Table 3 and Supplementary S4. |
| `code/legacy/deform_ss2d_prefix.py` | Earlier version (SHA-256 `cfed78de...`), used for the five partial placement configurations of Table 3, the recipe-selection attempts (Table 6, note 3), and the earlier runs compared in Supplementary S4. |
| `code/legacy/deform_ss2d_v2.py` | Bounded-path variant of Section V-E (SHA-256 `51520b33...`). |
| `code/apply_patches.py` | Turns a clean clone of [HZAI-ZJNU/Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO) into the code base used for the runs and writes all model YAML files. |
| `code/configs/` | The YAML files written by `apply_patches.py` (identical for both code versions). |
| `code/SHA256SUMS` | Checksums of the three module versions and of the YAML files. |
| `notebooks/` | The Colab and Kaggle notebooks that ran the experiments, with their outputs. |
| `results/per_class/` | Final metrics of every evaluated run (validation of `best.pt`, overall and per class). |
| `results/training_logs/` | Ultralytics `results.csv` of the five-seed runs (see below for the runs without a log). |
| `results/RUNS.csv` | Index of the 72 runs: dataset, configuration, seed, original run name, code version, notebook, files, notes. |
| `results/latency/`, `results/token_order/`, `results/diagnostics/`, `results/status/` | Latency benchmark (Table 4), token-order statistics (Section V-E, Fig. 6), single-seed runs of Section V-E, image and object sizes (Section IV-A), training status recorded by the Table 5 notebook. |
| `analysis/` | Analysis and figure scripts (see below). |
| `swrd/` | SWRD split and tiling scripts, the split itself, their outputs, and the protocol (`swrd/PROTOCOL.md`). |
| `figures/fig_arch.tex` | TikZ source of Fig. 1. |

## Where each result comes from

| Result in the paper | Data | Script | Notebook that produced the runs |
|---|---|---|---|
| Table 1, Table 2, Fig. 3 | `per_class_results_PREFIX.csv` (baseline), `_P1FIX.csv` (DeformSS2D, steel-pipe), `_swrdv1.csv` (SWRD), training logs | `reproduce_tables.py`, `make_figures.py` | `fase-2-colab-multiseed-150ep` (baseline), `fase-3a-ablasi-b-lokasi-deformable` (MODEL = "P1-fix"), `Paper_2_fase_4_validasi_swrdv1` (SWRD) |
| Table 3 | `per_class_results_PREFIX.csv`, training logs | `reproduce_tables.py` | `fase-3a-ablasi-b-lokasi-deformable` (earlier code version) |
| Table 4, Fig. 4 | `results/latency/fps_benchmark.csv` | `reproduce_tables.py`, `make_figures.py` | `fase-5-benchmark-fps-colab` (models built from YAML; no checkpoints needed) |
| Table 5, Fig. 5 | `per_class_results_gridfix.csv`, training logs | `reproduce_tables.py`, `make_figures.py` | `Paper_2_fase_13_rerun_gridfix_steelpipe` |
| Table 6 | `per_class_results_BASELINES.csv`, `_RECIPE.csv` | `reproduce_tables.py`, `count_detector_parameters.py` | `fase-8-detektor-pembanding`, `fase-9-lengan-resep-b2` |
| Parameter counts (Tables 3, 6; Section V-E) | | `count_parameters.py`, `count_detector_parameters.py` | |
| Section IV-A, image and object sizes | `results/diagnostics/verifikasi_dimensi_*.csv` (rows `DSF-YOLO`) | `reproduce_tables.py` | `fase-6b-verifikasi-dimensi-citra` |
| Section IV-D, design resolution | per-class CSVs | `power_simulation.py` | |
| Section V-E, Fig. 6 | `results/diagnostics/k5*.csv`, `per_class_results_RECIPE_v2.csv`, `results/token_order/` | `reproduce_tables.py`, `make_figures.py` | `fase-10-k5aug-auto`, `fase-11-kontrol-k5dup-auto`, `fase-12-path-terbatas-auto`, `Paper_2_fase_14_urutan_token_valnyata` |
| Fig. 2 | `results/dataset_figure/pilihan.json` | `make_dataset_figure.py` (needs the dataset images) | `Paper_2_fase_15_contoh_citra_dataset` |
| Supplementary S1 | `swrd/swrd_split_v1.csv`, `swrd/swrd_excluded_v1.csv`, `swrd/outputs/` | `swrd/*.py` | |
| Supplementary S2 | `per_class_results_swrdv1*.csv`, `diag_train_vs_val_swrdv1.csv`, `swrd/outputs/geometry_diagnostic/` | `reproduce_tables.py`, `swrd/swrd_geometry_diagnostic.py` | `Paper_2_fase_4_validasi_swrdv1` |
| Supplementary S3 | `results/diagnostics/` | `reproduce_tables.py` | `fase-10`, `fase-11`, `fase-12` |
| Supplementary S4 | `per_class_results_PREFIX.csv`, `_P1FIX.csv`, `_gridfix.csv`, training logs | `reproduce_tables.py` | `fase-2`, `fase-7-p1-noproj`, `fase-3a`, `Paper_2_fase_13` |

Model labels in the CSV files: `B2` = SS2D baseline (Mamba-YOLO-T); `P1`, `P1-fix` = DeformSS2D, full configuration
(earlier and grid-fixed code); `P1-NoProj` = configuration without the capacity constraint; `BackboneOnly`,
`NeckOnly`, `BB-Late`, `BB-Early`, `BB-Late-Alt` = placement variants of Table 3; `K5dup-auto` = control run of
Section V-E. Steel-pipe class names in the CSV files are those of the original label files; the paper uses
`hollow-bead`, `overlap` and `unfused` for `dense-gas-hole`, `non-fusion` and `unfused-hole`.

Some values in the files are in Indonesian:

- `status_latih` (`per_class_results_gridfix.csv`, `results/status/`): `sehat` = trained, `KOLAPS` = collapsed, as
  judged by the notebook during the run; `rasio_cls_e140_min` is the ratio of criterion (b). The paper uses the full
  criteria (a)-(c), which `reproduce_tables.py` applies to every log; for these runs both give the same outcome.
- `status_kelas` (SWRD files): `agregat` = all classes, `utama` = class interpreted in the paper, `deskriptif` =
  class reported but not interpreted (fewer than 40 validation radiographs).
- `verifikasi_dimensi_*.csv`: `rumus` = `BENAR (piksel)` (sizes in pixels, used in the paper) or `LAMA (asumsi
  persegi)` (an earlier calculation that assumed square images). The rows labeled `SWRD` in these two files come
  from a different dataset (a Roboflow export that notebook `fase-6b` refers to as SWRD) and are not used in the
  paper; the SWRD object statistics in the paper come from `swrd/outputs/geometry_diagnostic/`.

## Notes on the run records

- **Folder names ending in `lr0.0022`** (DeformSS2D seed 4 on the steel-pipe dataset, YOLOv8s in Table 6): these
  are second runs under the same name. Ultralytics appends `2` to an existing run folder, so `..._lr0.002` became
  `..._lr0.0022`; the learning rate was 0.002 (for DeformSS2D seed 4 the training log shows the same learning-rate
  schedule as every other steel-pipe run of the study recipe; the YOLOv8s log is not included).
- **DeformSS2D seed 2, steel-pipe**: the grid-fixed run was saved in the folder of the earlier-version seed-2 run
  (`p1_seed2_150ep_lr0.002`) and replaced that run's checkpoint and log. The earlier run's metrics in
  `per_class_results_PREFIX.csv` were recorded before.
- **Baseline seed 4, steel-pipe**: its `results.csv` has no rows for epochs 26, 27, 87 and 88; it is released as
  recorded. It does not affect the collapse criteria: epoch 140 is present, and the classification loss reaches its
  minimum over epochs 1-140 at epoch 140.
- **Runs without a training log here**: the five comparison detectors and the three default-recipe runs of Table 6,
  and the five earlier-version runs of the capacity-matched model (Supplementary S4). Their final metrics are in
  `results/per_class/`. The logs of the single-seed runs of Section V-E are in `results/diagnostics/`. The
  recipe-selection attempts of Table 6, note 3, are not included.
- **Notebook outputs** show only the last session of each notebook. `Paper_2_fase_13_rerun_gridfix_steelpipe` is
  released without outputs; its `MODEL` variable selects the configuration (`"P1-NoProj"` for Table 5).
  `Paper_2_fase_14_urutan_token_valnyata` needs the DeformSS2D checkpoints of seeds 1 and 3, which are not included;
  their SHA-256 values are recorded in `results/token_order/fase14_meta.json`.

## Rebuilding the models

```bash
git clone https://github.com/HZAI-ZJNU/Mamba-YOLO.git
python code/apply_patches.py Mamba-YOLO                       # grid-fixed DeformSS2D
# python code/apply_patches.py Mamba-YOLO --deform code/legacy/deform_ss2d_prefix.py   # earlier version (Table 3)
python analysis/count_parameters.py Mamba-YOLO params.json    # CPU only; reproduces every parameter count
```

Every patch asserts that its target occurs exactly once, so an incompatible upstream change stops the script. The
script copies the module into the clone and then patches it: it adds an option to remove the input projection of the
offset network and the block classes of the configuration without the capacity constraint. With the default option
the module is unchanged in behavior, and the SHA-256 values in `code/SHA256SUMS` refer to the files in `code/`, not
to the patched copy. The paper cites upstream commit 10d108c (December 2024); the notebooks clone the repository head,
and the six commits up to 51f7be9, the head recorded by the last notebook, change only the README.

## Training

The five-seed runs used the recipe of Section IV-C: SGD, lr0 = 0.002, linear decay (`cos_lr=False`), 150 epochs, no
early stopping (`patience=0`), batch 8, 640 px, AMP, training from scratch, TF32 disabled. The exceptions are the
default-recipe runs of Table 6 and the runs of Section V-E (`optimizer="auto"`, which selects AdamW), and the single
SWRD run with lr0 = 0.01 (Section V-F); `results/RUNS.csv` identifies them. Training ran on NVIDIA A100 GPUs (Colab)
with Ultralytics 8.2.29 as bundled with Mamba-YOLO and the selective-scan kernel compiled from the same repository;
the latency benchmark ran on an NVIDIA T4. The saved outputs of every training notebook record torch 2.11.0+cu128
(Python 3.12 in `fase-2`, 3.13 in the others). The later notebooks (`Paper_2_fase_4`, `Paper_2_fase_13`,
`Paper_2_fase_14`) install torch 2.11.0+cu128 if another version is present (`Paper_2_fase_13` and `Paper_2_fase_14`
also pin torchvision 0.26.0) and refuse to run unless the module has the expected SHA-256. The earlier notebooks do
not enforce a checksum (`fase-10` to `fase-12` print it), so the code version of every run is recorded in
`results/RUNS.csv`. Comments and printed messages in the notebooks are in Indonesian.

The notebooks read the datasets from the authors' Kaggle copies and write to a Google Drive folder; adjust these
paths to your own copies. Credentials are read from a file outside the notebook and are not part of this repository.

The backward pass of `grid_sample`, used by the deformable modules, has no deterministic CUDA implementation, so
deformable runs are not bit-exactly reproducible even with a fixed seed.

## Datasets

- **Steel-pipe weld dataset**: D. Yang *et al.*, "Deep learning based steel pipe weld defect detection," *Applied
  Artificial Intelligence*, vol. 35, no. 15, pp. 1237-1249, 2021, doi:10.1080/08839514.2021.1975391. Used as released
  (training 2,726 images, validation 682 images).
- **SWRD**: X. Zhao *et al.*, "SWRD: A dataset of radiographic image of seam weld for defect detection," *Journal of
  Nondestructive Evaluation*, vol. 44, art. 50, 2025, doi:10.1007/s10921-025-01186-w. The release has no official
  split; `swrd/` contains the scripts that build ours from the official release, the resulting split, and the outputs
  of the checks. Tiles are not redistributed here.

Both datasets are subject to the terms of their authors.

## License

The code is derived from Mamba-YOLO and Ultralytics, both under the GNU Affero General Public License v3.0, and is
released under the same license (`LICENSE`). Result files and documentation may be reused with attribution to the
paper.

## Citation

Please cite the paper above; full bibliographic details will be added after publication.
