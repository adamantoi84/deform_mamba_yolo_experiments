#!/usr/bin/env python3
"""apply_patches.py -- turn a clean clone of HZAI-ZJNU/Mamba-YOLO into the code base used in the paper.

    git clone https://github.com/HZAI-ZJNU/Mamba-YOLO.git
    python code/apply_patches.py Mamba-YOLO                       # grid-fixed DeformSS2D (all main results)
    python code/apply_patches.py Mamba-YOLO --deform code/legacy/deform_ss2d_prefix.py   # earlier version (Table 3)
    python code/apply_patches.py Mamba-YOLO --steelpipe-root /path/to/steel-tube-dataset-all/yolo

What it does (the patches and YAML files are identical, string for string, to cell "4b" of
notebooks/Paper_2_fase_13_rerun_gridfix_steelpipe.ipynb; the steel-pipe data YAML of that cell is written only when
--steelpipe-root is given; comments inside the copied block are in Indonesian):
  * copies the chosen deform_ss2d.py into ultralytics/nn/modules/ and checks its SHA-256;
  * patches 1.1-1.4 (SSM_CLASS indirection in VSSBlock/XSSBlock), 2.1, NP-1...NP-5 (variant without the capacity
    constraint; all new arguments default to the old behavior), 3.1-3.3 (block registration in tasks.py),
    4.1/4.2/5.1 (torch.load weights_only=False for torch >= 2.6), 6.1 (np.trapz -> np.trapezoid for numpy >= 2.4);
  * writes the model YAML files of every configuration to ultralytics/cfg/models/mamba-yolo/.
Every patch asserts that its target string occurs exactly once, so an incompatible upstream change fails loudly.
Upstream: the notebooks cloned the repository head; every commit from 10d108c (cited in the paper) to 51f7be9 (the
head recorded by the last notebook) changes only the README, so the code base is the same for all runs.
"""
import argparse, hashlib, os, re, shutil, sys, py_compile
from pathlib import Path

HERE = Path(__file__).resolve().parent
KNOWN = {"675445871a2e4d55400175ef5f6fda83c90ece651bf818e3f54481bc14a6d9aa": "grid-fixed (main results)",
         "cfed78de317a7781f69cf763d370e223908cf71165a30d0a0b300ec2de020ca8": "earlier version, half-pixel offset (Table 3)"}
ap = argparse.ArgumentParser()
ap.add_argument("repo", help="path to a clean clone of HZAI-ZJNU/Mamba-YOLO")
ap.add_argument("--deform", default=str(HERE / "deform_ss2d.py"), help="deform_ss2d.py to install")
ap.add_argument("--steelpipe-root", default=None, help="folder with images/ and labels/ of the steel-pipe dataset")
args = ap.parse_args()
REPO = os.path.abspath(args.repo)
STEEL_BASE = os.path.abspath(args.steelpipe_root) if args.steelpipe_root else None
STEEL_NAMES = ["air-hole", "bite-edge", "broken-arc", "crack", "dense-gas-hole", "non-fusion", "slag-inclusion", "unfused-hole"]
assert os.path.isfile(f"{REPO}/ultralytics/nn/modules/mamba_yolo.py"), f"{REPO} is not a Mamba-YOLO clone"

_sha = hashlib.sha256(open(args.deform, "rb").read()).hexdigest()
print(f"deform_ss2d.py: {args.deform}\n  sha256 {_sha}  ({KNOWN.get(_sha, 'UNKNOWN VERSION')})")
shutil.copy(args.deform, f"{REPO}/ultralytics/nn/modules/deform_ss2d.py")

# ------------------------------------------------------------------------------------------------------------------
# Verbatim from notebook cell 4b (Fase 13).
# ------------------------------------------------------------------------------------------------------------------
import re, py_compile
def patch(path, subs):
    p = f"{REPO}/{path}"
    src = open(p).read()
    for old, new, tag in subs:
        if new in src:
            print(f"  ~ {tag}: skip"); continue
        assert src.count(old) == 1, f"GAGAL {tag}: {src.count(old)}x (harus 1x)"
        src = src.replace(old, new)
        print(f"  OK {tag}")
    open(p, "w").write(src)

print("\n### Patch")
patch("ultralytics/nn/modules/mamba_yolo.py", [
    ("class XSSBlock(nn.Module):\n    def __init__(",
     "class XSSBlock(nn.Module):\n    SSM_CLASS = SS2D\n    def __init__(", "1.1"),
    ("nn.Sequential(*(SS2D(d_model=self.hidden_dim,",
     "nn.Sequential(*(self.SSM_CLASS(d_model=self.hidden_dim,", "1.2"),
    ("class VSSBlock(nn.Module):\n    def __init__(",
     "class VSSBlock(nn.Module):\n    SSM_CLASS = SS2D\n    def __init__(", "1.3"),
    ("            self.op = SS2D(",
     "            self.op = self.SSM_CLASS(", "1.4"),
])
patch("ultralytics/nn/modules/deform_ss2d.py", [
    ("from .mamba_yolo import SS2D",
     "from .mamba_yolo import SS2D, VSSBlock, XSSBlock", "2.1"),
])

# ---- Patch NP-1..NP-5: varian P1-NoProj (PATCH-p1-noproj.md), diterapkan RUNTIME ke salinan grid-fix ----
# Semua argumen baru default ke perilaku lama -> P1 & 5 varian parsial TIDAK berubah (dicek di pre-flight Sel 5:
# P1 tetap 5.930.869 params). sha256 yang dicatat di CSV = berkas sumber grid-fix SEBELUM patch ini.
patch("ultralytics/nn/modules/deform_ss2d.py", [
    ("    def __init__(self, in_ch: int, kernel_size: int):\n"
     "        super().__init__()\n"
     "        pad = kernel_size // 2 if kernel_size > 1 else 0\n"
     "        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1)\n",
     "    def __init__(self, in_ch: int, kernel_size: int, use_proj: bool = True):\n"
     "        super().__init__()\n"
     "        pad = kernel_size // 2 if kernel_size > 1 else 0\n"
     "        self.use_proj = use_proj\n"
     "        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1) if use_proj else nn.Identity()\n", "NP-1"),
    ("    def __init__(self, embed_dim: int, kernel_size: int = 5):\n"
     "        super().__init__()\n"
     "        self.conv_offset = ConvOffset(embed_dim, kernel_size)\n",
     "    def __init__(self, embed_dim: int, kernel_size: int = 5, use_proj: bool = True):\n"
     "        super().__init__()\n"
     "        self.conv_offset = ConvOffset(embed_dim, kernel_size, use_proj=use_proj)\n", "NP-2"),
    ("                 deform_kernel_size: int = None,\n"
     "                 **kwargs):\n",
     "                 deform_kernel_size: int = None,\n"
     "                 deform_use_proj: bool = True,\n"
     "                 **kwargs):\n", "NP-3a"),
    ("        self.deform_scans = nn.ModuleList([\n"
     "            DeformableScan(d_inner, kernel_size=ks) for _ in range(n_deform)\n"
     "        ])\n",
     "        self.deform_use_proj = deform_use_proj\n"
     "        self.deform_scans = nn.ModuleList([\n"
     "            DeformableScan(d_inner, kernel_size=ks, use_proj=deform_use_proj) for _ in range(n_deform)\n"
     "        ])\n", "NP-3b"),
    ('\nif __name__ == "__main__":',
     '\n\nclass DeformVSSBlockNP(VSSBlock):\n'
     '    """P1-NoProj (backbone): ssm_ratio TETAP dari YAML/VSSBlock (2.0) + ConvOffset tanpa proj.\n'
     '    Total model 6.329.230 params (+5,72% vs B2) -- sengaja di atas toleransi +/-5% (stress test)."""\n'
     '    SSM_CLASS = partial(DeformSS2D, deform_mode="k3_replace", deform_kernel_size=5, deform_use_proj=False)\n'
     '    # TIDAK meng-override __init__ (beda dgn DeformVSSBlock yg memaksa ssm_ratio=1.4)\n'
     '\n\nclass DeformXSSBlockNP(XSSBlock):\n'
     '    """P1-NoProj (neck). Lihat DeformVSSBlockNP."""\n'
     '    SSM_CLASS = partial(DeformSS2D, deform_mode="k3_replace", deform_kernel_size=5, deform_use_proj=False)\n'
     '\n\nif __name__ == "__main__":', "NP-4"),
])
patch("ultralytics/nn/tasks.py", [
    ("    XSSBlock\n)\n",
     "    XSSBlock\n)\nfrom ultralytics.nn.modules.deform_ss2d import DeformVSSBlock, DeformXSSBlock, DeformVSSBlockNP, DeformXSSBlockNP\n", "3.1"),
    ("SimpleStem, VisionClueMerge, VSSBlock, XSSBlock\n        }:",
     "SimpleStem, VisionClueMerge, VSSBlock, XSSBlock, DeformVSSBlock, DeformXSSBlock, DeformVSSBlockNP, DeformXSSBlockNP\n        }:", "3.2"),
    ("RepC3, XSSBlock}:",
     "RepC3, XSSBlock, DeformXSSBlock, DeformXSSBlockNP}:", "3.3"),
    ('            ckpt = torch.load(file, map_location="cpu")\n\n    except ModuleNotFoundError as e:',
     '            ckpt = torch.load(file, map_location="cpu", weights_only=False)\n\n    except ModuleNotFoundError as e:', "4.1"),
    ('        check_requirements(e.name)  # install missing module\n        ckpt = torch.load(file, map_location="cpu")',
     '        check_requirements(e.name)  # install missing module\n        ckpt = torch.load(file, map_location="cpu", weights_only=False)', "4.2"),
])
patch("ultralytics/utils/torch_utils.py", [
    ('    x = torch.load(f, map_location=torch.device("cpu"))\n    if "model" not in x:',
     '    x = torch.load(f, map_location=torch.device("cpu"), weights_only=False)\n    if "model" not in x:', "5.1"),
])

y = open(f"{REPO}/ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml").read()
y = re.sub(r"\bVSSBlock\b", "DeformVSSBlock", y)
y = re.sub(r"\bXSSBlock\b", "DeformXSSBlock", y)
open(f"{REPO}/ultralytics/cfg/models/mamba-yolo/DeMamba-YOLO-T.yaml", "w").write(y)

# Ablasi B: 2 varian campuran -- backbone-saja-deformable, neck-saja-deformable.
# Split di penanda "# Mamba-YOLO PAFPN" (awal section head/neck), regex-swap CUMA
# di satu bagian. Params sudah diverifikasi offline: BackboneOnly=5.952M(-0.58%),
# NeckOnly=5.966M(-0.36%) vs B2 -- keduanya dlm toleransi +-5% TANPA ssm_ratio baru.
base_yaml = open(f"{REPO}/ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml").read()
split_marker = "# Mamba-YOLO PAFPN"
idx = base_yaml.find(split_marker)
assert idx != -1, "penanda split YAML tidak ketemu -- struktur Mamba-YOLO-T.yaml berubah?"
backbone_part, head_part = base_yaml[:idx], base_yaml[idx:]
assert "VSSBlock" in backbone_part and "XSSBlock" not in backbone_part
assert "XSSBlock" in head_part and "VSSBlock" not in head_part

backbone_deform = re.sub(r"\bVSSBlock\b", "DeformVSSBlock", backbone_part)
open(f"{REPO}/ultralytics/cfg/models/mamba-yolo/DeMamba-YOLO-T-backbone-only.yaml", "w").write(
    backbone_deform + head_part)

head_deform = re.sub(r"\bXSSBlock\b", "DeformXSSBlock", head_part)
open(f"{REPO}/ultralytics/cfg/models/mamba-yolo/DeMamba-YOLO-T-neck-only.yaml", "w").write(
    backbone_part + head_deform)
print("  OK 2 YAML Ablasi B (backbone-only, neck-only) berhasil digenerate")

# Ablasi B-lanjut: 3 varian stage-wise di backbone (literatur-deformable-lintas-paradigma.md
# SS6.3). Neck TIDAK disentuh (tetap XSSBlock biasa di semua 3 varian ini) -- isolasi bersih
# dari pertanyaan Ablasi B sebelumnya. Tiap baris backbone punya komentar posisi unik
# ("# 1","# 3","# 5","# 7") dipakai utk menarget baris tepat, assert-guarded (fail loud
# kalau match != 1x, mis. krn upstream ubah format YAML).
def _swap_positions(text, positions):
    lines = text.split("\n")
    hit = {p: 0 for p in positions}
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        for p in positions:
            if stripped.endswith(f"# {p}") and "VSSBlock" in line and "Deform" not in line:
                lines[i] = line.replace("VSSBlock", "DeformVSSBlock")
                hit[p] += 1
    for p in positions:
        assert hit[p] == 1, f"posisi {p}: ditemukan {hit[p]}x (harus tepat 1x) -- format YAML berubah?"
    return "\n".join(lines)

stagewise_variants = {"bb-deep": [7], "bb-late": [5, 7], "bb-early": [1, 3]}
for vname, positions in stagewise_variants.items():
    result = _swap_positions(base_yaml, positions)
    open(f"{REPO}/ultralytics/cfg/models/mamba-yolo/DeMamba-YOLO-T-{vname}.yaml", "w").write(result)
print(f"  OK {len(stagewise_variants)} YAML Ablasi B-lanjut (bb-deep, bb-late, bb-early) berhasil digenerate")

# BB-Late-Alt (BERSYARAT, backlog Section 2d) -- sumbu kerapatan intra-stage. Posisi 5
# dipecah jadi 3 baris terpisah (D-S-D) krn parse_model bikin n INSTANCE TERPISAH dari
# kelas yg sama per baris (bukan loop internal) -- lihat tasks.py baris ~946. Memecah 1
# baris jadi 3 menggeser SEMUA indeks "from" eksplisit di bawahnya +2 -- WAJIB diperbaiki,
# assert-guarded (kalau salah, parse_model gagal keras karena channel Concat tak cocok,
# bukan diam-diam salah). Sudah diverifikasi offline: save-layers hasil parse_model =
# [3,7,10,13,16,19,22], cocok persis perhitungan manual.
_lines = base_yaml.split("\n")
_t5 = None
for _i, _l in enumerate(_lines):
    if _l.rstrip().endswith("# 5") and "VSSBlock" in _l and "Deform" not in _l:
        assert _t5 is None, "posisi 5: >1 baris cocok"
        _t5 = _i
assert _t5 is not None, "posisi 5 tidak ditemukan"
_dsd = [
    "  - [-1, 1, DeformVSSBlock, [512]]      # 5a (D)",
    "  - [-1, 1, VSSBlock, [512]]             # 5b (S)",
    "  - [-1, 1, DeformVSSBlock, [512]]      # 5c (D)",
]
_lines = _lines[:_t5] + _dsd + _lines[_t5+1:]
_hit7 = 0
for _i, _l in enumerate(_lines):
    if _l.rstrip().endswith("# 7") and "VSSBlock" in _l and "Deform" not in _l:
        _lines[_i] = _l.replace("VSSBlock", "DeformVSSBlock")
        _hit7 += 1
assert _hit7 == 1, f"posisi 7: {_hit7}x"
_text = "\n".join(_lines)
_reindex = [
    ("  - [[-1, 5], 1, Concat, [1]]  # cat backbone P4",
     "  - [[-1, 7], 1, Concat, [1]]  # cat backbone P4 (geser +2 krn split posisi 5)"),
    ("  - [[-1, 11], 1, Concat, [1]]  # cat head P4",
     "  - [[-1, 13], 1, Concat, [1]]  # cat head P4 (geser +2 krn split posisi 5)"),
    ("  - [[-1, 8], 1, Concat, [1]]  # cat head P5",
     "  - [[-1, 10], 1, Concat, [1]]  # cat head P5 (geser +2 krn split posisi 5)"),
    ("  - [[14, 17, 20], 1, Detect, [nc]]  # Detect(P3, P4, P5)",
     "  - [[16, 19, 22], 1, Detect, [nc]]  # Detect(P3, P4, P5) (geser +2 krn split posisi 5)"),
]
for _old, _new in _reindex:
    assert _text.count(_old) == 1, f"GAGAL cari saat reindex: {_old!r}"
    _text = _text.replace(_old, _new)
open(f"{REPO}/ultralytics/cfg/models/mamba-yolo/DeMamba-YOLO-T-bb-late-alt.yaml", "w").write(_text)
print("  OK 1 YAML BB-Late-Alt (D-S-D, bersyarat) berhasil digenerate, indeks head diperbaiki")

# ---- Patch 6.1: np.trapz DIHAPUS mulai numpy 2.4.0; Ultralytics 8.2.29 memakainya utk menghitung AP ----
# Tanpa patch ini, bila Colab sudah memakai numpy >= 2.4, training CRASH di validasi epoch pertama.
# np.trapezoid = fungsi yang SAMA (nama baru sejak numpy 2.0; np.trapz di 2.0-2.3 hanya memanggil np.trapezoid)
# -> angka mAP IDENTIK dgn run sebelumnya. Fallback np.trapz hanya utk numpy 1.x.
patch("ultralytics/utils/metrics.py", [
    ("ap = np.trapz(np.interp(x, mrec, mpre), x)",
     "ap = (np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)(np.interp(x, mrec, mpre), x)", "6.1"),
])
import numpy as _np
print(f"  numpy {_np.__version__}: np.trapz {'ada' if hasattr(_np, 'trapz') else 'TIDAK ADA (patch 6.1 mencegah crash)'}")

# P1-NoProj: posisi deformable SAMA dgn P1 (1,3,5,7,11,14,17,20), kelas *NP (ssm_ratio 2.0, tanpa proj)
_y = open(f"{REPO}/ultralytics/cfg/models/mamba-yolo/DeMamba-YOLO-T.yaml").read()
assert _y.count("DeformVSSBlock") == 4 and _y.count("DeformXSSBlock") == 4, "YAML P1 tak terduga"
_y = re.sub(r"\bDeformVSSBlock\b", "DeformVSSBlockNP", _y)
_y = re.sub(r"\bDeformXSSBlock\b", "DeformXSSBlockNP", _y)
open(f"{REPO}/ultralytics/cfg/models/mamba-yolo/DeMamba-YOLO-T-noproj.yaml", "w").write(_y)
print("  OK YAML P1-NoProj (DeMamba-YOLO-T-noproj.yaml) digenerate")

# data YAML steel-pipe (optional; only when --steelpipe-root is given). Class names = names in the original release.
if STEEL_BASE:
    import yaml as _yaml
    STEEL_YAML = f"{REPO}/ultralytics/cfg/datasets/steelpipe.yaml"
    with open(STEEL_YAML, "w") as _f:
        _yaml.safe_dump({"path": STEEL_BASE, "train": "images/train2021", "val": "images/val2021",
                         "nc": 8, "names": STEEL_NAMES}, _f, sort_keys=False, allow_unicode=True)
    print(f"  OK steelpipe.yaml written: {STEEL_YAML}")

for f in ["ultralytics/nn/modules/mamba_yolo.py", "ultralytics/nn/modules/deform_ss2d.py",
          "ultralytics/nn/tasks.py", "ultralytics/utils/torch_utils.py", "ultralytics/utils/metrics.py"]:
    py_compile.compile(f"{REPO}/{f}", doraise=True)
print("  OK sintaks semua file valid -- SEMUA PATCH + YAML OK")

print("Done.")
