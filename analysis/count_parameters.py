"""count_parameters.py -- parameter count of every Mamba-YOLO-T configuration in the paper (nc = 8), obtained by BUILDING
each model from its YAML in a clone patched with code/apply_patches.py. Runs on CPU: the selective-scan CUDA kernel is
replaced by a zero function, which only serves the forward pass Ultralytics uses to determine the strides.

    python code/apply_patches.py Mamba-YOLO
    python analysis/count_parameters.py Mamba-YOLO params.json

Expected (Tables 3 and 6): B2 5,986,912; P1 5,930,869; P1-NoProj 6,329,230; BackboneOnly 5,952,204; NeckOnly 5,965,577;
BB-Late 5,964,057; BB-Early 5,975,059; BB-Late-Alt 5,972,944; BB-Deep 5,990,718; K5 (Section V-E) 6,215,245."""
import sys, types, json
from functools import partial
REPO, OUT = sys.argv[1], sys.argv[2]; sys.path.insert(0, REPO)
_fk = types.ModuleType('selective_scan_cuda_core'); _fk.fwd = None; sys.modules['selective_scan_cuda_core'] = _fk
import ultralytics.nn.modules.common_utils_mbyolo as cu
cu.SelectiveScanCore.apply = staticmethod(lambda u, *a, **k: u * 0)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import deform_ss2d as dm
def cnt(y):
    m = DetectionModel(f"{REPO}/ultralytics/cfg/models/mamba-yolo/{y}", ch=3, nc=8, verbose=False)
    return sum(p.numel() for p in m.parameters()), sum(1 for x in m.modules() if isinstance(x, dm.DeformSS2D))
R = {k: cnt(y) for k, y in {"B2": "Mamba-YOLO-T.yaml", "P1": "DeMamba-YOLO-T.yaml", "P1-NoProj": "DeMamba-YOLO-T-noproj.yaml",
     "BackboneOnly": "DeMamba-YOLO-T-backbone-only.yaml", "NeckOnly": "DeMamba-YOLO-T-neck-only.yaml",
     "BB-Late": "DeMamba-YOLO-T-bb-late.yaml", "BB-Early": "DeMamba-YOLO-T-bb-early.yaml",
     "BB-Late-Alt": "DeMamba-YOLO-T-bb-late-alt.yaml", "BB-Deep": "DeMamba-YOLO-T-bb-deep.yaml"}.items()}
for c in (dm.DeformVSSBlockNP, dm.DeformXSSBlockNP):   # P1 without capacity matching (ssm_ratio 2.0, projection kept)
    c.SSM_CLASS = partial(dm.DeformSS2D, deform_mode="k3_replace", deform_kernel_size=5, deform_use_proj=True)
R["P1-uncapped"] = cnt("DeMamba-YOLO-T-noproj.yaml")
# Section V-E: additive K=5 configuration (4 fixed SS2D paths + 1 deformable), ssm_ratio 1.7, no projection
_k5 = partial(dm.DeformSS2D, deform_mode="k5_augment", deform_kernel_size=5, deform_use_proj=False)
for c in (dm.DeformVSSBlockNP, dm.DeformXSSBlockNP):
    c.SSM_CLASS = _k5
    _orig = c.__init__
    def _init(self, *a, ssm_ratio=2.0, __o=_orig, **k): __o(self, *a, ssm_ratio=1.7, **k)
    c.__init__ = _init
_m = DetectionModel(f"{REPO}/ultralytics/cfg/models/mamba-yolo/DeMamba-YOLO-T-noproj.yaml", ch=3, nc=8, verbose=False)
assert [x.K for x in _m.modules() if isinstance(x, dm.DeformSS2D)] == [5] * 10
R["K5"] = (sum(p.numel() for p in _m.parameters()), 10)
OFF = {"ConvOffset": sum(p.numel() for x in _m.modules() if isinstance(x, dm.ConvOffset) for p in x.parameters()),
       "DeformableScan": sum(p.numel() for x in _m.modules() if isinstance(x, dm.DeformableScan) for p in x.parameters())}
json.dump({**{k: {"params": v[0], "n_deform": v[1]} for k, v in R.items()}, "K5_offset_net": OFF}, open(OUT, "w"), indent=1)
for k, (n, d) in R.items(): print(f"{k:<13}{n:>11,}  {n/R['B2'][0]-1:+.2%}  deform={d}")
print("K5 offset network:", {k: f"{v:,}" for k, v in OFF.items()}, f"-> K5 without offset network {R['K5'][0]-OFF['DeformableScan']:,}")
