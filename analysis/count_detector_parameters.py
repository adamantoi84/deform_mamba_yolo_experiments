#!/usr/bin/env python3
"""count_detector_parameters.py -- "Parameters" column of Table 6 for the five comparison detectors.

Builds each detector exactly as the Ultralytics 8.2.29 trainer builds it when training FROM SCRATCH from its .yaml
(notebook fase-8): DetectionModel(cfg="<key>.yaml", nc=8). Parameter count = sum(p.numel()) of the unfused model, the
same way as the Mamba-YOLO-T numbers (B2 5,986,912; DeformSS2D 5,930,869).
Check: with nc=80, YOLOv8n and YOLOv8s must give the numbers documented by Ultralytics (3,157,200 and 11,166,560).

    pip install ultralytics==8.2.29
    python analysis/count_detector_parameters.py
"""
import argparse, re
from pathlib import Path

import pandas as pd
import ultralytics
from ultralytics.nn.tasks import DetectionModel, yaml_model_load

ap = argparse.ArgumentParser()
ap.add_argument("--baselines", default=str(Path(__file__).resolve().parents[1] / "results" / "per_class" / "per_class_results_BASELINES.csv"))
a = ap.parse_args()

assert ultralytics.__version__ == "8.2.29", ultralytics.__version__

# model key from the run folder name (.../output/<key>_seed1_150ep_.../weights/best.pt)
B = pd.read_csv(a.baselines); B = B[B["class"] == "ALL"]
KEYS = {r.model: re.search(r"/output/(.+?)_seed\d", r.ckpt_path).group(1) for r in B.itertuples()}
assert KEYS == {"YOLOv3-tiny": "yolov3-tiny", "YOLOv5s": "yolov5s", "YOLOv6n": "yolov6n",
                "YOLOv8n": "yolov8n", "YOLOv8s": "yolov8s"}, KEYS


def n_params(cfg, nc):
    m = DetectionModel(cfg, nc=nc, verbose=False)
    return sum(p.numel() for p in m.parameters()), m


# check against the numbers documented by Ultralytics (nc = 80)
DOK = {"yolov8n.yaml": 3_157_200, "yolov8s.yaml": 11_166_560}
for cfg, ref in DOK.items():
    got, _ = n_params(cfg, 80)
    print(f"check nc=80 {cfg:<14} {got:>12,}  documented {ref:>12,}  {'OK' if got == ref else 'FAIL'}")
    assert got == ref

print(f"\n{'Model':<12}{'cfg':<16}{'YAML used':<22}{'scale':<7}{'head':<10}{'nc=8':>12}{'nc=80':>13}")
rows = []
for name, key in KEYS.items():
    cfg = f"{key}.yaml"
    d = yaml_model_load(cfg)
    p8, m = n_params(cfg, 8)
    p80, _ = n_params(cfg, 80)
    head = type(m.model[-1]).__name__
    anchors = "anchors" in d
    rows.append((name, cfg, Path(d.get("yaml_file", cfg)).name, d.get("scale"), head, anchors, p8, p80))
    src = Path(ultralytics.__file__).parent / "cfg" / "models"
    used = [p for p in src.rglob("*.yaml") if p.name == re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", cfg)]
    print(f"{name:<12}{cfg:<16}{(used[0].relative_to(src).as_posix() if used else '?'):<22}{str(d.get('scale')):<7}"
          f"{head:<10}{p8:>12,}{p80:>13,}   anchor-based={anchors}")

print("\nMamba-YOLO-T (Table 6, nc=8): B2 5,986,912; DeformSS2D 5,930,869")
for name, *_, p8, _ in rows:
    print(f"  {name:<12} {p8/5_986_912:5.2f}× B2")

EXPECTED = {"YOLOv3-tiny": 12_136_240, "YOLOv5s": 9_125_288, "YOLOv6n": 4_238_936, "YOLOv8n": 3_012_408, "YOLOv8s": 11_138_696}
assert {r[0]: r[6] for r in rows} == EXPECTED, "parameter counts differ from Table 6"
print("\nTable 6: all five parameter counts match")
