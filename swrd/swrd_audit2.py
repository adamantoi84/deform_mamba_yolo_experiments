import json, re, sys, csv, hashlib
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent            # folder berisi Raw_data dan crop_weld_data
rawI, rawJ = ROOT/"Raw_data"/"images", ROOT/"Raw_data"/"json"
cj, ci = ROOT/"crop_weld_data"/"crop_weld_jsons", ROOT/"crop_weld_data"/"crop_weld_images"
pre = lambda s: re.sub(r"-?\d+$", "", s)
src = lambda s: "DJ-RT" if s.startswith("DJ-RT") else ("RRT" if s.startswith("RRT") else "lain:" + pre(s))
def info(p):
    try:
        im = Image.open(p); return im.mode, im.size[0], im.size[1]
    except Exception as e: return "ERR", 0, 0
def wcsv(name, rows):
    with open(ROOT/name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)

# ---------- 1. inventaris citra mentah ----------
jsons = {p.stem: p for p in rawJ.glob("*.json")}
raw = []
for p in sorted(p for p in rawI.iterdir() if p.is_file()):
    m, W, H = info(p); raw.append(dict(stem=p.stem, ext=p.suffix, bytes=p.stat().st_size, mode=m, W=W, H=H, src=src(p.stem), batch=pre(p.stem), has_json=p.stem in jsons))
wcsv("swrd_audit2_raw_inventory.csv", raw)
print("[1] citra mentah:", len(raw), "| ekstensi:", dict(Counter(r["ext"] for r in raw)))
print("[1] mode per sumber:", {k: dict(v) for k, v in {s: Counter(r["mode"] for r in raw if r["src"].split(":")[0] == s) for s in ("DJ-RT", "RRT", "lain")}.items()})
print("[1] tanpa JSON:", [(r["stem"], r["ext"], r["bytes"], r["mode"], r["W"], r["H"]) for r in raw if not r["has_json"]])
print("[1] sumber non-DJ-RT:", Counter(r["src"] for r in raw if r["src"] != "DJ-RT"))

# ---------- 2. duplikat: ukuran -> md5 4MB -> md5 penuh -> samakah JSON-nya ----------
def md5(p, n=None):
    h = hashlib.md5()
    with open(p, "rb") as f:
        h.update(f.read(n) if n else f.read()) if n else [h.update(c) for c in iter(lambda: f.read(8 << 20), b"")]
    return h.hexdigest()
bysize = defaultdict(list)
for r in raw: bysize[r["bytes"]].append(r["stem"])
paths = {p.stem: p for p in rawI.iterdir() if p.is_file()}
byfp = defaultdict(list)
for s, L in bysize.items():
    if len(L) > 1:
        for x in L: byfp[(s, md5(paths[x], 4_000_000))].append(x)
cand = [v for v in byfp.values() if len(v) > 1]
def shapes(s):
    if s not in jsons: return None
    d = json.load(open(jsons[s], encoding="utf-8")); return sorted(json.dumps([x["label"], x["points"]], sort_keys=True) for x in d["shapes"])
rows, full_groups = [], []
for g, v in enumerate(cand):
    byfull = defaultdict(list)
    for x in v: byfull[md5(paths[x])].append(x)
    for h, m in byfull.items():
        if len(m) > 1:
            full_groups.append(m); eq = len({json.dumps(shapes(x)) for x in m}) == 1
            for x in m: rows.append(dict(group=len(full_groups), stem=x, batch=pre(x), md5=h, json_identik=eq))
if rows: wcsv("swrd_audit2_duplikat.csv", rows)
print(f"[2] kelompok kandidat (ukuran+md5 4MB): {len(cand)} | identik penuh (md5 seluruh berkas): {len(full_groups)} | berkas di dalamnya: {sum(map(len, full_groups))}")
print("[2] JSON identik pada", sum(1 for g in full_groups if len({json.dumps(shapes(x)) for x in g}) == 1), "dari", len(full_groups), "kelompok")
print("[2] pasangan batch yang terlibat:", Counter(tuple(sorted({pre(x) for x in g})) for g in full_groups).most_common(12))
print("[2] kelompok dengan >2 anggota:", [g for g in full_groups if len(g) > 2][:5])

# ---------- 3. inventaris crop + pemetaan ke citra mentah + cek geometri 焊缝 ----------
rawset = set(paths) | set(jsons)
def raw_key(s, d):
    ip = Path(d.get("imagePath", "")).stem
    return next((c for c in (s, re.sub(r"^[AB]_", "", s), ip, re.sub(r"^[AB]_", "", ip)) if c in rawset), None)
def weld_boxes(s):
    d = json.load(open(jsons[s], encoding="utf-8")); out = []
    for x in d["shapes"]:
        if x["label"] == "焊缝":
            a = np.array(x["points"]); out.append((a[:, 0].max() - a[:, 0].min(), a[:, 1].max() - a[:, 1].min()))
    return out
crops, geo, miss_raw = [], [], {}
for sub in sorted(p for p in cj.glob("*/*") if p.is_dir()):
    tag = f"{sub.parent.name}/{sub.name}"; imd = ci/sub.parent.name/sub.name
    imgs = {p.stem: p for p in imd.iterdir() if p.is_file()} if imd.exists() else {}
    for jf in sorted(sub.glob("*.json")):
        d = json.load(open(jf, encoding="utf-8")); r = raw_key(jf.stem, d)
        m, W, H, b = ("-", 0, 0, 0)
        if jf.stem in imgs: m, W, H = info(imgs[jf.stem]); b = imgs[jf.stem].stat().st_size
        else: miss_raw.setdefault(tag, []).append(r)
        crops.append(dict(folder=tag, stem=jf.stem, raw=r, src=src(r or ""), has_image=jf.stem in imgs, mode=m, W=W, H=H, bytes=b, jsonW=d.get("imageWidth"), jsonH=d.get("imageHeight")))
        if r in jsons and d.get("imageWidth"):
            wb = weld_boxes(r)
            if wb:
                best = min(wb, key=lambda t: abs(t[0]-d["imageWidth"]) + abs(t[1]-d["imageHeight"]))
                geo.append((abs(best[0]-d["imageWidth"]), abs(best[1]-d["imageHeight"])))
wcsv("swrd_audit2_crop_inventory.csv", crops)
print("[3] crop per folder, mode, sumber:")
for k, v in sorted(Counter((c["folder"], c["mode"], c["src"].split(":")[0]) for c in crops).items()): print("   ", k, v)
print("[3] ukuran berkas crop (GB) per folder:", {f: round(sum(c["bytes"] for c in crops if c["folder"] == f) / 1e9, 1) for f in sorted({c["folder"] for c in crops})})
used = {c["raw"] for c in crops}
print("[3] citra mentah tanpa crop:", sorted(set(rawset) - used))
t2 = miss_raw.get("T/2", []); print("[3] T/2 tanpa citra:", len(t2), "| film mentah unik:", len(set(t2)), "| sumber:", Counter(src(x or '') for x in t2))
if geo:
    g = np.array(geo); print(f"[3] cek geometri 焊缝 vs crop (selisih piksel W,H): n={len(g)} median={np.median(g,0)} p95={np.percentile(g,95,axis=0)} cocok<=3px: {(g.max(1)<=3).mean():.3f}")
