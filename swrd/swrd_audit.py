import json, re, sys, csv, hashlib
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(sys.argv[1])            # folder SWXD_Data
OUT  = Path(sys.argv[2] if len(sys.argv) > 2 else "swrd_audit_raw.csv")
raw_j = ROOT/"Raw_data"/"json"; raw_i = ROOT/"Raw_data"/"images"
pre = lambda s: re.sub(r"-?\d+$", "", s)          # kunci batch: sama dengan skrip prefix sebelumnya

# ---------- A. citra mentah: per-berkas, per-label, per-batch ----------
rows, labels = [], Counter()
cls_obj, cls_film, cls_batch = Counter(), defaultdict(set), defaultdict(set)
for jf in sorted(raw_j.glob("*.json")):
    d = json.load(open(jf, encoding="utf-8")); c = Counter(s["label"] for s in d["shapes"])
    for k, v in c.items():
        labels[k] += v; cls_obj[k] += v; cls_film[k].add(jf.stem); cls_batch[k].add(pre(jf.stem))
    rows.append(dict(stem=jf.stem, batch=pre(jf.stem), W=d.get("imageWidth"), H=d.get("imageHeight"),
                     n_obj=len(d["shapes"]), **{f"n[{k}]": v for k, v in c.items()}))
cols = sorted({k for r in rows for k in r}, key=lambda x: (not x in ("stem","batch","W","H","n_obj"), x))
with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, cols, restval=0); w.writeheader(); w.writerows(rows)
print(f"[A] json mentah: {len(rows)}  batch unik: {len({r['batch'] for r in rows})}")
print("[A] label | objek | film | batch")
for k, v in labels.most_common():
    print(f"   {k!r:14} {v:7d} {len(cls_film[k]):6d} {len(cls_batch[k]):5d}")
bs = Counter(r["batch"] for r in rows)
print("[A] batch terbesar (share):", [(b, n, round(n/len(rows), 3)) for b, n in bs.most_common(5)])

# ---------- B. pasangan citra-json mentah, duplikat kasar ----------
ims = {p.stem: p for p in raw_i.iterdir() if p.is_file()}
print(f"[B] citra mentah: {len(ims)}; json tanpa citra: {len({r['stem'] for r in rows}-set(ims))}; citra tanpa json: {len(set(ims)-{r['stem'] for r in rows})}")
fp = defaultdict(list)
for s, p in ims.items():
    with open(p, "rb") as f: fp[(p.stat().st_size, hashlib.md5(f.read(4_000_000)).hexdigest())].append(s)
dup = [v for v in fp.values() if len(v) > 1]
print(f"[B] kelompok citra mentah dengan ukuran+md5(4MB awal) sama: {len(dup)}; contoh: {dup[:3]}")
try:
    from PIL import Image
    p = next(iter(ims.values())); im = Image.open(p); print("[B] contoh mentah:", p.name, im.mode, im.size)
except Exception as e: print("[B] PIL gagal:", e)

# ---------- C. crop: kelas, pasangan, pemetaan ke citra mentah ----------
rawset = set(ims) | {r["stem"] for r in rows}
cj, ci = ROOT/"crop_weld_data"/"crop_weld_jsons", ROOT/"crop_weld_data"/"crop_weld_images"
per_raw = defaultdict(list)
for sub in sorted(p for p in cj.glob("*/*") if p.is_dir()):
    js = {p.stem for p in sub.glob("*.json")}
    imd = ci/sub.parent.name/sub.name
    ig = {p.stem: p for p in imd.iterdir()} if imd.exists() else {}
    mapped = 0
    for s in js:
        d = json.load(open(sub/f"{s}.json", encoding="utf-8"))
        cand = [s, re.sub(r"^[AB]_", "", s), Path(d.get("imagePath","")).stem, re.sub(r"^[AB]_", "", Path(d.get("imagePath","")).stem)]
        hit = next((c for c in cand if c in rawset), None)
        if hit: mapped += 1; per_raw[hit].append(f"{sub.parent.name}/{sub.name}")
    print(f"[C] {sub.parent.name}/{sub.name}: json {len(js)}  citra {len(ig)}  json_tanpa_citra {len(js-set(ig))}  citra_tanpa_json {len(set(ig)-js)}  terpetakan_ke_mentah {mapped}")
    if ig:
        try:
            p = next(iter(ig.values())); im = Image.open(p); print(f"    contoh {p.name}: {im.mode} {im.size}")
        except Exception as e: print("    PIL gagal:", e)
dist = Counter(len(v) for v in per_raw.values())
print("[C] jumlah crop per citra mentah:", dict(sorted(dist.items())), "| citra mentah punya crop:", len(per_raw), "dari", len(rawset))
