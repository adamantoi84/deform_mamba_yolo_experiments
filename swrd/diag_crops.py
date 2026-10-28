"""diag_crops.py -- diagnosis ketidakcocokan label (verify_crops: 96.7% < 99.5%). Taruh di folder yang sama dengan verify_crops.py. python diag_crops.py
Menjawab: apakah label resmi yang 'tidak cocok' (1) objek yang terpotong tepi crop (aturan potong/tidak potong berbeda), (2) label yang diedit ulang
(koordinat bergeser/ukuran berubah di crop-JSON), atau (3) crop yang titik asalnya berbeda dari perkiraan?  Keluaran: diag_crops_summary.txt, diag_crops_unmatched.csv"""
import json, csv, sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
RAW_I, RAW_J = ROOT/"Raw_data"/"images", ROOT/"Raw_data"/"json"
CJ, CI = ROOT/"crop_weld_data"/"crop_weld_jsons", ROOT/"crop_weld_data"/"crop_weld_images"
WELD = "焊缝"; LINES = []
def say(s=""): print(s); LINES.append(s)
def bbox(sh):
    p = np.array(sh["points"], float)
    if sh.get("shape_type") == "circle" and len(p) == 2:
        r = float(np.hypot(*(p[1] - p[0]))); return (p[0][0] - r, p[0][1] - r, p[0][0] + r, p[0][1] + r)
    return (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
def split(d):
    w, o = [], []
    for sh in d.get("shapes", []): (w if sh.get("label") == WELD else o).append((sh.get("label"), bbox(sh), sh.get("shape_type"), len(sh["points"])))
    return w, o
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - ix*iy
    return ix*iy/u if u > 0 else 0
kept = {r["stem"] for r in csv.DictReader(open(ROOT/"swrd_split_v1.csv", encoding="utf-8-sig"))}
inv = defaultdict(list)
for r in csv.DictReader(open(ROOT/"swrd_audit2_crop_inventory.csv", encoding="utf-8-sig")):
    if r["raw"] in kept and r["jsonW"]: inv[r["raw"]].append(r)
nolabel = []; rows_out = []; C = Counter(); beyond = Counter(); devhist = Counter(); crops_all_bad = []; weird = []; shape_types = Counter(); npts_off = Counter(); npts_our = Counter()
cat = Counter(); catlab = defaultdict(Counter); iou_bins = Counter(); sizechg = []
for stem, rows in sorted(inv.items()):
    d = json.load(open(RAW_J/f"{stem}.json", encoding="utf-8")); W, H = int(d["imageWidth"]), int(d["imageHeight"]); welds, objs = split(d)
    ours = [(max(0, int(round(b[0]))), max(0, int(round(b[1]))), min(W, int(round(b[2]))), min(H, int(round(b[3])))) for _, b, _, _ in welds]
    used = set()
    for r in rows:
        Wc, Hc = int(r["jsonW"]), int(r["jsonH"])
        cand = [(abs(Wc-(o[2]-o[0]))+abs(Hc-(o[3]-o[1])), j) for j, o in enumerate(ours) if j not in used and abs(Wc-(o[2]-o[0])) <= 3 and abs(Hc-(o[3]-o[1])) <= 3]
        if not cand: continue
        _, j = min(cand); used.add(j); ox, oy = ours[j][0], ours[j][1]
        cd = json.load(open(CJ/r["folder"]/f"{r['stem']}.json", encoding="utf-8")); cw, co = split(cd)
        if not any(b[0] > 3 and b[1] > 3 and b[2] < Wc-3 and b[3] < Hc-3 for _, b, _, _ in co): nolabel.append((r['stem'], r['folder']))
        for _, b, _, _ in cw:
            if not (abs(b[0]) <= 1 and abs(b[1]) <= 1 and abs(b[2]-Wc) <= 2 and abs(b[3]-Hc) <= 2): weird.append((r["stem"], r["folder"], Wc, Hc, [round(v) for v in b], (ox, oy)))
        for l, b, st, npt in co:
            shape_types[st] += 1; npts_off[npt] += 1
            if b[0] < -0.5 or b[1] < -0.5 or b[2] > Wc + 0.5 or b[3] > Hc + 0.5: beyond["resmi: koordinat di luar crop"] += 1
        for l, b, st, npt in objs: npts_our[(st, npt)] += 0
        # jodohkan: label sama, kotak kita (tanpa potong) vs resmi; pakai IoU terbaik, serakah
        pairs = []
        for a, (l, b, st, npt) in enumerate(co):
            for k, (l2, c, st2, npt2) in enumerate(objs):
                if l != l2: continue
                cb = (c[0]-ox, c[1]-oy, c[2]-ox, c[3]-oy); v = iou(b, cb)
                if v > 0.05: pairs.append((v, a, k, cb))
        pairs.sort(reverse=True); ua, uk, pm = set(), set(), {}
        for v, a, k, cb in pairs:
            if a in ua or k in uk: continue
            ua.add(a); uk.add(k); pm[a] = (k, cb, v)
        nbad = 0
        for a, (l, b, st, npt) in enumerate(co):
            edge = b[0] <= 1.5 or b[1] <= 1.5 or b[2] >= Wc-1.5 or b[3] >= Hc-1.5
            if a in pm:
                k, cb, v = pm[a]; cc = (max(cb[0], 0), max(cb[1], 0), min(cb[2], Wc), min(cb[3], Hc)); dev_clip = max(abs(b[i]-cc[i]) for i in range(4)); dev_raw = max(abs(b[i]-cb[i]) for i in range(4))
                if dev_clip <= 1.5: cat["cocok (dengan potong tepi)"] += 1; continue
                if dev_raw <= 1.5: cat["cocok hanya TANPA potong (resmi tidak memotong)"] += 1; catlab["tanpa potong"][l] += 1; continue
                nbad += 1; key = "TEPI crop, deviasi >1.5 px" if edge else "INTERIOR, deviasi >1.5 px"; cat[key] += 1; catlab[key][l] += 1
                iou_bins["IoU>=0.9" if v >= .9 else "0.7-0.9" if v >= .7 else "0.5-0.7" if v >= .5 else "<0.5"] += 1
                dv = tuple(round(b[i]-cb[i]) for i in range(4)); devhist[dv] += 1
                sizechg.append(((b[2]-b[0])-(cb[2]-cb[0]), (b[3]-b[1])-(cb[3]-cb[1])))
                rows_out.append(dict(film=stem, crop=r["stem"], label=l, kategori=key, off_box=[round(x, 1) for x in b], our_box=[round(x, 1) for x in cb], iou=round(v, 2)))
            else:
                nbad += 1; cat["resmi tanpa pasangan IoU>0.05 (TEPI)" if edge else "resmi tanpa pasangan IoU>0.05 (INTERIOR)"] += 1; catlab["resmi tanpa pasangan"][l] += 1
                rows_out.append(dict(film=stem, crop=r["stem"], label=l, kategori="resmi tanpa pasangan", off_box=[round(x, 1) for x in b], our_box="", iou=0))
        if nbad and nbad >= 0.8 * max(len(co), 1) and len(co) >= 3: crops_all_bad.append((r["stem"], r["folder"], len(co), nbad))
        # objek kita yang beririsan tetapi tak dijodohkan resmi
        for k, (l2, c, st2, npt2) in enumerate(objs):
            if k in uk: continue
            cb = (c[0]-ox, c[1]-oy, c[2]-ox, c[3]-oy)
            if min(cb[2], Wc) > max(cb[0], 0) and min(cb[3], Hc) > max(cb[1], 0):
                ix = (min(cb[2], Wc)-max(cb[0], 0))*(min(cb[3], Hc)-max(cb[1], 0)); vis = ix/max((cb[2]-cb[0])*(cb[3]-cb[1]), 1e-9)
                cat["kita tanpa pasangan resmi, fraksi dalam crop " + ("<0.5" if vis < .5 else ">=0.5")] += 1; catlab["kita tanpa pasangan"][l2] += 1
say(f"bentuk label resmi (shape_type): {dict(shape_types)}; jumlah titik per bentuk: {dict(npts_off.most_common(6))}")
say(f"label resmi dengan koordinat di luar crop (tidak dipotong): {beyond['resmi: koordinat di luar crop']}")
say("\nkategori label (IoU-matching, tol 1.5 px):")
for k, v in cat.most_common(): say(f"   {k}: {v}   per kelas: {dict(catlab[k].most_common(5))}" if k in catlab else f"   {k}: {v}")
say(f"\nlabel resmi yang berpasangan tetapi deviasi >1.5 px: IoU {dict(iou_bins)}; perubahan ukuran median (dW,dH) = {np.median(np.array(sizechg),0) if sizechg else None}")
say(f"deviasi (resmi - kita) (dx0,dy0,dx1,dy1) terbanyak: {dict(devhist.most_common(10))}")
say(f"crop dengan >=80% label tak cocok (dan >=3 label): {len(crops_all_bad)} -> {crops_all_bad[:12]}")
say(f"crop resmi yang poligon 焊缝-nya bukan pas selebar crop: {len(weird)} -> {weird[:10]}")
with open(ROOT/"diag_crops_unmatched.csv", "w", newline="", encoding="utf-8-sig") as f:
    if rows_out: w = csv.DictWriter(f, rows_out[0].keys()); w.writeheader(); w.writerows(rows_out)
# pixel check untuk crop yang hampir semuanya tak cocok + crop 'aneh'
say("\ncek piksel pada crop bermasalah (asal kita, geser +-3, korelasi patch tengah 400x400):")
def best(raw, crop, ox, oy, R=3):
    h, w = crop.shape; ph, pw = min(h, 400), min(w, 400); y0, x0 = h//2-ph//2, w//2-pw//2
    c = crop[y0:y0+ph, x0:x0+pw].astype(np.float64); c -= c.mean(); cn = np.sqrt((c*c).sum()); b = (-2, 0, 0)
    for dy in range(-R, R+1):
        for dx in range(-R, R+1):
            yy, xx = oy+dy+y0, ox+dx+x0
            if yy < 0 or xx < 0 or yy+ph > raw.shape[0] or xx+pw > raw.shape[1]: continue
            rr = raw[yy:yy+ph, xx:xx+pw].astype(np.float64); rr -= rr.mean(); den = cn*np.sqrt((rr*rr).sum()); v = (c*rr).sum()/den if den > 0 else 0
            if v > b[0]: b = (v, dx, dy)
    return b
rg = np.random.default_rng(1); nl = [nolabel[i] for i in rg.permutation(len(nolabel))[:30]]
say(f'crop tanpa label bagian-dalam (tak teruji lewat label): {len(nolabel)}; 30 disampel untuk cek piksel')
chk = [(s, fo) for s, fo, *_ in crops_all_bad[:15]] + [(w[0], w[1]) for w in weird[:8]] + nl
for cs, fo in chk:
    row = next((x for x in sum(inv.values(), []) if x["stem"] == cs and x["folder"] == fo), None)
    if not row or row["has_image"] not in ("True", "1"): say(f"   {cs}: tanpa citra crop"); continue
    stem = row["raw"]; d = json.load(open(RAW_J/f"{stem}.json", encoding="utf-8")); W, H = int(d["imageWidth"]), int(d["imageHeight"]); welds, _ = split(d)
    crop = np.asarray(Image.open(next(p for p in (CI/fo).iterdir() if p.stem == cs))); Hc, Wc = crop.shape
    raw = np.asarray(Image.open(next(p for p in RAW_I.iterdir() if p.stem == stem)))
    outs = []
    for _, b, _, _ in welds:
        ox, oy = max(0, int(round(b[0]))), max(0, int(round(b[1])))
        if abs((min(W, int(round(b[2])))-ox) - Wc) <= 3 and abs((min(H, int(round(b[3])))-oy) - Hc) <= 3:
            v, dx, dy = best(raw, crop, ox, oy); eq = float((raw[oy:oy+Hc, ox:ox+Wc] == crop[:raw.shape[0]-oy, :raw.shape[1]-ox]).mean()) if oy+Hc <= raw.shape[0] and ox+Wc <= raw.shape[1] else float("nan")
            outs.append(f"asal kita=({ox},{oy}) pearson={v:.3f} geser terbaik=({dx},{dy}) identik@asal={eq:.4f}")
    say(f"   {cs} ({fo}): " + ("; ".join(outs) if outs else "tak ada kotak 焊缝 yang ukurannya cocok"))
open(ROOT/"diag_crops_summary.txt", "w", encoding="utf-8").write("\n".join(LINES) + "\n"); print("\nselesai. kirim diag_crops_summary.txt dan diag_crops_unmatched.csv")
