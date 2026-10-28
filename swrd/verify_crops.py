"""verify_crops.py  -- membuktikan (bukan mengasumsikan) bahwa crop dan label buatan kita = crop resmi SWRD.
Taruh di folder yang berisi Raw_data, crop_weld_data, swrd_split_v1.csv dan swrd_audit2_crop_inventory.csv.  python verify_crops.py
Bagian 1 (JSON, SEMUA film terpakai): untuk tiap crop resmi cari crop kita (kotak 焊缝 pada citra mentah, ukuran +-3 px), lalu
   bandingkan label crop-JSON resmi dengan label citra-mentah yang digeser ke koordinat crop. Titik asal crop yang sebenarnya diestimasi
   dari selisih koordinat (raw - crop) sehingga TIDAK bergantung pada asumsi pembulatan kita; lalu dilaporkan apakah asal itu = round/floor/ceil dari
   pojok poligon 焊缝, berapa label yang cocok / hanya ada di resmi / hanya ada di kita, dan aturan resmi untuk objek yang terpotong tepi crop.
Bagian 2 (piksel, sampel acak berstrata): bandingkan citra crop resmi dengan potongan citra mentah pada titik asal kita (dan geser +-3 px).
Keluaran: verify_crops_summary.txt, verify_crops_report.csv (per crop), verify_crops_pixels.csv (per crop sampel)."""
import json, csv, sys, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
RAW_I, RAW_J = ROOT/"Raw_data"/"images", ROOT/"Raw_data"/"json"
CJ, CI = ROOT/"crop_weld_data"/"crop_weld_jsons", ROOT/"crop_weld_data"/"crop_weld_images"
def opt(n, d): return type(d)(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d
SEED = opt("--seed", 12345); QUOTA = {"L/1": opt("--n-l1", 14), "L/2": opt("--n-l2", 6), "T/1": opt("--n-t1", 14), "T/2": opt("--n-t2", 6)}
WELD = "焊缝"; TOL = 3; COARSE = 8.0; FINE = 1.5
LINES = []
def say(s=""): print(s); LINES.append(s)

def bbox(sh):
    p = np.array(sh["points"], float)
    if sh.get("shape_type") == "circle" and len(p) == 2:
        r = float(np.hypot(*(p[1] - p[0]))); return (p[0][0] - r, p[0][1] - r, p[0][0] + r, p[0][1] + r)
    return (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
def split_shapes(d):
    w, o = [], []
    for sh in d.get("shapes", []): (w if sh.get("label") == WELD else o).append((sh.get("label"), bbox(sh)))
    return w, o
def match(off, ours, ox, oy, Wc, Hc, tol, clip):
    """penjodohan serakah per label: off=[(lab,box)] koordinat crop; ours=[(lab,box)] koordinat citra mentah, digeser (ox,oy)."""
    cand = []
    for i, (l, b) in enumerate(off):
        for j, (l2, c) in enumerate(ours):
            if l != l2: continue
            cb = (c[0] - ox, c[1] - oy, c[2] - ox, c[3] - oy)
            if clip: cb = (max(cb[0], 0), max(cb[1], 0), min(cb[2], Wc), min(cb[3], Hc))
            if cb[2] <= cb[0] or cb[3] <= cb[1]: continue
            dev = max(abs(b[k] - cb[k]) for k in range(4))
            if dev <= tol: cand.append((dev, i, j))
    cand.sort(); ui, uj, pairs = set(), set(), []
    for dev, i, j in cand:
        if i in ui or j in uj: continue
        ui.add(i); uj.add(j); pairs.append((i, j, dev))
    return pairs

kept = {r["stem"] for r in csv.DictReader(open(ROOT/"swrd_split_v1.csv", encoding="utf-8-sig"))}
inv = defaultdict(list)
for r in csv.DictReader(open(ROOT/"swrd_audit2_crop_inventory.csv", encoding="utf-8-sig")):
    if r["raw"] in kept and r["jsonW"] not in ("", None): inv[r["raw"]].append(r)
say(f"film terpakai (split_v1): {len(kept)}; film dengan crop resmi: {len(inv)}; crop resmi: {sum(len(v) for v in inv.values())}")
say(f"film tanpa crop resmi: {sorted(kept - set(inv))}")

t0 = time.time(); rep = []; film_info = {}
C = Counter(); lab_off = Counter(); lab_match = Counter(); lab_onlyoff = Counter(); lab_ours = Counter(); lab_onlyours = Counter()
dim_dev = Counter(); conv = {"round": Counter(), "floor": Counter(), "ceil": Counter()}; wdev = Counter(); onlyours_vis = Counter(); onlyours_ctr = Counter(); spread = []
unpaired_off, unpaired_ours = [], []
for n, (stem, rows) in enumerate(sorted(inv.items())):
    d = json.load(open(RAW_J/f"{stem}.json", encoding="utf-8")); W, H = int(d["imageWidth"]), int(d["imageHeight"])
    welds, objs = split_shapes(d)
    ours = []   # (x0f,y0f,x1f,y1f) float, dan versi bulat yang dipakai builder
    for _, b in welds: ours.append(dict(f=b, r=(max(0, int(round(b[0]))), max(0, int(round(b[1]))), min(W, int(round(b[2]))), min(H, int(round(b[3]))))))
    film_info[stem] = (d, objs, ours)
    pairs = []
    for i, r in enumerate(rows):
        for j, o in enumerate(ours):
            dw, dh = abs(int(r["jsonW"]) - (o["r"][2] - o["r"][0])), abs(int(r["jsonH"]) - (o["r"][3] - o["r"][1]))
            if dw <= TOL and dh <= TOL: pairs.append((dw + dh, i, j))
    pairs.sort(); ui, uj, pp = set(), set(), []
    for _, i, j in pairs:
        if i in ui or j in uj: continue
        ui.add(i); uj.add(j); pp.append((i, j))
    for i, r in enumerate(rows):
        if i not in ui: C["crop_resmi_tanpa_pasangan_kita"] += 1; unpaired_off.append(r["stem"])
    for j in range(len(ours)):
        if j not in uj: C["crop_kita_tanpa_pasangan_resmi"] += 1; unpaired_ours.append(f"{stem}#{j}")
    for i, j in pp:
        r, o = rows[i], ours[j]; Wc, Hc = int(r["jsonW"]), int(r["jsonH"]); C["crop_berpasangan"] += 1
        dim_dev[(Wc - (o["r"][2] - o["r"][0]), Hc - (o["r"][3] - o["r"][1]))] += 1
        cd = json.load(open(CJ/r["folder"]/f"{r['stem']}.json", encoding="utf-8")); cw, co = split_shapes(cd)
        for _, b in cw: wdev[(round(b[0]), round(b[1]), round(b[2] - Wc), round(b[3] - Hc))] += 1
        ox0, oy0 = o["r"][0], o["r"][1]
        # --- estimasi titik asal sebenarnya dari pasangan label bagian dalam crop ---
        pr = match(co, objs, ox0, oy0, Wc, Hc, COARSE, False); dxs, dys, sw = [], [], []
        for a, b2, _ in pr:
            ob, rb = co[a][1], objs[b2][1]
            if ob[0] > 3 and ob[1] > 3 and ob[2] < Wc - 3 and ob[3] < Hc - 3:
                dxs += [rb[0] - ob[0], rb[2] - ob[2]]; dys += [rb[1] - ob[1], rb[3] - ob[3]]; sw.append(abs((rb[0] - ob[0]) - (rb[2] - ob[2])))
        if dxs:
            tx, ty = float(np.median(dxs)), float(np.median(dys)); spread += sw
            for nm, fn in (("round", lambda v: int(round(v))), ("floor", lambda v: int(np.floor(v))), ("ceil", lambda v: int(np.ceil(v)))):
                conv[nm][(round(tx - fn(o["f"][0]), 2), round(ty - fn(o["f"][1]), 2))] += 1
        else: tx, ty = float(ox0), float(oy0); C["crop_tanpa_label_dalam_untuk_estimasi"] += 1
        # --- penjodohan akhir pada titik asal terestimasi ---
        pf = match(co, objs, tx, ty, Wc, Hc, FINE, True); mo, mr = {a for a, _, _ in pf}, {b2 for _, b2, _ in pf}
        for a, (l, b) in enumerate(co):
            lab_off[l] += 1; lab_match[l] += a in mo; lab_onlyoff[l] += a not in mo
        n_only_ours = 0
        for b2, (l, c) in enumerate(objs):
            ix0, iy0, ix1, iy1 = max(c[0] - tx, 0), max(c[1] - ty, 0), min(c[2] - tx, Wc), min(c[3] - ty, Hc)
            if ix1 <= ix0 or iy1 <= iy0: continue
            lab_ours[l] += 1
            if b2 in mr: continue
            lab_onlyours[l] += 1; n_only_ours += 1; area = max((c[2] - c[0]) * (c[3] - c[1]), 1e-9); vis = (ix1 - ix0) * (iy1 - iy0) / area
            onlyours_vis["<0.10" if vis < .1 else "0.10-0.50" if vis < .5 else "0.50-0.90" if vis < .9 else ">=0.90"] += 1
            cx, cy = (c[0] + c[2]) / 2 - tx, (c[1] + c[3]) / 2 - ty; onlyours_ctr["pusat di dalam crop" if 0 <= cx <= Wc and 0 <= cy <= Hc else "pusat di luar crop"] += 1
        n_off = len(co)
        rep.append(dict(film=stem, folder=r["folder"], crop=r["stem"], orient="H" if Wc >= Hc else "V", Wc=Wc, Hc=Hc, ours_x0=ox0, ours_y0=oy0, est_x0=tx, est_y0=ty,
                        n_official=n_off, n_match=len(pf), only_official=n_off - len(pf), only_ours=n_only_ours, estimasi_dari=len(dxs) // 2))
    if n % 500 == 0: print(f"   film {n}/{len(inv)}  {time.time()-t0:.0f}s", flush=True)
with open(ROOT/"verify_crops_report.csv", "w", newline="", encoding="utf-8-sig") as f:
    if rep: w = csv.DictWriter(f, rep[0].keys()); w.writeheader(); w.writerows(rep)

say(f"\n=== BAGIAN 1: JSON, semua film terpakai ({time.time()-t0:.0f}s) ===")
say(f"crop resmi berpasangan dengan crop kita (ukuran +-{TOL}): {C['crop_berpasangan']}; crop resmi tanpa pasangan: {C['crop_resmi_tanpa_pasangan_kita']} {unpaired_off[:8]}; crop kita (焊缝) tanpa pasangan resmi: {C['crop_kita_tanpa_pasangan_resmi']} {unpaired_ours[:8]}")
say(f"selisih ukuran (resmi - kita) (dW,dH): {dict(dim_dev.most_common(8))}")
say(f"poligon 焊缝 dalam crop-JSON resmi: (x0,y0, x1-W, y1-H) terbanyak: {dict(Counter(wdev).most_common(6))}")
for nm in ("round", "floor", "ceil"):
    c = conv[nm]; tot = sum(c.values()); z = c.get((0.0, 0.0), 0)
    say(f"titik asal sebenarnya (estimasi dari label) - {nm}(pojok 焊缝): tepat 0 pada {z}/{tot} crop ({z/max(tot,1):.2%}); sebaran: {dict(c.most_common(6))}")
say(f"crop tanpa label bagian-dalam untuk estimasi asal (tidak teruji lewat label): {C['crop_tanpa_label_dalam_untuk_estimasi']}")
say(f"selisih skala/peregangan (|dx0 - dx1|) median {np.median(spread) if spread else float('nan')}, maks {max(spread) if spread else float('nan')}")
say("\nlabel per kelas (resmi total / cocok / hanya-resmi | kita yang beririsan dengan crop / hanya-kita):")
for l in sorted(set(lab_off) | set(lab_ours), key=lambda k: -lab_off[k]):
    say(f"   {l}: {lab_off[l]} / {lab_match[l]} / {lab_onlyoff[l]} | {lab_ours[l]} / {lab_onlyours[l]}")
say(f"objek 'hanya-kita' menurut fraksi luas yang berada dalam crop: {dict(onlyours_vis)}; menurut letak pusat: {dict(onlyours_ctr)}")
tot_off, tot_m = sum(lab_off.values()), sum(lab_match.values())
say(f"TOTAL label resmi: {tot_off}; cocok dengan label kita (tol {FINE} px): {tot_m} ({tot_m/max(tot_off,1):.3%})")
bad = sorted([r for r in rep if r["only_official"] or r["only_ours"]], key=lambda r: -(r["only_official"] + r["only_ours"]))[:10]
say("crop dengan ketidakcocokan terbanyak: " + "; ".join(f"{r['crop']} resmi-saja {r['only_official']} kita-saja {r['only_ours']}" for r in bad))

# ---------- BAGIAN 2: piksel ----------
say("\n=== BAGIAN 2: piksel (sampel berstrata) ===")
rng = np.random.default_rng(SEED); sample = []
byfolder = defaultdict(list)
for r in rep:
    row = next(x for x in inv[r["film"]] if x["stem"] == r["crop"])
    if row["has_image"] in ("True", "1", True, 1) and row["mode"] not in ("-", "L", "P", "RGB"): byfolder[r["folder"]].append((r, row))
for fo, q in QUOTA.items():
    items = byfolder.get(fo, [])
    if fo == "T/1":   # separuh A_ separuh B_ (sekunder vertikal)
        a = [x for x in items if x[0]["crop"].startswith("A_")]; b = [x for x in items if x[0]["crop"].startswith("B_")]
        pick = [a[i] for i in rng.permutation(len(a))[:q // 2]] + [b[i] for i in rng.permutation(len(b))[:q - q // 2]]
    else: pick = [items[i] for i in rng.permutation(len(items))[:q]]
    sample += pick
allp = [x for v in byfolder.values() for x in v]
if allp:   # kasus ekstrem
    for key in (lambda x: -x[0]["Hc"] if x[0]["orient"] == "H" else 0, lambda x: x[0]["Hc"] if x[0]["orient"] == "H" else 1e9, lambda x: x[0]["Wc"] if x[0]["orient"] == "V" else 1e9):
        e = min(allp, key=key)
        if e not in sample: sample.append(e)
say(f"crop disampel: {len(sample)} (tersedia per folder: { {k: len(v) for k, v in byfolder.items()} })")
def pearson_search(raw, crop, ox, oy, R):
    h, w = crop.shape; ph, pw = min(h, 400), min(w, 400); y0, x0 = h // 2 - ph // 2, w // 2 - pw // 2
    c = crop[y0:y0 + ph, x0:x0 + pw].astype(np.float64); c -= c.mean(); cn = np.sqrt((c * c).sum()); best = (-2, 0, 0)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            yy, xx = oy + dy + y0, ox + dx + x0
            if yy < 0 or xx < 0 or yy + ph > raw.shape[0] or xx + pw > raw.shape[1]: continue
            r = raw[yy:yy + ph, xx:xx + pw].astype(np.float64); r -= r.mean(); den = cn * np.sqrt((r * r).sum())
            v = (c * r).sum() / den if den > 0 else 0
            if v > best[0]: best = (v, dx, dy)
    return best
def exact_frac(raw, crop, ox, oy):
    h, w = crop.shape
    if ox < 0 or oy < 0 or oy + h > raw.shape[0] or ox + w > raw.shape[1]: return float("nan")
    return float((raw[oy:oy + h, ox:ox + w] == crop).mean())
prow = []; byraw = defaultdict(list)
for r, row in sample: byraw[r["film"]].append((r, row))
for film, lst in sorted(byraw.items()):
    raw = np.asarray(Image.open(next(p for p in RAW_I.iterdir() if p.stem == film)))
    for r, row in lst:
        cp = next((p for p in (CI/r["folder"]).iterdir() if p.stem == r["crop"]), None)
        if cp is None: continue
        crop = np.asarray(Image.open(cp)); ox, oy = r["ours_x0"], r["ours_y0"]
        shape_ok = crop.shape == (r["Hc"], r["Wc"]); cc, dx, dy = pearson_search(raw, crop, ox, oy, 3)
        wide = None
        if cc < 0.9:
            try:
                import cv2
                h, w = crop.shape; ph, pw = min(h, 256), min(w, 256); y0, x0 = h // 2 - ph // 2, w // 2 - pw // 2
                R = 80; ya, xa = max(oy + y0 - R, 0), max(ox + x0 - R, 0)
                reg = raw[ya:oy + y0 + ph + R, xa:ox + x0 + pw + R].astype(np.float32); tpl = crop[y0:y0 + ph, x0:x0 + pw].astype(np.float32)
                res = cv2.matchTemplate(reg, tpl, cv2.TM_CCOEFF_NORMED); _, mv, _, ml = cv2.minMaxLoc(res); wide = (round(float(mv), 3), xa + ml[0] - x0 - ox, ya + ml[1] - y0 - oy)
            except Exception as e: wide = f"cv2 tak tersedia/gagal: {e}"
        prow.append(dict(film=film, folder=r["folder"], crop=r["crop"], orient=r["orient"], Wc=r["Wc"], Hc=r["Hc"], shape_crop_sama=shape_ok, dtype=str(crop.dtype),
                         ours_x0=ox, ours_y0=oy, est_label_x0=r["est_x0"], est_label_y0=r["est_y0"], best_dx=dx, best_dy=dy, pearson=round(cc, 4),
                         exact_at_ours=round(exact_frac(raw, crop, ox, oy), 6), exact_at_best=round(exact_frac(raw, crop, ox + dx, oy + dy), 6), wide_search=wide))
with open(ROOT/"verify_crops_pixels.csv", "w", newline="", encoding="utf-8-sig") as f:
    if prow: w = csv.DictWriter(f, prow[0].keys()); w.writeheader(); w.writerows(prow)
n = len(prow)
if n:
    say(f"ukuran array crop = ukuran JSON: {sum(p['shape_crop_sama'] for p in prow)}/{n}")
    say(f"piksel IDENTIK 100% pada titik asal kita (shift 0,0): {sum(p['exact_at_ours'] == 1.0 for p in prow)}/{n}")
    say(f"piksel identik >=99.99%: {sum(p['exact_at_ours'] >= 0.9999 for p in prow)}/{n}; pergeseran terbaik (dx,dy): {dict(Counter((p['best_dx'], p['best_dy']) for p in prow))}")
    say(f"korelasi Pearson patch tengah (min / median): {min(p['pearson'] for p in prow)} / {np.median([p['pearson'] for p in prow])}")
    sus = [p for p in prow if p["exact_at_ours"] != 1.0]
    for p in sus[:12]: say(f"   TIDAK identik: {p['crop']} ({p['folder']}, {p['orient']}) best=({p['best_dx']},{p['best_dy']}) pearson={p['pearson']} exact_at_best={p['exact_at_best']} wide={p['wide_search']} label-asal=({p['est_label_x0']},{p['est_label_y0']}) kita=({p['ours_x0']},{p['ours_y0']})")
say("\nKRITERIA LULUS (ditetapkan sebelum melihat hasil): (1) semua crop resmi berpasangan; (2) titik asal estimasi-label = round(pojok 焊缝) pada >=99.5% crop dan tidak ada yang selisih >1 px; "
    "(3) >=99.5% label resmi cocok (tol 1.5 px); (4) semua crop sampel identik piksel pada titik asal kita. Bila ada yang gagal, builder JANGAN dijalankan.")
open(ROOT/"verify_crops_summary.txt", "w", encoding="utf-8").write("\n".join(LINES) + "\n")
print("\nselesai. kirim verify_crops_summary.txt (dan verify_crops_report.csv, verify_crops_pixels.csv).")
