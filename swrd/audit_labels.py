"""audit_labels.py -- audit SETIAP label di swrd_tiles terhadap anotasi ASLI. Taruh di folder SWXD_Data (berisi Raw_data, crop_weld_data,
swrd_tiles, swrd_audit2_crop_inventory.csv).   python audit_labels.py [--tiles swrd_tiles]   (semua cek + lembar visual val)
                                              python audit_labels.py --sheets all (lembar visual train + val)
Cek per tile (semua 3.400, tanpa memakai kode builder):
 A. label .txt == label yang diturunkan ulang dari JSON crop RESMI (visibilitas >=0,5) -- jumlah, kelas, koordinat (tol 0,01 px), dengan aturan
    'kotak' (v3: kotak objek dipotong tepi tile) atau 'poligon' (v4: poligon objek dipotong tepi tile)
 E/F. (aturan apa pun) kotak yang tidak berisi poligon cacat asli kelas sama (E), atau yang cacat aslinya mengisi <50% kotak (F)
 B. setiap kotak berada di dalam objek kelas SAMA pada JSON citra MENTAH (anotasi asli sebelum di-crop), digeser ke koordinat tile
 C. PNG == hasil render ulang dari crop resmi/mentah (stretch persentil + CLAHE) -> gambar dan label tidak tertukar
 G. tile negatif tidak menyentuh cacat asli; H. cacat asli yang tampak di tile positif selalu berlabel
 D. skor kontras tiap kotak (beda rerata dalam kotak vs cincin sekitarnya / std cincin, pada data 16-bit asli) -> untuk mengurutkan
    kotak yang 'tampak kosong'. Skor rendah BUKAN berarti label salah; artinya cacat samar pada data asli.
Keluaran (folder audit_<nama folder tile>/): ringkasan.txt, kotak.csv (semua kotak), gagal.csv, lembar_<split>_NN.jpg (konteks anotasi asli)."""
import json, csv, sys, math, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
T = ROOT/(sys.argv[sys.argv.index("--tiles") + 1] if "--tiles" in sys.argv else "swrd_tiles"); OUT = ROOT/f"audit_{T.name}"; OUT.mkdir(exist_ok=True)
for _p in list(OUT.glob("lembar_*")): _p.unlink()   # buang lembar dari audit sebelumnya
RAW_I, RAW_J = ROOT/"Raw_data"/"images", ROOT/"Raw_data"/"json"
CJ, CI = ROOT/"crop_weld_data"/"crop_weld_jsons", ROOT/"crop_weld_data"/"crop_weld_images"
SHEETS = sys.argv[sys.argv.index("--sheets") + 1] if "--sheets" in sys.argv else "val"
NO_PIX = "--no-pixel" in sys.argv
CLS = {"气孔": 0, "夹渣": 1, "裂纹": 2, "咬边": 3, "未熔合": 4, "未焊透": 5}
NAMES = ["porosity", "inclusion", "crack", "undercut", "lack_of_fusion", "lack_of_penetration"]
COL = [(255, 60, 60), (60, 220, 60), (60, 140, 255), (255, 210, 0), (255, 0, 255), (0, 230, 230)]
try: FONT = ImageFont.load_default(size=15)
except TypeError: FONT = ImageFont.load_default()
try:
    import cv2; CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
except ImportError: sys.exit("butuh OpenCV (sama dengan saat render)")

def bbox(pts, st=None):
    p = np.array(pts, float)
    if st == "circle" and len(p) == 2:
        r = float(np.hypot(*(p[1] - p[0]))); return (p[0][0] - r, p[0][1] - r, p[0][0] + r, p[0][1] + r)
    return (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
def poly_pts(pts, b): return pts if pts and len(pts) >= 3 else [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]
def clip_bbox(P, x0, y0, x1, y1):
    """kotak pembatas EKSAK dari (poligon P irisan persegi), atau None bila irisan tak berluas (Liang-Barsky + pojok di dalam poligon)."""
    pts = [(float(p[0]), float(p[1])) for p in P]; n = len(pts); cand = []
    for i in range(n):
        (ax, ay), (bx, by) = pts[i - 1], pts[i]; dx, dy = bx - ax, by - ay; t0, t1 = 0.0, 1.0; ok = True
        for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
            if p == 0:
                if q < 0: ok = False; break
            else:
                t = q / p
                if p < 0:
                    if t > t1: ok = False; break
                    t0 = max(t0, t)
                else:
                    if t < t0: ok = False; break
                    t1 = min(t1, t)
        if ok: cand += [(ax + t0 * dx, ay + t0 * dy), (ax + t1 * dx, ay + t1 * dy)]
    def inside(x, y):
        c = False
        for i in range(n):
            (ax, ay), (bx, by) = pts[i - 1], pts[i]
            if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay): c = not c
        return c
    cand += [(x, y) for x in (x0, x1) for y in (y0, y1) if inside(x, y)]
    if not cand: return None
    xs, ys = [c[0] for c in cand], [c[1] for c in cand]
    if max(xs) - min(xs) <= 0 or max(ys) - min(ys) <= 0: return None
    return (min(xs), min(ys), max(xs), max(ys))
def clip_area(P, x0, y0, x1, y1):
    """luas (poligon P irisan persegi), Sutherland-Hodgman + shoelace (tepi degenerat hasil potong poligon tak-cembung berluas nol, jadi luas tetap benar)."""
    pts = [(float(p[0]), float(p[1])) for p in P]
    for k, (inside, inter) in enumerate((
            (lambda p: p[0] >= x0, lambda a, b: (x0, a[1] + (x0 - a[0]) * (b[1] - a[1]) / (b[0] - a[0]))),
            (lambda p: p[0] <= x1, lambda a, b: (x1, a[1] + (x1 - a[0]) * (b[1] - a[1]) / (b[0] - a[0]))),
            (lambda p: p[1] >= y0, lambda a, b: (a[0] + (y0 - a[1]) * (b[0] - a[0]) / (b[1] - a[1]), y0)),
            (lambda p: p[1] <= y1, lambda a, b: (a[0] + (y1 - a[1]) * (b[0] - a[0]) / (b[1] - a[1]), y1)))):
        out = []
        for i in range(len(pts)):
            a, b = pts[i - 1], pts[i]; ia, ib = inside(a), inside(b)
            if ib:
                if not ia: out.append(inter(a, b))
                out.append(b)
            elif ia: out.append(inter(a, b))
        pts = out
        if not pts: return 0.0
    return abs(sum(pts[i - 1][0] * pts[i][1] - pts[i][0] * pts[i - 1][1] for i in range(len(pts)))) / 2

def vis_poly(P, b, x0, y0, w, h, longx):   # aturan v4.1 (lihat build_swrd_tiles.py)
    cb = clip_bbox(P, x0, y0, x0 + w, y0 + h)
    if cb is None: return None, 0.0
    if longx: e = (cb[2] - cb[0]) / min(b[2] - b[0], w); sa = clip_area(P, x0, -1e9, x0 + w, 1e9)
    else: e = (cb[3] - cb[1]) / min(b[3] - b[1], h); sa = clip_area(P, -1e9, y0, 1e9, y0 + h)
    if sa > 1e-9: sh = clip_area(P, x0, y0, x0 + w, y0 + h) / sa
    else: sh = (cb[3] - cb[1]) / min(b[3] - b[1], h) if longx else (cb[2] - cb[0]) / min(b[2] - b[0], w)
    return cb, min(e, 1.0) * min(sh, 1.0)
def prep(a):
    lo, hi = np.percentile(a, (0.5, 99.5)); b = np.zeros(a.shape, np.uint8) if hi <= lo else (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return CLAHE.apply(b)
def stretch(a):
    lo, hi = np.percentile(a, (0.5, 99.5)); return np.zeros(a.shape, np.uint8) if hi <= lo else (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)

man = list(csv.DictReader(open(T/"manifest.csv", encoding="utf-8-sig")))
inv = {(r["folder"], r["stem"]): r for r in csv.DictReader(open(ROOT/"swrd_audit2_crop_inventory.csv", encoding="utf-8-sig"))}
imgs = {}
for fo in {m["folder"] for m in man}:
    if (CI/fo).exists():
        for p in (CI/fo).iterdir(): imgs[(fo, p.stem)] = p
fails, rows, C = [], [], Counter(); cjson, rjson = {}, {}
def fail(m, kind, msg): fails.append(dict(tile=m["name"], split=m["split"], jenis=kind, detail=msg)); C["GAGAL_" + kind] += 1

bycrop = defaultdict(list)
for m in man: bycrop[(m["folder"], m["crop"])].append(m)
t0 = time.time(); ctx = {}   # nama tile -> data untuk lembar visual
for n, (key, ms) in enumerate(sorted(bycrop.items())):
    fo, cs = key; film = ms[0]["film"]
    cd = json.load(open(CJ/fo/f"{cs}.json", encoding="utf-8")); Wc, Hc = int(cd["imageWidth"]), int(cd["imageHeight"])
    rd = json.load(open(RAW_J/f"{film}.json", encoding="utf-8"))
    off = [(sh["label"], bbox(sh["points"], sh.get("shape_type")), sh["points"] if sh.get("shape_type") not in ("rectangle", "circle") else None) for sh in cd["shapes"] if sh.get("label") != "焊缝"]
    raw_objs = [(CLS[sh["label"]], bbox(sh["points"], sh.get("shape_type")), sh["points"] if sh.get("shape_type") not in ("rectangle", "circle") else None) for sh in rd["shapes"] if sh.get("label") in CLS]
    crop_arr = raw_arr = None
    need_pix = (not NO_PIX) or any(m["split"] == SHEETS or SHEETS == "all" for m in ms)
    if need_pix:
        if key in imgs: crop_arr = np.asarray(Image.open(imgs[key]))
        if crop_arr is None or any(m["src"] == "mentah" for m in ms):
            raw_arr = np.asarray(Image.open(next(p for p in RAW_I.iterdir() if p.stem == film))) if raw_arr is None else raw_arr
    for m in ms:
        x, y, w, h, X, Y = (int(m[k]) for k in ("x", "y", "w", "h", "X", "Y")); ox, oy = X - x, Y - y
        # A. turunkan ulang label dari JSON crop resmi, dengan DUA aturan:
        #    'kotak'   = kotak pembatas objek dipotong tepi tile (builder v3)
        #    'poligon' = POLIGON objek dipotong tepi tile -> kotak pembatasnya; visibilitas v4.1 (bentang sumbu panjang x luas di pita sumbu pendek)
        longx = Wc >= Hc; exp_k, exp_p = [], []; amb_p = False
        for lab, b, pts in off:
            if lab not in CLS or b[2] - b[0] < 1 or b[3] - b[1] < 1: continue
            ix0, iy0, ix1, iy1 = max(b[0], x), max(b[1], y), min(b[2], x + w), min(b[3], y + h)
            if ix1 > ix0 and iy1 > iy0 and ((ix1 - ix0) / min(b[2] - b[0], w)) * ((iy1 - iy0) / min(b[3] - b[1], h)) >= 0.5: exp_k.append((CLS[lab], ix0 - x, iy0 - y, ix1 - x, iy1 - y))
            cb, v = vis_poly(poly_pts(pts, b), b, x, y, w, h, longx)
            if cb is not None:
                if v >= 0.5: exp_p.append((CLS[lab], cb[0] - x, cb[1] - y, cb[2] - x, cb[3] - y))
                elif v >= 0.1: amb_p = True
        got = []
        for line in open(T/"labels"/m["split"]/f"{m['name']}.txt").read().splitlines():
            if not line.strip(): continue
            c, cx, cy, bw, bh = line.split(); c = int(c); cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            got.append((c, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        def same(E):   # penjodohan satu-satu (tidak bergantung urutan sort; pembulatan 6 desimal YOLO -> tol 0,01 px)
            if len(E) != len(got): return False
            used = set()
            for g in got:
                j = next((j for j, a in enumerate(E) if j not in used and a[0] == g[0] and max(abs(a[i] - g[i]) for i in range(1, 5)) <= 0.01), None)
                if j is None: return False
                used.add(j)
            return True
        got.sort(); C["kotak"] += len(got); C["tile"] += 1
        rule = "keduanya" if same(exp_k) and same(exp_p) else "kotak" if same(exp_k) else "poligon" if same(exp_p) else None
        C["aturan_" + str(rule)] += 1
        if rule is None:
            fail(m, "A_beda_dengan_JSON_crop_resmi", f"diharapkan(poligon) {[(NAMES[e[0]],) + tuple(round(float(v),1) for v in e[1:]) for e in sorted(exp_p)]} | ada {[(NAMES[g[0]],) + tuple(round(float(v),1) for v in g[1:]) for g in got]}")
        elif rule == "poligon" and amb_p: fail(m, "A_tile_ambigu_terpilih", "")
        # E/F (tidak bergantung aturan): isi setiap kotak dibandingkan poligon ASLI kelas sama
        for g in got:
            bx = (g[1] + x, g[2] + y, g[3] + x, g[4] + y); parts = []
            for lab, b, pts in off:
                if CLS.get(lab) != g[0]: continue
                P = poly_pts(pts, b)
                if clip_area(P, *bx) > 1e-6: parts.append(clip_bbox(P, *bx))
            parts = [p for p in parts if p is not None]
            if not parts: fail(m, "E_kotak_tanpa_cacat_asli", f"{NAMES[g[0]]} {tuple(round(float(v),1) for v in g[1:])}: tak ada poligon asli {NAMES[g[0]]} di dalam kotak"); continue
            u = (min(p[0] for p in parts), min(p[1] for p in parts), max(p[2] for p in parts), max(p[3] for p in parts))
            iou = ((u[2] - u[0]) * (u[3] - u[1])) / ((bx[2] - bx[0]) * (bx[3] - bx[1]))   # u selalu di dalam kotak -> IoU = luas u / luas kotak
            if iou < 0.5: fail(m, "F_kotak_jauh_lebih_besar_dari_cacat", f"{NAMES[g[0]]} {tuple(round(float(v),1) for v in g[1:])}: cacat asli hanya mengisi {iou:.0%} kotak")
            elif iou < 0.9: C["kotak_longgar_IoU_0.5_0.9"] += 1
        # B. tiap kotak harus di dalam objek kelas sama di JSON mentah
        for g in got:
            rb = (g[1] + X, g[2] + Y, g[3] + X, g[4] + Y)
            ok = any(c == g[0] and rb[0] >= b[0] - 1.5 and rb[1] >= b[1] - 1.5 and rb[2] <= b[2] + 1.5 and rb[3] <= b[3] + 1.5 for c, b, _ in raw_objs)
            if not ok: fail(m, "B_tak_ada_di_JSON_mentah", f"{NAMES[g[0]]} {tuple(round(float(v),1) for v in g[1:])}")
            rows.append(dict(tile=m["name"], split=m["split"], film=film, kelas=NAMES[g[0]], x0=round(g[1], 1), y0=round(g[2], 1), x1=round(g[3], 1), y1=round(g[4], 1),
                             lebar=round(g[3] - g[1], 1), tinggi=round(g[4] - g[2], 1), ada_di_mentah=int(ok), kontras=None))
        # G. tile negatif tidak boleh menyentuh kotak cacat asli (JSON mentah) mana pun
        if m["positif"] == "0":
            if any(min(b[2], X + w) > max(b[0], X) and min(b[3], Y + h) > max(b[1], Y) for c, b, _ in raw_objs): fail(m, "G_negatif_menyentuh_cacat_asli", "")
        # H. tile positif: setiap cacat asli (poligon JSON mentah) yang tampak >=10% di tile harus beririsan dengan label kelas sama
        else:
            for c, b, pts in raw_objs:
                if b[2] - b[0] < 1 or b[3] - b[1] < 1: continue
                cb, v = vis_poly(poly_pts(pts, b), b, X, Y, w, h, longx)
                if cb is None: continue
                if v >= 0.1 and not any(g[0] == c and min(g[3] + X, cb[2]) > max(g[1] + X, cb[0]) and min(g[4] + Y, cb[3]) > max(g[2] + Y, cb[1]) for g in got):
                    fail(m, "H_cacat_asli_tampak_tanpa_label", f"{NAMES[c]} tampak {v:.2f}")
        # C/D. piksel
        if crop_arr is not None and m["src"] == "resmi": patch = crop_arr[y:y + h, x:x + w]
        elif raw_arr is not None: patch = raw_arr[Y:Y + h, X:X + w]
        else: patch = None
        if patch is not None:
            if not NO_PIX:
                png = np.asarray(Image.open(T/"images"/m["split"]/f"{m['name']}.png"))
                if png.shape != patch.shape or not np.array_equal(png, prep(patch.astype(np.float32))): fail(m, "C_PNG_beda_render_ulang", "")
            pf = patch.astype(np.float64); scores = []
            for k, g in enumerate(got):
                x0, y0, x1, y1 = max(0, int(math.floor(g[1]))), max(0, int(math.floor(g[2]))), min(w, int(math.ceil(g[3]))), min(h, int(math.ceil(g[4])))
                mg = max(5, int(0.5 * max(x1 - x0, y1 - y0)))
                inner = pf[y0:y1, x0:x1]; X0, Y0, X1, Y1 = max(0, x0 - mg), max(0, y0 - mg), min(w, x1 + mg), min(h, y1 + mg)
                ring = pf[Y0:Y1, X0:X1].copy(); msk = np.ones(ring.shape, bool); msk[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
                r = ring[msk]; s = abs(inner.mean() - r.mean()) / (r.std() + 1e-9) if inner.size and r.size > 10 else float("nan")
                scores.append(s); rows[-len(got) + k]["kontras"] = round(float(s), 3)
            if got and (SHEETS == "all" or m["split"] == SHEETS):
                M = 200; cx0, cy0 = max(0, x - M), max(0, y - M)
                if crop_arr is not None: cx1, cy1 = min(crop_arr.shape[1], x + w + M), min(crop_arr.shape[0], y + h + M); cimg = crop_arr[cy0:cy1, cx0:cx1]
                else: cx1, cy1 = min(Wc, x + w + M), min(Hc, y + h + M); cimg = raw_arr[oy + cy0:oy + cy1, ox + cx0:ox + cx1]
                ctx[m["name"]] = dict(m=m, got=got, sc=min(scores), stretch=stretch(cimg.astype(np.float32)), cxy=(cx0, cy0), off=off, win=(x, y, w, h))
    if n % 300 == 0: print(f"   crop {n}/{len(bycrop)}  {time.time()-t0:.0f}s", flush=True)

with open(OUT/"kotak.csv", "w", newline="", encoding="utf-8-sig") as f:
    if rows: wr = csv.DictWriter(f, rows[0].keys()); wr.writeheader(); wr.writerows(rows)
with open(OUT/"gagal.csv", "w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, ["tile", "split", "jenis", "detail"]); wr.writeheader(); wr.writerows(fails)

# ---------- lembar visual: kiri = tile seperti dilatih + kotak; kanan = konteks anotasi ASLI (poligon resmi) ----------
def panel(d):
    m, got = d["m"], d["got"]; png = Image.open(T/"images"/m["split"]/f"{m['name']}.png").convert("RGB"); dr = ImageDraw.Draw(png)
    for c, x0, y0, x1, y1 in got: dr.rectangle([x0, y0, x1, y1], outline=COL[c], width=2)
    cx0, cy0 = d["cxy"]; im = Image.fromarray(d["stretch"]).convert("RGB"); dr2 = ImageDraw.Draw(im)
    for lab, b, pts in d["off"]:
        if b[2] < cx0 or b[0] > cx0 + im.size[0] or b[3] < cy0 or b[1] > cy0 + im.size[1]: continue
        col = COL[CLS[lab]] if lab in CLS else (150, 150, 150); P = [(px - cx0, py - cy0) for px, py in poly_pts(pts, b)]
        if len(P) == 2: dr2.rectangle([min(P[0][0], P[1][0]), min(P[0][1], P[1][1]), max(P[0][0], P[1][0]), max(P[0][1], P[1][1])], outline=col, width=2)   # rectangle/circle -> kotak
        elif len(P) > 2: dr2.line(P + [P[0]], fill=col, width=2)
        dr2.text((b[0] - cx0, max(b[1] - cy0 - 16, 0)), (NAMES[CLS[lab]] if lab in CLS else ("pseudo" if lab == "伪缺陷" else str(lab)))[:12], fill=col, font=FONT)
    x, y, w, h = d["win"]; dr2.rectangle([x - cx0, y - cy0, x - cx0 + w, y - cy0 + h], outline=(255, 255, 255), width=1)
    H = 430; a = png.resize((max(1, int(png.size[0] * H / png.size[1])), H)) if png.size[1] else png
    sc = min(H / im.size[1], 900 / im.size[0]); b = im.resize((max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))))
    can = Image.new("RGB", (1500, H + 26), (25, 25, 25)); can.paste(a, (0, 26)); can.paste(b, (a.size[0] + 15, 26)); dc = ImageDraw.Draw(can)
    dc.text((4, 4), f"{m['name']} | {m['split']} | kelas: {', '.join(sorted({NAMES[g[0]] for g in got}))} | kontras min {d['sc']:.2f}  (kiri: tile dilatih; kanan: crop resmi +-200px, poligon ASLI, kotak putih = tile)" + ("  <<GAGAL>>" if m["name"] in bad_t else ""), fill=(255, 255, 255), font=FONT)
    return can
bad_t = {f["tile"] for f in fails}
order = sorted(ctx.values(), key=lambda d: (d["m"]["split"], d["m"]["name"] not in bad_t, d["sc"] if not math.isnan(d["sc"]) else -1))
for sp in sorted({d["m"]["split"] for d in order}):
    L = [d for d in order if d["m"]["split"] == sp]
    for p in range(0, len(L), 6):
        page = Image.new("RGB", (1500, 6 * 470), (10, 10, 10))
        for i, d in enumerate(L[p:p + 6]): page.paste(panel(d), (0, i * 470))
        page.save(OUT/f"lembar_{sp}_{p // 6 + 1:02d}.jpg", quality=90)

sc = [r["kontras"] for r in rows if r["kontras"] is not None and not math.isnan(r["kontras"])]
lines = [f"tile diperiksa: {C['tile']}; kotak: {C['kotak']}",
         f"A label != JSON crop resmi (aturan mana pun): {C['GAGAL_A_beda_dengan_JSON_crop_resmi']} tile",
         f"aturan label yang cocok per tile: kotak(v3) {C['aturan_kotak']}, poligon(v4) {C['aturan_poligon']}, keduanya-sama {C['aturan_keduanya']}, tidak-cocok {C['aturan_None']}",
         f"E kotak berlabel yang TIDAK berisi poligon cacat asli kelas sama: {C['GAGAL_E_kotak_tanpa_cacat_asli']} kotak",
         f"F kotak yang cacat aslinya mengisi <50% kotak: {C['GAGAL_F_kotak_jauh_lebih_besar_dari_cacat']} kotak; longgar (IoU 0,5-0,9, dicatat saja): {C['kotak_longgar_IoU_0.5_0.9']}",
         f"tile ambigu terpilih (aturan poligon): {C['GAGAL_A_tile_ambigu_terpilih']}",
         f"G tile negatif menyentuh cacat asli: {C['GAGAL_G_negatif_menyentuh_cacat_asli']} tile",
         f"H cacat asli tampak (>=10%) di tile positif tanpa label: {C['GAGAL_H_cacat_asli_tampak_tanpa_label']} kotak",
         f"B kotak tak ada di JSON mentah (anotasi asli): {C['GAGAL_B_tak_ada_di_JSON_mentah']} kotak",
         f"C PNG != render ulang: {C['GAGAL_C_PNG_beda_render_ulang']} tile" + (" (dilewati, --no-pixel)" if NO_PIX else ""),
         f"skor kontras kotak (kuantil 5/25/50/75/95): {np.percentile(sc, [5, 25, 50, 75, 95]).round(2).tolist() if sc else '-'}; kotak skor <0,5: {sum(s < 0.5 for s in sc)}",
         "skor kontras rendah per kelas (<0,5): " + str(dict(Counter(r["kelas"] for r in rows if r["kontras"] is not None and r["kontras"] < 0.5))),
         f"lembar visual: {len(ctx)} tile positif ({SHEETS}), tile GAGAL lebih dulu, lalu dari kontras terendah",
         "HASIL: " + ("LULUS - setiap label identik dengan anotasi resmi dan berasal dari anotasi asli" if not fails else f"ADA {len(fails)} KEGAGALAN - lihat gagal.csv")]
print("\n".join(lines)); open(OUT/"ringkasan.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
