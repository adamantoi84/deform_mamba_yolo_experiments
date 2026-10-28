"""Pembuat tile SWRD untuk YOLO (v4.1: label & piksel dari crop RESMI; kotak = bagian POLIGON cacat di dalam tile).  Taruh di folder yang berisi Raw_data, crop_weld_data,
swrd_split_v1.csv dan swrd_audit2_crop_inventory.csv.
  python build_swrd_tiles.py --plan-only   -> hanya memilih tile dari JSON (cepat), menulis swrd_tiles/manifest.csv + summary.txt
  python build_swrd_tiles.py               -> memilih lalu merender PNG + label YOLO + data.yaml + overlay_sheet.png
Dasar (verify_crops.py + diag_crops.py): piksel crop resmi = citra mentah[y0:y1+1, x0:x1+1] dengan (x0,y0,x1,y1) = kotak poligon 焊缝;
label crop resmi = label mentah yang dipotong oleh crop/poligon 焊缝 (selalu lebih kecil/sama, tak pernah lebih besar).
Aturan (dikunci di sini, jangan diubah setelah melihat hasil training):
 * unit = crop resmi (crop_weld_data). Label = kotak pembatas tiap bentuk pada JSON crop resmi. Piksel = TIFF crop resmi; bila crop resmi tak punya
   citra (T/2), piksel diambil dari citra mentah pada titik asal kotak 焊缝 (hanya bila ukuran cocok persis). Bila keduanya tersedia, setiap tile
   dicek identik piksel dengan citra mentah (tile yang tidak identik dibuang dan dicatat).
 * sumbu panjang crop dipotong jendela 640 px tanpa tumpang tindih (offset acak); sumbu pendek: seluruhnya bila <=640, bila lebih diambil
   pita 640 px pada posisi acak per jendela -> semua tile <=640x640, resolusi asli, tanpa penskalaan, tanpa rotasi
 * v4: poligon cacat (JSON resmi) dipotong tepi tile; kotak label = kotak pembatas bagian poligon di dalam tile (v3 memotong KOTAK objek,
   sehingga cacat diagonal/memanjang bisa menghasilkan kotak di area tile yang tidak berisi cacat). Negatif tetap konservatif (tak menyentuh kotak cacat).
 * cacat dipertahankan jika visibilitas >=0,5: [bentang di tile / min(bentang objek, tile)] pada sumbu panjang crop x [luas poligon di tile /
   luas poligon di pita tile] pada sumbu pendek (v4.1; v4 memakai bentang kotak global di sumbu pendek -> LoP melengkung salah dianggap terpotong)
 * tile dibuang bila: ada cacat resmi terlihat 10-50% (ambigu); ada label di luar enam kelas (夹钨/焊瘤/内凹/None); atau ada cacat pada JSON
   mentah yang terlihat >=10% di tile tetapi TIDAK tertutup label resmi kelas yang sama (cacat tampak tetapi tak berlabel di versi resmi)
 * kotak resmi degenerat (lebar atau tinggi < 1 px, serpihan pemotongan poligon) diabaikan dan dihitung
 * 伪缺陷 = latar belakang. Negatif = tidak menyentuh cacat nyata mana pun (resmi maupun mentah)
 * positif:negatif = 1:1 (mengikuti paper); maks 2 positif + 2 negatif per film; lantai kuota kelas langka"""
import json, csv, sys, math, time, zlib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
RAW_I, RAW_J = ROOT/"Raw_data"/"images", ROOT/"Raw_data"/"json"
CJ, CI = ROOT/"crop_weld_data"/"crop_weld_jsons", ROOT/"crop_weld_data"/"crop_weld_images"
OUT = ROOT/"swrd_tiles"
def opt(n, d): return type(d)(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d
PLAN_ONLY = "--plan-only" in sys.argv
SEED = opt("--seed", 2026); TILE = 640; TOL = 3
NPOS = dict(train=opt("--npos-train", 1350), val=opt("--npos-val", 350)); NNEG = dict(train=opt("--nneg-train", 1350), val=opt("--nneg-val", 350))
CAP_POS = CAP_NEG = 2; KEEP, SLIVER, FLOOR = 0.50, 0.10, 0.08
CLS = {"气孔": 0, "夹渣": 1, "裂纹": 2, "咬边": 3, "未熔合": 4, "未焊透": 5}
NAMES = ["porosity", "inclusion", "crack", "undercut", "lack_of_fusion", "lack_of_penetration"]; WELD, PSEUDO = "焊缝", "伪缺陷"
INV = defaultdict(list)
for r in csv.DictReader(open(ROOT/"swrd_audit2_crop_inventory.csv", encoding="utf-8-sig")):
    if r["jsonW"] not in ("", None): INV[r["raw"]].append(r)
IMG = {}   # (folder, stem) -> path citra crop resmi
for fo in {r["folder"] for v in INV.values() for r in v}:
    if (CI/fo).exists():
        for p in (CI/fo).iterdir():
            if p.is_file() and p.suffix.lower() in (".tif", ".tiff", ".png", ".bmp", ".jpg"): IMG[(fo, p.stem)] = p

def bbox(sh):
    p = np.array(sh["points"], float)
    if sh.get("shape_type") == "circle" and len(p) == 2:
        r = float(np.hypot(*(p[1] - p[0]))); return (p[0][0] - r, p[0][1] - r, p[0][0] + r, p[0][1] + r)
    return (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())

def pts_of(sh, b):
    """titik poligon bentuk; persegi/lingkaran/poligon < 3 titik -> 4 pojok kotak pembatasnya"""
    if sh.get("shape_type") in ("rectangle", "circle") or len(sh["points"]) < 3: return [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]
    return [(float(p[0]), float(p[1])) for p in sh["points"]]

def clip_bbox(P, x0, y0, x1, y1):
    """kotak pembatas EKSAK dari (poligon P irisan persegi [x0,x1]x[y0,y1]) atau None bila irisan tak berluas.
    Titik ekstrem irisan selalu potongan sisi poligon di dalam persegi (Liang-Barsky) atau pojok persegi di dalam poligon. Diuji vs shapely (36k poligon)."""
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

BIG = 1e9
def vis_poly(P, b, x0, y0, w, h, longx):
    """visibilitas cacat (poligon P, kotak penuh b) di tile (x0,y0,w,h). Sumbu panjang crop (arah jendela digeser): bentang bagian poligon di tile
    dibagi min(bentang objek, ukuran tile). Sumbu pendek: LUAS poligon di tile dibagi luas poligon di 'pita' tile (tile diperpanjang tak hingga ke
    arah sumbu pendek) -> cacat panjang yang melengkung tidak dianggap terpotong hanya karena kotak globalnya tinggi. Kembalikan (kotak terpotong, vis)."""
    cb = clip_bbox(P, x0, y0, x0 + w, y0 + h)
    if cb is None: return None, 0.0
    if longx: e = (cb[2] - cb[0]) / min(b[2] - b[0], w); sa = clip_area(P, x0, -BIG, x0 + w, BIG)
    else: e = (cb[3] - cb[1]) / min(b[3] - b[1], h); sa = clip_area(P, -BIG, y0, BIG, y0 + h)
    if sa > 1e-9: sh = clip_area(P, x0, y0, x0 + w, y0 + h) / sa
    else: sh = (cb[3] - cb[1]) / min(b[3] - b[1], h) if longx else (cb[2] - cb[0]) / min(b[2] - b[0], w)   # poligon tanpa luas: kembali ke bentang
    return cb, min(e, 1.0) * min(sh, 1.0)

def windows(Wc, Hc, rng):
    longx = Wc >= Hc; L, S = (Wc, Hc) if longx else (Hc, Wc); n = L // TILE; off = int(rng.integers(0, L - n * TILE + 1)); out = []
    for k in range(n):
        p = off + k * TILE
        if S <= TILE: q, s = 0, S
        else: s = TILE; q = int(rng.integers(0, S - TILE + 1))
        out.append((p, q, TILE, s) if longx else (q, p, s, TILE))  # (x0,y0,w,h) dalam koordinat crop
    return out

def inter(b, x0, y0, w, h): return max(b[0], x0), max(b[1], y0), min(b[2], x0 + w), min(b[3], y0 + h)

def film_pool(stem, split, batch, stats):
    d = json.load(open(RAW_J/f"{stem}.json", encoding="utf-8")); W, H = int(d["imageWidth"]), int(d["imageHeight"])
    welds = [bbox(sh) for sh in d["shapes"] if sh.get("label") == WELD]
    raw_real = [(CLS[sh["label"]], bbox(sh), pts_of(sh, bbox(sh))) for sh in d["shapes"] if sh.get("label") in CLS]
    ours = [(max(0, int(round(b[0]))), max(0, int(round(b[1]))), min(W, int(round(b[2])) + 1), min(H, int(round(b[3])) + 1)) for b in welds]
    rows = INV.get(stem, []); pairs = []
    for i, r in enumerate(rows):
        for j, o in enumerate(ours):
            dw, dh = int(r["jsonW"]) - (o[2] - o[0]), int(r["jsonH"]) - (o[3] - o[1])
            if abs(dw) <= TOL and abs(dh) <= TOL: pairs.append((abs(dw) + abs(dh), i, j))
    pairs.sort(); ui, uj, crops = set(), set(), []
    for _, i, j in pairs:
        if i in ui or j in uj: continue
        ui.add(i); uj.add(j); crops.append((rows[i], ours[j]))
    stats["crop_resmi_tanpa_pasangan"] += len(rows) - len(ui); stats["焊缝_mentah_tanpa_crop_resmi"] += len(ours) - len(uj)
    rng = np.random.default_rng(SEED + zlib.crc32(stem.encode())); pool = []
    for r, o in sorted(crops, key=lambda t: t[0]["stem"]):
        Wc, Hc = int(r["jsonW"]), int(r["jsonH"]); ox, oy = o[0], o[1]; exact = (o[2] - o[0], o[3] - o[1]) == (Wc, Hc)
        img = IMG.get((r["folder"], r["stem"]))
        if img is None and not exact: stats["crop_tanpa_citra_dan_ukuran_tak_persis(dilewati)"] += 1; continue
        stats["crop_dipakai_" + ("citra_resmi" if img else "dari_mentah")] += 1; stats["crop_horizontal" if Wc >= Hc else "crop_vertikal"] += 1
        cd = json.load(open(CJ/r["folder"]/f"{r['stem']}.json", encoding="utf-8")); off = []
        for sh in cd.get("shapes", []):
            lab = sh.get("label")
            if lab == WELD: continue
            b = bbox(sh)
            if b[2] - b[0] < 1 or b[3] - b[1] < 1:
                stats["kotak_resmi_degenerat_diabaikan"] += 1; continue
            off.append((lab, b, pts_of(sh, b)))
        rr = [(c, (b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy), [(px - ox, py - oy) for px, py in P]) for c, b, P in raw_real]
        longx = Wc >= Hc
        for (x0, y0, w, h) in windows(Wc, Hc, rng):
            objs, amb, taint, touch, pseudo, unl, offc = [], False, False, False, False, False, []
            for lab, b, P in off:
                ix0, iy0, ix1, iy1 = inter(b, x0, y0, w, h)
                if ix1 <= ix0 or iy1 <= iy0: continue
                if lab in CLS:
                    touch = True                          # negatif tetap konservatif: tidak boleh menyentuh KOTAK cacat mana pun
                    cb, vis = vis_poly(P, b, x0, y0, w, h, longx)   # v4: bagian POLIGON cacat yang benar-benar ada di tile
                    if cb is None: continue
                    offc.append((CLS[lab], cb))
                    if vis >= KEEP: objs.append((CLS[lab], cb[0] - x0, cb[1] - y0, cb[2] - x0, cb[3] - y0))
                    elif vis >= SLIVER: amb = True
                elif lab == PSEUDO: pseudo = True
                else: taint = True
            for c, b, P in rr:   # cacat mentah yang tampak di tile tetapi tak tertutup label resmi kelas sama
                ix0, iy0, ix1, iy1 = inter(b, x0, y0, w, h)
                if ix1 <= ix0 or iy1 <= iy0: continue
                touch = True
                if b[2] - b[0] < 1 or b[3] - b[1] < 1: continue
                cb, vis = vis_poly(P, b, x0, y0, w, h, longx)
                if cb is None or vis < SLIVER: continue
                if not any(k == c and min(ob[2], cb[2]) > max(ob[0], cb[0]) and min(ob[3], cb[3]) > max(ob[1], cb[1]) for k, ob in offc): unl = True
            stats["jendela"] += 1; stats["ambigu"] += amb; stats["tercemar"] += taint; stats["cacat_mentah_tak_berlabel_resmi"] += unl
            pos_ok = bool(objs) and not (amb or taint or unl); neg_ok = (not touch) and not taint
            stats["pos_ok"] += pos_ok; stats["neg_ok"] += neg_ok
            if pos_ok or neg_ok:
                pool.append(dict(film=stem, split=split, batch=batch, crop=r["stem"], folder=r["folder"], src="resmi" if img else "mentah", x=x0, y=y0, w=w, h=h,
                                 X=ox + x0, Y=oy + y0, objs=objs, pos=pos_ok, neg=neg_ok, pseudo=pseudo, W=W, H=H, Wc=Wc, Hc=Hc, orient="H" if Wc >= Hc else "V"))
    return pool

def select(pool, npos, nneg, rng, tag):
    pos = [t for t in pool if t["pos"]]; neg = [t for t in pool if t["neg"]]
    cnt = Counter(o[0] for t in pos for o in t["objs"]); avail = Counter(c for t in pos for c in {o[0] for o in t["objs"]})
    w = {c: math.sqrt(cnt[c]) for c in range(6)}; s = sum(w.values())
    q = {c: max(int(round(npos * w[c] / s)), min(avail[c], int(round(FLOOR * npos)))) for c in range(6)}
    order = sorted(range(6), key=lambda c: cnt[c]); rng.shuffle(pos); rng.shuffle(neg)
    chosen, ids, fn, ct = [], set(), Counter(), Counter()
    def add(t):
        ids.add(id(t)); chosen.append(t); fn[t["film"]] += 1
        for c in {o[0] for o in t["objs"]}: ct[c] += 1
    for cap in range(1, CAP_POS + 1):
        for c in order:
            for t in pos:
                if ct[c] >= q[c] or len(chosen) >= npos: break
                if id(t) not in ids and fn[t["film"]] < cap and any(o[0] == c for o in t["objs"]): add(t)
    for cap in range(1, CAP_POS + 1):
        for t in pos:
            if len(chosen) >= npos: break
            if id(t) not in ids and fn[t["film"]] < cap: add(t)
    nn, fnn = [], Counter()
    for cap in range(1, CAP_NEG + 1):
        for t in neg:
            if len(nn) >= nneg: break
            if id(t) not in ids and fnn[t["film"]] < cap: ids.add(id(t)); nn.append(t); fnn[t["film"]] += 1
    print(f"[{tag}] positif {len(chosen)}/{npos} (tersedia {len(pos)}), negatif {len(nn)}/{nneg} (tersedia {len(neg)}) | kuota/tersedia per kelas:", {NAMES[c]: (q[c], avail[c]) for c in range(6)})
    return chosen + nn

if __name__ == "__main__":
    if any((OUT/"images").rglob("*.png")) or any((OUT/"labels").rglob("*.txt")):   # jangan menimpa/mencampur hasil lama
        sys.exit(f"{OUT} sudah berisi tile lama. Ganti nama folder lama (mis. swrd_tiles_v3) lalu jalankan ulang.")
    t0 = time.time(); rows = list(csv.DictReader(open(ROOT/"swrd_split_v1.csv", encoding="utf-8-sig")))
    stats = Counter(); pools = {"train": [], "val": []}
    for r in sorted(rows, key=lambda r: r["stem"]): pools[r["split"]] += film_pool(r["stem"], r["split"], r["batch"], stats)
    print(f"[plan] {len(rows)} film dibaca ({time.time()-t0:.0f}s). statistik:", dict(stats))
    sel = []
    for k, sp in enumerate(("train", "val")): sel += select(pools[sp], NPOS[sp], NNEG[sp], np.random.default_rng(SEED + 17 + k), sp)
    for t in sel: t["name"] = f"{t['crop']}_x{t['x']}_y{t['y']}"
    assert len({t["name"] for t in sel}) == len(sel), "nama tile ganda"
    OUT.mkdir(exist_ok=True)
    HDR = ["name", "split", "film", "batch", "folder", "crop", "src", "orient", "x", "y", "X", "Y", "w", "h"]
    def write_manifest(items):
        with open(OUT/"manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(HDR + ["positif", "pseudo"] + [f"n_{n}" for n in NAMES])
            for t in items:
                c = Counter(o[0] for o in t["objs"]); w.writerow([t[k] for k in HDR] + [int(bool(t["objs"])), int(t["pseudo"])] + [c[i] for i in range(6)])
    def summarize(items, extra=""):
        lines = []
        for sp in ("train", "val"):
            S = [t for t in items if t["split"] == sp]
            if not S: continue
            P = [t for t in S if t["objs"]]; N = [t for t in S if not t["objs"]]
            co = Counter(o[0] for t in P for o in t["objs"]); ct = Counter(c for t in P for c in {o[0] for o in t["objs"]})
            bt = Counter(t["batch"] for t in S); ft = Counter(t["film"] for t in S)
            lines.append(f"{sp}: tile {len(S)} (positif {len(P)}, negatif {len(N)}); film berbeda {len(ft)}, maks tile/film {max(ft.values())}; batch berbeda {len(bt)}, batch terbesar {max(bt.values())/len(S):.1%}; negatif ber-pseudo-defect {sum(t['pseudo'] for t in N)/max(len(N),1):.1%}; tile vertikal {sum(t['orient']=='V' for t in S)}; piksel dari mentah (crop tanpa citra) {sum(t['src']=='mentah' for t in S)}")
            lines.append("   objek per kelas: " + ", ".join(f"{NAMES[c]} {co[c]}" for c in range(6)) + " | tile per kelas: " + ", ".join(f"{NAMES[c]} {ct[c]}" for c in range(6)))
            lines.append("   FILM per kelas: " + ", ".join(f"{NAMES[c]} {len({t['film'] for t in P if any(o[0]==c for o in t['objs'])})}" for c in range(6)))
            lines.append(f"   tile lebar min/median/maks: {min(t['w'] for t in S)}/{int(np.median([t['w'] for t in S]))}/{max(t['w'] for t in S)}; tinggi min/median/maks: {min(t['h'] for t in S)}/{int(np.median([t['h'] for t in S]))}/{max(t['h'] for t in S)}")
        print("\n".join(lines)); open(OUT/"summary.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n" + json.dumps(dict(stats), ensure_ascii=False) + "\n" + extra)
    write_manifest(sel); summarize(sel)
    if PLAN_ONLY: print("--plan-only: selesai (manifest.csv dan summary.txt ditulis)."); sys.exit()

    # ---------- render ----------
    try: import cv2
    except ImportError: sys.exit("butuh OpenCV: pip install opencv-python-headless")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for sp in ("train", "val"):
        (OUT/"images"/sp).mkdir(parents=True, exist_ok=True); (OUT/"labels"/sp).mkdir(parents=True, exist_ok=True)
    def prep(a):
        lo, hi = np.percentile(a, (0.5, 99.5)); b = np.zeros(a.shape, np.uint8) if hi <= lo else (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        return clahe.apply(b)
    byfilm = defaultdict(list)
    for t in sel: byfilm[t["film"]].append(t)
    rawp = {p.stem: p for p in RAW_I.iterdir() if p.is_file() and p.stem in byfilm}; dropped = []; t0 = time.time()
    for n, (film, ts) in enumerate(sorted(byfilm.items())):
        raw = np.asarray(Image.open(rawp[film])) if film in rawp else None
        if raw is not None and raw.shape != (ts[0]["H"], ts[0]["W"]): raw = None; print("   ukuran citra mentah tidak cocok JSON:", film)
        cache = {}
        for t in ts:
            key = (t["folder"], t["crop"])
            if t["src"] == "resmi":
                if key not in cache:
                    a = np.asarray(Image.open(IMG[key])); cache[key] = a if a.shape == (t["Hc"], t["Wc"]) else None
                    if cache[key] is None: print("   ukuran citra crop resmi tidak cocok JSON:", key, a.shape)
                if cache[key] is None: dropped.append((t["name"], "ukuran crop resmi")); continue
                patch = cache[key][t["y"]:t["y"] + t["h"], t["x"]:t["x"] + t["w"]]
                if raw is not None and not np.array_equal(patch, raw[t["Y"]:t["Y"] + t["h"], t["X"]:t["X"] + t["w"]]): dropped.append((t["name"], "tidak identik dengan mentah")); continue
            else:
                if raw is None: dropped.append((t["name"], "mentah tak terbaca")); continue
                patch = raw[t["Y"]:t["Y"] + t["h"], t["X"]:t["X"] + t["w"]]
            if patch.shape != (t["h"], t["w"]): dropped.append((t["name"], f"bentuk {patch.shape}")); continue
            Image.fromarray(prep(patch.astype(np.float32))).save(OUT/"images"/t["split"]/f"{t['name']}.png", compress_level=6)
            with open(OUT/"labels"/t["split"]/f"{t['name']}.txt", "w") as f:
                for c, x0, y0, x1, y1 in t["objs"]: f.write(f"{c} {(x0+x1)/2/t['w']:.6f} {(y0+y1)/2/t['h']:.6f} {(x1-x0)/t['w']:.6f} {(y1-y0)/t['h']:.6f}\n")
        if n % 200 == 0: print(f"   render film {n}/{len(byfilm)}  {time.time()-t0:.0f}s", flush=True)
    dn = {d[0] for d in dropped}; sel = [t for t in sel if t["name"] not in dn]
    write_manifest(sel); summarize(sel, extra="tile dibuang saat render: " + json.dumps(Counter(d[1] for d in dropped), ensure_ascii=False) + "\n" + "\n".join(f"{a}\t{b}" for a, b in dropped) + "\n")
    open(OUT/"data.yaml", "w", encoding="utf-8").write("# path: diisi saat dipakai (mis. /kaggle/input/swrd-tiles-v1)\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    rg = np.random.default_rng(SEED); posv = [t for t in sel if t["objs"]]; hh = [t for t in posv if t["orient"] == "H"]; vv = [t for t in posv if t["orient"] == "V"]
    pick = [hh[i] for i in rg.permutation(len(hh))[:8]] + [vv[i] for i in rg.permutation(len(vv))[:4]]; COL = [(255, 0, 0), (0, 200, 0), (0, 120, 255), (255, 200, 0), (255, 0, 255), (0, 220, 220)]; cells = []
    for t in pick:
        im = Image.open(OUT/"images"/t["split"]/f"{t['name']}.png").convert("RGB"); d = ImageDraw.Draw(im)
        for c, x0, y0, x1, y1 in t["objs"]: d.rectangle([x0, y0, x1, y1], outline=COL[c], width=2); d.text((x0 + 2, max(y0 - 11, 0)), NAMES[c][:5], fill=COL[c])
        sc = 420 / max(im.size); cells.append(im.resize((int(im.size[0] * sc), int(im.size[1] * sc))))
    sheet = Image.new("RGB", (4 * 430, 3 * 430), (30, 30, 30))
    for i, im in enumerate(cells): sheet.paste(im, ((i % 4) * 430, (i // 4) * 430))
    sheet.save(OUT/"overlay_sheet.png")
    print(f"selesai. tile dibuang saat render: {len(dropped)} {dict(Counter(d[1] for d in dropped))}. keluaran: {OUT} (periksa overlay_sheet.png)")
