"""Tahap 4: bedakan 'film kembar' dari 'pola bersama' pada pasangan NCC tertinggi.
Masukan: swrd_split_v1.csv dan swrd_audit3_ncc_top.csv (di folder ini, sebelah Raw_data). Keluaran: teks + swrd_audit4_panel.png"""
import csv, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent; rawI = ROOT/"Raw_data"/"images"
N_TOP, N_RAND, DS, HP, SHAPE, P, SRCH = 24, 16, 16, 9, (144, 576), 384, 40
top = list(csv.DictReader(open(ROOT/"swrd_audit3_ncc_top.csv", encoding="utf-8-sig")))
split = {r["stem"]: r["split"] for r in csv.DictReader(open(ROOT/"swrd_split_v1.csv", encoding="utf-8-sig"))}
paths = {p.stem: p for p in rawI.iterdir() if p.is_file() and p.stem in split}
pairs = [(float(r["ncc"]), r["val"], r["train"]) for r in top[:N_TOP]]
from collections import Counter
cv = Counter(r["val"] for r in top); ct = Counter(r["train"] for r in top)
print(f"[0] dari {len(top)} baris teratas: film val berbeda {len(cv)}, film train berbeda {len(ct)}; film train paling sering muncul: {ct.most_common(3)}; val paling sering: {cv.most_common(3)}")
rng = np.random.default_rng(1); vals = [s for s in split if split[s] == "val"]; trs = [s for s in split if split[s] == "train"]
rand = [(None, rng.choice(vals), rng.choice(trs)) for _ in range(N_RAND)]
def load(s): return np.asarray(Image.open(paths[s])).astype(np.float32)
def boxblur(a, k):
    p = k // 2; a = np.pad(a, p, mode="edge"); c = np.cumsum(np.cumsum(a, 0, dtype=np.float64), 1, dtype=np.float64); c = np.pad(c, ((1, 0), (1, 0)))
    return ((c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)).astype(np.float32)
def ds_sig(a):
    h, w = a.shape[0] // DS * DS, a.shape[1] // DS * DS; b = a[:h, :w].reshape(h // DS, DS, w // DS, DS).mean((1, 3)); b = b - boxblur(b, HP)
    b = (b - b.mean()) / (b.std() + 1e-6); o = np.zeros(SHAPE, np.float32); hh, ww = min(b.shape[0], SHAPE[0]), min(b.shape[1], SHAPE[1]); o[:hh, :ww] = b[:hh, :ww]; return o
def align(a, b):
    sa, sb = ds_sig(a), ds_sig(b); c = np.fft.irfft2(np.fft.rfft2(sa) * np.conj(np.fft.rfft2(sb)), s=SHAPE)
    i = int(np.argmax(c)); dy, dx = np.unravel_index(i, SHAPE); dy = dy - SHAPE[0] if dy > SHAPE[0] // 2 else dy; dx = dx - SHAPE[1] if dx > SHAPE[1] // 2 else dx
    return float(c.max() / np.sqrt((sa**2).sum() * (sb**2).sum())), int(dy) * DS, int(dx) * DS
def patch_ncc(ha, hb, dy, dx):
    """ha=val, hb=train (sudah high-pass, resolusi asli). Titik val p cocok dengan titik train p-(dy,dx), cari ±SRCH."""
    H, W = ha.shape; out = []; cy = H // 2
    for cx in np.linspace(P, W - P, 7):
        y0, x0 = int(cy - P // 2), int(cx - P // 2); ty, tx = y0 - dy - SRCH, x0 - dx - SRCH; R = P + 2 * SRCH
        if ty < 0 or tx < 0 or ty + R > hb.shape[0] or tx + R > hb.shape[1] or y0 + P > H or x0 + P > W: continue
        A = ha[y0:y0 + P, x0:x0 + P]; A = A - A.mean(); Rg = hb[ty:ty + R, tx:tx + R]
        FA = np.fft.rfft2(A, s=(R, R)); FR = np.fft.rfft2(Rg); c = np.fft.irfft2(FR * np.conj(FA), s=(R, R))[:2 * SRCH + 1, :2 * SRCH + 1]
        ii = np.pad(np.cumsum(np.cumsum(Rg.astype(np.float64) ** 2, 0), 1), ((1, 0), (1, 0))); sq = ii[P:P + 2*SRCH+1, P:P+2*SRCH+1] - ii[:2*SRCH+1, P:P+2*SRCH+1] - ii[P:P+2*SRCH+1, :2*SRCH+1] + ii[:2*SRCH+1, :2*SRCH+1]
        # catatan: energi lokal dihitung dari Rg tanpa pengurangan rerata; Rg sudah high-pass (rerata ~0)
        out.append(float((c / np.sqrt((A.astype(np.float64) ** 2).sum() * np.maximum(sq, 1e-9))).max()))
    return out
def clean(a):
    h = a - boxblur(a, 31)                                   # high-pass resolusi asli
    return h - h.mean(0, keepdims=True) - h.mean(1, keepdims=True)   # buang pola garis kolom/baris sepanjang film (artefak pemindai)
def run(plist, tag):
    res = []
    for ncc, v, t in plist:
        a, b = load(v), load(t); n2, dy, dx = align(a, b); pn = patch_ncc(clean(a), clean(b), dy, dx)
        res.append((v, t, n2, dy, dx, pn)); m = f"{np.median(pn):.3f}" if pn else "n/a"
        print(f"[{tag}] {v} ~ {t} | NCC x16 {n2:.3f} geser(px) {dy},{dx} | NCC patch asli: median {m} maks {max(pn):.3f} (n={len(pn)})" if pn else f"[{tag}] {v} ~ {t} | NCC x16 {n2:.3f} geser {dy},{dx} | patch tak tumpang tindih")
    return res
R1 = run(pairs, "TOP"); R2 = run(rand, "ACAK")
med = lambda R: [np.median(r[5]) for r in R if r[5]]
print(f"[ringkas] TOP: median patch NCC {np.median(med(R1)):.3f} (maks {np.max(med(R1)):.3f}) | ACAK: median {np.median(med(R2)):.3f} (maks {np.max(med(R2)):.3f})")
# panel visual: 10 pasangan teratas, val di atas, train di bawah (kontras diregangkan)
def thumb(s, W=560):
    a = load(s); a = a[:a.shape[0] // 8 * 8, :a.shape[1] // 8 * 8].reshape(a.shape[0] // 8, 8, a.shape[1] // 8, 8).mean((1, 3)); lo, hi = np.percentile(a, [1, 99.5]); im = Image.fromarray((np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8))
    return im.resize((W, max(40, int(im.size[1] * W / im.size[0]))))
rows = []
for (v, t, n2, dy, dx, pn) in R1[:10]:
    a, b = thumb(v), thumb(t); h = a.size[1] + b.size[1] + 18; im = Image.new("L", (560, h), 0); d = ImageDraw.Draw(im)
    d.text((2, 0), f"VAL {v}", fill=255); im.paste(a, (0, 9)); d.text((2, 9 + a.size[1]), f"TRAIN {t}  NCC16={n2:.2f} patch={np.median(pn) if pn else float('nan'):.2f}", fill=255); im.paste(b, (0, 18 + a.size[1])); rows.append(im)
Hs = [r.size[1] for r in rows]; sheet = Image.new("L", (1130, sum(max(Hs[i], Hs[i + 1]) + 6 for i in range(0, len(rows) - 1, 2))), 40); y = 0
for i in range(0, len(rows) - 1, 2):
    sheet.paste(rows[i], (0, y)); sheet.paste(rows[i + 1], (570, y)); y += max(Hs[i], Hs[i + 1]) + 6
sheet.save(ROOT/"swrd_audit4_panel.png"); print("panel disimpan: swrd_audit4_panel.png")
