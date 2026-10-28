"""Cek kebocoran val<->train SWRD (tahap 3). Taruh di folder yang berisi Raw_data dan swrd_split_v1.csv.
A) kemiripan geometri label (hanya JSON)   B) korelasi silang 2D citra (high-pass, x16) val vs train, dengan kontrol sintetis."""
import json, sys, csv, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
rawI, rawJ = ROOT/"Raw_data"/"images", ROOT/"Raw_data"/"json"
CLS = {"气孔", "夹渣", "裂纹", "咬边", "未熔合", "未焊透"}
rows = list(csv.DictReader(open(ROOT/"swrd_split_v1.csv", encoding="utf-8-sig")))
split = {r["stem"]: r["split"] for r in rows}; batch = {r["stem"]: r["batch"] for r in rows}
paths = {p.stem: p for p in rawI.iterdir() if p.is_file() and p.stem in split}
print(f"film: {len(split)} (train {sum(v=='train' for v in split.values())}, val {sum(v=='val' for v in split.values())}); berkas citra ditemukan: {len(paths)}")

# ---------- A. geometri label ----------
def boxes(s):
    d = json.load(open(rawJ/f"{s}.json", encoding="utf-8")); out = set()
    for x in d["shapes"]:
        if x["label"] in CLS:
            a = np.array(x["points"]); out.add((x["label"], *[int(round(v / 2)) for v in (a[:,0].min(), a[:,1].min(), a[:,0].max(), a[:,1].max())]))
    return out
B = {s: boxes(s) for s in split}; inv = defaultdict(list)
for s, bs in B.items():
    for b in bs: inv[b].append(s)
share = Counter()
for b, L in inv.items():
    if 1 < len(L) < 6:
        for i in range(len(L)):
            for j in range(i+1, len(L)): share[(L[i], L[j])] += 1
hits = []
for (a, b), k in share.items():
    m = min(len(B[a]), len(B[b]))
    if k >= 2 or (k >= 1 and m <= 2): hits.append((a, b, k, m))
print(f"[A] pasangan film dengan kotak cacat identik (±2px): {len(hits)}; lintas-batch: {sum(batch[a]!=batch[b] for a,b,_,_ in hits)}; lintas-split train/val: {sum(split[a]!=split[b] for a,b,_,_ in hits)}")
with open(ROOT/"swrd_audit3_labeldup.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["a","b","batch_a","batch_b","split_a","split_b","kotak_sama","kotak_min"])
    for a, b, k, m in sorted(hits, key=lambda t: -t[2]): w.writerow([a, b, batch[a], batch[b], split[a], split[b], k, m])
print("[A] pasangan lintas-batch (maks 10):", [(a, b, k) for a, b, k, m in hits if batch[a] != batch[b]][:10])

# ---------- B. korelasi silang 2D ----------
DS, HP, SHAPE = 16, 9, (144, 576)
def boxblur(a, k):
    p = k // 2; a = np.pad(a, p, mode="edge"); c = np.cumsum(np.cumsum(a, 0), 1); c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)
def sig(p):
    a = np.asarray(Image.open(p)).astype(np.float32); h, w = a.shape[0] // DS * DS, a.shape[1] // DS * DS
    a = a[:h, :w].reshape(h // DS, DS, w // DS, DS).mean((1, 3)); return a - boxblur(a, HP)
def pad(a):
    a = (a - a.mean()) / (a.std() + 1e-6); o = np.zeros(SHAPE, np.float32); h, w = min(a.shape[0], SHAPE[0]), min(a.shape[1], SHAPE[1]); o[:h, :w] = a[:h, :w]; return o, (h, w)
t0 = time.time(); S = {}; DIM = {}
for n, s in enumerate(sorted(paths)):
    try: S[s], DIM[s] = pad(sig(paths[s]))
    except Exception as e: print("   gagal baca", s, e)
    if n % 300 == 0: print(f"   baca {n}/{len(paths)}  {time.time()-t0:.0f}s", flush=True)
names = sorted(S); F = {s: np.fft.rfft2(S[s]).astype(np.complex64) for s in names}; E = {s: float((S[s] ** 2).sum()) for s in names}
tr = [s for s in names if split[s] == "train"]; va = [s for s in names if split[s] == "val"]
Ftr = np.stack([F[s] for s in tr]); Etr = np.array([E[s] for s in tr])
def peaks(Fi, Ei):
    out = np.empty(len(tr), np.float32)
    for k in range(0, len(tr), 400):
        c = np.fft.irfft2(Fi[None] * np.conj(Ftr[k:k+400]), s=SHAPE, axes=(1, 2))
        out[k:k+400] = c.reshape(len(c), -1).max(1) / np.sqrt(Ei * Etr[k:k+400])
    return out
allp, top = [], []
for n, s in enumerate(va):
    p = peaks(F[s], E[s]); allp.append(p); j = np.argsort(-p)[:3]
    top += [(float(p[k]), s, tr[k]) for k in j]
    if n % 40 == 0: print(f"   val {n}/{len(va)}  {time.time()-t0:.0f}s", flush=True)
allp = np.concatenate(allp); top.sort(reverse=True)
print(f"[B] pasangan val x train: {len(allp)} | peak NCC kuantil 50/99/99.9/maks: {np.quantile(allp,[.5,.99,.999]).round(3).tolist()} {allp.max():.3f}")
# kontrol sintetis: salinan bergeser + derau, dibandingkan dengan aslinya
rng = np.random.default_rng(0); ctl = []
for s in rng.choice(va, size=min(30, len(va)), replace=False):
    dy, dx = int(rng.integers(-6, 7)), int(rng.integers(-40, 41)); h, w = DIM[s]; a = S[s].copy(); a[:h, :w] += rng.normal(0, 1.0, (h, w)).astype(np.float32); a = np.roll(a, (dy, dx), (0, 1))
    Fa = np.fft.rfft2(a).astype(np.complex64); c = np.fft.irfft2(Fa * np.conj(F[s]), s=SHAPE).max() / np.sqrt(float((a**2).sum()) * E[s]); ctl.append(float(c))
print(f"[B] kontrol sintetis (geser<=±6/±40 sel + derau SNR=1): NCC min/median {min(ctl):.3f}/{np.median(ctl):.3f}  -> ambang deteksi harus < min kontrol dan > kuantil 99.9 pasangan acak")
with open(ROOT/"swrd_audit3_ncc_top.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["ncc","val","train","batch_val","batch_train"])
    for p, a, b in top[:300]: w.writerow([round(p, 4), a, b, batch[a], batch[b]])
print("[B] 15 pasangan teratas:"); [print(f"   {p:.3f} {a} ~ {b}  ({batch[a]} | {batch[b]})") for p, a, b in top[:15]]
