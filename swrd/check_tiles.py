"""check_tiles.py -- pemeriksaan akhir swrd_tiles sebelum diunggah ke Kaggle. Taruh di folder yang sama. python check_tiles.py [--zip]"""
import csv, sys, hashlib, zipfile
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent; T = ROOT/"swrd_tiles"; E = []
def bad(m):
    E.append(m)
    if len(E) <= 20: print("GAGAL:", m)
man = list(csv.DictReader(open(T/"manifest.csv", encoding="utf-8-sig")))
split = {r["stem"]: r for r in csv.DictReader(open(ROOT/"swrd_split_v1.csv", encoding="utf-8-sig"))}
NAMES = ["porosity", "inclusion", "crack", "undercut", "lack_of_fusion", "lack_of_penetration"]
md5 = defaultdict(list); fs = defaultdict(set); bs = defaultdict(set); ncls = Counter()
for m in man:
    sp, nm = m["split"], m["name"]; fs[m["film"]].add(sp); bs[m["batch"]].add(sp)
    if split.get(m["film"], {}).get("split") != sp: bad(f"split tile != swrd_split_v1: {nm}")
    ip, lp = T/"images"/sp/f"{nm}.png", T/"labels"/sp/f"{nm}.txt"
    if not ip.exists() or not lp.exists(): bad(f"berkas hilang {nm}"); continue
    im = Image.open(ip)
    if im.mode != "L" or im.size != (int(m["w"]), int(m["h"])): bad(f"mode/ukuran {nm} {im.mode} {im.size}")
    a = np.asarray(im)
    if a.std() < 1: bad(f"citra datar {nm}")
    md5[hashlib.md5(a.tobytes()).hexdigest()].append(sp + "/" + nm)
    rows = [l.split() for l in lp.read_text().splitlines() if l.strip()]
    if (len(rows) > 0) != (m["positif"] == "1"): bad(f"positif/label tak konsisten {nm}")
    c = Counter()
    for r in rows:
        if len(r) != 5: bad(f"format label {nm}"); continue
        k = int(r[0]); x, y, w, h = map(float, r[1:]); c[k] += 1; ncls[(sp, k)] += 1
        if not (0 <= k <= 5 and w > 0 and h > 0 and x - w / 2 >= -1e-6 and y - h / 2 >= -1e-6 and x + w / 2 <= 1 + 1e-6 and y + h / 2 <= 1 + 1e-6): bad(f"kotak di luar [0,1] {nm} {r}")
    if any(c[i] != int(m[f"n_{NAMES[i]}"]) for i in range(6)): bad(f"jumlah objek != manifest {nm}")
for sp in ("train", "val"):
    want = {m["name"] for m in man if m["split"] == sp}
    if {p.stem for p in (T/"images"/sp).glob("*")} != want: bad(f"isi images/{sp} != manifest")
    if {p.stem for p in (T/"labels"/sp).glob("*")} != want: bad(f"isi labels/{sp} != manifest")
if any(len(v) > 1 for v in fs.values()): bad("film muncul di train dan val")
if any(len(v) > 1 for v in bs.values()): bad("batch muncul di train dan val")
dup = [v for v in md5.values() if len(v) > 1]
print(f"tile {len(man)} | duplikat piksel identik: {len(dup)} kelompok (lintas split: {sum(len({x.split('/')[0] for x in v}) > 1 for v in dup)}) {dup[:3]}")
print("objek per kelas:", {sp: {NAMES[k]: ncls[(sp, k)] for k in range(6)} for sp in ("train", "val")})
print("HASIL:", "LULUS" if not E else f"GAGAL ({len(E)} masalah)")
if "--zip" in sys.argv and not E:
    z = ROOT/"swrd_tiles_v1.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as f:
        for p in sorted(T.rglob("*")):
            if p.is_file(): f.write(p, p.relative_to(T))
    print(f"zip: {z} ({z.stat().st_size/1e9:.2f} GB)")
