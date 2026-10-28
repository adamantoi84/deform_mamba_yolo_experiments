"""Split SWRD v1: unit = batch tanggal, ~90/10 train/val, stratifikasi jumlah film per kelas.
Masukan: in_labels.csv (swrd_audit_raw.csv), in_inventory.csv, in_dup.csv. Keluaran: swrd_split_v1.csv, swrd_excluded_v1.csv."""
import pandas as pd, numpy as np, re, sys
SEED_LIST = range(40); VAL_FRAC = 0.10
lab = pd.read_csv("in_labels.csv", encoding="utf-8-sig"); inv = pd.read_csv("in_inventory.csv", encoding="utf-8-sig"); du = pd.read_csv("in_dup.csv", encoding="utf-8-sig")
conf_groups = set(du.loc[~du.json_identik, "group"])
conf_films = set(du[du.group.isin(conf_groups) & (du.batch == "DJ-RT-20230321")].stem)          # salinan batch awal dari pasangan berlabel konflik
dup_copy = set(du[du.batch == "DJ-RT-20230909"].stem)                                           # seluruh batch 20230909 = salinan 20230321
reason = {}
for _, r in inv.iterrows():
    s = r.stem
    if r.ext == ".db": reason[s] = "bukan citra (Thumbs.db)"
    elif r.src != "DJ-RT": reason[s] = "sumber non-DJ-RT (RRT/bam)"
    elif r["mode"] == "L": reason[s] = "8-bit (domain pindaian lain)"
    elif r["mode"] == "ERR": reason[s] = "berkas tidak terbaca"
    elif s in dup_copy: reason[s] = "salinan identik (md5) dari batch 20230321"
    elif s in conf_films: reason[s] = "duplikat identik dengan label konflik"
    elif not r.has_json: reason[s] = "tanpa JSON"
ex = pd.DataFrame(sorted(reason.items()), columns=["stem", "alasan"]); ex.to_csv("swrd_excluded_v1.csv", index=False, encoding="utf-8-sig")
keep = lab[~lab.stem.isin(reason)].copy()
cls = {"气孔": "por", "夹渣": "incl", "裂纹": "crack", "咬边": "under", "未熔合": "LoF", "未焊透": "LoP"}
X = pd.DataFrame({v: (keep[f"n[{k}]"] > 0).astype(int) for k, v in cls.items()}); X["films"] = 1
yr = keep.batch.str.extract(r"(\d{4})")[0]
for y in ("2022", "2023", "2024"): X["y" + y] = (yr == y).astype(int).values
X["batch"] = keep.batch.values
B = X.groupby("batch").sum(); M = B.values.astype(float); T = M.sum(0)
w = np.ones(M.shape[1]); w[list(B.columns).index("films")] = 4
def loss(a):
    v = M[a == 1].sum(0); return float((w * ((v - VAL_FRAC * T) / (VAL_FRAC * T + 1)) ** 2).sum())
best = None
for seed in SEED_LIST:
    rng = np.random.default_rng(seed); a = (rng.random(len(B)) < VAL_FRAC).astype(int); cur = loss(a)
    for _ in range(20000):
        i = rng.integers(len(B)); a[i] ^= 1; n = loss(a)
        if n <= cur: cur = n
        else: a[i] ^= 1
    if best is None or cur < best[0]: best = (cur, seed, a.copy())
cur, seed, a = best
smap = dict(zip(B.index, np.where(a == 1, "val", "train")))
keep["split"] = keep.batch.map(smap)
keep[["stem", "batch", "split"]].to_csv("swrd_split_v1.csv", index=False, encoding="utf-8-sig")
print("seed", seed, "loss", round(cur, 4), "| excluded:", len(ex), "| kept films:", len(keep), "| batches:", len(B))
for s in ("train", "val"):
    v = pd.Series(M[(np.array([smap[b] for b in B.index]) == s)].sum(0), index=B.columns).astype(int)
    print(s, "batches", int((pd.Series(smap) == s).sum()), v.to_dict())
