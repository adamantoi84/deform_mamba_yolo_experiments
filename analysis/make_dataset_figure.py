#!/usr/bin/env python3
"""
make_dataset_figure.py: Fig. 2 (ground-truth examples, Section IV-A), from the examples chosen in notebook
Paper_2_fase_15_contoh_citra_dataset.

Inputs, in the folder given by the CONTOH environment variable (default ./contoh_dataset):
    pilihan.json                     the chosen examples (released as results/dataset_figure/pilihan.json)
    asli/SP/<file>.jpg + <stem>.txt  steel-pipe validation images and labels, as distributed (not released here)
    asli/SW/<file>.png + <stem>.txt  SWRD validation tiles and labels, built with swrd/build_swrd_tiles.py
Output: figures/fig_dataset.pdf and .png
Checks: the SHA-256 of every image, its size, and its boxes must equal those recorded in pilihan.json. A panel whose
image is missing is drawn as a placeholder, and its file name is printed.

    CONTOH=results/dataset_figure python analysis/make_dataset_figure.py   # placeholders only (no images)

Comments below are in Indonesian.
"""
import hashlib, json, os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

BASE = os.environ.get("CONTOH", "contoh_dataset")   # folder with pilihan.json and asli/SP, asli/SW (images not released)
OUT = "figures"; os.makedirs(OUT, exist_ok=True)
INK, MUTED = "#0b0b0b", "#52514e"
TARGET, LAIN = "#ffd400", "#33c3ff"
IMGSZ, FAKTOR, MIN_IN = 640, 3.0, 64
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "text.color": INK,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600})

M = json.load(open(f"{BASE}/pilihan.json"))
assert M["split"] == "val" and M["seed"] == 2026 and M["imgsz"] == IMGSZ
URUT = {"steel-pipe": ["air-hole", "bite-edge", "broken-arc", "crack", "hollow-bead", "overlap", "slag-inclusion",
                       "unfused"],
        "SWRD": ["porosity", "inclusion", "crack", "undercut", "lack of fusion", "lack of penetration"]}
assert M["steel_names"] == URUT["steel-pipe"]
assert sorted(M["swrd_names"].values()) == sorted(URUT["SWRD"])
P = {(p["dataset"], p["kelas"]): p for p in M["pilihan"]}
# Klaim caption: contoh dipilih penulis di antara 8 citra pertama urutan acak (halaman 0 notebook) tiap kelas;
# potongan (a) = 3 x sisi panjang kotak, min 64 px masukan jaringan.
assert all(1 <= p["rank_acak"] <= 8 for p in M["pilihan"]), "ada pilihan di luar 8 kandidat pertama"
assert M["zoom_faktor"] == FAKTOR == 3.0 and M["zoom_min_in"] == MIN_IN == 64
assert len(P) == len(M["pilihan"]) == 14, "harus tepat satu pilihan per kelas (8 + 6)"
SUB = {"steel-pipe": "SP", "SWRD": "SW"}
NAMES = {"steel-pipe": M["steel_names"], "SWRD": [M["swrd_names"][str(i)] for i in range(6)]}


def muat(p):
    """Citra asli + verifikasi; None bila belum tersedia."""
    f = f"{BASE}/asli/{SUB[p['dataset']]}/{p['file']}"
    if not os.path.exists(f):
        return None
    assert hashlib.sha256(open(f, "rb").read()).hexdigest() == p["sha256"], f"sha256 berbeda: {f}"
    im = Image.open(f)
    assert im.size == (p["W"], p["H"]), f"ukuran berbeda: {f}"
    lab = os.path.splitext(f)[0] + ".txt"
    if os.path.exists(lab):
        bx = []
        for ln in open(lab).read().splitlines():
            if ln.strip():
                c, xc, yc, w, h = ln.split(); xc, yc, w, h = map(float, (xc, yc, w, h))
                bx.append((NAMES[p["dataset"]][int(c)], (xc - w / 2) * p["W"], (yc - h / 2) * p["H"],
                           (xc + w / 2) * p["W"], (yc + h / 2) * p["H"]))
        js = [(b["kelas"], b["x1"], b["y1"], b["x2"], b["y2"]) for b in p["semua_kotak"]]
        assert len(bx) == len(js) and all(a[0] == b[0] and np.allclose(a[1:], b[1:], atol=0.02) for a, b in zip(bx, js)), \
            f"label .txt tidak sama dengan pilihan.json: {lab}"
    return np.asarray(im.convert("L"))


def potong(p):
    g = p["kotak_zoom"]; W, H = p["W"], p["H"]; s = IMGSZ / max(W, H)
    sisi = min(max(FAKTOR * max(g["w_in"], g["h_in"]), MIN_IN) / s, min(W, H))
    cx, cy = (g["x1"] + g["x2"]) / 2, (g["y1"] + g["y2"]) / 2
    return min(max(cx - sisi / 2, 0), W - sisi), min(max(cy - sisi / 2, 0), H - sisi), sisi


def kotak(ax, p, lw):
    for b in sorted(p["semua_kotak"], key=lambda b: b["kelas"] == p["kelas"]):   # kelas panel digambar terakhir
        t = b["kelas"] == p["kelas"]
        r = mpatches.Rectangle((b["x1"], b["y1"]), b["x2"] - b["x1"], b["y2"] - b["y1"], fill=False,
                               edgecolor=TARGET if t else LAIN, lw=lw if t else 0.75 * lw,
                               ls="-" if t else (0, (2.5, 1.5)), joinstyle="miter")
        r.set_path_effects([pe.withStroke(linewidth=(lw if t else 0.75 * lw) + 0.7, foreground="black")])
        ax.add_patch(r)


def panel(ax, ds, kelas, crop):
    p = P[(ds, kelas)]
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4); sp.set_color(MUTED)
    g = p["kotak_zoom"]
    ket = f"{g['w_in']:.0f}×{g['h_in']:.0f} px, AR {g['ar']:.1f}"
    a = muat(p)
    if a is None:
        print(f"  placeholder: {p['id']} ({p['file']}) belum ada")
        ax.set_facecolor("#e4e3df"); ax.set_box_aspect(1)
        ax.text(0.5, 0.5, "menunggu\ncitra", ha="center", va="center", fontsize=6, color=MUTED, transform=ax.transAxes)
    else:
        ax.imshow(a, cmap="gray", vmin=0, vmax=255, interpolation="bilinear" if crop else "antialiased")
        kotak(ax, p, 0.6 if crop else 0.5)
        if crop:
            x0, y0, s = potong(p)
            ax.set_xlim(x0, x0 + s); ax.set_ylim(y0 + s, y0)
        else:
            ax.set_xlim(0, p["W"]); ax.set_ylim(p["H"], 0)
    # label dlm offset POIN dari tepi bawah sumbu -> jarak sama utk panel setinggi apa pun
    ax.annotate(kelas, (0.5, 0), xycoords="axes fraction", xytext=(0, -2.5), textcoords="offset points",
                ha="center", va="top", fontsize=7.5)
    ax.annotate(ket, (0.5, 0), xycoords="axes fraction", xytext=(0, -11.5), textcoords="offset points",
                ha="center", va="top", fontsize=6.5, color=MUTED)
    return p


# Tata letak dalam inci (lebar 2 kolom IEEE = 7.16 in): panel persegi selebar mungkin, dasar panel sejajar.
WFIG, GAP, TIT, LAB, SEP = 7.16, 0.07, 0.17, 0.27, 0.10
A = (WFIG - 7 * GAP) / 8                     # sisi panel (a)
B = (WFIG - 5 * GAP) / 6                     # lebar panel (b)
HFIG = TIT + A + LAB + SEP + TIT + B + LAB
fig = plt.figure(figsize=(WFIG, HFIG))
def kotak_ax(x, y_bawah, w, h):              # inci -> pecahan gambar
    return fig.add_axes([x / WFIG, y_bawah / HFIG, w / WFIG, h / HFIG])
ya = HFIG - TIT - A                          # dasar baris (a)
for i, k in enumerate(URUT["steel-pipe"]):
    panel(kotak_ax(i * (A + GAP), ya, A, A), "steel-pipe", k, crop=True)
yb = LAB                                     # dasar baris (b)
for i, k in enumerate(URUT["SWRD"]):
    p = P[("SWRD", k)]
    h = B * p["H"] / p["W"] if p["W"] >= p["H"] else B
    panel(kotak_ax(i * (B + GAP), yb, B, h), "SWRD", k, crop=False)
fig.text(0, 1.0, "(a) Steel-pipe weld dataset", ha="left", va="top", fontsize=8, fontweight="bold")
fig.text(0, (yb + B + TIT) / HFIG, "(b) SWRD tiles", ha="left", va="top", fontsize=8, fontweight="bold")
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig_dataset.{ext}", pad_inches=0.01, bbox_inches="tight")
print("ditulis figures/fig_dataset")
for p in M["pilihan"]:
    print(f"  {p['id']:28s} rank {p['rank_acak']:2d}/{p['pool']:3d}  {p['file']}")
