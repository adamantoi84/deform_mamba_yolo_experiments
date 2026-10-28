"""
diagnostik_geometri_swrd.py  --  batas struktural head YOLOv8/Mamba-YOLO pada kotak SWRD tiles

Tanpa GPU, tanpa torch, tanpa membuka piksel (ukuran gambar dibaca dari header PNG).
Hanya butuh numpy. Menjawab: berapa bagian kotak GT yang SECARA GEOMETRIS tidak bisa
dipelajari / tidak bisa diprediksi dengan IoU >= 0.5 oleh head Detect (Ultralytics 8.2.29,
yang dipakai Mamba-YOLO), terlepas dari arsitektur backbone (B2 atau P1 sama saja).

Aturan yang dimodelkan (diverifikasi langsung dari kode repo HZAI-ZJNU/Mamba-YOLO):
  1. load_image: sisi panjang gambar di-resize ke imgsz (tile kita sisi panjang = 640 -> r = 1).
  2. TaskAlignedAssigner.select_candidates_in_gts: anchor hanya boleh positif bila PUSAT anchor
     berada di dalam kotak GT (tegas, eps 1e-9). Pusat anchor = (i + 0.5) * stride, stride 8/16/32.
  3. bbox2dist / DFL: jarak tiap sisi dari pusat anchor maksimal (reg_max - 1 - 0.01) * stride
     = 14.99 * stride  ->  P3 120 px, P4 240 px, P5 480 px per sisi.
  4. RandomPerspective.box_candidates (augmentasi train): kotak dibuang dari target bila
     rasio aspek >= 100 atau lebar/tinggi <= 2 px setelah transformasi (skala acak U(0.5, 1.5)).

Plafon IoU per kotak = maks atas semua anchor yang pusatnya di dalam GT dari IoU antara GT dan
kotak terbaik yang bisa dikeluarkan anchor itu (GT dipotong ke jangkauan anchor). Posisi grid
relatif terhadap kotak bergantung pada padding letterbox / translasi, jadi dihitung rata-rata
atas OFF x OFF pergeseran seragam dalam [0, 32) px.

Pemakaian (Windows, folder berisi swrd_tiles):
    python diagnostik_geometri_swrd.py --tiles swrd_tiles
    python diagnostik_geometri_swrd.py --tiles swrd_tiles --reg_max 32     # skenario "bagaimana jika"
Keluaran: folder diag_geometri_<tiles>[_rm<reg_max>]/ berisi ringkasan.txt, per_kelas.csv, per_kotak.csv
"""
import argparse, csv, struct, sys
from pathlib import Path
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NAMES_DEFAULT = ["porosity", "inclusion", "crack", "undercut", "lack_of_fusion", "lack_of_penetration"]
STRIDES = (8, 16, 32)


def png_size(p):
    with open(p, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise ValueError(f"bukan PNG valid: {p}")
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def read_names(tiles):
    y = tiles / "data.yaml"
    try:
        import yaml
        d = yaml.safe_load(open(y, encoding="utf-8"))
        n = d["names"]
        return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)
    except Exception:
        return NAMES_DEFAULT


def best_frac_1d(a, b, s, o, R):
    """Untuk interval GT (a, b) (array), grid anchor (i+0.5)*s + o: kembalikan
    (ada_anchor_di_dalam, fraksi_terbaik_yang_terjangkau). Anchor terbaik selalu salah satu dari
    dua anchor yang mengapit pusat interval (lihat catatan di bawah)."""
    c = 0.5 * (a + b)
    L = b - a
    i0 = np.floor((c - o) / s - 0.5)
    best = np.zeros_like(a)
    found = np.zeros(a.shape, bool)
    for di in (0.0, 1.0):
        ax = (i0 + di + 0.5) * s + o
        inside = ((ax - a) > 1e-9) & ((b - ax) > 1e-9)
        f = (np.minimum(b, ax + R) - np.maximum(a, ax - R)) / np.maximum(L, 1e-12)
        f = np.where(inside, np.clip(f, 0, 1), 0.0)
        best = np.maximum(best, f)
        found |= inside
    return found, best
# Catatan: fraksi f(ax) unimodal dengan puncak di pusat interval; bila ada anchor di dalam (a, b),
# anchor yang paling dekat ke pusat dari sisi kiri/kanan juga di dalam (a, b). Jadi cukup cek 2 anchor.
# IoU kotak terpotong (selalu di dalam GT) = fx * fy, dan grid anchor berupa hasil kali kolom x baris,
# sehingga maks atas anchor = (maks fx atas kolom) * (maks fy atas baris) per stride.


def ceiling(x1, y1, x2, y2, reg_max, n_off):
    R = {s: (reg_max - 1 - 0.01) * s for s in STRIDES}
    offs = (np.arange(n_off) + 0.5) * (32.0 / n_off)
    n = len(x1)
    ious = np.zeros((n, n_off * n_off))
    noanc = np.zeros((n, n_off * n_off), bool)
    k = 0
    for ox in offs:
        fx = {s: best_frac_1d(x1, x2, s, ox, R[s]) for s in STRIDES}
        for oy in offs:
            best = np.zeros(n)
            anyanc = np.zeros(n, bool)
            for s in STRIDES:
                hx, bx = fx[s]
                hy, by = best_frac_1d(y1, y2, s, oy, R[s])
                ok = hx & hy
                best = np.maximum(best, np.where(ok, bx * by, 0.0))
                anyanc |= ok
            ious[:, k] = best
            noanc[:, k] = ~anyanc
            k += 1
    return ious, noanc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="swrd_tiles")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--reg_max", type=int, default=16)
    ap.add_argument("--off", type=int, default=16, help="jumlah pergeseran grid per sumbu (16 -> langkah 2 px)")
    a = ap.parse_args()
    tiles = Path(a.tiles)
    names = read_names(tiles)
    out = Path(f"diag_geometri_{tiles.name}" + ("" if a.reg_max == 16 else f"_rm{a.reg_max}"))
    out.mkdir(exist_ok=True)

    rows = []
    for split in ("train", "val"):
        imgs = sorted((tiles / "images" / split).glob("*.png"))
        if not imgs:
            sys.exit(f"tidak ada PNG di {tiles / 'images' / split}")
        for p in imgs:
            lab = tiles / "labels" / split / (p.stem + ".txt")
            if not lab.exists():
                continue
            W, H = png_size(p)
            r = a.imgsz / max(W, H)
            for ln in lab.read_text(encoding="utf-8").split("\n"):
                t = ln.split()
                if len(t) != 5:
                    continue
                c, cx, cy, w, h = int(t[0]), *map(float, t[1:])
                rows.append((split, p.stem, c, (cx - w / 2) * W * r, (cy - h / 2) * H * r,
                             (cx + w / 2) * W * r, (cy + h / 2) * H * r))
    sp = np.array([x[0] for x in rows]); st = [x[1] for x in rows]
    cl = np.array([x[2] for x in rows])
    x1, y1, x2, y2 = (np.array([x[i] for x in rows]) for i in (3, 4, 5, 6))
    bw, bh = x2 - x1, y2 - y1
    lo, sh = np.maximum(bw, bh), np.minimum(bw, bh)
    ar = np.maximum(bw / (bh + 1e-16), bh / (bw + 1e-16))

    ious, noanc = ceiling(x1, y1, x2, y2, a.reg_max, a.off)
    m_iou = ious.mean(1); p50 = (ious >= 0.5).mean(1); p75 = (ious >= 0.75).mean(1); pno = noanc.mean(1)
    # peluang kotak dibuang dari target train oleh box_candidates (tanpa mosaic; skala k ~ U(0.5,1.5))
    p_side = np.clip(2.0 / np.maximum(sh, 1e-9) - 0.5, 0, 1)
    p_drop = np.where(ar >= 100, 1.0, p_side)

    with open(out / "per_kotak.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["split", "tile", "kelas", "x1", "y1", "x2", "y2", "lebar", "tinggi", "rasio_aspek",
                    "plafon_iou_rata2", "p_iou_ge_0.5", "p_iou_ge_0.75", "p_tanpa_anchor", "p_dibuang_aug"])
        for i in range(len(rows)):
            w.writerow([sp[i], st[i], names[cl[i]], *(f"{v:.2f}" for v in (x1[i], y1[i], x2[i], y2[i], bw[i], bh[i], ar[i])),
                        *(f"{v:.4f}" for v in (m_iou[i], p50[i], p75[i], pno[i], p_drop[i]))])

    pct = lambda v, q: float(np.percentile(v, q)) if len(v) else float("nan")
    cols = ["split", "kelas", "n_kotak", "sisi_panjang_p50", "sisi_panjang_p90", "sisi_pendek_p10", "sisi_pendek_p50",
            "aspek_p50", "aspek_p90", "pct_aspek_ge100", "pct_pendek_le2", "pct_panjang_gt240", "pct_panjang_gt480",
            "p_dibuang_aug", "plafon_iou_rata2", "plafon_recall_iou50", "plafon_recall_iou75", "p_tanpa_anchor"]
    res = []
    for split in ("train", "val"):
        for c in range(len(names)):
            m = (sp == split) & (cl == c)
            if not m.any():
                continue
            res.append([split, names[c], int(m.sum()), pct(lo[m], 50), pct(lo[m], 90), pct(sh[m], 10), pct(sh[m], 50),
                        pct(ar[m], 50), pct(ar[m], 90), 100 * (ar[m] >= 100).mean(), 100 * (sh[m] <= 2).mean(),
                        100 * (lo[m] > 240).mean(), 100 * (lo[m] > 480).mean(), p_drop[m].mean(),
                        m_iou[m].mean(), p50[m].mean(), p75[m].mean(), pno[m].mean()])
        m = sp == split
        res.append([split, "SEMUA", int(m.sum()), pct(lo[m], 50), pct(lo[m], 90), pct(sh[m], 10), pct(sh[m], 50),
                    pct(ar[m], 50), pct(ar[m], 90), 100 * (ar[m] >= 100).mean(), 100 * (sh[m] <= 2).mean(),
                    100 * (lo[m] > 240).mean(), 100 * (lo[m] > 480).mean(), p_drop[m].mean(),
                    m_iou[m].mean(), p50[m].mean(), p75[m].mean(), pno[m].mean()])
    with open(out / "per_kelas.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(cols)
        for r_ in res:
            w.writerow([r_[0], r_[1], r_[2], *(f"{v:.4f}" for v in r_[3:])])

    hdr = (f"{'split':5} {'kelas':20} {'n':>5} {'panjang p50/p90':>16} {'pendek p10/p50':>15} {'aspek p50':>9} "
           f"{'%asp>=100':>9} {'%pjg>240':>8} {'%pjg>480':>8} {'p_buang':>7} {'IoU_maks':>8} {'rec@.5':>6} {'rec@.75':>7} {'tnp_anc':>7}")
    lines = [f"Diagnostik geometri -- {tiles}  (imgsz {a.imgsz}, reg_max {a.reg_max}, jangkauan/sisi "
             + ", ".join(f"P{int(np.log2(s))} {(a.reg_max - 1.01) * s:.0f}px" for s in STRIDES) + ")", hdr]
    for r_ in res:
        lines.append(f"{r_[0]:5} {r_[1]:20} {r_[2]:5d} {r_[3]:7.0f}/{r_[4]:<8.0f} {r_[5]:6.1f}/{r_[6]:<8.1f} {r_[7]:9.1f} "
                     f"{r_[9]:9.1f} {r_[11]:8.1f} {r_[12]:8.1f} {r_[13]:7.3f} {r_[14]:8.3f} {r_[15]:6.3f} {r_[16]:7.3f} {r_[17]:7.3f}")
    lines += ["",
              "Kolom: panjang/pendek = sisi kotak (px, setelah resize ke imgsz); %asp>=100 = kotak yang SELALU dibuang",
              "augmentasi train (box_candidates); p_buang = peluang rata-rata kotak dibuang dari target train per",
              "penyajian (tanpa mosaic); IoU_maks = plafon IoU rata-rata yang bisa dicapai prediksi anchor positif;",
              "rec@.5 / rec@.75 = bagian kotak yang secara geometris BISA ditangkap pada IoU 0.5 / 0.75",
              "(batas atas recall -> batas atas kasar AP50 / AP75); tnp_anc = peluang tidak ada satu pun pusat anchor",
              "di dalam kotak (kotak tidak pernah mendapat sinyal positif pada posisi grid itu)."]
    txt = "\n".join(lines)
    print(txt)
    (out / "ringkasan.txt").write_text(txt + "\n", encoding="utf-8")
    print(f"\n-> {out}/ringkasan.txt, per_kelas.csv, per_kotak.csv")


if __name__ == "__main__":
    main()
