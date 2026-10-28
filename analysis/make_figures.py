#!/usr/bin/env python3
"""
make_figures.py: data figures of the paper (Figs. 3-6). Every number is read from the files in results/; every
number that is also quoted in the paper is asserted here.

Output: fig_seeds, fig_latency, fig_training, fig_token_order as PDF (vector, Type 42 fonts) and PNG (600 dpi),
written to ./figures or to the folder given by the FIG_OUT environment variable. Widths: 1 column = 3.5 in,
2 columns = 7.16 in (IEEE). Colors: blue = deformable runs that trained, orange = collapsed runs, gray = baseline;
marker shape and dashed lines act as a second channel for black-and-white printing.
Usage: python analysis/make_figures.py        (comments below are in Indonesian)
"""
import glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from pathlib import Path
RES = Path(__file__).resolve().parents[1] / "results"
PC, TL = RES / "per_class", RES / "training_logs"
OUT = os.environ.get("FIG_OUT", "figures"); os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8d8c88", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 600, "legend.frameon": False, "axes.titleweight": "bold", "axes.titlelocation": "left"})

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig); print("ditulis", name)

def load(p):
    r = pd.read_csv(p, skipinitialspace=True); r.columns = [c.strip() for c in r.columns]
    return r.apply(pd.to_numeric, errors="coerce")
def grid(ax, axis="y"):
    ax.grid(axis=axis, color="#e4e3df", linewidth=0.5); ax.set_axisbelow(True)

# ================================================================ Gambar: sebaran per seed
P = pd.read_csv(PC / "per_class_results_PREFIX.csv"); F = pd.read_csv(PC / "per_class_results_P1FIX.csv")
G = pd.read_csv(PC / "per_class_results_gridfix.csv"); G = G[G.model == "P1-NoProj"]; S = pd.read_csv(PC / "per_class_results_swrdv1.csv")
def col(d, m): return d[(d.model == m) & (d["class"] == "ALL")].set_index("seed")["mAP50_95"].sort_index()
b2 = col(P, "B2")
STEEL = [("Baseline", b2, set()), ("Early\n(2)", col(P, "BB-Early"), set()), ("Late,\ninterl. (3)", col(P, "BB-Late-Alt"), set()),
         ("Neck\n(4)", col(P, "NeckOnly"), set()), ("Late\n(4)", col(P, "BB-Late"), set()),
         ("Backbone\n(6)", col(P, "BackboneOnly"), set()), ("Full\n(10)", col(F, "P1-fix"), {3}),
         ("Full, uncon-\nstrained (10)", col(G, G.model.unique()[0]), {2, 4, 5})]
SWRD = [("Baseline", col(S, "B2"), set()), ("Full\n(10)", col(S, "P1"), {2})]
assert f"{b2.mean():.4f}" == "0.5662" and f"{b2.min():.4f}" == "0.5428" and f"{b2.max():.4f}" == "0.5799"
assert f"{STEEL[6][1].drop(3).mean():.4f}" == "0.5533" and f"{SWRD[1][1].drop(2).mean():.4f}" == "0.0602"
assert f"{SWRD[0][1].mean():.4f}" == "0.0594"

fig = plt.figure(figsize=(7.16, 2.55))
gs = fig.add_gridspec(2, 2, width_ratios=[4.6, 1.25], height_ratios=[5.2, 0.75], hspace=0.06, wspace=0.16)
JIT = {1: -0.16, 2: -0.08, 3: 0.0, 4: 0.08, 5: 0.16}
def strip(ax_hi, ax_lo, groups, ylim, base_range, title, ylabel):
    for i, (lab, v, coll) in enumerate(groups):
        ok_ = v.drop(list(coll)) if coll else v
        c = GRAY if i == 0 else BLUE
        ax_hi.scatter([i + JIT[s] for s in ok_.index], ok_.values, s=14, color=c, edgecolors="white", linewidths=0.4, zorder=3)
        if len(ok_) >= 3:   # rerata hanya bila >=3 run terlatih (2 run tak-terkendala dilaporkan terpisah, §5.4)
            ax_hi.plot([i - 0.24, i + 0.24], [ok_.mean()] * 2, color=INK, linewidth=0.9, zorder=4)
        for k, s in enumerate(sorted(coll)):
            xk = i if len(coll) == 1 else i - 0.22 + 0.44 * k / (len(coll) - 1)
            ax_lo.scatter(xk, 0.5, marker="x", s=16, color=ORANGE, linewidths=1.0, zorder=3)
    ax_hi.axhspan(*base_range, color=GRAY, alpha=0.13, linewidth=0, zorder=0)
    ax_hi.set_ylim(*ylim); ax_hi.set_xlim(-0.6, len(groups) - 0.4); ax_hi.set_xticks([]); ax_hi.spines["bottom"].set_visible(False)
    ax_hi.set_ylabel(ylabel); ax_hi.set_title(title, pad=3); grid(ax_hi)
    ax_lo.set_ylim(0, 1); ax_lo.set_yticks([0.5]); ax_lo.set_yticklabels(["collapsed"], color=ORANGE)
    ax_lo.set_xlim(-0.6, len(groups) - 0.4); ax_lo.set_xticks(range(len(groups)))
    ax_lo.set_xticklabels([g[0] for g in groups]); ax_lo.tick_params(axis="y", length=0)
    for ax in (ax_hi, ax_lo):  # marka putus sumbu
        pass
    d = 0.012
    kw = dict(transform=ax_hi.transAxes, color=MUTED, clip_on=False, linewidth=0.6)
    ax_hi.plot((-d, +d), (-0.035, 0.0), **kw)
a_hi, a_lo = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
strip(a_hi, a_lo, STEEL, (0.49, 0.605), (b2.min(), b2.max()), "(a) Steel-pipe dataset", "mAP50:95")
a_hi.axvline(6.5, color=MUTED, linewidth=0.5, linestyle=(0, (2, 2))); a_lo.axvline(6.5, color=MUTED, linewidth=0.5, linestyle=(0, (2, 2)))
a_hi.text(-0.45, 0.5806, "baseline seed-to-seed range", fontsize=6.5, color=MUTED, va="bottom")
a_hi.text(3.5, 0.6035, "capacity-matched placements (number of deformable blocks)", fontsize=6.5, color=MUTED, ha="center", va="top")
a_hi.text(7.0, 0.6035, "+5.72%\nparameters", fontsize=6.5, color=MUTED, ha="center", va="top", linespacing=1.0)
sb = SWRD[0][1]
b_hi, b_lo = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
strip(b_hi, b_lo, SWRD, (0.045, 0.075), (sb.min(), sb.max()), "(b) SWRD", "")
b_lo.set_yticklabels([""])
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=GRAY, mec="white", ms=4.5, label="baseline run"),
                    Line2D([], [], marker="o", ls="", color=BLUE, mec="white", ms=4.5, label="deformable run, trained"),
                    Line2D([], [], marker="x", ls="", color=ORANGE, ms=4.5, mew=1.0, label="deformable run, collapsed"),
                    Line2D([], [], color=INK, lw=0.9, label="mean (≥3 trained runs)")],
           loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07), handletextpad=0.3, columnspacing=1.2)
save(fig, "fig_seeds")

# ================================================================ Gambar: latensi vs jumlah blok deformable
FB = pd.read_csv(RES / "latency" / "fps_benchmark.csv")
lr = stats.linregress(FB.n_deform, FB.lat_median_ms)
assert f"{lr.intercept:.1f}" == "24.7" and f"{lr.slope:.2f}" == "1.46" and f"{lr.rvalue**2:.2f}" == "0.93"
NAMES = {"B2": "baseline", "BB-Deep": "backbone deep", "BB-Early": "backbone early", "BB-Late-Alt": "backbone late, interl.",
         "BB-Late": "backbone late", "NeckOnly": "neck only", "BackboneOnly": "backbone only", "P1": "full"}
OFF = {"B2": (5, -3), "BB-Deep": (5, -3), "BB-Early": (5, -3), "BB-Late-Alt": (-4, 4), "BB-Late": (-4, 4),
       "NeckOnly": (5, -6), "BackboneOnly": (5, -6), "P1": (-4, 4)}
OFF["BB-Early"] = (-5, 4)
HA = {"BB-Late-Alt": "right", "BB-Late": "right", "P1": "right", "BB-Early": "right"}
fig, ax = plt.subplots(figsize=(3.5, 2.3))
xx = np.linspace(-0.3, 10.3, 50)
ax.plot(xx, lr.intercept + lr.slope * xx, color=GRAY, linewidth=0.9, linestyle=(0, (4, 2)), zorder=1)
ax.scatter(FB.n_deform, FB.lat_median_ms, s=18, color=BLUE, edgecolors="white", linewidths=0.5, zorder=3)
for r in FB.itertuples():
    ax.annotate(NAMES[r.variant], (r.n_deform, r.lat_median_ms), xytext=OFF[r.variant], textcoords="offset points",
                fontsize=6.5, color=MUTED, ha=HA.get(r.variant, "left"), va="center")
ax.text(4.3, 23.0, f"fit: {lr.intercept:.1f} + {lr.slope:.2f}$n$ ms ($R^2$ = {lr.rvalue**2:.2f})", fontsize=7, color=MUTED)
ax.set_xlabel("Number of deformable blocks, $n$"); ax.set_ylabel("Latency (ms)")
ax.set_xticks(range(0, 11, 2)); ax.set_xlim(-0.6, 10.8); ax.set_ylim(21, 41); grid(ax, "both")
save(fig, "fig_latency")

# ================================================================ Gambar: kurva loss klasifikasi
STEEL_B2 = {s: load(TL / "steelpipe" / "baseline" / f"seed{s}_results.csv") for s in range(1, 6)}
STEEL_P1 = {s: load(TL / "steelpipe" / "deformss2d" / f"seed{s}_results.csv") for s in range(1, 6)}
STEEL_NP = {s: load(TL / "steelpipe" / "deformss2d_unconstrained" / f"seed{s}_results.csv") for s in range(1, 6)}
SW_B2 = {s: load(TL / "swrd" / "baseline" / f"seed{s}_results.csv") for s in range(1, 6)}
SW_P1 = {s: load(TL / "swrd" / "deformss2d" / f"seed{s}_results.csv") for s in range(1, 6)}
def b2ref(R): return np.mean([r.loc[r.epoch == 140, "train/cls_loss"].item() for r in R.values()])
assert f"{b2ref(STEEL_B2):.4f}" == "0.5452"
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.25), sharey=True)
PAN = [("(a) Steel-pipe, capacity-matched", STEEL_B2, STEEL_P1, {3}),
       ("(b) Steel-pipe, unconstrained", STEEL_B2, STEEL_NP, {2, 4, 5}),
       ("(c) SWRD, capacity-matched", SW_B2, SW_P1, {2})]
for ax, (title, RB, RD, coll) in zip(axes, PAN):
    for r in RB.values():
        ax.plot(r.epoch, r["train/cls_loss"], color=GRAY, linewidth=0.6, zorder=1)
    for s, r in RD.items():
        if s in coll:
            ax.plot(r.epoch, r["train/cls_loss"], color=ORANGE, linewidth=1.0, linestyle=(0, (3, 1.5)), zorder=3)
        else:
            ax.plot(r.epoch, r["train/cls_loss"], color=BLUE, linewidth=0.7, zorder=2)
    ref = b2ref(RB)
    ax.axhline(1.5 * ref, color=MUTED, linewidth=0.5, linestyle=(0, (1, 1.5)))
    ax.axvline(140, color=MUTED, linewidth=0.5, linestyle=(0, (1, 1.5)))
    ax.set_yscale("log"); ax.set_ylim(0.4, 60); ax.set_xlim(0, 150); ax.set_xticks([0, 50, 100, 140])
    ax.set_title(title, pad=3); ax.set_xlabel("Epoch"); grid(ax)
axes[0].set_ylabel("Training classification loss")
axes[0].set_yticks([0.5, 1, 2, 5, 10, 50]); axes[0].set_yticklabels(["0.5", "1", "2", "5", "10", "50"])
fig.tight_layout(w_pad=0.6)
fig.legend(handles=[Line2D([], [], color=GRAY, lw=0.6, label="baseline"), Line2D([], [], color=BLUE, lw=0.7, label="deformable, trained"),
                    Line2D([], [], color=ORANGE, lw=1.0, ls=(0, (3, 1.5)), label="deformable, collapsed"),
                    Line2D([], [], color=MUTED, lw=0.5, ls=(0, (1, 1.5)), label="1.5 × baseline mean at epoch 140; epoch 140")],
           loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07), handlelength=2.2, columnspacing=1.2)
save(fig, "fig_training")

# ================================================================ Gambar: urutan token (Fase 14)
T = pd.read_csv(RES / "token_order" / "fase14_urutan_token_ringkas.csv"); T = T[T["mode"] == "fp32"]
ORDER = list(dict.fromkeys(T.module))
assert len(ORDER) == 10
LAYER = [m.split(".")[1] + ("·" + m.split(".")[2] if m.split(".")[2].isdigit() and m.startswith("model.5.") else "") for m in ORDER]
fig, ax = plt.subplots(figsize=(3.5, 2.25))
for s, c, mk, dx in ((1, BLUE, "o", -0.12), (3, ORANGE, "s", 0.12)):
    x = T[T.seed == s].set_index("module").loc[ORDER]
    frac = 100 * (1 - x.moved_frac_mean); w = (x.N * (1 - x.moved_frac_mean)).sum() / x.N.sum()
    assert f"{100*w:.1f}" == {1: "96.6", 3: "88.6"}[s]
    ax.scatter(np.arange(10) + dx, frac, s=16, marker=mk, color=c, edgecolors="white", linewidths=0.4, zorder=3,
               label={1: f"seed 1 ({100*w:.1f}% overall)", 3: f"seed 3, pre-collapse ({100*w:.1f}% overall)"}[s])
    assert x.frac_img_identical.iloc[0] == 1.0  # peta 160×160 identik raster di semua citra
ax.set_xticks(range(10))
ax.set_xticklabels([f"{l}\n{h}²" for l, h in zip(LAYER, T[T.seed == 1].set_index("module").loc[ORDER].H)], fontsize=6.5)
ax.axvline(5.5, color=MUTED, linewidth=0.5, linestyle=(0, (2, 2)))
ax.text(2.5, 10, "backbone", fontsize=6.5, color=MUTED, ha="center"); ax.text(7.5, 10, "neck", fontsize=6.5, color=MUTED, ha="center")
ax.set_ylim(0, 104); ax.set_xlim(-0.6, 9.6); ax.set_ylabel("Tokens at raster position (%)")
ax.set_xlabel("Deformable block (layer index, feature-map size)"); grid(ax)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.08), handletextpad=0.2)
save(fig, "fig_token_order")
