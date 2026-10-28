#!/usr/bin/env python3
"""reproduce_tables.py -- recompute every number in Tables 1-6 of the paper, the statistics quoted in the text, and
the numbers of Supplementary S2-S4, from the released results. No GPU, no training, no dataset needed.

    python analysis/reproduce_tables.py                 # prints all tables and tests
    python analysis/reproduce_tables.py --json out.json # also writes the values to a JSON file

Inputs (paths relative to the repository root):
    results/per_class/*.csv                 final metrics of every run (re-validation of best.pt, notebook "Sel 9")
    results/training_logs/<dataset>/<config>/seed<n>_results.csv   Ultralytics results.csv of every run
    results/latency/fps_benchmark.csv       latency benchmark (Table 4)
    results/diagnostics/k5*.csv             single-seed diagnostic runs of Section V-E and Supplementary S3
    results/diagnostics/verifikasi_dimensi_ringkasan.csv   object sizes, steel-pipe (Section IV-A)
    swrd/outputs/geometry_diagnostic/       per-box geometry of the SWRD tiles (Supplementary S2)

Conventions (as in the paper):
    * mAP = mAP50:95 on the validation split; Delta = baseline - other (positive favors the baseline).
    * A run "collapsed" if (a) a training loss (box/cls/dfl) was ever non-finite, or (b) the training
      classification loss at epoch 140 is >= 1.5 x its own minimum over epochs 1-140, or (c) it is >= 1.5 x the
      mean epoch-140 classification loss of the five baseline runs on the same dataset (Section IV-D).
    * Means and confirmatory paired tests use the runs that trained; the paired tests use the seeds at which both
      models trained. Exploratory: Welch (95% CI + p), exact Mann-Whitney U, exact permutation test on |difference
      of means| (all splits enumerated).
    * The five partial placement configurations (Table 3) were trained with the earlier version of the code
      (code/legacy/deform_ss2d_prefix.py); everything else deformable uses code/deform_ss2d.py.
"""
import argparse, itertools, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SEEDS = [1, 2, 3, 4, 5]
OUT = {}

# Parameter counts (built with nc=8; reproduced by analysis/count_parameters.py and count_detector_parameters.py)
PARAMS = {"B2": 5_986_912, "P1": 5_930_869, "P1-NoProj": 6_329_230, "BackboneOnly": 5_952_204, "NeckOnly": 5_965_577,
          "BB-Late": 5_964_057, "BB-Early": 5_975_059, "BB-Late-Alt": 5_972_944, "BB-Deep": 5_990_718,
          "YOLOv3-tiny": 12_136_240, "YOLOv5s": 9_125_288, "YOLOv6n": 4_238_936, "YOLOv8n": 3_012_408,
          "YOLOv8s": 11_138_696}
STEEL_NAMES = {"air-hole": "air-hole", "bite-edge": "bite-edge", "broken-arc": "broken-arc", "crack": "crack",
               "dense-gas-hole": "hollow-bead", "non-fusion": "overlap", "slag-inclusion": "slag-inclusion",
               "unfused-hole": "unfused"}          # name in the CSVs (original label files) -> name in the paper
SWRD_CLASSES = ["porosity", "inclusion", "crack", "undercut", "lack_of_fusion", "lack_of_penetration"]


# ---------------------------------------------------------------------------------------------------- helpers
def pc(name):
    d = pd.read_csv(RES / "per_class" / name)
    d.columns = d.columns.str.strip()
    return d


def agg(d, model, dataset=None):
    q = (d.model == model) & (d["class"] == "ALL")
    if dataset is not None:
        q &= d.dataset == dataset
    s = d[q].set_index("seed")["mAP50_95"].sort_index()
    assert list(s.index) == SEEDS, (model, list(s.index))
    return s


def log(path):
    r = pd.read_csv(path, skipinitialspace=True)
    r.columns = [c.strip() for c in r.columns]
    return r.apply(pd.to_numeric, errors="coerce")


def cls140(r):
    e = int(r.epoch[r.epoch <= 140].max())       # 140, or the last logged epoch <= 140 if a row is missing
    return float(r.loc[r.epoch == e, "train/cls_loss"].iloc[0])


def collapse(r, ref):
    """Collapse status of one run. ref = mean epoch-140 classification loss of the baseline runs."""
    tl = r[["train/box_loss", "train/cls_loss", "train/dfl_loss"]].to_numpy(float)
    a = not np.isfinite(tl).all()
    c140 = cls140(r)
    rb = c140 / float(np.nanmin(r.loc[r.epoch <= 140, "train/cls_loss"])) if np.isfinite(c140) else np.nan
    rc = c140 / ref if np.isfinite(c140) else np.nan
    crit = [k for k, hit in (("a", a), ("b", np.isfinite(rb) and rb >= 1.5), ("c", np.isfinite(rc) and rc >= 1.5),
                             ("a", not np.isfinite(c140))) if hit]
    return {"collapsed": bool(crit), "criteria": "".join(sorted(set(crit))), "ratio_b": rb, "ratio_c": rc}


def logs(dataset, config):
    return {s: log(RES / "training_logs" / dataset / config / f"seed{s}_results.csv") for s in SEEDS}


def welch(a, b):
    w = stats.ttest_ind(a, b, equal_var=False)
    se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    d, q = np.mean(a) - np.mean(b), stats.t.ppf(0.975, w.df)
    return d - q * se, d + q * se, float(w.pvalue)


def perm(a, b):
    v, n = np.r_[a, b], len(a)
    d = abs(np.mean(a) - np.mean(b))
    hits = tot = 0
    for idx in itertools.combinations(range(len(v)), n):
        m = np.zeros(len(v), bool); m[list(idx)] = True
        tot += 1; hits += abs(v[m].mean() - v[~m].mean()) >= d - 1e-12
    return hits / tot, tot


def bf(a, b):
    return float(stats.levene(a, b, center="median").pvalue)          # Brown-Forsythe


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]]).pvalue)


def f4(x):
    return f"{x:.4f}"


def sgn(x, nd=4):
    return f"{x:+.{nd}f}".replace("-", "−")


def show(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


# ---------------------------------------------------------------------------------------------------- data
PRE, FIX, SW = pc("per_class_results_PREFIX.csv"), pc("per_class_results_P1FIX.csv"), pc("per_class_results_swrdv1.csv")
GF, BASE = pc("per_class_results_gridfix.csv"), pc("per_class_results_BASELINES.csv")
REC, SWD = pc("per_class_results_RECIPE.csv"), pc("per_class_results_swrdv1_diag.csv")

LOG = {"steel": {"base": logs("steelpipe", "baseline"), "deform": logs("steelpipe", "deformss2d"),
                 "uncon": logs("steelpipe", "deformss2d_unconstrained"),
                 "uncon_old": logs("steelpipe", "legacy_deformss2d_unconstrained")},
       "swrd": {"base": logs("swrd", "baseline"), "deform": logs("swrd", "deformss2d")}}
for v in ("backboneonly", "neckonly", "bb-late", "bb-early", "bb-late-alt"):
    LOG["steel"][v] = logs("steelpipe", f"legacy_{v}")
REF = {ds: float(np.mean([cls140(r) for r in LOG[ds]["base"].values()])) for ds in LOG}
STATUS = {(ds, cfg, s): collapse(r, REF[ds]) for ds in LOG for cfg in LOG[ds] for s, r in LOG[ds][cfg].items()}


def trained(ds, cfg):
    return [s for s in SEEDS if not STATUS[(ds, cfg, s)]["collapsed"]]


# ---------------------------------------------------------------------------------------------------- collapse status
show("Training outcome (collapse criteria a/b/c, Section IV-D)")
print(f"reference: mean epoch-140 classification loss of the baseline = {REF['steel']:.4f} (steel-pipe), "
      f"{REF['swrd']:.4f} (SWRD); threshold (c) = 1.5 x reference")
for (ds, cfg, s), st in STATUS.items():
    print(f"  {ds:5s} {cfg:10s} seed {s}: ratio(b) {st['ratio_b']:6.3f}  ratio(c) {st['ratio_c']:6.3f}  "
          f"-> {'COLLAPSED (' + st['criteria'] + ')' if st['collapsed'] else 'trained'}")
OUT["status"] = {f"{ds}/{cfg}/seed{s}": st for (ds, cfg, s), st in STATUS.items()}


# ---------------------------------------------------------------------------------------------------- Table 1
def main_row(name, base, deform, ds):
    tr_d = trained(ds, "deform"); tr_b = trained(ds, "base"); pair = [s for s in tr_b if s in tr_d]
    b, d = base.loc[tr_b].values, deform.loc[tr_d].values
    bp, dp = base.loc[pair].values, deform.loc[pair].values
    lo, hi, pw = welch(b, d)
    pp, ntot = perm(b, d)
    row = {"baseline": (b.mean(), b.std(ddof=1)), "deform": (d.mean(), d.std(ddof=1)), "delta": b.mean() - d.mean(),
           "paired_t": float(stats.ttest_rel(bp, dp).pvalue),
           "wilcoxon": float(stats.wilcoxon(bp, dp, method="exact").pvalue), "welch_ci": (lo, hi), "welch": pw,
           "mann_whitney": float(stats.mannwhitneyu(b, d, method="exact").pvalue), "permutation": pp,
           "n_permutations": ntot, "paired_seeds": pair, "mean_paired_difference": float(np.mean(bp - dp)),
           "brown_forsythe": bf(b, d)}
    # sensitivity: include the best retained checkpoint of collapsed runs
    ba, da = base.values, deform.values
    lo2, hi2, pw2 = welch(ba, da)
    row["with_collapsed"] = {"deform": (da.mean(), da.std(ddof=1)), "paired_t": float(stats.ttest_rel(ba, da).pvalue),
                             "wilcoxon": float(stats.wilcoxon(ba, da, method="exact").pvalue), "welch": pw2,
                             "mann_whitney": float(stats.mannwhitneyu(ba, da, method="exact").pvalue),
                             "permutation": perm(ba, da)[0]}
    print(f"{name:22s} {f4(row['baseline'][0])} ± {f4(row['baseline'][1])} | {f4(row['deform'][0])} ± "
          f"{f4(row['deform'][1])} | {sgn(row['delta'])} | paired t {f4(row['paired_t'])} | Wilcoxon "
          f"{f4(row['wilcoxon'])} | CI [{sgn(lo)}, {sgn(hi)}] | Welch {f4(pw)} | MWU {f4(row['mann_whitney'])} | "
          f"perm {f4(pp)} ({ntot})")
    print(f"{'':22s} paired seeds {pair}, mean paired difference {sgn(row['mean_paired_difference'])}; "
          f"Brown-Forsythe p = {row['brown_forsythe']:.2f} (s.d. {row['deform'][1]:.4f} vs {row['baseline'][1]:.4f})")
    w = row["with_collapsed"]
    print(f"{'':22s} incl. collapsed: {f4(w['deform'][0])} ± {f4(w['deform'][1])}; min p over five tests = "
          f"{min(w['paired_t'], w['wilcoxon'], w['welch'], w['mann_whitney'], w['permutation']):.3f}")
    return row


show("Table 1 -- main comparison (mAP50:95)")
B2_STEEL, P1_STEEL = agg(PRE, "B2"), agg(FIX, "P1-fix")
B2_SWRD, P1_SWRD = agg(SW, "B2", "swrdv1"), agg(SW, "P1", "swrdv1")
OUT["table1"] = {"steel-pipe": main_row("Steel-pipe (8 classes)", B2_STEEL, P1_STEEL, "steel"),
                 "SWRD": main_row("SWRD (6 classes)", B2_SWRD, P1_SWRD, "swrd")}


# ---------------------------------------------------------------------------------------------------- Table 2
show("Table 2 -- per-class mAP50:95, mean over the runs that trained (Delta computed before rounding)")
OUT["table2"] = {}
for ds, d_base, m_base, d_def, m_def, classes, nd in (
        ("steel", PRE, "B2", FIX, "P1-fix", list(STEEL_NAMES), 3), ("swrd", SW, "B2", SW, "P1", SWRD_CLASSES, 4)):
    tb, td = trained(ds, "base"), trained(ds, "deform"); ahead = 0; rows = {}
    for c in classes:
        b = d_base[(d_base.model == m_base) & (d_base["class"] == c) & d_base.seed.isin(tb)].mAP50_95.mean()
        x = d_def[(d_def.model == m_def) & (d_def["class"] == c) & d_def.seed.isin(td)].mAP50_95.mean()
        ahead += b > x; name = STEEL_NAMES.get(c, c.replace("_", " "))
        rows[name] = (b, x, b - x)
        print(f"  {name:22s} {b:.{nd}f}  {x:.{nd}f}  {sgn(b - x, nd)}")
    print(f"  baseline ahead: {ahead}/{len(classes)}")
    OUT["table2"][ds] = {"rows": rows, "baseline_ahead": f"{ahead}/{len(classes)}"}


# ---------------------------------------------------------------------------------------------------- Table 3
show("Table 3 -- placement ablation, steel-pipe (partial configurations: earlier code version)")
CFG = [("Backbone early (stages 1, 3)", "BB-Early", 2), ("Backbone late, interleaved", "BB-Late-Alt", 3),
       ("Neck only", "NeckOnly", 4), ("Backbone late (stages 5, 7)", "BB-Late", 4), ("Backbone only", "BackboneOnly", 6)]
print(f"  {'SS2D baseline':30s} 0  {PARAMS['B2']:,}  {f4(B2_STEEL.mean())} ± {f4(B2_STEEL.std())}")
OUT["table3"] = {}; SD = {"SS2D baseline": (0, B2_STEEL.std())}
for label, m, nb in CFG:
    v = agg(PRE, m); lo, hi, pw = welch(B2_STEEL.values, v.values)
    pt = float(stats.ttest_rel(B2_STEEL.values, v.values).pvalue)
    n_tr = len(trained("steel", m.lower()))
    OUT["table3"][m] = {"mean": v.mean(), "sd": v.std(), "delta": B2_STEEL.mean() - v.mean(), "paired_t": pt,
                        "welch": pw, "runs_trained": n_tr}
    SD[label] = (nb, v.std())
    print(f"  {label:30s} {nb}  {PARAMS[m]:,}  {f4(v.mean())} ± {f4(v.std())}  {sgn(B2_STEEL.mean() - v.mean())}  "
          f"paired t {f4(pt)}  Welch {f4(pw)}  (trained {n_tr}/5)")
t1 = OUT["table1"]["steel-pipe"]
print(f"  {'Backbone + neck (full)':30s} 10 {PARAMS['P1']:,}  {f4(t1['deform'][0])} ± {f4(t1['deform'][1])}  "
      f"{sgn(t1['delta'])}  paired t {f4(t1['paired_t'])}  Welch {f4(t1['welch'])}  (four runs that trained)")
SD["full"] = (10, t1["deform"][1])
nb_, sd_ = np.array([v[0] for v in SD.values()]), np.array([v[1] for v in SD.values()])
pr, sr = stats.pearsonr(nb_, sd_), stats.spearmanr(nb_, sd_)
print(f"  s.d. vs number of deformable blocks (7 configurations): Pearson r = {pr[0]:.2f} (p = {pr[1]:.2f}), "
      f"Spearman rho = {sr[0]:.2f} (p = {sr[1]:.2f})")
groups = {"B2": B2_STEEL.values, **{m: agg(PRE, m).values for _, m, _ in CFG},
          "full": P1_STEEL.loc[trained("steel", "deform")].values}
pairs = [(a, b, bf(groups[a], groups[b])) for a, b in itertools.combinations(groups, 2)]
amin = min(pairs, key=lambda t: t[2])
print(f"  Brown-Forsythe over {len(pairs)} pairs of configurations: smallest p = {amin[2]:.4f} ({amin[0]} vs {amin[1]}); "
      f"Bonferroni threshold 0.05/{len(pairs)} = {0.05 / len(pairs):.4f}")
OUT["table3_spread"] = {"pearson_r": pr[0], "spearman_rho": sr[0], "bf_min": amin}
n_old = sum(len(trained("steel", m.lower())) for _, m, _ in CFG)
print(f"  partial configurations: {n_old}/25 runs trained (earlier code version)")

# interaction backbone x neck (2x2; partial configurations from the earlier code version)
bo, no = agg(PRE, "BackboneOnly"), agg(PRE, "NeckOnly"); full = P1_STEEL.loc[trained("steel", "deform")]
d_bo, d_no, d_full = bo.mean() - B2_STEEL.mean(), no.mean() - B2_STEEL.mean(), full.mean() - B2_STEEL.mean()
rows = [(0, 0, x) for x in B2_STEEL] + [(1, 0, x) for x in bo] + [(0, 1, x) for x in no] + [(1, 1, x) for x in full]
X = np.array([[1, b, n, b * n] for b, n, _ in rows], float); y = np.array([r[2] for r in rows])
beta, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ beta; dof = len(y) - 4
se = np.sqrt(res @ res / dof * np.linalg.inv(X.T @ X)[3, 3]); p_int = 2 * stats.t.sf(abs(beta[3] / se), dof)
s4 = [s for s in SEEDS if s in full.index]
contrast = [full[s] - bo[s] - no[s] + B2_STEEL[s] for s in s4]
p_con = float(stats.ttest_1samp(contrast, 0).pvalue)
print(f"  interaction: backbone-only {sgn(d_bo)}, neck-only {sgn(d_no)} -> additive {sgn(d_bo + d_no)}; full "
      f"{sgn(d_full)}; two-way model p = {p_int:.2f}; per-seed contrast (seeds {s4}) p = {p_con:.2f}")
OUT["interaction"] = {"additive": d_bo + d_no, "full": d_full, "p_two_way": p_int, "p_contrast": p_con}


# ---------------------------------------------------------------------------------------------------- Table 4
show("Table 4 -- latency (NVIDIA T4, batch 1, 640x640, FP32; median of 300 runs after 50 warm-up runs)")
F = pd.read_csv(RES / "latency" / "fps_benchmark.csv")
LAB = {"B2": "SS2D baseline", "BB-Deep": "Backbone deep (stage 7)", "BB-Early": "Backbone early",
       "BB-Late-Alt": "Backbone late, interleaved", "NeckOnly": "Neck only", "BB-Late": "Backbone late",
       "BackboneOnly": "Backbone only", "P1": "Backbone + neck (full)"}
F = F.set_index("variant").loc[list(LAB)].reset_index()          # row order of Table 4
for _, r in F.iterrows():
    print(f"  {LAB[r.variant]:28s} {int(r.n_deform):2d}  {r.lat_median_ms:6.2f} ms  {r.fps_median:5.1f} FPS  "
          f"{r.slowdown_vs_B2:.2f}x")
fit = stats.linregress(F.n_deform, F.lat_median_ms)
print(f"  latency ~ {fit.intercept:.2f} + {fit.slope:.3f} x n_deform ms (R^2 = {fit.rvalue ** 2:.2f})")
OUT["table4"] = {"intercept": fit.intercept, "slope": fit.slope, "r2": fit.rvalue ** 2}


# ---------------------------------------------------------------------------------------------------- Table 5
show("Table 5 -- capacity constraint removed (grid-fixed code), steel-pipe")
OUT["table5"] = {}
for s in SEEDS:
    r = GF[(GF.model == "P1-NoProj") & (GF.seed == s) & (GF["class"] == "ALL")].iloc[0]
    st = STATUS[("steel", "uncon", s)]
    OUT["table5"][s] = {"mAP": r.mAP50_95, "recall": r.recall, "collapsed": st["collapsed"]}
    print(f"  seed {s}: {r.mAP50_95:.4f}  recall {r.recall:.3f}  {'collapsed (' + st['criteria'] + ')' if st['collapsed'] else 'completed'}")
k_new = len(trained("steel", "uncon")); k_old = len(trained("steel", "uncon_old"))
k_b2 = len(trained("steel", "base")); k_p1 = len(trained("steel", "deform"))
print(f"  completed {k_new}/5; Fisher vs baseline ({k_b2}/5) p = {fisher(k_new, 5, k_b2, 5):.2f}; vs capacity-matched "
      f"({k_p1}/5) p = {fisher(k_new, 5, k_p1, 5):.2f}")
k_m = len(trained("steel", "deform")) + len(trained("swrd", "deform")); k_b = len(trained("steel", "base")) + len(trained("swrd", "base"))
print(f"  capacity-matched, both datasets: {k_m}/10 completed vs baseline {k_b}/10, Fisher p = {fisher(k_m, 10, k_b, 10):.2f}")
OUT["fisher"] = {"uncon_vs_base": fisher(k_new, 5, k_b2, 5), "uncon_vs_matched": fisher(k_new, 5, k_p1, 5),
                 "matched_vs_base_both": fisher(k_m, 10, k_b, 10)}


# ---------------------------------------------------------------------------------------------------- Table 6
show("Table 6 -- detectors on the steel-pipe dataset, seed 1: study recipe (SGD, lr0 0.002) vs default (AdamW, auto)")
def s1(d, m):
    return float(d[(d.model == m) & (d.seed == 1) & (d["class"] == "ALL")].mAP50_95.iloc[0])
OUT["table6"] = {}
for m, lab in (("YOLOv3-tiny", "YOLOv3-tiny (anchor-free)"), ("YOLOv5s", "YOLOv5s (anchor-free)"),
               ("YOLOv6n", "YOLOv6n (Ultralytics)"), ("YOLOv8n", "YOLOv8n"), ("YOLOv8s", "YOLOv8s")):
    st = s1(BASE, m); de = s1(REC, f"{m}-auto") if f"{m}-auto" in set(REC.model) else None
    OUT["table6"][m] = (st, de)
    print(f"  {lab:28s} {PARAMS[m]:>11,}  {st:.4f}  " + (f"{de:.4f}  {sgn(de - st)}" if de else "n.r."))
st, de = s1(PRE, "B2"), s1(REC, "B2-auto"); OUT["table6"]["B2"] = (st, de)
print(f"  {'Mamba-YOLO-T (SS2D)':28s} {PARAMS['B2']:>11,}  {st:.4f}  {de:.4f}  {sgn(de - st)}")
st = s1(FIX, "P1-fix"); OUT["table6"]["P1"] = (st, None)
print(f"  {'DeformSS2D':28s} {PARAMS['P1']:>11,}  {st:.4f}  did not train (see note 3 of Table 6)")
b, d = s1(SW, "B2"), s1(SWD, "B2")
cb = float(SW[(SW.model == "B2") & (SW.seed == 1) & (SW["class"] == "crack")].mAP50_95.iloc[0])
cd = float(SWD[(SWD.model == "B2") & (SWD["class"] == "crack")].mAP50_95.iloc[0])
print(f"  SWRD baseline seed 1, lr0 0.002 -> 0.01: {b:.4f} -> {d:.4f} ({sgn(d - b, 3)}); crack {cb:.3f} -> {cd:.3f} "
      f"({sgn(cd - cb, 3)}); best study-recipe run {B2_SWRD.max():.4f}")
OUT["swrd_lr"] = {"all": (b, d), "crack": (cb, cd)}


# ---------------------------------------------------------------------------------------------------- Supplementary S2
show("Supplementary S2 -- why SWRD scores are low")
TV = pc("diag_train_vs_val_swrdv1.csv")
def tv(m, s, split, tag="swrdv1", c="ALL"):
    return float(TV[(TV.dataset == tag) & (TV.model == m) & (TV.seed == s) & (TV.split == split)
                    & (TV["class"] == c)].mAP50_95.iloc[0])
trb, trd = trained("swrd", "base"), trained("swrd", "deform")
tb = np.array([tv("B2", s, "train") for s in trb]); td = np.array([tv("P1", s, "train") for s in trd])
rat = np.r_[tb / B2_SWRD.loc[trb].values, td / P1_SWRD.loc[trd].values]
print(f"  training split, selected checkpoints: baseline {tb.mean():.3f} ± {tb.std(ddof=1):.3f} (seeds {trb}), "
      f"deformable {td.mean():.3f} ± {td.std(ddof=1):.3f} (seeds {trd}); train/val ratio {rat.min():.2f}-{rat.max():.2f}")
t02, t01 = tv("B2", 1, "train"), tv("B2", 1, "train", "swrdv1_diag_lr0.01")
print(f"  baseline seed 1, lr0 0.002 -> 0.01: training split {t02:.3f} -> {t01:.3f}; validation {b:.3f} -> {d:.3f}")
for c in ("porosity", "crack", "inclusion", "lack_of_fusion"):
    v02 = float(SW[(SW.model == "B2") & (SW.seed == 1) & (SW["class"] == c)].mAP50_95.iloc[0])
    v01 = float(SWD[(SWD.model == "B2") & (SWD["class"] == c)].mAP50_95.iloc[0])
    a02, a01 = tv("B2", 1, "train", c=c), tv("B2", 1, "train", "swrdv1_diag_lr0.01", c)
    print(f"    {c:16s} validation {v02:.3f} -> {v01:.3f} ({sgn(v01 - v02, 3)}); training {a02:.3f} -> {a01:.3f} "
          f"(x{a01 / a02:.1f}); training/validation at lr0 0.01 = {a01 / v01:.1f}")
for m, ser, tr_ in (("B2", B2_SWRD, trb), ("P1", P1_SWRD, trd)):
    cm = {c: SW[(SW.model == m) & (SW["class"] == c) & SW.seed.isin(tr_)].mAP50_95.mean() for c in SWRD_CLASSES}
    print(f"  class means above 0.1, {m}: {[c for c, v in cm.items() if v > 0.1]}")
def map_ratio(d, model, dataset=None):
    q = (d.model == model) & (d["class"] == "ALL") & (d.dataset == dataset if dataset else True)
    x = d[q].sort_values("seed"); return (x.mAP50_95 / x.mAP50).to_numpy()
rs, rp = map_ratio(SW, "B2", "swrdv1"), map_ratio(PRE, "B2")
print(f"  mAP50:95 / mAP50, baseline: SWRD {rs.mean():.3f} ({rs.min():.3f}-{rs.max():.3f}), "
      f"steel-pipe {rp.mean():.3f} ({rp.min():.3f}-{rp.max():.3f})")
GEO = ROOT / "swrd" / "outputs" / "geometry_diagnostic"
K = pd.read_csv(GEO / "per_kotak.csv", encoding="utf-8-sig"); KC = pd.read_csv(GEO / "per_kelas.csv", encoding="utf-8-sig")
ar = np.maximum(K.lebar, K.tinggi) / np.minimum(K.lebar, K.tinggi); eqs = np.sqrt(K.lebar * K.tinggi)
n_tr = K[K.split == "train"].kelas.value_counts()
print(f"  SWRD boxes: {len(K):,} ({(K.split == 'val').sum()} validation); training porosity {n_tr['porosity']:,}, other classes "
      f"{n_tr.drop('porosity').min()}-{n_tr.drop('porosity').max()}")
print(f"  SWRD aspect ratio > 5: {(ar > 5).sum()} ({100 * (ar > 5).mean():.1f}%); 90th percentile {ar.quantile(.9):.2f}; "
      f"median equivalent side {eqs.median():.1f} px")
SP_ = pd.read_csv(RES / "diagnostics" / "verifikasi_dimensi_ringkasan.csv")
sp = SP_[(SP_.dataset == "DSF-YOLO") & SP_.rumus.str.startswith("BENAR")].iloc[0]
print(f"  steel-pipe aspect ratio > 5: {int(sp.n_ar_gt5)} of {int(sp.n):,} ({sp.pct_ar_gt5:.2f}%); 90th percentile "
      f"{sp.ar_p90:.2f}; median equivalent side {sp.side640_median:.1f} px; {sp.pct_small:.1f}% below 32 px")
IM = pd.read_csv(RES / "diagnostics" / "verifikasi_dimensi_ukuran_citra.csv"); IM = IM[IM.dataset == "DSF-YOLO"]
print(f"  steel-pipe image sizes (Section IV-A): {len(IM)} resolutions, aspect ratio "
      f"{(IM.W / IM.H).min():.2f}-{(IM.W / IM.H).max():.2f}")
edge = (K.x1 <= 0.5) | (K.y1 <= 0.5) | (K.x2 >= 639.5) | (K.y2 >= 639.5)   # 640-px tile edges only: a lower bound
for c in ("lack_of_penetration", "lack_of_fusion", "crack", "porosity"):
    e = [100 * edge[(K.kelas == c) & (K.split == s)].mean() for s in ("train", "val")]
    print(f"    {c:20s} boxes touching a 640-px tile edge: {min(e):.0f}-{max(e):.0f}% (training, validation)")
kc = KC[KC.kelas != "SEMUA"]
print(f"  head can match the labels: min over class x split {100 * kc.plafon_recall_iou50.min():.1f}% at IoU 0.5, "
      f"{100 * kc.plafon_recall_iou75.min():.1f}% at IoU 0.75; boxes always removed by the augmentation filter "
      f"{kc.pct_aspek_ge100.max():.1f}%")
iou = lambda h, dd: (h - dd) / (h + dd)          # box shifted by dd along its short side h
thr = np.arange(0.50, 0.951, 0.05)
for w_, h_ in ((400, 30), (100, 100)):
    print(f"    {w_} x {h_} px box shifted 10 px along its short side: IoU {iou(h_, 10):.2f}, counted at "
          f"{int((iou(h_, 10) >= thr - 1e-9).sum())} of 10 thresholds")
OUT["s2"] = {"train_base": (tb.mean(), tb.std(ddof=1)), "train_deform": (td.mean(), td.std(ddof=1)),
             "ratio": (rat.min(), rat.max()), "lr_train": (t02, t01), "map_ratio_swrd": rs.mean(),
             "map_ratio_steel": rp.mean(), "ar_gt5_pct": 100 * (ar > 5).mean()}


# ---------------------------------------------------------------------------------------------------- Section V-E
show("Section V-E and Supplementary S3 -- additive configuration (K = 5), default recipe, seed 1 (single runs)")
DIA = RES / "diagnostics"
A, BP, DUP = (log(DIA / f) for f in ("k5aug_seed1_auto_results.csv", "k5bp_results.csv", "k5dup_seed1_results.csv"))
ca = A["train/cls_loss"]
print(f"  learned path (original parameterization): cls loss minimum {ca.min():.3f} at epoch {int(A.epoch[ca.idxmin()])}; "
      f"max mAP {A['metrics/mAP50-95(B)'].max():.5f}; last logged epoch {int(A.epoch.max())}")
mb = BP["metrics/mAP50-95(B)"]; at = lambda e: float(mb[BP.epoch == e].iloc[0])
nan_ep = int(BP.epoch[BP[["train/box_loss", "train/cls_loss", "train/dfl_loss"]].isna().any(axis=1)].min())
print(f"  bounded path: peak mAP {mb.max():.3f} at epoch {int(BP.epoch[mb.idxmax()])}; epoch 50 {at(50):.3f}; "
      f"epoch 53 {at(53):.3f}; epoch 54 {at(54):.3f}; first non-finite training loss at epoch {nan_ep}")
M = pd.read_csv(DIA / "k5bp_offset_monitor.csv"); M11 = M[(M.epoch >= 11) & (M.epoch <= 59)]
mv = M11.groupby("epoch").moved_frac.mean()
mw = M11.groupby("epoch").apply(lambda g: np.average(g.moved_frac, weights=g.N), include_groups=False)
print(f"  bounded path, epochs 11-59: share of tokens that changed position, mean over the {M.block.nunique()} blocks "
      f"{100 * mv.min():.0f}-{100 * mv.max():.0f}% (weighted by tokens per block: {100 * mw.min():.0f}-{100 * mw.max():.0f}%); "
      f"largest displacement {M.disp_max_rows.max():.2f} rows")
dup, b2a = s1(pc("per_class_results_RECIPE_v2.csv"), "K5dup-auto"), s1(REC, "B2-auto")
print(f"  control (fixed copy of the row-major path): mAP {dup:.4f} vs baseline {b2a:.4f} ({sgn(dup - b2a)}); "
      f"baseline seed-to-seed s.d. under the study recipe {B2_STEEL.std():.4f}")
OUT["section5e"] = {"aug_cls_min": float(ca.min()), "aug_map_max": float(A["metrics/mAP50-95(B)"].max()),
                    "bounded_peak": float(mb.max()), "bounded_nan_epoch": nan_ep,
                    "moved_block_mean": (float(mv.min()), float(mv.max())), "control": dup, "baseline_auto": b2a}


# ---------------------------------------------------------------------------------------------------- Supplementary S4
show("Supplementary S4 -- earlier code version vs grid-fixed code (exploratory)")
old = agg(PRE, "P1"); tr = trained("steel", "deform")
pt = float(stats.ttest_rel(old.loc[tr].values, P1_STEEL.loc[tr].values).pvalue)   # seeds at which the corrected runs trained
pw = welch(old.values, P1_STEEL.loc[tr].values)[2]                                  # all five earlier runs vs corrected runs that trained
print(f"  capacity-matched model: paired t over seeds {tr} p = {pt:.2f}; Welch (five earlier runs vs the {len(tr)} corrected "
      f"runs that trained) p = {pw:.2f}")
changed = [s for s in SEEDS if STATUS[("steel", "uncon_old", s)]["collapsed"] != STATUS[("steel", "uncon", s)]["collapsed"]]
only_c = [s for s in SEEDS if STATUS[("steel", "uncon_old", s)]["criteria"] == "c"]
print(f"  without the capacity constraint: earlier {k_old}/5 completed vs corrected {k_new}/5, Fisher p = "
      f"{fisher(k_old, 5, k_new, 5):.1f}; outcome changed at seeds {changed}; earlier collapses found only by (c): {only_c}")
OUT["s4"] = {"paired_t": pt, "welch": pw, "fisher": fisher(k_old, 5, k_new, 5), "changed": changed}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--json", default=None); a = ap.parse_args()
    if a.json:
        json.dump(OUT, open(a.json, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print(f"\nwritten {a.json}")
