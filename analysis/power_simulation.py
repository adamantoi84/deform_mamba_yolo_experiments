#!/usr/bin/env python3
"""power_simulation.py -- design-resolution numbers of Section IV-D, recomputed from the per-seed data.

  (1) Type-I error of the PAIRED t-test when paired seeds share no source of variability (the seed changes only the
      initialization, Section IV-C): two independent groups, H0 true, s.d. = observed s.d. of each model, n = 5 and
      n = 4 pairs (n = 4: collapsed run excluded). Welch on the same data for comparison.
  (2) 80% power of the two-sided paired t-test: effect in units of the s.d. of the paired differences (exact,
      noncentral t).
  (3) Minimum detectable difference of Welch's test (80% power, alpha = 0.05) at the observed group sizes and s.d.
      (baseline 5 runs, deformable 4 runs that trained).

    python analysis/power_simulation.py [--out power_simulation.json]
Fixed RNG seed (2026): results repeat exactly.
"""
import argparse, json
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import brentq

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="power_simulation.json")
a = ap.parse_args()
N_SIM, ALPHA = 40_000, 0.05

from pathlib import Path
PC = Path(__file__).resolve().parents[1] / "results" / "per_class"
pre = pd.read_csv(PC / "per_class_results_PREFIX.csv"); fx = pd.read_csv(PC / "per_class_results_P1FIX.csv")
sw = pd.read_csv(PC / "per_class_results_swrdv1.csv")
def col(d, m, drop=None):
    x = d[(d.model == m) & (d["class"] == "ALL")].set_index("seed")["mAP50_95"].sort_index()
    return x.drop(drop) if drop else x
DATA = {"steel-pipe": (col(pre, "B2"), col(fx, "P1-fix", drop=3)),
        "SWRD": (col(sw, "B2"), col(sw, "P1", drop=2))}
for k, (b, p) in DATA.items():
    assert len(b) == 5 and len(p) == 4, (k, len(b), len(p))

rng = np.random.default_rng(2026)
out = {"n_sim": N_SIM, "alpha": ALPHA, "rng_seed": 2026}

# (1) type-I error
t1 = {}
for k, (b, p) in DATA.items():
    sa, sb = b.std(ddof=1), p.std(ddof=1)
    for n in (5, 4):
        x = rng.normal(0, sa, (N_SIM, n)); y = rng.normal(0, sb, (N_SIM, n))
        t1[f"{k}_n{n}"] = {"sd_base": round(sa, 5), "sd_deform": round(sb, 5),
                           "paired_t": float((stats.ttest_rel(x, y, axis=1).pvalue < ALPHA).mean()),
                           "welch": float((stats.ttest_ind(x, y, axis=1, equal_var=False).pvalue < ALPHA).mean())}
out["type1"] = t1

# (2) power of the paired t-test (noncentral t, exact)
def pw_paired(d, n):
    df = n - 1; tc = stats.t.ppf(1 - ALPHA / 2, df); nc = d * np.sqrt(n)
    return stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)
out["paired_effect_sd_80pct"] = {f"n{n}": brentq(lambda d: pw_paired(d, n) - 0.8, 0.5, 4) for n in (5, 4)}

# (3) Welch MDE, 5 vs 4, observed s.d.; common random numbers make power monotone in d; the 80% point is found
#     by bisection (tolerance 1e-5)
def mde80(sa, sb, na=5, nb=4):
    zx = rng.standard_normal((N_SIM, na)); zy = rng.standard_normal((N_SIM, nb))
    pw = lambda d: float((stats.ttest_ind(sa * zx, d + sb * zy, axis=1, equal_var=False).pvalue < ALPHA).mean())
    lo, hi = 0.0, 0.2
    while hi - lo > 1e-5:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pw(mid) < 0.8 else (lo, mid)
    return hi, pw(hi)
mde = {}
for k, (b, p) in DATA.items():
    d80, p80 = mde80(b.std(ddof=1), p.std(ddof=1))
    mde[k] = {"mde_80": round(d80, 5), "power_at_mde": p80, "base_mean": round(b.mean(), 4), "mde_frac_of_base": d80 / b.mean()}
out["welch_mde"] = mde

json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(out, indent=1))
