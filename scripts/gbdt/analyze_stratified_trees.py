"""Analyze the stratified ensemble (memo _284): importance by feature/family +
what actually co-occurs inside trees (same root->leaf path = real interaction).

Usage:
  uv run python -m scripts.gbdt.analyze_stratified_trees [artifacts.pkl]

Default artifact path is the runs/ output of scripts.gbdt.stratified_boosting.
"""
import json, pickle, sys
from collections import Counter, defaultdict
import numpy as np

ART = sys.argv[1] if len(sys.argv) > 1 else \
    "runs/gbdt/stratified/sp500_up_20pct_50d_dd10pct_maxtune/artifacts.pkl"
art = pickle.load(open(ART, "rb"))
records, feat_cols, groups = art["records"], art["feat_cols"], art["groups"]
fam_of = {c: g for g, cols in groups.items() for c in cols}
N = len(records)

# ---------- 1) ensemble importance ----------
gain = Counter(); used = Counter(); avail = Counter()
for r in records:
    for c in r["cols"]:
        avail[c] += 1
    for c, v in r["total_gain"].items():
        gain[c] += v; used[c] += 1
total_gain = sum(gain.values())

print(f"=== STRATIFIED ENSEMBLE ({N} trees) — importance ===")
print(f"\n--- top 25 features by total gain (share of {total_gain:,.0f}) ---")
print(f"{'feature':38}{'family':16}{'gain%':>7}{'avail':>6}{'used':>6}{'use%':>6}")
for c, v in gain.most_common(25):
    print(f"{c:38}{fam_of[c]:16}{100*v/total_gain:>6.2f}%{avail[c]:>6}{used[c]:>6}"
          f"{100*used[c]/avail[c]:>5.0f}%")

# family shares: stratified vs iter-0 (same classifier, apples-to-apples)
it0 = json.load(open("results/gbdt/experiments/sp500_up_20pct_50d_dd10pct_maxtune/"
                     "loop/iter_0_request.json"))["diagnostics"]["feature_importance"]
it0 = {k: float(v) for k, v in it0.items()}
t0sum = sum(it0.values())
fam_gain = defaultdict(float); fam_it0 = defaultdict(float)
for c, v in gain.items():   fam_gain[fam_of[c]] += v
for c, v in it0.items():    fam_it0[fam_of.get(c, "?")] += v
print(f"\n--- family gain share: stratified vs iter-0 baseline ---")
print(f"{'family':16}{'n_feat':>7}{'strat%':>8}{'iter0%':>8}{'shift':>8}")
for g in sorted(fam_gain, key=lambda x: -fam_gain[x]):
    s, b = 100*fam_gain[g]/total_gain, 100*fam_it0[g]/t0sum
    print(f"{g:16}{len(groups[g]):>7}{s:>7.1f}%{b:>7.1f}%{s-b:>+8.1f}")

# per-feature usage-rate leaders/laggards (min 20 avail)
rate = {c: used[c]/avail[c] for c in avail if avail[c] >= 20}
print(f"\n--- 'when offered, how often used' (avail>=20) — top 10 / bottom 10 ---")
for c in sorted(rate, key=lambda x: -rate[x])[:10]:
    print(f"  {rate[c]:>5.0%}  {c} [{fam_of[c]}] (avail {avail[c]})")
print("  ...")
for c in sorted(rate, key=lambda x: rate[x])[:10]:
    print(f"  {rate[c]:>5.0%}  {c} [{fam_of[c]}] (avail {avail[c]})")

# ---------- 2) same-path co-occurrence ----------
pair_path = Counter(); pair_avail = Counter()
for r in records:
    cs = set(r["cols"])
    for a, b in r["path_pairs"]:
        pair_path[(a, b)] += 1
    # co-availability only for pairs that ever co-path (keep counter small later)
for r in records:
    cs = sorted(set(r["cols"]))
    s = set(cs)
    for (a, b) in pair_path:
        if a in s and b in s:
            pair_avail[(a, b)] += 1

print(f"\n=== SAME-PATH CO-OCCURRENCE (features on one root->leaf path) ===")
cross = [(p, n) for p, n in pair_path.items() if fam_of[p[0]] != fam_of[p[1]]]
within = [(p, n) for p, n in pair_path.items() if fam_of[p[0]] == fam_of[p[1]]]
print(f"\n--- top 15 CROSS-family path pairs (count | co-avail | rate) ---")
for (a, b), n in sorted(cross, key=lambda x: -x[1])[:15]:
    ca = pair_avail[(a, b)]
    print(f"  {n:>4} |{ca:>5} |{n/ca:>5.0%}  {a} [{fam_of[a]}]  x  {b} [{fam_of[b]}]")
print(f"\n--- top 10 WITHIN-family path pairs (the <=2 cap in action) ---")
for (a, b), n in sorted(within, key=lambda x: -x[1])[:10]:
    ca = pair_avail[(a, b)]
    print(f"  {n:>4} |{ca:>5} |{n/ca:>5.0%}  {a}  x  {b}  [{fam_of[a]}]")

# family x family matrix (share of all path-pair counts)
fp = Counter()
for (a, b), n in pair_path.items():
    fa, fb = sorted((fam_of[a], fam_of[b]))
    fp[(fa, fb)] += n
tot_fp = sum(fp.values())
fams = sorted(groups, key=lambda g: -fam_gain[g])
print(f"\n--- family x family same-path share (% of all path-pairs) ---")
print(f"{'':16}" + "".join(f"{g[:7]:>8}" for g in fams))
for i, ga in enumerate(fams):
    row = f"{ga:16}"
    for j, gb in enumerate(fams):
        if j < i: row += f"{'':>8}"
        else:
            k = tuple(sorted((ga, gb)))
            row += f"{100*fp.get(k,0)/tot_fp:>7.1f}%"
    print(row)

# ---------- 3) calendar pair integrity ----------
PAIRS = [("dow_sin","dow_cos"), ("dom_sin","dom_cos"), ("moy_sin","moy_cos"),
         ("moq_sin","moq_cos"), ("qoy_sin","qoy_cos")]
print(f"\n=== CALENDAR sin/cos pair integrity ===")
print(f"{'pair':12}{'offered':>8}{'both_used':>10}{'one_used':>9}{'same_path':>10}")
for a, b in PAIRS:
    off = bu = ou = sp = 0
    for r in records:
        if a not in r["cols"]: continue
        off += 1
        ua, ub = a in r["total_gain"], b in r["total_gain"]
        bu += ua and ub; ou += ua != ub
        if (min(a,b), max(a,b)) in {tuple(p) for p in r["path_pairs"]}: sp += 1
    print(f"{a[:-4]:12}{off:>8}{bu:>10}{ou:>9}{sp:>10}")

# ---------- 4) F18 partners on paths ----------
f18_partner = Counter()
for (a, b), n in pair_path.items():
    fa, fb = fam_of[a], fam_of[b]
    if fa == "F18" and fb != "F18": f18_partner[b] += n
    elif fb == "F18" and fa != "F18": f18_partner[a] += n
print(f"\n=== top 12 non-F18 features co-pathing with fundamentals ===")
for c, n in f18_partner.most_common(12):
    print(f"  {n:>5}  {c} [{fam_of[c]}]")
