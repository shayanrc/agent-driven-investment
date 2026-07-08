import json
from collections import defaultdict

RUN = "results/gbdt/experiments/sp500_up_20pct_50d_dd10pct_maxtune"
imp = json.load(open(f"{RUN}/loop/iter_0_request.json"))["diagnostics"]["feature_importance"]
imp = {k: float(v) for k, v in imp.items()}
total = sum(imp.values())

def classify(f):
    # opt-in pools (exact by prefix)
    if f.startswith("fund_"):                              return ("fundamentals", "F18 fundamentals")
    if f.startswith("vwap_dev"):                           return ("vwap", "F20 vwap-dev")
    if f.startswith("moq_") or f.startswith("qoy_"):       return ("calendar2", "F21 calendar2")
    # technical, ordered so specific patterns win
    if "_xs_" in f:                                        return ("technical", "cross-sectional")
    if "outside_band" in f or "outside" in f:              return ("technical", "persistence/bands (F16)")
    if any(s in f for s in ("garman_klass","parkinson","yang_zhang","realized_vol","vol_pct","volatility","_vol_","vol_change","vol_ret_corr","vol_of_vol")):
                                                           return ("technical", "volatility")
    if "drawdown" in f:                                    return ("technical", "drawdown/regime")
    if any(s in f for s in ("volume","dollar_move","amihud","turnover","illiq","obv")):
                                                           return ("technical", "volume/liquidity")
    if any(s in f for s in ("sma_distance","ema_","macd","sma_","price_to","above_sma","bollinger","dist_")):
                                                           return ("technical", "trend/moving-avg")
    if f in ("fiscal_year_end_week","budget_week","diwali_week","fomc_week") or any(f.startswith(s) for s in ("dow_","dom_","moy_","week","is_")):
                                                           return ("technical", "calendar (F15)")
    if any(s in f for s in ("stock_return","rel_strength","index_return","_return_","momentum","roc_","reversal","runup","beta","skew","kurt")):
                                                           return ("technical", "returns/momentum")
    if any(s in f for s in ("true_range","atr","high_low","_range","gap")):
                                                           return ("technical", "range/gap")
    if "zscore" in f:                                      return ("technical", "zscore/level")
    return ("technical", "other")

pool_imp = defaultdict(float); pool_n = defaultdict(int)
grp_imp = defaultdict(float); grp_n = defaultdict(int); grp_top = {}
grp_pool = {}
other = []
for f, v in imp.items():
    pool, grp = classify(f)
    pool_imp[pool] += v; pool_n[pool] += 1
    grp_imp[grp] += v; grp_n[grp] += 1; grp_pool[grp] = pool
    if grp not in grp_top or v > grp_top[grp][1]:
        grp_top[grp] = (f, v)
    if grp == "other":
        other.append((f, v))

print(f"=== iter-0 baseline · total gain across {len(imp)} features = {total:.1f} ===\n")
print("=== BY TOP-LEVEL POOL ===")
print(f"{'pool':13}{'n':>4}{'gain%':>8}{'sum':>9}{'mean':>8}")
for p in sorted(pool_imp, key=lambda x: -pool_imp[x]):
    print(f"{p:13}{pool_n[p]:>4}{100*pool_imp[p]/total:>7.1f}%{pool_imp[p]:>9.1f}{pool_imp[p]/pool_n[p]:>8.2f}")

print("\n=== BY SEMANTIC GROUP ===")
print(f"{'group':26}{'pool':13}{'n':>4}{'gain%':>8}{'mean':>8}  top[gain]")
for g in sorted(grp_imp, key=lambda x: -grp_imp[x]):
    tf, tv = grp_top[g]
    print(f"{g:26}{grp_pool[g]:13}{grp_n[g]:>4}{100*grp_imp[g]/total:>7.1f}%{grp_imp[g]/grp_n[g]:>8.2f}  {tf}[{tv:.1f}]")

print(f"\n=== 'other' bucket ({len(other)} feats, verify no misclass) ===")
for f, v in sorted(other, key=lambda x: -x[1]):
    print(f"  {f}  {v:.2f}")
