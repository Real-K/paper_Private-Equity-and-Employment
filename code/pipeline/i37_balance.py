# -*- coding: utf-8 -*-
"""I-37 균형표 (Table 1 Panel) — 처치 vs 매칭대조 vs 매칭 전 모집단.

referee 필수 요구. 매칭이 무엇을 달성했는지 보이려면 **매칭 전/후를 함께** 내야 한다.
품질 지표는 표준화 차이 ND = (m_t − m_c) / sqrt((v_t + v_c)/2). Imbens–Rubin 기준 |ND| < 0.25.
매칭 전 열은 각 이벤트 시점의 적격 never-treated 에서 200개를 무작위 추출해 추정한다.
"""
import numpy as np
from h30_common import load, deals, build, emit, SEED, widx, flow, dflow

rng = np.random.default_rng(SEED)
NPOP = 200
print("[I-37] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, cache = build(G, allt, PE)
Ev, Hv, Sv, idx, mset = G["Ev"], G["Hv"], G["Sv"], G["idx"], G["mset"]
never = np.flatnonzero(~idx.isin(PE))
adpt = G["adpt_arr"]
print(f"  이벤트 {len(EV)} · never-treated {len(never):,}")

def feats(row, m0):
    """사전 특성 8종."""
    c = cache.get(m0)
    if c is None: return None
    Ep, g, sb, gb, ageb = c
    e = Ep[row]
    if not (np.isfinite(e) and e >= 5): return None
    w = widx(G, m0, -12, -1)
    if len(w) != 12: return None
    h, s, ee = Hv[row, w], Sv[row, w], Ev[row, w]
    if not (np.isfinite(h).all() and np.isfinite(s).all() and np.isfinite(ee).all()): return None
    den = np.nanmean(ee)
    if not (np.isfinite(den) and den > 0): return None
    w2 = widx(G, m0, -24, -13)
    z2 = float((Hv[row, w2] == 0).mean()) if (len(w2) == 12 and np.isfinite(Hv[row, w2]).all()) else np.nan
    age = (m0 - adpt[row]) / 12.0 if np.isfinite(adpt[row]) else np.nan
    return dict(emp=float(e), logemp=float(np.log(e)),
                growth=float(g[row]) if np.isfinite(g[row]) else np.nan,
                age=age, hire=float(np.nansum(h) / den), sep=float(np.nansum(s) / den),
                zero=float((h == 0).mean()), zero_pre=z2)

VARS = [("emp", "Employees", 1), ("logemp", "Log employees", 3),
        ("growth", "Pre-deal employment growth", 3), ("age", "Firm age (years)", 1),
        ("hire", "Hiring rate, prior 12 months", 3),
        ("sep", "Separation rate, prior 12 months", 3),
        ("zero", "Share of no-hire months, −12 to −1", 3),
        ("zero_pre", "Share of no-hire months, −24 to −13", 3)]

T, C, Pp = {k: [] for k, _, _ in VARS}, {k: [] for k, _, _ in VARS}, {k: [] for k, _, _ in VARS}
for e in EV:
    ft = feats(e["ti"], e["m0"])
    if ft is None: continue
    fc = [feats(k, e["m0"]) for k in e["ctrls"]]
    fc = [x for x in fc if x]
    if not fc: continue
    for k, _, _ in VARS:
        T[k].append(ft[k]); C[k].append(float(np.nanmean([x[k] for x in fc])))
    for i in rng.choice(never, size=min(NPOP, len(never)), replace=False):
        fp = feats(int(i), e["m0"])
        if fp:
            for k, _, _ in VARS: Pp[k].append(fp[k])
n_ev = len(T["emp"])
print(f"  균형표 이벤트 {n_ev} · 모집단 관측 {len(Pp['emp']):,}")

def nd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    v = (np.var(a, ddof=1) + np.var(b, ddof=1)) / 2
    return float((a.mean() - b.mean()) / np.sqrt(v)) if v > 0 else np.nan
def ms(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean()), float(np.median(x)), float(x.std(ddof=1))

rows, worst = [], 0.0
print(f"\n  {'변수':<34} {'처치':>9} {'매칭대조':>9} {'ND':>7} {'모집단':>10} {'ND(전)':>8}")
for k, lab, dec in VARS:
    mt, md, st = ms(T[k]); mc, _, _ = ms(C[k]); mp, _, _ = ms(Pp[k])
    n1, n0 = nd(T[k], C[k]), nd(T[k], Pp[k])
    worst = max(worst, abs(n1))
    rows.append(dict(var=lab, treated=round(mt, dec), treated_sd=round(st, dec),
                     matched=round(mc, dec), nd_matched=round(n1, 3),
                     pool=round(mp, dec), nd_pool=round(n0, 3), dec=dec))
    print(f"  {lab:<34} {mt:>9.{dec}f} {mc:>9.{dec}f} {n1:>+7.3f} {mp:>10.{dec}f} {n0:>+8.3f}")
print(f"\n  매칭 후 최대 |ND| = {worst:.3f}  (Imbens–Rubin 기준 0.25 "
      f"{'✓ 충족' if worst < 0.25 else '🔴 초과'})")
ind = [G["ind_arr"][e["ti"]] for e in EV]
emit("I-37", "균형표 (Table 1 Panel D)", "GO" if worst < 0.25 else "PARTIAL",
     {"rows": rows, "n_events": n_ev, "n_pool_obs": len(Pp["emp"]), "pool_draws_per_event": NPOP,
      "max_abs_nd_matched": round(worst, 3), "imbens_rubin_threshold": 0.25,
      "n_industries": int(len(set(ind))),
      "nd_definition": "(mean_t - mean_c) / sqrt((var_t + var_c)/2)"},
     "매칭 전/후 표준화 차이로 매칭 품질을 제시한다",
     f"매칭 후 최대 |ND| {worst:.3f} (기준 0.25) · 이벤트 {n_ev} · 산업 {len(set(ind))}개",
     kill_met=False, n=n_ev)
