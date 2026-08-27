# -*- coding: utf-8 -*-
"""I-57 두 번째 추정량 + 재배치 쌍대비.

목적 둘.
(1) **방법론이 다른 두 번째 추정량**으로 상태의존성을 확인한다. 매칭차분(창 대 창)과 이산시간
    hazard(월별, 이벤트 FE)는 가정이 다르다. 둘이 같은 답을 주면 사양 의존성이 크게 줄어든다.
    I-25 의 삼중교호는 구 지표(무채용월 비중)였다 — 확정 지표(사전 채용률)로 재실행한다.
(2) **재배치 서사의 검정.** 채용 gradient 는 +0.48 인데 고용 gradient 는 없다(오히려 음수).
    회계상 이직이 따라 늘어야 한다. 이를 **쌍대비**로 검정한다 — 같은 기업에서 두 결과대상의
    gradient 차이는 각각을 따로 추정해 빼는 것보다 분산이 작다.

Panel A  hazard 삼중교호 — 확정 상태지표 (연속 · 3분위)
Panel B  결과대상 배터리 — 채용·이직·churn·고용, **양측** RI
Panel C  ★ 쌍대비 — (채용 gradient) − (고용 gradient) 를 같은 이벤트에서 직접 추정
Panel D  churn 초과분 — 물량 벤치마크를 넘어선 순환 증가가 있는가

[메모리] hazard 는 I-02 규율 준수 — grouped binomial 셀접기.
"""
import gc
import numpy as np, pandas as pd
import statsmodels.api as sm
from h30_common import load, deals, build, emit, SEED, qci, NB, widx
from h39_common import SIZE_B

rng = np.random.default_rng(SEED)
NDRAW = 2000
print("[I-57] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Sv, Ev, adpt = G["Hv"], G["Sv"], G["Ev"], G["adpt_arr"]


def blk(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return None
    h, s, e = Hv[row, c].astype(float), Sv[row, c].astype(float), Ev[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(s).all() and np.isfinite(e).all()): return None
    if np.mean(e) < 5: return None
    return float(h.sum()), float(s.sum()), float(np.mean(e))


OUTS = {"hire":  lambda H, S, E: (np.log(H / E) if H > 0 else None),
        "sep":   lambda H, S, E: (np.log(S / E) if S > 0 else None),
        "churn": lambda H, S, E: (np.log((H + S) / E) if (H + S) > 0 else None),
        "emp":   lambda H, S, E: np.log(E)}


def unit(focal, others, m0, gi):
    st = blk(focal, m0, -24, -13)
    if st is None: return None
    po, pr = blk(focal, m0, 1, 12), blk(focal, m0, -12, -1)
    if po is None or pr is None: return None
    rec = {"g": gi, "S": -float(np.log1p(st[0] / st[2])), "lsize": np.log(st[2]),
           "ind": str(G["ind_arr"][focal])[:1],
           "age": ((m0 - adpt[focal]) / 12.0 if np.isfinite(adpt[focal]) else np.nan)}
    w36 = blk(focal, m0, -36, -25)
    rec["grow"] = np.log(st[2] / w36[2]) if (w36 and w36[2] > 0) else np.nan
    for k, f in OUTS.items():
        a_, b_ = f(*pr), f(*po)
        if a_ is None or b_ is None: rec[k] = np.nan; continue
        cs = []
        for o in others:
            p2, r2 = blk(o, m0, 1, 12), blk(o, m0, -12, -1)
            if p2 is None or r2 is None: continue
            x_, y_ = f(*r2), f(*p2)
            if x_ is not None and y_ is not None: cs.append(y_ - x_)
        rec[k] = (b_ - a_) - float(np.mean(cs)) if cs else np.nan
    return rec


# ── 교정 설계 (I-56·I-58): 상태균형 매칭 + 거울 위약 ──
idx = G["idx"]; mset, ind_arr = G["mset"], G["ind_arr"]
NOTPE = np.asarray(~idx.isin(set(PE)))
_c, _s = {}, {}
def cellarr(m0):
    if m0 in _c: return _c[m0]
    iw = [mset[m] for m in range(m0 - 6, m0) if m in mset]
    i18 = [mset[m] for m in range(m0 - 18, m0 - 12) if m in mset]
    if not iw or not i18: _c[m0] = None; return None
    with np.errstate(all="ignore"):
        Ep = np.nanmean(Ev[:, iw], axis=1); g = Ep / np.nanmean(Ev[:, i18], axis=1) - 1
    _c[m0] = (Ep, g, np.digitize(Ep, SIZE_B, right=False),
              np.where(np.isnan(g), -1, np.digitize(g, [-0.10, 0.10])),
              np.where(np.isnan(adpt), -1, np.digitize((m0 - adpt) / 12.0, [5, 15])))
    return _c[m0]
def Sall(m0):
    if m0 in _s: return _s[m0]
    c = widx(G, m0, -24, -13)
    if len(c) != 12: _s[m0] = (None, None); return _s[m0]
    h = Hv[:, c].astype(float); e = Ev[:, c].astype(float)
    ok = np.isfinite(h).all(1) & np.isfinite(e).all(1) & (np.nanmean(e, 1) >= 5)
    S = np.full(Hv.shape[0], np.nan)
    S[ok] = -np.log1p(h[ok].sum(1) / np.nanmean(e[ok], 1))
    fin = np.isfinite(S); b = np.full(Hv.shape[0], -9)
    if fin.sum() >= 50:
        q1, q2 = np.percentile(S[fin], [33.33, 66.67])
        b = np.where(fin, np.digitize(S, [q1, q2]), -9)
    _s[m0] = (S, b); return _s[m0]
def match(focal, m0, k=5):
    c = cellarr(m0)
    if c is None: return None
    Ep, g, sb, gb, ageb = c
    if not (np.isfinite(Ep[focal]) and Ep[focal] >= 5): return None
    S, bins = Sall(m0)
    if S is None or not np.isfinite(S[focal]) or bins[focal] == -9: return None
    same = (NOTPE & (ind_arr == ind_arr[focal]) & (sb == sb[focal]) & (gb == gb[focal])
            & (ageb == ageb[focal]) & (Ep >= 5) & np.isfinite(Ep) & (bins == bins[focal]))
    cand = np.flatnonzero(same); cand = cand[cand != focal]
    if len(cand) == 0: return None
    gt = g[focal] if np.isfinite(g[focal]) else 0.0
    gc = np.where(np.isfinite(g[cand]), g[cand], 0.0)
    d = ((np.log(Ep[cand]) - np.log(Ep[focal])) / 0.9) ** 2 + \
        ((np.clip(gc, -1, 2) - np.clip(gt, -1, 2)) / 0.35) ** 2
    return cand[np.argsort(d)[:k]]

T, P = [], []
for gi, e in enumerate(EV):
    ct = match(e["ti"], e["m0"])
    if ct is None: continue
    u = unit(e["ti"], [int(x) for x in ct], e["m0"], gi)
    if u: T.append(u)
    for k in ct:                                   # 거울 위약: k 자신을 재매칭
        ck = match(int(k), e["m0"])
        if ck is None: continue
        v = unit(int(k), [int(x) for x in ck], e["m0"], gi)
        if v: P.append(v)
print(f"  처치 {len(T)} · 유사처치 {len(P)}  (상태균형 매칭 + 거울 위약)")


def design(rows):
    cols = [np.ones(len(rows)), np.array([r["lsize"] for r in rows])]
    for k in ("grow", "age"):
        v = np.array([r[k] for r in rows], float); m = np.isfinite(v)
        cols.append(np.where(m, v, np.median(v[m]) if m.any() else 0.0))
    for s_ in sorted({r["ind"] for r in rows})[1:]:
        cols.append(np.array([1.0 if r["ind"] == s_ else 0.0 for r in rows]))
    return np.column_stack(cols)


def grad(rows, key, cuts=None):
    sub = [r for r in rows if np.isfinite(r.get(key, np.nan))]
    if len(sub) < 30: return None, 0
    y = np.array([r[key] for r in sub]); x = np.array([r["S"] for r in sub])
    if cuts is not None: y = np.clip(y, cuts[0], cuts[1])
    C = design(sub)
    r_ = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    yr, xr = r_(y), r_(x); d = float(np.sum(xr * xr))
    return (float(np.sum(xr * yr) / d) if d > 0 else None), len(sub)


def ri2(key, tag, two_sided=True):
    sub = [r for r in T if np.isfinite(r.get(key, np.nan))]
    if len(sub) < 30: print(f"  {tag}: 표본 부족"); return None
    cuts = tuple(np.percentile([r[key] for r in sub], [5, 95]))
    obs, n_t = grad(T, key, cuts)
    cells = sorted({r["g"] for r in P}); byg = {c: [r for r in P if r["g"] == c] for c in cells}
    null = []
    for _ in range(NDRAW):
        d_ = []
        for i in rng.permutation(len(cells)):
            d_ += byg[cells[i]]
            if len(d_) >= n_t: break
        v_, _ = grad(d_[:n_t], key, cuts)
        if v_ is not None: null.append(v_)
    null = np.array(null)
    pu = (int((null >= obs).sum()) + 1) / (len(null) + 1)
    pl = (int((null <= obs).sum()) + 1) / (len(null) + 1)
    p2 = min(1.0, 2 * min(pu, pl))
    out = {"observed": round(obs, 4), "n": n_t, "null_mean": round(float(null.mean()), 4),
           "null_sd": round(float(null.std()), 4), "null_ci": qci(null),
           "RI_p_upper": round(float(pu), 4), "RI_p_lower": round(float(pl), 4),
           "RI_p_two_sided": round(float(p2), 4),
           "z": round(float((obs - null.mean()) / null.std()), 2),
           "excess": round(float(obs - null.mean()), 4),
           "sig": bool((p2 if two_sided else pu) < 0.05)}
    mark = "✓" if out["sig"] else "✗"
    print(f"  {tag:<26} obs {out['observed']:>+7.4f} · null {out['null_mean']:>+7.4f}"
          f"(SD {out['null_sd']:.4f}) · 초과 {out['excess']:>+7.4f} · z={out['z']:>5.2f} · "
          f"p(상){out['RI_p_upper']:.3f} p(양측){out['RI_p_two_sided']:.3f} {mark}  n={n_t}")
    return out


print("\n[Panel B] 결과대상 배터리 — 양측 RI")
PB = {}
for k, tag in (("hire", "Δlog 채용률"), ("sep", "Δlog 이직률"),
               ("churn", "Δlog churn (H+S)/E"), ("emp", "Δlog 고용")):
    PB[k] = ri2(k, tag)

print("\n[Panel C] ★ 쌍대비 — (채용 − 고용) gradient, 같은 이벤트")
PC = {}
for a, b, tag in (("hire", "emp", "채용 − 고용"), ("hire", "sep", "채용 − 이직"),
                  ("churn", "emp", "churn − 고용")):
    for r in T + P:
        r[f"{a}_{b}"] = (r[a] - r[b]) if (np.isfinite(r.get(a, np.nan))
                                          and np.isfinite(r.get(b, np.nan))) else np.nan
    PC[tag] = ri2(f"{a}_{b}", tag)

print("\n[Panel A] hazard 삼중교호 — 확정 상태지표")
PA = {}
try:
    Svals = np.array([r["S"] for r in T])
    q1, q2 = np.percentile(Svals, [33.33, 66.67])
    Smap = {}
    for gi, e in enumerate(EV):
        st = blk(e["ti"], e["m0"], -24, -13)
        if st: Smap[gi] = -float(np.log1p(st[0] / st[2]))
    rows = []
    for gi, e in enumerate(EV):
        if gi not in Smap: continue
        S = Smap[gi]; hi = 1.0 if S > q2 else 0.0
        ctb = match(e["ti"], e["m0"])
        if ctb is None: continue
        for row, tr in [(e["ti"], 1)] + [(int(k), 0) for k in ctb]:
            for post, (a, b) in ((0, (-12, -1)), (1, (1, 12))):
                c = widx(G, e["m0"], a, b)
                if len(c) != 12: continue
                h = Hv[row, c].astype(float)
                if not np.isfinite(h).all(): continue
                rows.append((gi, tr, post, hi, S, float((h > 0).sum()), 12.0))
    D = pd.DataFrame(rows, columns=["ev", "tr", "post", "hi", "S", "k", "n"])
    D = D.groupby(["ev", "tr", "post", "hi", "S"], as_index=False)[["k", "n"]].sum()
    print(f"  셀 {len(D)} (grouped binomial)")
    for tag, col in (("연속 S", "S"), ("상위3분위", "hi")):
        X = pd.get_dummies(D["ev"].astype("category"), drop_first=True, dtype=float).to_numpy()
        base = np.column_stack([D["tr"], D["post"], D["tr"] * D["post"],
                                D[col], D[col] * D["post"], D["tr"] * D[col],
                                D["tr"] * D["post"] * D[col]]).astype(float)
        Xf = np.column_stack([base, X]).astype(np.float32)
        m = sm.GLM(np.column_stack([D["k"], D["n"] - D["k"]]).astype(float), Xf,
                   family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit(
                   cov_type="cluster", cov_kwds={"groups": D["ev"].to_numpy()}, maxiter=100)
        b_, se_ = float(m.params[6]), float(m.bse[6])
        PA[tag] = {"coef": round(b_, 4), "se": round(se_, 4),
                   "HR": round(float(np.exp(b_)), 4),
                   "HR_ci": [round(float(np.exp(b_ - 1.96 * se_)), 4),
                             round(float(np.exp(b_ + 1.96 * se_)), 4)],
                   "z": round(b_ / se_, 2), "sig": bool(abs(b_ / se_) > 1.96),
                   "n_cells": int(len(D))}
        print(f"  {tag:<12} treated×post×S {b_:>+7.4f} (SE {se_:.4f}) HR {PA[tag]['HR']:.4f} "
              f"{PA[tag]['HR_ci']} z={PA[tag]['z']} {'✓' if PA[tag]['sig'] else '✗'}")
        del Xf, X, base; gc.collect()
except Exception as ex:
    PA["error"] = str(ex); print("  실패:", ex)

hire, emp = PB.get("hire"), PB.get("emp")
pair = PC.get("채용 − 고용")
verdict = (
    f"[A] hazard 삼중교호(확정 지표): 연속 {PA.get('연속 S', {}).get('HR', '-')} "
    f"{PA.get('연속 S', {}).get('HR_ci', '')} · 3분위 {PA.get('상위3분위', {}).get('HR', '-')} "
    f"{PA.get('상위3분위', {}).get('HR_ci', '')}. "
    f"[B] 채용 {hire['observed']:+.4f}(p상 {hire['RI_p_upper']:.3f}) vs 고용 "
    f"{emp['observed']:+.4f}(p양측 {emp['RI_p_two_sided']:.3f}, z {emp['z']}). "
    f"[C] ★ 쌍대비 채용−고용 {pair['observed']:+.4f} (귀무 {pair['null_mean']:+.4f}, "
    f"z {pair['z']}, p상 {pair['RI_p_upper']:.4f}) "
    + ("— 재배치 서사가 쌍대비에서 직접 지지된다." if pair["RI_p_upper"] < 0.05 else
       "— 쌍대비도 유의하지 않다. 재배치는 여전히 미확인."))
emit("I-57", "두 번째 추정량(hazard) + 재배치 쌍대비",
     "GO" if (pair and pair["RI_p_upper"] < 0.05) else "PARTIAL",
     {"panelA_hazard_triple": PA, "panelB_outcomes": PB, "panelC_paired": PC,
      "n_treated": len(T), "n_placebo": len(P), "n_draws": NDRAW},
     "방법론이 다른 두 번째 추정량이 같은 답을 주는가 · 재배치가 쌍대비에서 지지되는가",
     verdict, kill_met=False, n=len(T))
