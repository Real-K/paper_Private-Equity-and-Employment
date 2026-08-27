# -*- coding: utf-8 -*-
"""I-62 두 가지 설계 취약점의 직접 검정.

(1) 사전추세의 공유기준 문제. 표 7 의 사전추세는 결과창을 (−24:−13)→(−12:−1) 로 잡는데,
    state 자체가 (−24:−13) 에서 산출된다. 분모/기준을 공유하므로 기계적 음의 상관이 생긴다.
    → state 를 (−36:−25) 로 옮겨 결과창과 완전히 분리한 뒤 다시 잰다.
(2) 무채용 구간 탈락. 주 사양은 사전·사후 창 모두 채용>0 을 요구해 0 채용 firm-window 를
    표본에서 뺀다. 결과변수에 대한 선택이다. → asinh/log1p 로 0 을 살려 재추정한다.

Panel A  사전추세: 초기 state(−36:−25) → 결과 (−24:−13)→(−12:−1)
Panel B  본 추정: 초기 state(−36:−25) → 결과 (−12:−1)→(+1:+12)
Panel C  asinh 결과 (0 채용 창 포함)
Panel D  log1p 결과 (0 채용 창 포함)
Panel E  양수 채용 요건이 떨어뜨리는 표본
"""
import numpy as np, json
from h30_common import load, deals, build, emit, SEED, qci, widx
from h39_common import SIZE_B
rng = np.random.default_rng(SEED); NDRAW = 2000
print("[I-66] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Ev, adpt, idx = G["Hv"], G["Ev"], G["adpt_arr"], G["idx"]
mset, ind_arr = G["mset"], G["ind_arr"]; NOTPE = np.asarray(~idx.isin(set(PE)))
_c, _s = {}, {}
def cellarr(m0):
    if m0 in _c: return _c[m0]
    iw = [mset[m] for m in range(m0-6, m0) if m in mset]
    i18 = [mset[m] for m in range(m0-18, m0-12) if m in mset]
    if not iw or not i18: _c[m0] = None; return None
    with np.errstate(all="ignore"):
        Ep = np.nanmean(Ev[:, iw], axis=1); g = Ep/np.nanmean(Ev[:, i18], axis=1) - 1
    _c[m0] = (Ep, g, np.digitize(Ep, SIZE_B, right=False),
              np.where(np.isnan(g), -1, np.digitize(g, [-0.10, 0.10])),
              np.where(np.isnan(adpt), -1, np.digitize((m0-adpt)/12.0, [5, 15])))
    return _c[m0]
def Sall(m0, a, b):
    k = (m0, a, b)
    if k in _s: return _s[k]
    c = widx(G, m0, a, b)
    if len(c) != (b-a+1): _s[k] = (None, None); return _s[k]
    h = Hv[:, c].astype(float); e = Ev[:, c].astype(float)
    ok = np.isfinite(h).all(1) & np.isfinite(e).all(1) & (np.nanmean(e, 1) >= 5)
    S = np.full(Hv.shape[0], np.nan); S[ok] = -np.log1p(h[ok].sum(1)/np.nanmean(e[ok], 1))
    fin = np.isfinite(S); bn = np.full(Hv.shape[0], -9)
    if fin.sum() >= 50:
        q1, q2 = np.percentile(S[fin], [33.33, 66.67]); bn = np.where(fin, np.digitize(S, [q1, q2]), -9)
    _s[k] = (S, bn); return _s[k]
def match(focal, m0, sa, sb, k=5):
    c = cellarr(m0)
    if c is None: return None
    Ep, g, szb, gb, ageb = c
    if not (np.isfinite(Ep[focal]) and Ep[focal] >= 5): return None
    S, bins = Sall(m0, sa, sb)
    if S is None or not np.isfinite(S[focal]) or bins[focal] == -9: return None
    same = (NOTPE & (ind_arr == ind_arr[focal]) & (szb == szb[focal]) & (gb == gb[focal])
            & (ageb == ageb[focal]) & (Ep >= 5) & np.isfinite(Ep) & (bins == bins[focal]))
    cand = np.flatnonzero(same); cand = cand[cand != focal]
    if len(cand) == 0: return None
    gt = g[focal] if np.isfinite(g[focal]) else 0.0
    gc = np.where(np.isfinite(g[cand]), g[cand], 0.0)
    d = ((np.log(Ep[cand])-np.log(Ep[focal]))/0.9)**2 + ((np.clip(gc,-1,2)-np.clip(gt,-1,2))/0.35)**2
    return cand[np.argsort(d)[:k]]
def blk(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b-a+1): return None
    h = Hv[row, c].astype(float); e = Ev[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.mean(e) < 5: return None
    return float(h.sum()), float(np.mean(e))
TRANS = {"log": lambda r: np.log(r), "asinh": lambda r: np.arcsinh(r), "log1p": lambda r: np.log1p(r)}
def y_of(row, m0, pre, post, tr):
    po, pr = blk(row, m0, *post), blk(row, m0, *pre)
    if po is None or pr is None: return None
    if tr == "log" and (po[0] <= 0 or pr[0] <= 0): return None
    f = TRANS[tr]
    return float(f(po[0]/po[1]) - f(pr[0]/pr[1]))
def unit(focal, ctrls, m0, gi, sa, sb, pre, post, tr):
    st = blk(focal, m0, sa, sb); yt = y_of(focal, m0, pre, post, tr)
    if st is None or yt is None: return None
    cs = [y_of(int(o), m0, pre, post, tr) for o in ctrls]
    cs = [v for v in cs if v is not None]
    if not cs: return None
    w36 = blk(focal, m0, -36, -25) if (sa, sb) != (-36, -25) else st
    return dict(g=gi, eff=yt - float(np.mean(cs)), S=-float(np.log1p(st[0]/st[1])),
                lsize=np.log(st[1]),
                grow=(np.log(st[1]/w36[1]) if (w36 and w36[1] > 0) else np.nan),
                age=((m0-adpt[focal])/12.0 if np.isfinite(adpt[focal]) else np.nan),
                ind=str(ind_arr[focal])[:1])
def assemble(sa, sb, pre, post, tr):
    T, P = [], []
    for gi, e in enumerate(EV):
        ct = match(e["ti"], e["m0"], sa, sb)
        if ct is None: continue
        u = unit(e["ti"], [int(x) for x in ct], e["m0"], gi, sa, sb, pre, post, tr)
        if u: T.append(u)
        for k in ct:
            ck = match(int(k), e["m0"], sa, sb)
            if ck is None: continue
            v = unit(int(k), [int(x) for x in ck], e["m0"], gi, sa, sb, pre, post, tr)
            if v: P.append(v)
    return T, P
def sl(rows, cuts=None):
    if len(rows) < 30: return None
    y = np.array([r["eff"] for r in rows]); x = np.array([r["S"] for r in rows])
    if cuts is not None: y = np.clip(y, cuts[0], cuts[1])
    cols = [np.ones(len(y)), np.array([r["lsize"] for r in rows])]
    for k in ("grow", "age"):
        v = np.array([r[k] for r in rows], float); m = np.isfinite(v)
        cols.append(np.where(m, v, np.median(v[m]) if m.any() else 0.0))
    for s_ in sorted({r["ind"] for r in rows})[1:]:
        cols.append(np.array([1.0 if r["ind"] == s_ else 0.0 for r in rows]))
    C = np.column_stack(cols); r_ = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    yr, xr = r_(y), r_(x); d = float(np.sum(xr*xr))
    return float(np.sum(xr*yr)/d) if d > 0 else None
def ri(T, P, tag, wins=(5, 95)):
    cuts = tuple(np.percentile([r["eff"] for r in T], wins)) if wins else None
    obs = sl(T, cuts); n_t = len(T)
    if obs is None: print(f"  {tag}: 표본 부족 n={n_t}"); return None
    cells = sorted({r["g"] for r in P}); byg = {c: [r for r in P if r["g"] == c] for c in cells}
    null = []
    for _ in range(NDRAW):
        d_ = []
        for i in rng.permutation(len(cells)):
            d_ += byg[cells[i]]
            if len(d_) >= n_t: break
        v = sl(d_[:n_t], cuts)
        if v is not None: null.append(v)
    null = np.array(null); sd = float(null.std())
    p_hi = (int((null >= obs).sum())+1)/(len(null)+1); p_lo = (int((null <= obs).sum())+1)/(len(null)+1)
    o = {"observed": round(obs, 4), "n": n_t, "n_placebo": len(P),
         "null_mean": round(float(null.mean()), 4), "null_sd": round(sd, 4), "null_ci": qci(null),
         "p_two_sided": round(float(min(1.0, 2*min(p_hi, p_lo))), 4),
         "z": round(float((obs-null.mean())/sd), 2), "excess": round(float(obs-null.mean()), 4),
         "sig": bool(min(p_hi, p_lo) < 0.025)}
    ex = obs - null.mean(); lo, hi = ex - 1.96*sd, ex + 1.96*sd
    o["excess_ci"] = [round(float(lo), 4), round(float(hi), 4)]
    # 등가성(rule 11): 점추정이 아니라 구간으로 판정한다. SESOI 는 본 gradient 0.7101 과 그 절반.
    o["equiv_vs_effect_0710"] = bool(lo > -0.7101 and hi < 0.7101)
    o["equiv_vs_half_effect_0355"] = bool(lo > -0.3551 and hi < 0.3551)
    o["one_sided_upper_below_effect"] = bool(hi < 0.7101)
    print(f"  {tag:<40} obs {o['observed']:>+7.4f} · null {o['null_mean']:>+7.4f}"
          f"(SD {o['null_sd']:.4f}) · z={o['z']:>5.2f} · p2 {o['p_two_sided']:.4f} "
          f"{'✓' if o['sig'] else '✗'}  n={n_t}")
    return o

print("\n[A] 사전추세 — state (−36:−25), 결과 (−24:−13)→(−12:−1)")
TA, PA_ = assemble(-36, -25, (-24, -13), (-12, -1), "log"); A = ri(TA, PA_, "pre-trend, non-overlapping state")
print("\n[B] 본 추정 — state (−36:−25), 결과 (−12:−1)→(+1:+12)")
TB, PB_ = assemble(-36, -25, (-12, -1), (1, 12), "log"); B = ri(TB, PB_, "post gradient, early state")
print("\n[C/D] 0 채용 창을 살린 변환")
TC, PC_ = assemble(-24, -13, (-12, -1), (1, 12), "asinh"); C = ri(TC, PC_, "asinh(hires/emp)")
TD, PD_ = assemble(-24, -13, (-12, -1), (1, 12), "log1p"); D = ri(TD, PD_, "log1p(hires/emp)")
print("\n[E] 양수 채용 요건이 떨어뜨리는 표본")
TL, PL_ = assemble(-24, -13, (-12, -1), (1, 12), "log"); L = ri(TL, PL_, "log (primary, for reference)")
E = {"n_log": len(TL), "n_asinh": len(TC), "n_dropped_by_positivity": len(TC)-len(TL),
     "share_dropped": round((len(TC)-len(TL))/len(TC), 4) if len(TC) else None,
     "n_placebo_log": len(PL_), "n_placebo_asinh": len(PC_)}
print(f"  log n={E['n_log']} · asinh n={E['n_asinh']} → 양수요건 탈락 {E['n_dropped_by_positivity']}건 ({E['share_dropped']:.1%})")

ok = bool(A and not A["sig"]) and bool(B and B["sig"]) and bool(C and C["sig"]) and bool(D and D["sig"])
emit("I-66", "비중첩 사전추세와 0 채용 창을 살린 강건성", "GO" if ok else "PARTIAL",
     {"panelA_pretrend_early_state": A, "panelB_post_early_state": B,
      "panelC_asinh": C, "panelD_log1p": D, "panelE_positivity_selection": E, "n_draws": NDRAW,
      "design": "I-60 과 동일 매칭·FWL. state 창과 결과 변환만 바꾼다."},
     "state 창을 결과와 분리해도 사전추세는 없고 본 gradient 는 남는가; 0 채용 창을 살려도 남는가",
     f"사전추세 {A['observed'] if A else None} (p2 {A['p_two_sided'] if A else None}); "
     f"초기 state 본 추정 {B['observed'] if B else None} (p2 {B['p_two_sided'] if B else None}); "
     f"asinh {C['observed'] if C else None}, log1p {D['observed'] if D else None}; "
     f"양수요건 탈락 {E['n_dropped_by_positivity']}건.",
     kill_met=False, n=(B or {}).get("n"))
