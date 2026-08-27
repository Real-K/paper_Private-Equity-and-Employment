# -*- coding: utf-8 -*-
"""I-63 상태균형 설계에서의 상대고용 경로 (+12 / +24 / +36 개월).

원고 §6.5 와 표 4 Panel E 는 상대고용 경로를 -0.0564 / -0.0208 / -0.1199 로 인용하는데,
출처 `I55_employment_horizons.json` 은 스크립트 없이 inline 으로 산출됐고 처치 n 이
295 / 251 / 210 이라 주 설계(286)와 다르다. 같은 매칭·같은 위약으로 다시 낸다.

결과: log E(m0+h) - log E(기준창 -6:-1), 대조군 평균 차감, state 에 회귀 (FWL).
"""
import numpy as np, json
from h30_common import load, deals, build, emit, SEED, qci, widx, rel_log
from h39_common import SIZE_B
rng = np.random.default_rng(SEED); NDRAW = 2000
print("[I-67] 로딩...")
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
def Sall(m0):
    if m0 in _s: return _s[m0]
    c = widx(G, m0, -24, -13)
    if len(c) != 12: _s[m0] = (None, None); return _s[m0]
    h = Hv[:, c].astype(float); e = Ev[:, c].astype(float)
    ok = np.isfinite(h).all(1) & np.isfinite(e).all(1) & (np.nanmean(e, 1) >= 5)
    S = np.full(Hv.shape[0], np.nan); S[ok] = -np.log1p(h[ok].sum(1)/np.nanmean(e[ok], 1))
    fin = np.isfinite(S); b = np.full(Hv.shape[0], -9)
    if fin.sum() >= 50:
        q1, q2 = np.percentile(S[fin], [33.33, 66.67]); b = np.where(fin, np.digitize(S, [q1, q2]), -9)
    _s[m0] = (S, b); return _s[m0]
def match(focal, m0, k=5):
    c = cellarr(m0)
    if c is None: return None
    Ep, g, szb, gb, ageb = c
    if not (np.isfinite(Ep[focal]) and Ep[focal] >= 5): return None
    S, bins = Sall(m0)
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
def unit(focal, ctrls, m0, gi, h):
    st = blk(focal, m0, -24, -13)
    if st is None: return None
    y = rel_log(G, focal, np.asarray(ctrls, int), m0, k=h)
    if not np.isfinite(y): return None
    w36 = blk(focal, m0, -36, -25)
    return dict(g=gi, eff=float(y), S=-float(np.log1p(st[0]/st[1])), lsize=np.log(st[1]),
                grow=(np.log(st[1]/w36[1]) if (w36 and w36[1] > 0) else np.nan),
                age=((m0-adpt[focal])/12.0 if np.isfinite(adpt[focal]) else np.nan),
                ind=str(ind_arr[focal])[:1])
def assemble(h):
    T, P = [], []
    for gi, e in enumerate(EV):
        ct = match(e["ti"], e["m0"])
        if ct is None: continue
        u = unit(e["ti"], [int(x) for x in ct], e["m0"], gi, h)
        if u: T.append(u)
        for k in ct:
            ck = match(int(k), e["m0"])
            if ck is None: continue
            v = unit(int(k), [int(x) for x in ck], e["m0"], gi, h)
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
    if obs is None: return None
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
    ex = obs - null.mean()
    o = {"observed": round(obs, 4), "n_treated": n_t, "n_placebo": len(P),
         "null_mean": round(float(null.mean()), 4), "null_sd": round(sd, 4), "null_ci": qci(null),
         "excess": round(float(ex), 4), "excess_ci": [round(float(ex-1.96*sd), 4), round(float(ex+1.96*sd), 4)],
         "z": round(float(ex/sd), 2), "p_two_sided": round(float(min(1.0, 2*min(p_hi, p_lo))), 4),
         "sig": bool(min(p_hi, p_lo) < 0.025)}
    print(f"  {tag:<26} obs {o['observed']:>+7.4f} · null {o['null_mean']:>+7.4f}(SD {o['null_sd']:.4f})"
          f" · z={o['z']:>5.2f} · p2 {o['p_two_sided']:.4f} {'✓' if o['sig'] else '✗'}  n={n_t}")
    return o
R = {}
for h in (12, 24, 36):
    T, P = assemble(h); R[f"h{h}"] = ri(T, P, f"상대고용 +{h}개월")
sig = [k for k, v in R.items() if v and v["sig"]]
emit("I-67", "교정 설계에서의 상대고용 경로", "GO" if sig else "PARTIAL",
     {"horizons": R, "n_draws": NDRAW,
      "design": "I-60/I-61 과 동일한 상태균형 매칭 + 거울 위약. 결과만 상대고용 경로로 교체."},
     "상대고용 경로의 state gradient 가 교정 설계에서도 음(-)으로 유지되는가",
     " · ".join(f"+{h[1:]}m {R[h]['observed']:+.4f} (z {R[h]['z']}, p2 {R[h]['p_two_sided']})"
               for h in ("h12", "h24", "h36") if R[h]),
     kill_met=False, n=(R.get("h12") or {}).get("n_treated"))
