# -*- coding: utf-8 -*-
"""I-60 교정 설계(상태균형 매칭 + 거울 위약)에서의 사양곡선.

I-56 의 사양곡선은 공변량 lsize 를 [−12,−1] 고용으로 잡아 I-47/I-53/I-58 관례([−24,−13])와
달랐다. 표와 본문이 한 관례를 써야 하므로 I-58 관례로 전량 재산출한다.
윈저 절단점은 처치표본에서 산출해 위약에 동일 절대값으로 적용한다.
"""
import numpy as np, json
from h30_common import load, deals, build, emit, SEED, qci, widx
from h39_common import SIZE_B
rng = np.random.default_rng(SEED); NDRAW = 2000
print("[I-60] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Sv, Ev, adpt, idx = G["Hv"], G["Sv"], G["Ev"], G["adpt_arr"], G["idx"]
mset, ind_arr = G["mset"], G["ind_arr"]; NOTPE = np.asarray(~idx.isin(set(PE)))
Astar = np.zeros_like(Hv)
Astar[:, 1:] = np.maximum(0.0, np.diff(Ev, axis=1) + Sv[:, 1:]); Astar[:, 0] = Hv[:, 0]
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
    d = ((np.log(Ep[cand])-np.log(Ep[focal]))/0.9)**2 + ((np.clip(gc,-1,2)-np.clip(gt,-1,2))/0.35)**2
    return cand[np.argsort(d)[:k]]
def blk(row, m0, a, b, M=None):
    c = widx(G, m0, a, b)
    if len(c) != (b-a+1): return None
    h = (M if M is not None else Hv)[row, c].astype(float); e = Ev[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.mean(e) < 5: return None
    return float(h.sum()), float(np.mean(e))
def unit(focal, ctrls, m0, gi, M=None):
    st = blk(focal, m0, -24, -13)
    po, pr = blk(focal, m0, 1, 12, M), blk(focal, m0, -12, -1, M)
    if st is None or po is None or pr is None or po[0] <= 0 or pr[0] <= 0: return None
    cs = []
    for o in ctrls:
        p2, r2 = blk(int(o), m0, 1, 12, M), blk(int(o), m0, -12, -1, M)
        if p2 and r2 and p2[0] > 0 and r2[0] > 0:
            cs.append(np.log(p2[0]/p2[1]) - np.log(r2[0]/r2[1]))
    if not cs: return None
    w36 = blk(focal, m0, -36, -25)
    return dict(g=gi, eff=(np.log(po[0]/po[1]) - np.log(pr[0]/pr[1])) - float(np.mean(cs)),
                S=-float(np.log1p(st[0]/st[1])), lsize=np.log(st[1]),
                grow=(np.log(st[1]/w36[1]) if (w36 and w36[1] > 0) else np.nan),
                age=((m0-adpt[focal])/12.0 if np.isfinite(adpt[focal]) else np.nan),
                ind=str(ind_arr[focal])[:1])
def assemble(M=None):
    T, P = [], []
    for gi, e in enumerate(EV):
        ct = match(e["ti"], e["m0"])
        if ct is None: continue
        u = unit(e["ti"], [int(x) for x in ct], e["m0"], gi, M)
        if u: T.append(u)
        for k in ct:
            ck = match(int(k), e["m0"])
            if ck is None: continue
            v = unit(int(k), [int(x) for x in ck], e["m0"], gi, M)
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
def ri(T, P, wins, tag):
    cuts = tuple(np.percentile([r["eff"] for r in T], wins)) if wins else None
    obs = sl(T, cuts); n_t = len(T)
    cells = sorted({r["g"] for r in P}); byg = {c: [r for r in P if r["g"] == c] for c in cells}
    null = []
    for _ in range(NDRAW):
        d_ = []
        for i in rng.permutation(len(cells)):
            d_ += byg[cells[i]]
            if len(d_) >= n_t: break
        v = sl(d_[:n_t], cuts)
        if v is not None: null.append(v)
    null = np.array(null); p = (int((null >= obs).sum())+1)/(len(null)+1)
    o = {"observed": round(obs,4), "n": n_t, "null_mean": round(float(null.mean()),4),
         "null_sd": round(float(null.std()),4), "null_ci": qci(null),
         "RI_p": round(float(p),4), "z": round(float((obs-null.mean())/null.std()),2),
         "excess": round(float(obs-null.mean()),4), "sig": bool(p < 0.05),
         "MDE_RI_80": round(float(2.487*null.std()),4)}
    print(f"  {tag:<34} obs {o['observed']:>+7.4f} · null {o['null_mean']:>+7.4f}"
          f"(SD {o['null_sd']:.4f}) · z={o['z']:>5.2f} · RI p {o['RI_p']:.4f} "
          f"{'✓' if o['sig'] else '✗'}  n={n_t}")
    return o
print("\n[사양곡선] 교정 설계 · I-58 공변량 관례")
T, P = assemble()
R = {"winsor_5_95": ri(T, P, (5,95), "winsor 5/95 (주사양)"),
     "raw": ri(T, P, None, "무조정"),
     "winsor_1_99": ri(T, P, (1,99), "winsor 1/99"),
     "winsor_10_90": ri(T, P, (10,90), "winsor 10/90")}
Ta, Pa = assemble(Astar)
R["implied_hires"] = ri(Ta, Pa, (5,95), "암묵 채용 max(0,ΔE+S)")
m = R["winsor_5_95"]
emit("I-60", "교정 설계 사양곡선", "GO" if m["sig"] else "PARTIAL",
     {"specs": R, "n_draws": NDRAW,
      "design": "상태균형 매칭 + 거울 위약 + FWL(log규모[−24,−13]·사전성장·업력·산업)"},
     "교정 설계에서 사양 선택이 결론을 바꾸는가",
     f"주사양 {m['observed']:+.4f} (귀무 {m['null_mean']:+.4f}, z {m['z']}, RI p {m['RI_p']:.4f}). "
     f"5사양 중 {sum(1 for v in R.values() if v['sig'])}종 유의.",
     kill_met=False, n=m["n"])
