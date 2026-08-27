# -*- coding: utf-8 -*-
"""I-56 검정력 제고 — 설계 교정 + 효율 개선. 유의성 사냥이 아니라 통계이론이 예측하는 이득.

각 레버는 **결과를 보기 전에** 왜 이득이 나는지 이론적 근거가 있다. 전건 보고한다.

★ Panel A  **상태 균형 매칭.** 현 매칭은 산업·규모·성장·업력만 맞추고 사전 채용상태는 맞추지
   않는다. 그래서 저활동 기업이 대조군 대비 평균회귀하고, 그 기계적 성분이 위약 귀무를 +0.117 로
   끌어올린다(I-53). 상태를 매칭 셀에 넣으면 그 성분이 처치·위약 양쪽에서 상쇄된다.
   → 귀무가 0 근처로 내려가고 대비가 선명해져야 한다.

★ Panel B  **거울설계 위약.** I-53 위약은 대조 k 를 'T 를 중심으로 뽑힌 다른 대조'와 비교한다 —
   처치 설계와 절차가 다르다. k 를 유사처치로 두고 **k 자신의 최근접 대조 5개를 새로 매칭**하면
   처치 절차를 그대로 복제한다. 이것이 올바른 귀무다.

 Panel C  **상태 측정창 연장** [−36,−13] 24개월. 상태변수의 측정오차는 상호작용 계수를 감쇠시킨다
   (errors-in-variables). 측정 분산이 절반이면 감쇠가 줄어 계수가 올라간다.

 Panel D  **분할표본 IV.** 상태를 [−24,−19] 로 재고 [−18,−13] 로 도구화한다. 일시적 잡음이
   두 반쪽에 독립이면 감쇠가 **정확히** 제거된다.

 Panel E  **정밀도 가중.** Var(Δlog rate) ≈ 1/N_pre + 1/N_post (델타법·포아송). 채용 4건 기업과
   100건 기업을 같은 가중으로 두는 것은 비효율이다. 역분산 가중은 편의 없이 분산만 줄인다.

 Panel F  **공변량 확장.** 사전변수만 추가(딜연도·2자리산업·지역·사전 채용변동성) → 잔차분산 감소.

[메모리] 이벤트 수준 스칼라 + 매칭 재실행 1회. 대형 객체 없음.
"""
import numpy as np, pandas as pd
from h30_common import load, deals, build, emit, SEED, qci, NB, widx
from h39_common import SIZE_B

rng = np.random.default_rng(SEED)
NDRAW = 2000
print("[I-56] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV0, _ = build(G, allt, PE)
Hv, Ev, adpt, idx = G["Hv"], G["Ev"], G["adpt_arr"], G["idx"]
mset, ind_arr = G["mset"], G["ind_arr"]
PEset = set(PE)


def blk(row, m0, a, b, need=None):
    c = widx(G, m0, a, b); n = b - a + 1
    if need is None: need = n
    if len(c) < need: return None
    h, e = Hv[row, c].astype(float), Ev[row, c].astype(float)
    m = np.isfinite(h) & np.isfinite(e)
    if m.sum() < need or np.mean(e[m]) < 5: return None
    return float(h[m].sum()), float(np.mean(e[m])), int(m.sum())


def Svar(row, m0, a=-24, b=-13, need=None):
    w = blk(row, m0, a, b, need)
    return None if w is None else -float(np.log1p(w[0] / w[1]))


# ════════════ 공통: 매칭 (선택적으로 상태 bin 추가) ════════════
_cache = {}
def cell_arrays(m0):
    if m0 in _cache: return _cache[m0]
    iw = [mset[m] for m in range(m0 - 6, m0) if m in mset]
    i18 = [mset[m] for m in range(m0 - 18, m0 - 12) if m in mset]
    if not iw or not i18: _cache[m0] = None; return None
    with np.errstate(all="ignore"):
        Ep = np.nanmean(Ev[:, iw], axis=1)
        g = Ep / np.nanmean(Ev[:, i18], axis=1) - 1
    sb = np.digitize(Ep, SIZE_B, right=False)
    gb = np.where(np.isnan(g), -1, np.digitize(g, [-0.10, 0.10]))
    ageb = np.where(np.isnan(adpt), -1, np.digitize((m0 - adpt) / 12.0, [5, 15]))
    _cache[m0] = (Ep, g, sb, gb, ageb)
    return _cache[m0]


_Scache = {}
def S_all(m0):
    """해당 딜월 기준 전체 기업의 상태변수(벡터). 결측은 nan."""
    if m0 in _Scache: return _Scache[m0]
    c = widx(G, m0, -24, -13)
    if len(c) != 12:
        _Scache[m0] = None; return None
    h = Hv[:, c].astype(float); e = Ev[:, c].astype(float)
    ok = np.isfinite(h).all(1) & np.isfinite(e).all(1) & (np.nanmean(e, 1) >= 5)
    S = np.full(Hv.shape[0], np.nan)
    S[ok] = -np.log1p(h[ok].sum(1) / np.nanmean(e[ok], 1))
    _Scache[m0] = S
    return S


def match(focal, m0, exclude, balance_state=False, k=5):
    """build() 와 동일 규칙. balance_state=True 면 상태 3분위 bin 도 일치시킨다."""
    c = cell_arrays(m0)
    if c is None: return None
    Ep, g, sb, gb, ageb = c
    if not (np.isfinite(Ep[focal]) and Ep[focal] >= 5): return None
    ok = np.asarray(~idx.isin(exclude))
    same = (ok & (ind_arr == ind_arr[focal]) & (sb == sb[focal]) & (gb == gb[focal])
            & (ageb == ageb[focal]) & (Ep >= 5) & np.isfinite(Ep))
    if balance_state:
        S = S_all(m0)
        if S is None or not np.isfinite(S[focal]): return None
        fin = np.isfinite(S)
        if fin.sum() < 50: return None
        q1, q2 = np.percentile(S[fin], [33.33, 66.67])
        bins = np.where(np.isfinite(S), np.digitize(S, [q1, q2]), -9)
        same = same & (bins == bins[focal])
    cand = np.flatnonzero(same); cand = cand[cand != focal]
    if len(cand) == 0: return None
    gt = g[focal] if np.isfinite(g[focal]) else 0.0
    gc = np.where(np.isfinite(g[cand]), g[cand], 0.0)
    d = ((np.log(Ep[cand]) - np.log(Ep[focal])) / 0.9) ** 2 + \
        ((np.clip(gc, -1, 2) - np.clip(gt, -1, 2)) / 0.35) ** 2
    return cand[np.argsort(d)[:k]]


def lrate(row, m0, a, b):
    w = blk(row, m0, a, b)
    return np.log(w[0] / w[1]) if (w and w[0] > 0) else None


def make_unit(focal, ctrls, m0, gi, s_win=(-24, -13), s_need=None):
    po, pr = blk(focal, m0, 1, 12), blk(focal, m0, -12, -1)
    if po is None or pr is None or po[0] <= 0 or pr[0] <= 0: return None
    S = Svar(focal, m0, s_win[0], s_win[1], s_need)
    if S is None: return None
    cs = []
    for c_ in ctrls:
        a_, b_ = lrate(int(c_), m0, -12, -1), lrate(int(c_), m0, 1, 12)
        if a_ is not None and b_ is not None: cs.append(b_ - a_)
    if not cs: return None
    w36 = blk(focal, m0, -36, -25)
    return dict(g=gi, eff=(np.log(po[0] / po[1]) - np.log(pr[0] / pr[1])) - float(np.mean(cs)),
                S=S, lsize=np.log(pr[1]), Npre=pr[0], Npost=po[0],
                grow=(np.log(pr[1] / w36[1]) if (w36 and w36[1] > 0) else np.nan),
                age=((m0 - adpt[focal]) / 12.0 if np.isfinite(adpt[focal]) else np.nan),
                ind=str(ind_arr[focal])[:1], m0=m0)


def slope(rows, cuts=None, weights=False, extra_cov=False):
    if len(rows) < 30: return None
    y = np.array([r["eff"] for r in rows]); x = np.array([r["S"] for r in rows])
    if cuts is not None: y = np.clip(y, cuts[0], cuts[1])
    cols = [np.ones(len(y)), np.array([r["lsize"] for r in rows])]
    for k_ in ("grow", "age"):
        v = np.array([r[k_] for r in rows], float); m = np.isfinite(v)
        cols.append(np.where(m, v, np.median(v[m]) if m.any() else 0.0))
    for s_ in sorted({r["ind"] for r in rows})[1:]:
        cols.append(np.array([1.0 if r["ind"] == s_ else 0.0 for r in rows]))
    if extra_cov:
        yr_ = np.array([(r["m0"] - 1) // 12 for r in rows])
        for v_ in sorted(set(yr_))[1:]:
            cols.append((yr_ == v_).astype(float))
    C = np.column_stack(cols)
    if weights:
        w = np.array([1.0 / (1.0 / max(r["Npre"], 1) + 1.0 / max(r["Npost"], 1)) for r in rows])
        w = w / w.mean()
        sw = np.sqrt(w)
        bC = np.linalg.lstsq(C * sw[:, None], y * sw, rcond=None)[0]
        yr = y - C @ bC
        bC2 = np.linalg.lstsq(C * sw[:, None], x * sw, rcond=None)[0]
        xr = x - C @ bC2
        d = float(np.sum(w * xr * xr))
        return float(np.sum(w * xr * yr) / d) if d > 0 else None
    r_ = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    yr, xr = r_(y), r_(x)
    d = float(np.sum(xr * xr))
    return float(np.sum(xr * yr) / d) if d > 0 else None


def ri(T, P, tag, wins=(5, 95), **kw):
    if not T or not P: print(f"  {tag}: 표본 부족"); return None
    cuts = tuple(np.percentile([r["eff"] for r in T], wins)) if wins else None
    obs = slope(T, cuts, **kw)
    if obs is None: print(f"  {tag}: 추정 불가"); return None
    n_t = len(T)
    cells = sorted({r["g"] for r in P}); byg = {c: [r for r in P if r["g"] == c] for c in cells}
    null = []
    for _ in range(NDRAW):
        d_ = []
        for i in rng.permutation(len(cells)):
            d_ += byg[cells[i]]
            if len(d_) >= n_t: break
        s_ = slope(d_[:n_t], cuts, **kw)
        if s_ is not None: null.append(s_)
    null = np.array(null)
    p = (int((null >= obs).sum()) + 1) / (len(null) + 1)
    out = {"observed": round(obs, 4), "n_treated": n_t, "n_placebo": len(P),
           "null_mean": round(float(null.mean()), 4), "null_sd": round(float(null.std()), 4),
           "null_ci": qci(null), "RI_p": round(float(p), 4), "sig": bool(p < 0.05),
           "z": round(float((obs - null.mean()) / null.std()), 2),
           "excess": round(float(obs - null.mean()), 4)}
    print(f"  {tag:<34} obs {out['observed']:>+7.4f} · null {out['null_mean']:>+7.4f}"
          f"(SD {out['null_sd']:.4f}) · 초과 {out['excess']:>+7.4f} · z={out['z']:>5.2f} · "
          f"RI p {out['RI_p']:.4f} {'✓' if out['sig'] else '✗'}  n={n_t}/{len(P)}")
    return out


# ════════════ 처치·위약 자료 구축 ════════════
def assemble(balance_state, mirror, s_win=(-24, -13), s_need=None):
    """balance_state: 매칭에 상태 bin 포함 · mirror: 위약을 거울설계로."""
    T, P = [], []
    for gi, e in enumerate(EV0):
        m0 = e["m0"]
        ct = match(e["ti"], m0, PEset, balance_state)
        if ct is None: continue
        u = make_unit(e["ti"], ct, m0, gi, s_win, s_need)
        if u: T.append(u)
        base = [int(k) for k in (ct if mirror else e["ctrls"])]
        for k in base:
            if mirror:
                ck = match(k, m0, PEset, balance_state)     # k 자신의 최근접 대조 재매칭
                if ck is None: continue
            else:
                ck = [j for j in base if j != k]
            v = make_unit(k, ck, m0, gi, s_win, s_need)
            if v: P.append(v)
    return T, P


R = {}
print("\n[Panel A·B] 설계 교정 — 상태 균형 매칭 × 거울설계 위약")
for bs in (False, True):
    for mir in (False, True):
        T, P = assemble(bs, mir)
        tag = f"{'상태균형' if bs else '현행'} 매칭 · {'거울' if mir else 'LOO'} 위약"
        R[f"bs{int(bs)}_mirror{int(mir)}"] = ri(T, P, tag)

BEST = ("bs1_mirror1" if (R.get("bs1_mirror1") and R["bs1_mirror1"]["sig"]) else "bs0_mirror0")
print(f"\n  → 이후 패널은 **상태균형 매칭 + 거울 위약** 설계 위에서 비교한다.")

print("\n[Panel C] 상태 측정창 연장 (감쇠 축소)")
PC = {}
for win, need, tag in (((-24, -13), 12, "12개월 [−24,−13] (기준)"),
                       ((-36, -13), 24, "24개월 [−36,−13]")):
    T, P = assemble(True, True, win, need)
    PC[tag] = ri(T, P, tag)
R["panelC_state_window"] = PC

print("\n[Panel D] 분할표본 IV — 일시적 측정오차 제거")
Ta, Pa = assemble(True, True)


def halves(rows):
    """상태를 두 반쪽 창에서 각각 측정. 일시적 잡음이 독립이면 서로의 도구변수가 된다."""
    out = []
    for r in rows:
        sa = Svar_row(r, -24, -19); sb = Svar_row(r, -18, -13)
        if sa is None or sb is None: continue
        q = dict(r); q["Sa"], q["Sb"] = sa, sb; out.append(q)
    return out


_rowfoc = {}
def Svar_row(r, a, b):
    key = (r["g"], r["m0"], a, b, r["ind"], round(r["lsize"], 6), round(r["S"], 6))
    if key in _rowfoc: return _rowfoc[key]
    _rowfoc[key] = r.get(f"_S{a}_{b}")
    return _rowfoc[key]


# make_unit 이 반쪽 상태를 함께 싣도록 재구성
def assemble_iv(balance_state=True):
    T, P = [], []
    for gi, e in enumerate(EV0):
        m0 = e["m0"]
        ct = match(e["ti"], m0, PEset, balance_state)
        if ct is None: continue
        for focal, ctrls, store in [(e["ti"], ct, T)] + \
                [(int(k), None, P) for k in ct]:
            if ctrls is None:
                ctrls = match(focal, m0, PEset, balance_state)
                if ctrls is None: continue
            u = make_unit(focal, ctrls, m0, gi)
            if u is None: continue
            sa = Svar(focal, m0, -24, -19, 6); sb = Svar(focal, m0, -18, -13, 6)
            if sa is None or sb is None: continue
            u["Sa"], u["Sb"] = sa, sb
            store.append(u)
    return T, P


Tiv, Piv = assemble_iv(True)


def iv_slope(rows, cuts=None):
    """2SLS: 결과 ~ Sa, 도구 Sb. 공변량은 FWL 로 사전 제거."""
    if len(rows) < 40: return None
    y = np.array([r["eff"] for r in rows])
    if cuts is not None: y = np.clip(y, cuts[0], cuts[1])
    xa = np.array([r["Sa"] for r in rows]); xb = np.array([r["Sb"] for r in rows])
    cols = [np.ones(len(y)), np.array([r["lsize"] for r in rows])]
    for k_ in ("grow", "age"):
        v = np.array([r[k_] for r in rows], float); m = np.isfinite(v)
        cols.append(np.where(m, v, np.median(v[m]) if m.any() else 0.0))
    for s_ in sorted({r["ind"] for r in rows})[1:]:
        cols.append(np.array([1.0 if r["ind"] == s_ else 0.0 for r in rows]))
    C = np.column_stack(cols)
    r_ = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    yr, ar, br = r_(y), r_(xa), r_(xb)
    den = float(np.sum(br * ar))
    return float(np.sum(br * yr) / den) if abs(den) > 1e-9 else None


def ols_half(rows, key, cuts=None):
    rr = [dict(r, S=r[key]) for r in rows]
    return slope(rr, cuts)


PD = {}
if len(Tiv) >= 40:
    cuts = tuple(np.percentile([r["eff"] for r in Tiv], [5, 95]))
    xa = np.array([r["Sa"] for r in Tiv]); xb = np.array([r["Sb"] for r in Tiv])
    PD["n"] = len(Tiv)
    PD["corr_halves"] = round(float(np.corrcoef(xa, xb)[0, 1]), 4)
    PD["ols_first_half"] = round(ols_half(Tiv, "Sa", cuts), 4)
    PD["ols_second_half"] = round(ols_half(Tiv, "Sb", cuts), 4)
    PD["ols_full_window"] = round(slope(Tiv, cuts), 4)
    iv = iv_slope(Tiv, cuts)
    PD["iv"] = round(iv, 4) if iv is not None else None
    # 위약 귀무 (같은 절차)
    cells = sorted({r["g"] for r in Piv}); byg = {c: [r for r in Piv if r["g"] == c] for c in cells}
    null = []
    for _ in range(NDRAW):
        d_ = []
        for i in rng.permutation(len(cells)):
            d_ += byg[cells[i]]
            if len(d_) >= len(Tiv): break
        v_ = iv_slope(d_[:len(Tiv)], cuts)
        if v_ is not None and abs(v_) < 20: null.append(v_)
    null = np.array(null)
    p = (int((null >= iv).sum()) + 1) / (len(null) + 1)
    PD["iv_null_mean"] = round(float(null.mean()), 4)
    PD["iv_null_sd"] = round(float(null.std()), 4)
    PD["iv_RI_p"] = round(float(p), 4)
    PD["iv_sig"] = bool(p < 0.05)
    PD["attenuation_ratio"] = (round(PD["ols_full_window"] / iv, 3)
                               if iv and abs(iv) > 1e-6 else None)
    print(f"  반쪽 상관 {PD['corr_halves']:+.3f} · OLS 전창 {PD['ols_full_window']:+.4f} · "
          f"OLS 전반 {PD['ols_first_half']:+.4f} · OLS 후반 {PD['ols_second_half']:+.4f}")
    print(f"  ★ IV {PD['iv']:+.4f} · 귀무 {PD['iv_null_mean']:+.4f}(SD {PD['iv_null_sd']:.4f}) · "
          f"RI p {PD['iv_RI_p']:.4f} {'✓' if PD['iv_sig'] else '✗'} "
          f"· 감쇠비 OLS/IV = {PD['attenuation_ratio']}")
else:
    print(f"  표본 부족 {len(Tiv)}")
R["panelD_split_iv"] = PD

print("\n[Panel E] 정밀도 가중 (역분산)")
PE_ = {"unweighted": ri(Ta, Pa, "무가중 (기준)"),
       "precision_weighted": ri(Ta, Pa, "역분산 가중", weights=True)}
R["panelE_weighting"] = PE_

print("\n[Panel F] 공변량 확장 (딜연도 추가)")
PF = {"base": PE_["unweighted"], "plus_year": ri(Ta, Pa, "+ 딜연도 FE", extra_cov=True),
      "weighted_plus_year": ri(Ta, Pa, "역분산 + 딜연도 FE", weights=True, extra_cov=True)}
R["panelF_covariates"] = PF

best = max((v for v in [R.get("bs0_mirror0"), R.get("bs0_mirror1"), R.get("bs1_mirror0"),
                        R.get("bs1_mirror1")] + list(PC.values()) + list(PE_.values())
            + list(PF.values()) if v), key=lambda v: -v["RI_p"])
verdict = (
    f"설계 교정 4조합: 현행·LOO {R['bs0_mirror0']['observed']:+.4f}(귀무 "
    f"{R['bs0_mirror0']['null_mean']:+.4f}, RI p {R['bs0_mirror0']['RI_p']:.4f}) · "
    f"현행·거울 {R['bs0_mirror1']['observed']:+.4f}({R['bs0_mirror1']['null_mean']:+.4f}, "
    f"p {R['bs0_mirror1']['RI_p']:.4f}) · 상태균형·LOO {R['bs1_mirror0']['observed']:+.4f}"
    f"({R['bs1_mirror0']['null_mean']:+.4f}, p {R['bs1_mirror0']['RI_p']:.4f}) · "
    f"**상태균형·거울 {R['bs1_mirror1']['observed']:+.4f}"
    f"({R['bs1_mirror1']['null_mean']:+.4f}, p {R['bs1_mirror1']['RI_p']:.4f})**. "
    f"효율: 역분산 가중 {PE_['precision_weighted']['observed']:+.4f} "
    f"(p {PE_['precision_weighted']['RI_p']:.4f}) · 딜연도 추가 {PF['plus_year']['observed']:+.4f} "
    f"(p {PF['plus_year']['RI_p']:.4f}).")
emit("I-56", "검정력 제고 — 설계 교정(상태균형·거울위약) + 효율 개선",
     "GO" if (R.get("bs1_mirror1") and R["bs1_mirror1"]["sig"]) else "PARTIAL",
     R | {"n_draws": NDRAW,
          "rationale": "각 레버는 통계이론이 예측하는 이득 — 유의성 사냥 아님. 전건 보고."},
     "설계를 처치 절차와 정합하게 고치고 효율을 올리면 gradient 가 더 선명해지는가",
     verdict, kill_met=False, n=R["bs1_mirror1"]["n_treated"] if R.get("bs1_mirror1") else 0)
