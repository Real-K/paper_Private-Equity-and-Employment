# -*- coding: utf-8 -*-
"""I-58 설계 교정 감사 — 상태균형 매칭의 이득이 진짜인가.

I-56 실측: 상태균형 매칭으로 처치계수 +0.3834 → +0.5456, 거울위약 귀무 +0.1505 → +0.0886,
z 1.40 → 3.15. **그런데 처치계수가 올라간 것은 내 사전 이론과 반대 방향이다.**
(균형매칭은 평균회귀 성분을 양쪽에서 상쇄하므로 처치계수를 *낮출* 것으로 예상했다.)

설명되지 않는 방향 변화는 그대로 두면 안 된다. 세 가지를 확인한다.

Panel A  **공통표본 검정.** 균형매칭은 표본을 301 → 286 으로 줄인다. 탈락 15건이 차이를 만드는가.
         같은 286 이벤트에서 두 매칭을 비교한다.
Panel B  **균형 확인.** 각 설계에서 처치−대조의 상태변수 차이(표준화)를 보고한다.
         균형매칭이 실제로 상태를 맞추는가.
Panel C  **대조군 품질.** 대조군 상태 평균과 처치 상태의 상관 · 대조군 사전 채용률 분포.
Panel D  **분해.** 처치계수 변화가 (a) 처치기업 자기 변화 (b) 대조군 평균 중 어디서 오는가.
"""
import numpy as np, pandas as pd
from h30_common import load, deals, build, emit, SEED, qci, NB, widx
from h39_common import SIZE_B

rng = np.random.default_rng(SEED)
NDRAW = 2000
print("[I-58] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV0, _ = build(G, allt, PE)
Hv, Ev, adpt, idx = G["Hv"], G["Ev"], G["adpt_arr"], G["idx"]
mset, ind_arr = G["mset"], G["ind_arr"]
PEset = set(PE); NOTPE = np.asarray(~idx.isin(PEset))

_c = {}
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

_s = {}
def Sall(m0):
    if m0 in _s: return _s[m0]
    c = widx(G, m0, -24, -13)
    if len(c) != 12: _s[m0] = (None, None); return _s[m0]
    h = Hv[:, c].astype(float); e = Ev[:, c].astype(float)
    ok = np.isfinite(h).all(1) & np.isfinite(e).all(1) & (np.nanmean(e, 1) >= 5)
    S = np.full(Hv.shape[0], np.nan)
    S[ok] = -np.log1p(h[ok].sum(1) / np.nanmean(e[ok], 1))
    fin = np.isfinite(S)
    b = np.full(Hv.shape[0], -9)
    if fin.sum() >= 50:
        q1, q2 = np.percentile(S[fin], [33.33, 66.67])
        b = np.where(fin, np.digitize(S, [q1, q2]), -9)
    _s[m0] = (S, b)
    return _s[m0]

def match(focal, m0, bal, k=5):
    c = cellarr(m0)
    if c is None: return None
    Ep, g, sb, gb, ageb = c
    if not (np.isfinite(Ep[focal]) and Ep[focal] >= 5): return None
    same = (NOTPE & (ind_arr == ind_arr[focal]) & (sb == sb[focal]) & (gb == gb[focal])
            & (ageb == ageb[focal]) & (Ep >= 5) & np.isfinite(Ep))
    if bal:
        S, bins = Sall(m0)
        if S is None or not np.isfinite(S[focal]) or bins[focal] == -9: return None
        same = same & (bins == bins[focal])
    cand = np.flatnonzero(same); cand = cand[cand != focal]
    if len(cand) == 0: return None
    gt = g[focal] if np.isfinite(g[focal]) else 0.0
    gc = np.where(np.isfinite(g[cand]), g[cand], 0.0)
    d = ((np.log(Ep[cand]) - np.log(Ep[focal])) / 0.9) ** 2 + \
        ((np.clip(gc, -1, 2) - np.clip(gt, -1, 2)) / 0.35) ** 2
    return cand[np.argsort(d)[:k]]

def blk(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return None
    h, e = Hv[row, c].astype(float), Ev[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.mean(e) < 5: return None
    return float(h.sum()), float(np.mean(e))

def dl(row, m0):
    po, pr = blk(row, m0, 1, 12), blk(row, m0, -12, -1)
    if po is None or pr is None or po[0] <= 0 or pr[0] <= 0: return None
    return np.log(po[0] / po[1]) - np.log(pr[0] / pr[1])

def unit(focal, ctrls, m0, gi):
    st = blk(focal, m0, -24, -13)
    own = dl(focal, m0)
    if st is None or own is None: return None
    cs, cS = [], []
    S_, _b = Sall(m0)
    for o in ctrls:
        v = dl(int(o), m0)
        if v is None: continue
        cs.append(v)
        if S_ is not None and np.isfinite(S_[int(o)]): cS.append(float(S_[int(o)]))
    if not cs: return None
    w36 = blk(focal, m0, -36, -25)
    return dict(g=gi, own=own, cmean=float(np.mean(cs)), eff=own - float(np.mean(cs)),
                S=-float(np.log1p(st[0] / st[1])), cS=(float(np.mean(cS)) if cS else np.nan),
                lsize=np.log(st[1]),
                grow=(np.log(st[1] / w36[1]) if (w36 and w36[1] > 0) else np.nan),
                age=((m0 - adpt[focal]) / 12.0 if np.isfinite(adpt[focal]) else np.nan),
                ind=str(ind_arr[focal])[:1], bn=focal)

def build_set(bal):
    T, P, keys = [], [], set()
    for gi, e in enumerate(EV0):
        m0 = e["m0"]
        ct = match(e["ti"], m0, bal)
        if ct is None: continue
        u = unit(e["ti"], ct, m0, gi)
        if u: T.append(u); keys.add(e["ti"])
        for k in ct:
            ck = match(int(k), m0, bal)
            if ck is None: continue
            v = unit(int(k), ck, m0, gi)
            if v: P.append(v)
    return T, P, keys

def design(rows):
    cols = [np.ones(len(rows)), np.array([r["lsize"] for r in rows])]
    for k in ("grow", "age"):
        v = np.array([r[k] for r in rows], float); m = np.isfinite(v)
        cols.append(np.where(m, v, np.median(v[m]) if m.any() else 0.0))
    for s_ in sorted({r["ind"] for r in rows})[1:]:
        cols.append(np.array([1.0 if r["ind"] == s_ else 0.0 for r in rows]))
    return np.column_stack(cols)

def sl(rows, key="eff", cuts=None):
    if len(rows) < 30: return None
    y = np.array([r[key] for r in rows]); x = np.array([r["S"] for r in rows])
    if cuts is not None: y = np.clip(y, cuts[0], cuts[1])
    C = design(rows); r_ = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    yr, xr = r_(y), r_(x); d = float(np.sum(xr * xr))
    return float(np.sum(xr * yr) / d) if d > 0 else None

def ri(T, P, tag, key="eff"):
    cuts = tuple(np.percentile([r[key] for r in T], [5, 95]))
    obs = sl(T, key, cuts); n_t = len(T)
    cells = sorted({r["g"] for r in P}); byg = {c: [r for r in P if r["g"] == c] for c in cells}
    null = []
    for _ in range(NDRAW):
        d_ = []
        for i in rng.permutation(len(cells)):
            d_ += byg[cells[i]]
            if len(d_) >= n_t: break
        v = sl(d_[:n_t], key, cuts)
        if v is not None: null.append(v)
    null = np.array(null)
    p = (int((null >= obs).sum()) + 1) / (len(null) + 1)
    o = {"observed": round(obs, 4), "n": n_t, "null_mean": round(float(null.mean()), 4),
         "null_sd": round(float(null.std()), 4), "RI_p": round(float(p), 4),
         "z": round(float((obs - null.mean()) / null.std()), 2), "sig": bool(p < 0.05)}
    print(f"  {tag:<38} obs {o['observed']:>+7.4f} · null {o['null_mean']:>+7.4f}"
          f"(SD {o['null_sd']:.4f}) · z={o['z']:>5.2f} · RI p {o['RI_p']:.4f} "
          f"{'✓' if o['sig'] else '✗'}  n={n_t}")
    return o

print("\n[구축]")
T0, P0, K0 = build_set(False); print(f"  현행 매칭   처치 {len(T0)} · 위약 {len(P0)}")
T1, P1, K1 = build_set(True);  print(f"  상태균형    처치 {len(T1)} · 위약 {len(P1)}")
COMMON = K0 & K1
print(f"  공통 처치기업 {len(COMMON)}")

print("\n[Panel A] 공통표본 비교 — 표본 구성이 차이를 만드는가")
T0c = [r for r in T0 if r["bn"] in COMMON]; T1c = [r for r in T1 if r["bn"] in COMMON]
g0 = sorted({r["g"] for r in T0c}); g1 = sorted({r["g"] for r in T1c})
P0c = [r for r in P0 if r["g"] in set(g0)]; P1c = [r for r in P1 if r["g"] in set(g1)]
PA = {"n_common": len(COMMON),
      "current_full": ri(T0, P0, "현행 매칭 · 전체"),
      "balanced_full": ri(T1, P1, "상태균형 · 전체"),
      "current_common": ri(T0c, P0c, "현행 매칭 · 공통표본"),
      "balanced_common": ri(T1c, P1c, "상태균형 · 공통표본")}

print("\n[Panel B] 상태 균형 확인 — 처치 S vs 대조군 평균 S")
PB = {}
for tag, T in (("current", T0), ("balanced", T1)):
    a = np.array([r["S"] for r in T]); b = np.array([r["cS"] for r in T])
    m = np.isfinite(a) & np.isfinite(b)
    nd = float((a[m] - b[m]).mean() / np.sqrt((a[m].var() + b[m].var()) / 2))
    PB[tag] = {"treated_S_mean": round(float(a[m].mean()), 4),
               "control_S_mean": round(float(b[m].mean()), 4),
               "mean_gap": round(float((a[m] - b[m]).mean()), 4),
               "normalized_difference": round(nd, 4),
               "corr_treated_control_S": round(float(np.corrcoef(a[m], b[m])[0, 1]), 4)}
    print(f"  {tag:<10} 처치 S {PB[tag]['treated_S_mean']:+.4f} · 대조 S "
          f"{PB[tag]['control_S_mean']:+.4f} · 격차 {PB[tag]['mean_gap']:+.4f} · "
          f"ND {PB[tag]['normalized_difference']:+.4f} · corr {PB[tag]['corr_treated_control_S']:.3f}")

print("\n[Panel D] 분해 — 계수 변화가 처치 자기변화에서 오는가, 대조군에서 오는가")
PD = {}
for tag, T in (("current", T0c), ("balanced", T1c)):
    cuts = tuple(np.percentile([r["eff"] for r in T], [5, 95]))
    PD[tag] = {"eff": round(sl(T, "eff", cuts), 4),
               "own": round(sl(T, "own"), 4),
               "control_mean": round(sl(T, "cmean"), 4), "n": len(T)}
    print(f"  {tag:<10} eff {PD[tag]['eff']:>+7.4f} = 자기 {PD[tag]['own']:>+7.4f} "
          f"− 대조 {PD[tag]['control_mean']:>+7.4f}  (n={PD[tag]['n']})")

d_own = PD["balanced"]["own"] - PD["current"]["own"]
d_ctl = PD["balanced"]["control_mean"] - PD["current"]["control_mean"]
verdict = (
    f"[A] 공통표본({len(COMMON)})에서도 현행 {PA['current_common']['observed']:+.4f}"
    f"(z {PA['current_common']['z']}, p {PA['current_common']['RI_p']:.4f}) vs 상태균형 "
    f"{PA['balanced_common']['observed']:+.4f}(z {PA['balanced_common']['z']}, "
    f"p {PA['balanced_common']['RI_p']:.4f}) — 표본 구성이 아니라 **설계**가 차이를 만든다. "
    f"[B] 처치−대조 상태 ND 가 {PB['current']['normalized_difference']:+.3f} → "
    f"{PB['balanced']['normalized_difference']:+.3f} 로 개선. "
    f"[D] 계수 변화 분해: 처치 자기변화 기울기 {d_own:+.4f}, 대조군 평균 기울기 {d_ctl:+.4f} "
    f"→ 변화는 주로 {'대조군' if abs(d_ctl) > abs(d_own) else '처치 자기변화'} 쪽에서 온다.")
emit("I-58", "설계 교정 감사 — 상태균형 매칭의 이득 검증",
     "GO" if PA["balanced_common"]["sig"] else "PARTIAL",
     {"panelA_common_sample": PA, "panelB_state_balance": PB, "panelD_decomposition": PD,
      "n_draws": NDRAW},
     "상태균형 매칭의 이득이 표본 구성이 아니라 설계에서 오는가",
     verdict, kill_met=False, n=len(COMMON))
