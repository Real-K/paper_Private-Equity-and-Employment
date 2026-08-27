# -*- coding: utf-8 -*-
"""I-46 (C-1) 휴면 vs 사전 물량 — 동일 비겹침 창에서의 경마.

리뷰 3 MC1: "dormancy 가 독립된 economic state 인가, 아니면 low prior hiring volume 을 새 이름으로
부른 것인가." 우리는 I-44 에서 지표별 수렴 타당도만 보였고 **동일 창 경마는 돌리지 않았다.**
두 변수 모두 [−24,−13] 에서 재면 결과 기준창 [−12,−1] 과 겹치지 않아 regression fallacy 가 없다.

단순 경마는 강한 버전에 구조적으로 불리하다. dormancy 와 log N 은 같은 창에서 상관이 매우 높고,
게다가 **N < 12 인 기업은 매달 채용하는 것이 애초에 불가능**하다(휴면이 물량에 기계적으로 갇힌다).
그래서 세 가지를 함께 건다.

Panel A  단순 경마 — dorm + log(1+N) + X            (리뷰어 최소 요구)
Panel B  ★ **물량 정화 휴면(excess dormancy)** — §7 벤치마크를 사전기간에 적용.
         expected_dorm = (1/12)·Σ_j (1−w_j)^N. excess = actual − expected.
         구성상 물량 성분이 제거된 지표이므로, 이것이 예측하면 휴면은 독립 내용을 갖는다.
Panel C  ★ **휴면이 자유변수인 부표본** — N ≥ 12 (매달 채용이 가능했던 기업)만
Panel D  사전 물량 5분위 **내부** 휴면 기울기
Panel E  물량 × 휴면 2차원 bins
Panel F  물량·규모에 residualize 한 휴면
Panel G  size×industry 셀 내부
Panel H  영향점·winsorization 민감도

기각조건: A·B·C 가 모두 미검출이면 강한 버전 기각 → construct 를 '낮은 사전 채용 활동'으로 넓히고
제목에서 dormancy 특권화를 제거한다.
[메모리] 이벤트 수준 스칼라만.
"""
import numpy as np, pandas as pd
from h30_common import load, deals, build, emit, SEED, qci, NB, widx

rng = np.random.default_rng(SEED)
rng_fig = np.random.default_rng(SEED + 1)
SESOI = 0.3472

print("[I-46] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Ev, adpt = G["Hv"], G["Ev"], G["adpt_arr"]


def win(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return None
    h, e = Hv[row, c].astype(float), Ev[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.mean(e) < 5: return None
    return h, e


def lrate(row, m0, a, b):
    w = win(row, m0, a, b)
    if w is None: return np.nan
    N, E = w[0].sum(), np.mean(w[1])
    return float(np.log(N / E)) if N > 0 else np.nan


for e in EV:
    a, b = lrate(e["ti"], e["m0"], -12, -1), lrate(e["ti"], e["m0"], 1, 12)
    t = b - a if (np.isfinite(a) and np.isfinite(b)) else np.nan
    cs = [lrate(k, e["m0"], 1, 12) - lrate(k, e["m0"], -12, -1) for k in e["ctrls"]]
    cs = [x for x in cs if np.isfinite(x)]
    e["eff"] = t - float(np.mean(cs)) if (np.isfinite(t) and cs) else np.nan
    w = win(e["ti"], e["m0"], -24, -13)            # 상태 창 (결과 기준창 밖)
    if w is None:
        e["dorm"] = e["lN"] = e["xdorm"] = e["N"] = np.nan
    else:
        h, emp = w
        N = float(h.sum()); wj = emp / emp.sum()
        exp_zero = float(np.sum((1.0 - wj) ** N)) / 12.0      # 무작위 배분 시 기대 무채용 비중
        e["N"] = N
        e["dorm"] = float((h == 0).mean())
        e["lN"] = float(np.log1p(N))
        e["edorm"] = exp_zero
        e["xdorm"] = e["dorm"] - exp_zero                      # ★ 물량 정화 휴면
        e["lsize"] = float(np.log(np.mean(emp)))
    w36 = win(e["ti"], e["m0"], -36, -25)
    e["grow"] = (float(np.log(np.mean(w["1" if False else 1]) / np.mean(w36[1])))
                 if (w and w36 and np.mean(w36[1]) > 0) else np.nan)
    e["age"] = (e["m0"] - adpt[e["ti"]]) / 12.0 if np.isfinite(adpt[e["ti"]]) else np.nan
    e["ind1"] = str(G["ind_arr"][e["ti"]])[:1]

U = [e for e in EV if np.isfinite(e["eff"]) and np.isfinite(e.get("dorm", np.nan))]
y = np.array([e["eff"] for e in U])
d = np.array([e["dorm"] for e in U]); ln = np.array([e["lN"] for e in U])
xd = np.array([e["xdorm"] for e in U]); Nv = np.array([e["N"] for e in U])
ed = np.array([e["edorm"] for e in U])
print(f"  분석표본 {len(U)}/{len(EV)} · corr(dorm, logN)[−24,−13] = {np.corrcoef(d, ln)[0,1]:+.3f}")
print(f"  사전창 평균: 총채용 {Nv.mean():.1f} · 실제 휴면 {d.mean():.3f} · "
      f"무작위배분 기대 휴면 {ed.mean():.3f} · 초과 휴면 {xd.mean():+.3f}")
print(f"  N ≥ 12 (매달 채용이 가능했던 기업): {int((Nv >= 12).sum())} / {len(U)}")


def X_of(idx, extra):
    # extra 는 **이미 idx 에 정렬된** 배열이어야 한다 (부분표본 호출 대응)
    cols = [np.ones(len(idx))] + [np.asarray(v, float) for v in extra]
    for k in ("lsize", "grow", "age"):
        v = np.array([U[i][k] for i in idx], float)
        v = np.where(np.isfinite(v), v, np.nanmedian(v[np.isfinite(v)]))
        cols.append(v)
    for s_ in sorted({U[i]["ind1"] for i in idx})[1:]:
        cols.append(np.array([1.0 if U[i]["ind1"] == s_ else 0.0 for i in idx]))
    return np.column_stack(cols)


def ols_ci(X, yy, j, R=NB):
    b = np.linalg.lstsq(X, yy, rcond=None)[0]
    bb = []
    for _ in range(R):
        i = rng.integers(0, len(yy), len(yy))
        try: bb.append(np.linalg.lstsq(X[i], yy[i], rcond=None)[0][j])
        except np.linalg.LinAlgError: pass
    ci = qci(np.array(bb))
    return {"coef": round(float(b[j]), 4), "ci": ci, "n": len(yy),
            "sig": bool(ci[0] > 0 or ci[1] < 0)}


def gd(v_hi, v_lo, R=NB):
    v_hi, v_lo = np.asarray(v_hi, float), np.asarray(v_lo, float)
    if len(v_hi) < 8 or len(v_lo) < 8: return None
    dd = float(v_hi.mean() - v_lo.mean())
    bo = np.array([v_hi[rng.integers(0, len(v_hi), len(v_hi))].mean()
                   - v_lo[rng.integers(0, len(v_lo), len(v_lo))].mean() for _ in range(R)])
    ci = qci(bo)
    return {"diff": round(dd, 4), "ci": ci, "n_hi": len(v_hi), "n_lo": len(v_lo),
            "sig": bool(ci[0] > 0 or ci[1] < 0)}


idx = list(range(len(U)))
R = {}

# ── Panel A 단순 경마 ──
print("\n[Panel A] 단순 경마 — dorm + log(1+N), 동일 [−24,−13] 창")
XA = X_of(idx, [d, ln])
R["panelA_horserace"] = {"dorm": ols_ci(XA, y, 1), "logN": ols_ci(XA, y, 2),
                         "corr": round(float(np.corrcoef(d, ln)[0, 1]), 4)}
XA0 = X_of(idx, [d]); XA1 = X_of(idx, [ln])
R["panelA_horserace"]["dorm_alone"] = ols_ci(XA0, y, 1)
R["panelA_horserace"]["logN_alone"] = ols_ci(XA1, y, 1)
for k, lab in (("dorm_alone", "휴면 단독"), ("logN_alone", "log(1+N) 단독"),
               ("dorm", "휴면 | logN"), ("logN", "logN | 휴면")):
    v = R["panelA_horserace"][k]
    print(f"  {lab:<16} {v['coef']:>+8.4f} {str(v['ci']):<22} {'✓' if v['sig'] else '✗'}")

# ── Panel B ★ 물량 정화 휴면 ──
print("\n[Panel B] ★ 물량 정화 휴면 (excess dormancy) — §7 벤치마크를 사전기간에 적용")
XB = X_of(idx, [xd])
R["panelB_excess_dormancy"] = {"slope": ols_ci(XB, y, 1),
                               "slope_raw": ols_ci(np.column_stack([np.ones(len(y)), xd]), y, 1),
                               "corr_with_logN": round(float(np.corrcoef(xd, ln)[0, 1]), 4),
                               "mean": round(float(xd.mean()), 4)}
q1, q2 = np.percentile(xd, [33.33, 66.67])
R["panelB_excess_dormancy"]["tercile"] = gd(y[xd > q2], y[xd <= q1])
XB2 = X_of(idx, [xd, ln])
R["panelB_excess_dormancy"]["slope_given_logN"] = ols_ci(XB2, y, 1)
v = R["panelB_excess_dormancy"]
print(f"  corr(초과휴면, logN) = {v['corr_with_logN']:+.3f}  (원 휴면은 {R['panelA_horserace']['corr']:+.3f})")
print(f"  기울기 단독 {v['slope_raw']['coef']:>+8.4f} {v['slope_raw']['ci']} "
      f"{'✓' if v['slope_raw']['sig'] else '✗'}")
print(f"  기울기 +X   {v['slope']['coef']:>+8.4f} {v['slope']['ci']} {'✓' if v['slope']['sig'] else '✗'}")
print(f"  기울기 +X+logN {v['slope_given_logN']['coef']:>+8.4f} {v['slope_given_logN']['ci']} "
      f"{'✓' if v['slope_given_logN']['sig'] else '✗'}")
if v["tercile"]:
    print(f"  3분위 상−하 {v['tercile']['diff']:>+8.4f} {v['tercile']['ci']} "
          f"{'✓' if v['tercile']['sig'] else '✗'} (n {v['tercile']['n_hi']}/{v['tercile']['n_lo']})")

# ── Panel C ★ 휴면이 자유변수인 부표본 ──
print("\n[Panel C] ★ N ≥ 12 부표본 — 매달 채용이 물리적으로 가능했던 기업만")
for thr in (12, 24):
    m = np.where(Nv >= thr)[0]
    if len(m) < 40: print(f"  N≥{thr}: 표본 {len(m)} 부족"); continue
    Xc = X_of(list(m), [d[m], ln[m]])
    r1 = ols_ci(Xc, y[m], 1); r2 = ols_ci(X_of(list(m), [d[m]]), y[m], 1)
    qq1, qq2 = np.percentile(d[m], [33.33, 66.67])
    t = gd(y[m][d[m] > qq2], y[m][d[m] <= qq1])
    R[f"panelC_free_N{thr}"] = {"n": len(m), "dorm_alone": r2, "dorm_given_logN": r1, "tercile": t}
    print(f"  N≥{thr} (n={len(m)}) 휴면 단독 {r2['coef']:>+8.4f} {str(r2['ci']):<22}"
          f"{'✓' if r2['sig'] else '✗'} · |logN {r1['coef']:>+8.4f} {str(r1['ci']):<22}"
          f"{'✓' if r1['sig'] else '✗'}"
          + (f" · 3분위 {t['diff']:+.4f} {t['ci']} {'✓' if t['sig'] else '✗'}" if t else ""))

# ── Panel D 물량 5분위 내부 ──
print("\n[Panel D] 사전 물량 5분위 내부 휴면 기울기")
cut = np.percentile(ln, [20, 40, 60, 80])
qbin = np.digitize(ln, cut)
PD = {}
for q in range(5):
    m = np.where(qbin == q)[0]
    if len(m) < 30: PD[f"Q{q+1}"] = {"n": int(len(m)), "note": "표본 부족"}; continue
    r = ols_ci(np.column_stack([np.ones(len(m)), d[m]]), y[m], 1)
    PD[f"Q{q+1}"] = {"n": int(len(m)), "slope": r,
                     "mean_N": round(float(Nv[m].mean()), 1)}
    print(f"  Q{q+1} (n={len(m):>3}, 평균N {Nv[m].mean():>6.1f}) 기울기 {r['coef']:>+8.4f} "
          f"{str(r['ci']):<22} {'✓' if r['sig'] else '✗'}")
# 5분위 고정효과 하에서의 통합 기울기
Dq = np.column_stack([np.ones(len(y)), d] + [(qbin == q).astype(float) for q in range(1, 5)])
PD["pooled_within_quintile"] = ols_ci(Dq, y, 1)
print(f"  5분위 FE 하 통합 기울기 {PD['pooled_within_quintile']['coef']:>+8.4f} "
      f"{PD['pooled_within_quintile']['ci']} {'✓' if PD['pooled_within_quintile']['sig'] else '✗'}")
R["panelD_within_volume_quintile"] = PD

# ── Panel E 2차원 bins ──
print("\n[Panel E] 물량 × 휴면 2×2 (중앙값 분할)")
mln, md = np.median(ln), np.median(d)
PE_ = {}
for li, ll in ((0, "저물량"), (1, "고물량")):
    for di_, dl in ((0, "저휴면"), (1, "고휴면")):
        m = np.where(((ln > mln).astype(int) == li) & ((d > md).astype(int) == di_))[0]
        PE_[f"{ll}×{dl}"] = {"n": int(len(m)),
                             "mean": (round(float(y[m].mean()), 4) if len(m) else None)}
lo_hi = np.where((ln <= mln) & (d > md))[0]; lo_lo = np.where((ln <= mln) & (d <= md))[0]
hi_hi = np.where((ln > mln) & (d > md))[0]; hi_lo = np.where((ln > mln) & (d <= md))[0]
PE_["휴면효과|저물량"] = gd(y[lo_hi], y[lo_lo])
PE_["휴면효과|고물량"] = gd(y[hi_hi], y[hi_lo])
for k in ("저물량×저휴면", "저물량×고휴면", "고물량×저휴면", "고물량×고휴면"):
    print(f"  {k}  n={PE_[k]['n']:>3}  평균 {PE_[k]['mean']}")
for k in ("휴면효과|저물량", "휴면효과|고물량"):
    v = PE_[k]
    print(f"  {k} {v['diff']:>+8.4f} {v['ci']} {'✓' if v['sig'] else '✗'}" if v else f"  {k} 표본부족")
R["panelE_2d_bins"] = PE_

# ── Panel F residualized ──
print("\n[Panel F] 물량·규모에 residualize 한 휴면")
Xr = np.column_stack([np.ones(len(y)), ln, np.array([U[i]["lsize"] for i in idx])])
dres = d - Xr @ np.linalg.lstsq(Xr, d, rcond=None)[0]
r2_ = 1 - dres.var() / d.var()
rf = ols_ci(np.column_stack([np.ones(len(y)), dres]), y, 1)
q1r, q2r = np.percentile(dres, [33.33, 66.67])
tf = gd(y[dres > q2r], y[dres <= q1r])
R["panelF_residualized"] = {"r2_explained_by_volume_size": round(float(r2_), 4),
                            "slope": rf, "tercile": tf}
print(f"  물량·규모가 휴면 분산의 {r2_:.1%} 설명 · 잔차 기울기 {rf['coef']:>+8.4f} {rf['ci']} "
      f"{'✓' if rf['sig'] else '✗'}"
      + (f" · 3분위 {tf['diff']:+.4f} {tf['ci']} {'✓' if tf['sig'] else '✗'}" if tf else ""))

# ── Panel G size×industry 셀 내부 ──
print("\n[Panel G] size×industry 셀 내부 (셀 FE)")
sz = np.array([U[i]["lsize"] for i in idx])
szb = np.digitize(sz, np.percentile(sz, [33.33, 66.67]))
cell = np.array([f"{U[i]['ind1']}_{szb[i]}" for i in idx])
cols = [np.ones(len(y)), d] + [(cell == c).astype(float) for c in sorted(set(cell))[1:]]
rg = ols_ci(np.column_stack(cols), y, 1)
R["panelG_within_size_industry"] = {"slope": rg, "n_cells": int(len(set(cell)))}
print(f"  셀 {len(set(cell))}개 · 셀 FE 하 휴면 기울기 {rg['coef']:>+8.4f} {rg['ci']} "
      f"{'✓' if rg['sig'] else '✗'}")

# ── Panel H 영향점·winsorization ──
print("\n[Panel H] 영향점·winsorization 민감도 (휴면 단독 기울기)")
PH = {"raw": R["panelA_horserace"]["dorm_alone"]["coef"]}
for lo, hi, tag in ((1, 99, "w1_99"), (5, 95, "w5_95")):
    yw = np.clip(y, *np.percentile(y, [lo, hi]))
    PH[tag] = ols_ci(X_of(idx, [d]), yw, 1)["coef"]
drop = np.argsort(-np.abs(y - y.mean()))[:5]
keep = np.setdiff1d(np.arange(len(y)), drop)
PH["drop_top5"] = ols_ci(X_of(list(keep), [d[keep]]), y[keep], 1)["coef"]
R["panelH_influence"] = PH
print(f"  원 {PH['raw']:+.4f} · w1/99 {PH['w1_99']:+.4f} · w5/95 {PH['w5_95']:+.4f} · "
      f"상위5 제거 {PH['drop_top5']:+.4f}")

# ── Panel I 상태변수 재정의 + 공변량 조정 헤드라인 ──
print("\n[Panel I] 상태변수 후보별 헤드라인 — 원 / 공변량 조정")
lr_pre = np.array([np.log1p(U[i]["N"] / np.exp(U[i]["lsize"])) for i in idx])   # log(1+사전채용률)
CAND = (("dormancy", d, +1), ("neg_log_volume", -ln, +1), ("neg_log_rate", -lr_pre, +1))
PI = {}
for nm, x, sgn in CAND:
    raw = ols_ci(np.column_stack([np.ones(len(y)), x]), y, 1)
    adj = ols_ci(X_of(idx, [x]), y, 1)                     # X 에 lsize·grow·age·industry 포함
    qa, qb = np.percentile(x, [33.33, 66.67])
    t_raw = gd(y[x > qb], y[x <= qa])
    # 공변량 조정 대비: 결과를 **상태변수를 뺀** 공변량에만 회귀한 잔차로
    Xc = X_of(idx, [])
    yres = y - Xc @ np.linalg.lstsq(Xc, y, rcond=None)[0]
    t_adj = gd(yres[x > qb], yres[x <= qa])
    PI[nm] = {"slope_raw": raw, "slope_adj": adj, "tercile_raw": t_raw, "tercile_adj": t_adj}
    print(f"  {nm:<16} 기울기 원 {raw['coef']:>+8.4f}{'✓' if raw['sig'] else '✗'} · "
          f"조정 {adj['coef']:>+8.4f}{'✓' if adj['sig'] else '✗'} | 3분위 원 "
          f"{t_raw['diff']:>+7.4f} {str(t_raw['ci']):<20}{'✓' if t_raw['sig'] else '✗'} · 조정 "
          f"{t_adj['diff']:>+7.4f} {str(t_adj['ci']):<20}{'✓' if t_adj['sig'] else '✗'}")
R["panelI_state_definition"] = PI

# ── Panel J ★ 미처치 위약에서 각 상태변수의 기울기 ──
print("\n[Panel J] ★ 미처치 위약 — 각 상태변수 기울기가 처치에서만 나타나는가")
rows = []
for gi, e in enumerate(EV):
    ok = [k for k in e["ctrls"]
          if np.isfinite(lrate(k, e["m0"], -12, -1)) and np.isfinite(lrate(k, e["m0"], 1, 12))]
    if len(ok) < 3: continue
    ch = {k: lrate(k, e["m0"], 1, 12) - lrate(k, e["m0"], -12, -1) for k in ok}
    for k in ok:
        w = win(k, e["m0"], -24, -13)
        if w is None: continue
        h, emp = w; N = float(h.sum()); wj = emp / emp.sum()
        E_ = float(np.mean(emp))
        rows.append((gi, ch[k] - float(np.mean([ch[j] for j in ok if j != k])),
                     float((h == 0).mean()), float(np.log1p(N)),
                     float(np.log1p(N / E_)),
                     float((h == 0).mean()) - float(np.sum((1.0 - wj) ** N)) / 12.0))
gidx = np.array([r[0] for r in rows]); py = np.array([r[1] for r in rows])
PX = {"dormancy": np.array([r[2] for r in rows]), "neg_log_volume": -np.array([r[3] for r in rows]),
      "neg_log_rate": -np.array([r[4] for r in rows]), "excess_dormancy": np.array([r[5] for r in rows])}
GU = np.unique(gidx); byg = {g_: np.where(gidx == g_)[0] for g_ in GU}
PJ = {"n_pseudo": int(len(py)), "n_clusters": int(len(GU))}
for nm, x in PX.items():
    X_ = np.column_stack([np.ones(len(py)), x])
    b = np.linalg.lstsq(X_, py, rcond=None)[0][1]
    bb = []
    for _ in range(NB):
        p_ = np.concatenate([byg[GU[i]] for i in rng.integers(0, len(GU), len(GU))])
        try: bb.append(np.linalg.lstsq(X_[p_], py[p_], rcond=None)[0][1])
        except np.linalg.LinAlgError: pass
    ci = qci(np.array(bb))
    qa, qb = np.percentile(x, [33.33, 66.67])
    dt = float(py[x > qb].mean() - py[x <= qa].mean())
    tb = []
    for _ in range(NB):
        p_ = np.concatenate([byg[GU[i]] for i in rng.integers(0, len(GU), len(GU))])
        a_, b_ = py[p_][x[p_] > qb], py[p_][x[p_] <= qa]
        if len(a_) and len(b_): tb.append(a_.mean() - b_.mean())
    tci = qci(np.array(tb))
    PJ[nm] = {"slope": round(float(b), 4), "ci": ci, "sig": bool(ci[0] > 0 or ci[1] < 0),
              "tercile": round(dt, 4), "tercile_ci": tci,
              "tercile_sig": bool(tci[0] > 0 or tci[1] < 0)}
    print(f"  {nm:<16} 기울기 {b:>+8.4f} {str(ci):<22}{'✓' if PJ[nm]['sig'] else '✗'} · "
          f"3분위 {dt:>+7.4f} {str(tci):<22}{'✓' if PJ[nm]['tercile_sig'] else '✗'}")
print(f"  (유사처치 {len(py)}건 / 셀 {len(GU)}개, 셀 군집 부트)")
R["panelJ_untreated_placebo_by_state"] = PJ

# ── 판정 ──
strong = [R["panelA_horserace"]["dorm"]["sig"],
          R["panelB_excess_dormancy"]["slope"]["sig"],
          bool(R.get("panelC_free_N12", {}).get("dorm_given_logN", {}).get("sig")),
          R["panelD_within_volume_quintile"]["pooled_within_quintile"]["sig"],
          R["panelF_residualized"]["slope"]["sig"],
          R["panelG_within_size_industry"]["slope"]["sig"]]
n_strong = sum(bool(x) for x in strong)
verdict = (
    f"동일 [−24,−13] 창 경마: 휴면 단독 {R['panelA_horserace']['dorm_alone']['coef']:+.4f}"
    f"{R['panelA_horserace']['dorm_alone']['ci']}, logN 통제 시 "
    f"{R['panelA_horserace']['dorm']['coef']:+.4f}{R['panelA_horserace']['dorm']['ci']} "
    f"(corr {R['panelA_horserace']['corr']:+.3f}). "
    f"★물량 정화 휴면 기울기 {R['panelB_excess_dormancy']['slope']['coef']:+.4f}"
    f"{R['panelB_excess_dormancy']['slope']['ci']}. "
    f"물량5분위 FE 하 {R['panelD_within_volume_quintile']['pooled_within_quintile']['coef']:+.4f}. "
    f"size×industry 셀 FE 하 {R['panelG_within_size_industry']['slope']['coef']:+.4f}. "
    f"6개 강한버전 관문 중 {n_strong} 통과.")
emit("I-46", "휴면 vs 사전 물량 — 동일 비겹침 창 경마 (리뷰3 MC1)",
     "GO" if n_strong >= 3 else ("PARTIAL" if n_strong >= 1 else "KILL"),
     R | {"n": len(U), "SESOI": SESOI,
          "state_window": "[-24,-13]", "outcome_base_window": "[-12,-1]",
          "note": "두 변수 모두 결과 기준창 밖 — regression fallacy 없음"},
     "동일 비겹침 창에서 사전 물량을 통제해도 휴면이 반응을 예측하는가",
     verdict, kill_met=bool(n_strong == 0), n=len(U))
