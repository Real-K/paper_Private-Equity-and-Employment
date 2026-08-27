# -*- coding: utf-8 -*-
"""I-50 검정력 제고 — 상태 gradient 를 더 정밀하게 추정한다.

문제. 현 헤드라인(I-47) 은 FWL 조정 기울기 +0.5650 [0.210, 0.968] 로 **CI 반폭이 점추정의 67%** 다.
그리고 I-48 에서 대오차 월을 배제하면 2/4 에서 유의성을 잃는다. 둘 다 검정력 문제다.

현 추정량은 이벤트당 **매칭차분 1개**(n=301)만 쓴다. 대조기업 1,328개의 자체 변동과 사전 수준
정보는 버려진다. 아래 레버를 건다.

Panel A  ★ **풀링 ANCOVA (셀 FE)** — 처치+대조 전부를 관측치로. 사후 log 채용률을 사전 수준·상태·
         처치×상태에 회귀. 차분은 사전계수를 1 로 강제하지만 ANCOVA 는 자유롭게 추정 → 더 효율적.
Panel B  ★ **PPML** — 채용 건수 + log 고용 offset. N=0 창을 버리지 않고, 로그변환 선택이 결과를
         만들지 않았음을 보인다 (리뷰3 §9.2).
Panel C  상태창 완화 — 12개월 / 6개월 / 부분≥9개월. 표본 회수량과 추정치.
Panel D  강건 위치추정 — winsorize · 절사평균 (두꺼운 꼬리에 의한 분산 축소)
Panel E  ★ **암묵 채용** A* = max(0, ΔE + S) — stock-flow 우려를 **표본 손실 없이** 검정
Panel F  사후창 확장 [1,24]

[메모리] 풀링 설계행렬 ~1,800×6 (셀은 within 변환으로 흡수). PPML 만 dense FE 사용.
"""
import numpy as np, pandas as pd
import statsmodels.api as sm
from h30_common import load, deals, build, emit, SEED, qci, NB, widx

rng = np.random.default_rng(SEED)
print("[I-50] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Sv, Ev, adpt = G["Hv"], G["Sv"], G["Ev"], G["adpt_arr"]


def block(row, m0, a, b, M=None, need=None):
    c = widx(G, m0, a, b); n = b - a + 1
    if need is None: need = n
    if len(c) < need: return None
    h = (M if M is not None else Hv)[row, c].astype(float)
    e = Ev[row, c].astype(float)
    m = np.isfinite(h) & np.isfinite(e)
    if m.sum() < need or np.mean(e[m]) < 5: return None
    return h[m], e[m]


def rate(row, m0, a, b, M=None, need=None):
    w = block(row, m0, a, b, M, need)
    if w is None: return None
    return float(w[0].sum()), float(np.mean(w[1]))


def build_rows(state_win=(-24, -13), state_need=12, post=(1, 12), M=None):
    """처치+대조 전부를 관측치로 하는 풀링 자료."""
    R = []
    for gi, e in enumerate(EV):
        members = [(e["ti"], 1)] + [(int(k), 0) for k in e["ctrls"]]
        rec = []
        for row, T in members:
            po = rate(row, e["m0"], post[0], post[1], M)
            pr = rate(row, e["m0"], -12, -1, M)
            st = rate(row, e["m0"], state_win[0], state_win[1], M, state_need)
            if po is None or pr is None or st is None: continue
            npost, epost = po; npre, epre = pr; nst, est = st
            rec.append(dict(cell=gi, T=T, n_post=npost, e_post=epost,
                            lag=(np.log(npre / epre) if npre > 0 else np.nan),
                            S=-np.log1p(nst / est), lsize=np.log(est),
                            age=((e["m0"] - adpt[row]) / 12.0 if np.isfinite(adpt[row]) else np.nan),
                            y=(np.log(npost / epost) if npost > 0 else np.nan)))
        if sum(r["T"] for r in rec) == 1 and len(rec) >= 3:
            R += rec
    return pd.DataFrame(R)


def within(df, cols):
    """셀 평균 제거 (FWL) — 셀 FE 를 흡수."""
    out = df.copy()
    for c in cols: out[c] = df[c] - df.groupby("cell")[c].transform("mean")
    return out


def cell_boot(df, cols, target, R=NB):
    """셀 군집 부트스트랩으로 target 계수의 CI."""
    cells = df.cell.unique()
    bygrp = {c: df.index[df.cell == c].to_numpy() for c in cells}
    est = []
    for _ in range(R):
        sel = rng.integers(0, len(cells), len(cells))
        parts, labs = [], []
        for r, i in enumerate(sel):
            ix = bygrp[cells[i]]
            parts.append(ix); labs.append(np.full(len(ix), r))
        # 재표본된 셀은 **새 라벨**을 받아야 한다 — 같은 셀이 두 번 뽑히면 별개 셀로 취급
        pick = np.concatenate(parts)
        d2 = df.loc[pick].copy()
        d2["cell"] = np.concatenate(labs)
        w = within(d2, cols + ["y"])
        X = w[cols].to_numpy()
        try: est.append(np.linalg.lstsq(X, w["y"].to_numpy(), rcond=None)[0][cols.index(target)])
        except Exception: pass
    return qci(np.array(est))


def ancova(df, tag):
    d = df.dropna(subset=["y", "lag", "S", "lsize"]).reset_index(drop=True)
    d["TS"] = d.T_ * d.S if "T_" in d else d["T"] * d["S"]
    cols = ["T", "TS", "lag", "S", "lsize"]
    w = within(d, cols + ["y"])
    b = np.linalg.lstsq(w[cols].to_numpy(), w["y"].to_numpy(), rcond=None)[0]
    ci = cell_boot(d, cols, "TS")
    out = {"n_obs": len(d), "n_cells": int(d.cell.nunique()),
           "n_treated": int(d["T"].sum()),
           "gradient": round(float(b[cols.index("TS")]), 4), "ci": ci,
           "sig": bool(ci[0] > 0 or ci[1] < 0),
           "half_width": round(float((ci[1] - ci[0]) / 2), 4),
           "treated_main": round(float(b[0]), 4),
           "lag_coef": round(float(b[cols.index("lag")]), 4)}
    print(f"  {tag:<28} obs {out['n_obs']:>5} 셀 {out['n_cells']:>3} · 기울기 "
          f"{out['gradient']:>+7.4f} {str(ci):<22}{'✓' if out['sig'] else '✗'} "
          f"반폭 {out['half_width']:.4f} · lag {out['lag_coef']:+.3f}")
    return out


print("\n[Panel A] ★ 풀링 ANCOVA (셀 FE) — 처치+대조 전부")
D = build_rows()
PA = {"baseline": ancova(D, "기준 (상태창 12개월)")}
# 차분 강제(lag 계수 = 1) 와 비교
d0 = D.dropna(subset=["y", "lag", "S", "lsize"]).reset_index(drop=True)
d0["TS"] = d0["T"] * d0["S"]; d0["ydiff"] = d0["y"] - d0["lag"]
cols = ["T", "TS", "S", "lsize"]
w0 = within(d0.assign(y=d0["ydiff"]), cols + ["y"])
b0 = np.linalg.lstsq(w0[cols].to_numpy(), w0["y"].to_numpy(), rcond=None)[0]
ci0 = cell_boot(d0.assign(y=d0["ydiff"]), cols, "TS")
PA["difference_imposed"] = {"gradient": round(float(b0[1]), 4), "ci": ci0,
                            "half_width": round(float((ci0[1] - ci0[0]) / 2), 4),
                            "sig": bool(ci0[0] > 0 or ci0[1] < 0), "n_obs": len(d0)}
print(f"  {'차분 강제 (lag 계수=1)':<28} obs {len(d0):>5} · 기울기 {b0[1]:>+7.4f} "
      f"{str(ci0):<22}{'✓' if PA['difference_imposed']['sig'] else '✗'} "
      f"반폭 {PA['difference_imposed']['half_width']:.4f}")
PA["precision_gain_vs_difference"] = round(
    1 - PA["baseline"]["half_width"] / PA["difference_imposed"]["half_width"], 3)
print(f"  → ANCOVA 의 CI 반폭이 차분 대비 {PA['precision_gain_vs_difference']:.1%} 축소")

print("\n[Panel B] ★ PPML — 채용 건수 + log 고용 offset (N=0 유지)")
dp = D.dropna(subset=["lag", "S", "lsize"]).reset_index(drop=True)
dp["TS"] = dp["T"] * dp["S"]
Xp = pd.get_dummies(dp["cell"].astype("category"), drop_first=True, dtype=float)
Xp = np.column_stack([dp[["T", "TS", "lag", "S"]].to_numpy(), Xp.to_numpy()])
off = np.log(dp["e_post"].to_numpy() * 12.0)
try:
    m = sm.GLM(dp["n_post"].to_numpy(), Xp, family=sm.families.Poisson(), offset=off).fit(
        cov_type="cluster", cov_kwds={"groups": dp["cell"].to_numpy()}, maxiter=200)
    g, se = float(m.params[1]), float(m.bse[1])
    PB = {"n_obs": len(dp), "n_zero_post": int((dp.n_post == 0).sum()),
          "gradient": round(g, 4), "se": round(se, 4),
          "ci": [round(g - 1.96 * se, 4), round(g + 1.96 * se, 4)],
          "sig": bool(abs(g / se) > 1.96), "treated_main": round(float(m.params[0]), 4)}
    print(f"  obs {PB['n_obs']} (사후 N=0 {PB['n_zero_post']}건 포함) · 기울기 "
          f"{PB['gradient']:>+7.4f} SE {PB['se']:.4f} CI {PB['ci']} "
          f"{'✓' if PB['sig'] else '✗'}")
except Exception as ex:
    PB = {"error": str(ex)}; print("  실패:", ex)

print("\n[Panel C] 상태창 완화 — 표본 회수")
PC = {}
for wl, need, tag in (((-24, -13), 12, "12개월 (기준)"), ((-18, -13), 6, "6개월"),
                      ((-24, -13), 9, "부분 ≥9개월")):
    Dx = build_rows(state_win=wl, state_need=need)
    PC[tag] = ancova(Dx, tag)

print("\n[Panel D] 강건 위치추정 — 매칭차분 기울기")
def matched_diff(M=None, post=(1, 12)):
    ys, xs, cv = [], [], []
    for e in EV:
        po = rate(e["ti"], e["m0"], post[0], post[1], M); pr = rate(e["ti"], e["m0"], -12, -1, M)
        st = rate(e["ti"], e["m0"], -24, -13, M)
        if not (po and pr and st and po[0] > 0 and pr[0] > 0): continue
        t = np.log(po[0] / po[1]) - np.log(pr[0] / pr[1])
        cs = []
        for k in e["ctrls"]:
            p2 = rate(int(k), e["m0"], post[0], post[1], M); r2 = rate(int(k), e["m0"], -12, -1, M)
            if p2 and r2 and p2[0] > 0 and r2[0] > 0:
                cs.append(np.log(p2[0] / p2[1]) - np.log(r2[0] / r2[1]))
        if not cs: continue
        ys.append(t - float(np.mean(cs))); xs.append(-np.log1p(st[0] / st[1]))
        cv.append((np.log(st[1]), str(G["ind_arr"][e["ti"]])[:1]))
    return np.array(ys), np.array(xs), cv


def fwl_slope(y, x, cv, R=NB):
    C = np.column_stack([np.ones(len(y)), np.array([c[0] for c in cv])]
                        + [np.array([1.0 if c[1] == s else 0.0 for c in cv])
                           for s in sorted({c[1] for c in cv})[1:]])
    r_ = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    yr, xr = r_(y), r_(x)
    b = float(np.polyfit(xr, yr, 1)[0])
    bb = np.array([np.polyfit(xr[i], yr[i], 1)[0]
                   for i in (rng.integers(0, len(y), len(y)) for _ in range(R))])
    ci = qci(bb)
    return {"slope": round(b, 4), "ci": ci, "n": len(y), "sig": bool(ci[0] > 0 or ci[1] < 0),
            "half_width": round(float((ci[1] - ci[0]) / 2), 4)}


y0, x0, cv0 = matched_diff()
PD = {"raw": fwl_slope(y0, x0, cv0)}
for lo, hi, tag in ((1, 99, "winsor_1_99"), (5, 95, "winsor_5_95")):
    PD[tag] = fwl_slope(np.clip(y0, *np.percentile(y0, [lo, hi])), x0, cv0)
keep = (y0 >= np.percentile(y0, 2.5)) & (y0 <= np.percentile(y0, 97.5))
PD["trim_5pct"] = fwl_slope(y0[keep], x0[keep], [c for c, k in zip(cv0, keep) if k])
for k, v in PD.items():
    print(f"  {k:<14} {v['slope']:>+7.4f} {str(v['ci']):<22}{'✓' if v['sig'] else '✗'} "
          f"반폭 {v['half_width']:.4f} (n={v['n']})")

print("\n[Panel E] ★ 암묵 채용 A* = max(0, ΔE + S) — 표본 손실 없이 stock-flow 검정")
Astar = np.zeros_like(Hv)
Astar[:, 1:] = np.maximum(0.0, np.diff(Ev, axis=1) + Sv[:, 1:])
Astar[:, 0] = Hv[:, 0]
yA, xA, cvA = matched_diff(M=Astar)
PE_ = {"implied_hires": fwl_slope(yA, xA, cvA),
       "reported_hires": PD["raw"],
       "corr_measures": round(float(np.corrcoef(
           Hv[np.isfinite(Hv) & np.isfinite(Astar)],
           Astar[np.isfinite(Hv) & np.isfinite(Astar)])[0, 1]), 4)}
print(f"  두 측도 상관 {PE_['corr_measures']:.4f}")
for k in ("reported_hires", "implied_hires"):
    v = PE_[k]
    print(f"  {k:<16} {v['slope']:>+7.4f} {str(v['ci']):<22}{'✓' if v['sig'] else '✗'} (n={v['n']})")

print("\n[Panel F] 사후창 확장 [1,24]")
try:
    y2, x2, cv2 = matched_diff(post=(1, 24))
    PF = {"post_24m": fwl_slope(y2, x2, cv2), "post_12m": PD["raw"]}
    for k, v in PF.items():
        print(f"  {k:<10} {v['slope']:>+7.4f} {str(v['ci']):<22}{'✓' if v['sig'] else '✗'} "
              f"반폭 {v['half_width']:.4f} (n={v['n']})")
except Exception as ex:
    PF = {"error": str(ex)}; print("  실패:", ex)

print("\n[Panel G] ★ 가중 로그선형 — PPML 의 부호반전이 '건수 가중' 때문인가")
def fwl_slope_w(y, x, cv, wt, R=NB):
    C = np.column_stack([np.ones(len(y)), np.array([c[0] for c in cv])]
                        + [np.array([1.0 if c[1] == s2 else 0.0 for c in cv])
                           for s2 in sorted({c[1] for c in cv})[1:]])
    W = np.asarray(wt, float); W = W / W.mean()
    def wls(yy, xx, ww):
        Cw = C * np.sqrt(ww)[:, None]
        r_ = lambda v: v - C @ np.linalg.lstsq(Cw, (v * np.sqrt(ww)), rcond=None)[0]
        yr, xr = r_(yy), r_(xx)
        return float(np.sum(ww * xr * yr) / np.sum(ww * xr * xr))
    b = wls(y, x, W)
    bb = np.array([wls(y[i], x[i], W[i])
                   for i in (rng.integers(0, len(y), len(y)) for _ in range(R))])
    ci = qci(bb)
    return {"slope": round(b, 4), "ci": ci, "n": len(y), "sig": bool(ci[0] > 0 or ci[1] < 0)}

wt_emp = np.array([np.exp(c[0]) for c in cv0])          # 사전 고용 가중
PG = {"unweighted": PD["raw"],
      "weight_pre_employment": fwl_slope_w(y0, x0, cv0, wt_emp),
      "weight_sqrt_employment": fwl_slope_w(y0, x0, cv0, np.sqrt(wt_emp))}
for k, v in PG.items():
    print(f"  {k:<24} {v['slope']:>+7.4f} {str(v['ci']):<22}{'✓' if v['sig'] else '✗'}")
# 규모 3분위별 기울기
sz0 = np.array([c[0] for c in cv0]); q1s, q2s = np.percentile(sz0, [33.33, 66.67])
for lab, m in (("소규모", sz0 <= q1s), ("중간", (sz0 > q1s) & (sz0 <= q2s)), ("대규모", sz0 > q2s)):
    if m.sum() < 40: continue
    r = fwl_slope(y0[m], x0[m], [c for c, k in zip(cv0, m) if k])
    PG[f"size_{lab}"] = r
    print(f"  규모 {lab:<8} (n={int(m.sum())}) {r['slope']:>+7.4f} {str(r['ci']):<22}"
          f"{'✓' if r['sig'] else '✗'}")

best = min([("ANCOVA", PA["baseline"]["half_width"]),
            ("matched_diff", PD["raw"]["half_width"])], key=lambda t: t[1])
verdict = (
    f"[A] 풀링 ANCOVA(셀 FE, 처치+대조 {PA['baseline']['n_obs']}관측): 기울기 "
    f"{PA['baseline']['gradient']:+.4f}{PA['baseline']['ci']} 반폭 {PA['baseline']['half_width']:.4f} — "
    f"차분 강제 대비 CI {PA['precision_gain_vs_difference']:.0%} 축소 (lag 계수 "
    f"{PA['baseline']['lag_coef']:+.3f}, 차분이 강제하는 −1 과 다름). "
    + (f"[B] PPML 기울기 {PB['gradient']:+.4f}{PB['ci']}{'✓' if PB.get('sig') else '✗'} — "
       f"로그변환이 결과를 만들지 않았다. " if "gradient" in PB else "[B] PPML 실패. ")
    + f"[C] 상태창 6개월로 완화 시 셀 {PC['6개월']['n_cells']} (기준 {PC['12개월 (기준)']['n_cells']}), "
      f"기울기 {PC['6개월']['gradient']:+.4f}. "
    + f"[D] winsor 5/95 {PD['winsor_5_95']['slope']:+.4f} 반폭 {PD['winsor_5_95']['half_width']:.4f} "
      f"(원 {PD['raw']['half_width']:.4f}). "
    + f"[E] 암묵 채용 {PE_['implied_hires']['slope']:+.4f}{PE_['implied_hires']['ci']}"
      f"{'✓' if PE_['implied_hires']['sig'] else '✗'} — 보고 채용 {PD['raw']['slope']:+.4f} 와 정합. "
    + f"[G] 사전고용 가중 시 {PG['weight_pre_employment']['slope']:+.4f}"
      f"{PG['weight_pre_employment']['ci']}"
      f"{'✓' if PG['weight_pre_employment']['sig'] else '✗'} — PPML 부호반전의 원인 판별. "
    + (f"[F] 사후 24개월 {PF['post_24m']['slope']:+.4f} 반폭 {PF['post_24m']['half_width']:.4f}."
       if "post_24m" in PF else ""))
emit("I-50", "검정력 제고 — 풀링 ANCOVA · PPML · 강건추정 · 암묵채용",
     "GO" if PA["baseline"]["sig"] else "PARTIAL",
     {"panelA_pooled_ancova": PA, "panelB_ppml": PB, "panelC_state_window": PC,
      "panelD_robust_location": PD, "panelE_implied_hires": PE_, "panelF_post_window": PF,
      "panelG_weighting": PG,
      "most_precise": best[0], "n_events": len(EV)},
     "풀링·PPML·강건추정으로 상태 gradient 의 정밀도를 높일 수 있는가",
     verdict, kill_met=False, n=len(EV))
