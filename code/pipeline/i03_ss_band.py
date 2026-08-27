# -*- coding: utf-8 -*-
"""I-03 (S,s) 무행동 밴드폭 추정.

[I-02가 남긴 문제] 원 지속기간 의존성이 음(-)이다 (d=1 기준 HR: d>=12 0.295).
순수 (S,s)는 갭이 벌어질수록 hazard 상승을 예측하므로 정면 충돌. 두 가지 가능성:
  (i) 이질성 오염 — 저채용 성향 기업이 긴 spell과 낮은 hazard를 동시에 만든다
  (ii) d 자체가 갭의 나쁜 대리변수 — 이직률이 낮으면 오래 안 뽑아도 갭이 안 쌓인다

[해결] 갭을 직접 관측한다. **g = (마지막 채용 이후 누적 이직) / (spell 시작 시점 고용)**.
이직이 고용을 잠식해 갭이 벌어지고, 임계에 닿으면 일괄 채용한다 — 채용 (S,s)의 자연스러운 상태변수.

Panel A  이질성 분리 — 지속기간 프로파일 (풀링 vs 기업FE within)
Panel B  갭 조건부 hazard h(g) — (S,s) 정합성 검정 (기업FE within 포함)
Panel C  **밴드폭 = trigger gap** (채용이 실제로 발생한 달의 g) 의 DiD  ← 주 결과
Panel D  미완결 spell의 최대 도달 갭 (우측 절단 보완)

[한계 명시] g를 원화 고정비용으로 환산하려면 검증 불가한 구조 매핑이 필요하므로 하지 않는다.
밴드폭은 고용 대비 비율(로그포인트가 아닌 이직누적/고용) 단위로만 보고한다.
[메모리] 기업FE는 더미가 아니라 within 변환으로 흡수. 설계행렬 48,853 x 6 float32.
"""
import gc
import numpy as np, pandas as pd
import statsmodels.api as sm
from h30_common import load, deals, build, boot_did_ci, emit, SEED, qci, NB

rng = np.random.default_rng(SEED)
print("[I-03] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
EV, _ = build(G, allt, PE)
Hv, Sv, Ev, mset = G["Hv"], G["Sv"], G["Ev"], G["mset"]
print(f"  이벤트 {len(EV)}건")

DCUT = [1, 2, 3, 4, 6, 12]
DLAB = ["d=1", "d=2", "d=3", "d=4-5", "d=6-11", "d>=12"]
GCUT = [-1e-9, 0.0001, 0.02, 0.05, 0.10, 0.20]
GLAB = ["g=0", "g<=2%", "2-5%", "5-10%", "10-20%", "g>20%"]
NBK, NGK = len(DLAB), len(GLAB)


def walk(row, m0):
    """[(k, hire01, dbucket, gbucket, g, trig_or_nan)] — 버닌 후 사전/사후 창만."""
    out, d, warm = [], 1, 0
    cum, Eref = 0.0, np.nan
    for k in range(-24, 13):
        j = mset.get(m0 + k)
        if j is None:
            d, warm, cum, Eref = 1, 0, 0.0, np.nan
            continue
        h, s, e = Hv[row, j], Sv[row, j], Ev[row, j]
        if not (np.isfinite(h) and np.isfinite(s) and np.isfinite(e)):
            d, warm, cum, Eref = 1, 0, 0.0, np.nan
            continue
        g = cum / Eref if (np.isfinite(Eref) and Eref >= 5) else np.nan
        hire = 1 if h > 0 else 0
        if warm >= 6 and k != 0 and (-12 <= k <= -1 or 1 <= k <= 12) and np.isfinite(g):
            gb = int(np.searchsorted(GCUT, g, side="right") - 1)
            out.append((k, hire, int(np.searchsorted(DCUT, min(d, 24), side="right") - 1),
                        min(max(gb, 0), NGK - 1), g, g if hire else np.nan))
        if hire:
            d, cum, Eref = 1, 0.0, e
        else:
            d += 1; cum += s
        warm += 1
    return out


rec = []
for ei, e in enumerate(EV):
    for who, row in [(1, e["ti"])] + [(0, c) for c in e["ctrls"]]:
        for k, y, db, gb, g, trig in walk(row, e["m0"]):
            rec.append((ei, who, 1 if k > 0 else 0, y, db, gb, g, trig, row))
P = pd.DataFrame(rec, columns=["ev", "tr", "po", "hire", "db", "gb", "g", "trig", "row"])
del rec; gc.collect()
NE, NF, NOBS = P.ev.nunique(), P.row.nunique(), len(P)
print(f"  기업-월 {NOBS:,} (이벤트 {NE}, 기업 {NF:,})  갭 관측률 {np.isfinite(P.g).mean():.3f}")


def within(y, codes):
    """기업FE within 변환 (더미 없이)."""
    y = np.asarray(y, np.float64)
    sm_ = np.bincount(codes, weights=y, minlength=codes.max() + 1)
    ct = np.bincount(codes, minlength=codes.max() + 1)
    return y - (sm_ / np.maximum(ct, 1))[codes]


fcodes = pd.Categorical(P.row).codes.astype(np.int64)


def profile(bcol, nb, labs, tag):
    """풀링 LPM vs 기업FE within LPM — 버킷 프로파일 (기준=버킷0)."""
    D = np.zeros((NOBS, nb - 1), np.float32)
    b = P[bcol].values
    m = b > 0
    D[np.flatnonzero(m), b[m] - 1] = 1.0
    out = {}
    for lab, X, yv in [("pooled", sm.add_constant(D.astype(np.float64)), P.hire.values.astype(float)),
                       ("firmFE", np.column_stack([within(D[:, i], fcodes) for i in range(nb - 1)]),
                        within(P.hire.values, fcodes))]:
        r = sm.OLS(yv, X).fit(cov_type="cluster", cov_kwds={"groups": P.ev.values})
        off = 1 if lab == "pooled" else 0
        co = [round(float(r.params[off + i]), 4) for i in range(nb - 1)]
        se = [round(float(r.bse[off + i]), 4) for i in range(nb - 1)]
        out[lab] = {"coef_vs_base": co, "se": se}
        print(f"  {tag} {lab:<7} " + "  ".join(f"{labs[i+1]}:{co[i]:+.3f}" for i in range(nb - 1)))
        del r
    del D; gc.collect()
    return out


print("\n[Panel A] 지속기간 프로파일 — 이질성 분리 (기준 d=1, LPM 계수 = hazard pp 차)")
PA = profile("db", NBK, DLAB, "dur")
flip_A = (PA["firmFE"]["coef_vs_base"][-1] > PA["pooled"]["coef_vs_base"][-1])
print(f"  -> 기업FE 후 장기(d>=12) 계수 {PA['pooled']['coef_vs_base'][-1]:+.3f} "
      f"→ {PA['firmFE']['coef_vs_base'][-1]:+.3f} ({'완화' if flip_A else '악화'})")

print("\n[Panel B] 갭 조건부 hazard h(g) — (S,s) 정합성 (기준 g=0)")
PB = profile("gb", NGK, GLAB, "gap")
mono = all(PB["firmFE"]["coef_vs_base"][i] <= PB["firmFE"]["coef_vs_base"][i + 1] + 0.02
           for i in range(NGK - 2))
print(f"  -> 기업FE 갭 프로파일 단조증가(허용 0.02): {'✓ (S,s) 정합' if mono else '✗ 불정합'}")
# 원자료 h(g) 4그룹
print("  원자료 h(g):")
PBraw = {}
for who, wl in [(1, "처치"), (0, "대조")]:
    for pz, pl in [(0, "사전"), (1, "사후")]:
        S = P[(P.tr == who) & (P.po == pz)]
        v = [round(float(S.loc[S.gb == b, "hire"].mean()), 4) if (S.gb == b).sum() > 50 else None
             for b in range(NGK)]
        PBraw[f"{wl}_{pl}"] = v
        print(f"    {wl}{pl}: " + "  ".join(f"{GLAB[b]}={v[b]}" for b in range(NGK) if v[b] is not None))

# ---------- Panel C : 밴드폭 = trigger gap ----------
print("\n[Panel C] 밴드폭 = trigger gap (채용 발생 월의 누적이직/고용) DiD  ← 주 결과")
TG = np.full((len(EV), 2, 2), np.nan)   # ev, treated, post
CN = np.zeros((len(EV), 2, 2))
sub = P[np.isfinite(P.trig)]
agg = sub.groupby(["ev", "tr", "po"])["trig"].agg(["mean", "size"])
for (ev, tr, po), r in agg.iterrows():
    if r["size"] >= 3:
        TG[ev, tr, po] = r["mean"]; CN[ev, tr, po] = r["size"]
t, c = [], []
for i in range(len(EV)):
    a, b = TG[i, 1, 0], TG[i, 1, 1]
    cd = TG[i, 0, 1] - TG[i, 0, 0]
    if np.isfinite(a) and np.isfinite(b) and np.isfinite(cd):
        t.append(b - a); c.append(cd)
ptC, ciC, nC = boot_did_ci(t, c, rng)
sigC = "✓" if (ciC and (ciC[0] > 0 or ciC[1] < 0)) else "✗"
base_pre = float(np.nanmean(TG[:, 1, 0]))
print(f"  처치 사전 평균 trigger gap = {base_pre:.4f} (고용 대비 누적이직 비율)")
print(f"  DiD  n={nC}  {ptC:+.4f} {ciC} {sigC}"
      f"   → 밴드폭 변화 {100*ptC/base_pre:+.1f}%" if ptC else "")
PC = {"n_ev": nC, "DiD": ptC, "ci": ciC, "sig": sigC == "✓",
      "treated_pre_mean": round(base_pre, 4),
      "pct_change": round(100 * ptC / base_pre, 2) if ptC else None,
      "cells": {"t_pre": round(float(np.nanmean(TG[:, 1, 0])), 4),
                "t_post": round(float(np.nanmean(TG[:, 1, 1])), 4),
                "c_pre": round(float(np.nanmean(TG[:, 0, 0])), 4),
                "c_post": round(float(np.nanmean(TG[:, 0, 1])), 4)}}
print(f"  셀 평균: {PC['cells']}")

# ---------- Panel D : 미완결 spell 최대 도달 갭 ----------
print("\n[Panel D] 창내 미완결(우측절단) spell의 최대 도달 갭")
mx = P.groupby(["ev", "tr", "po"])["g"].max()
MX = np.full((len(EV), 2, 2), np.nan)
for (ev, tr, po), v in mx.items():
    MX[ev, tr, po] = v
t2, c2 = [], []
for i in range(len(EV)):
    a, b, cd = MX[i, 1, 0], MX[i, 1, 1], MX[i, 0, 1] - MX[i, 0, 0]
    if np.isfinite(a) and np.isfinite(b) and np.isfinite(cd):
        t2.append(b - a); c2.append(cd)
ptD, ciD, nD = boot_did_ci(t2, c2, rng)
sigD = "✓" if (ciD and (ciD[0] > 0 or ciD[1] < 0)) else "✗"
print(f"  최대 도달 갭 DiD  n={nD}  {ptD:+.4f} {ciD} {sigD}")
PD = {"n_ev": nD, "DiD": ptD, "ci": ciD, "sig": sigD == "✓"}

# ---------- 판정 ----------
narrowed = bool(PC["sig"] and ptC < 0)
if narrowed and mono:
    status, shape = "GO", "(S,s) 정합 + 밴드 축소"
elif mono:
    status, shape = "PARTIAL", "(S,s) 정합하나 밴드 축소 미검출"
elif narrowed:
    status, shape = "PARTIAL", "밴드 축소는 검출되나 (S,s) 갭 단조성 불성립"
else:
    status, shape = "KILL", "(S,s) 불정합 + 밴드 축소 미검출"
verdict = (f"갭 프로파일 기업FE 단조증가 {'✓' if mono else '✗'} | "
           f"trigger gap DiD {ptC}{ciC} {sigC} (처치사전 {base_pre:.4f}) | "
           f"지속기간 음의존성은 기업FE 후 {PA['pooled']['coef_vs_base'][-1]:+.3f}→"
           f"{PA['firmFE']['coef_vs_base'][-1]:+.3f} | 판정: {shape}")
emit("I-03", "(S,s) 무행동 밴드폭 추정", status,
     {"panelA_duration_profile": PA, "panelB_gap_profile": PB, "panelB_raw_hg": PBraw,
      "panelC_trigger_gap_band": PC, "panelD_max_gap_censored": PD,
      "duration_labels": DLAB, "gap_labels": GLAB},
     "갭(마지막 채용 이후 누적이직/고용) 조건부 hazard가 단조증가하면 (S,s) 정합. "
     "PE가 밴드를 좁히면 trigger gap DiD < 0",
     verdict, kill_met=(status == "KILL"), n=len(EV),
     extra={"shape": shape, "ss_monotone": mono, "band_narrowed": narrowed,
            "scope_limit": "g를 원화 고정비용으로 환산하는 구조 매핑은 검증 불가하여 수행하지 않음",
            "n_firm_months": NOBS, "n_firms": int(NF)})
