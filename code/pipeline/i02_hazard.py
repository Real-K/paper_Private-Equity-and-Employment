# -*- coding: utf-8 -*-
"""I-02 추정대상을 hazard로 전환.

지금까지의 결과대상(채용률 = 유량/평균재고)은 규모 변동에 오염되고 메커니즘(관성)과 단위가 다르다.
관성의 자연스러운 언어는 **다음 채용까지의 대기시간에 대한 이산시간 hazard** 다.

관측단위: 기업-월. d = 직전 연속 무채용 개월수 + 1. 사건 = 해당 월 신규채용 >= 1.
버닌 [-24,-13] 으로 d를 확립하고 [-12,-1] U [1,12] 만 기록. 딜 당월(k=0) 제외.

Panel A   지속기간 버킷별 hazard DiD — 풀링 + 이벤트 부트스트랩 (절대 pp / 비율 둘 다)
Panel B   cloglog + 이벤트FE + 지속기간더미, treated x post -> hazard ratio (이벤트 군집 SE)
Panel B2  삼중교호 treated x post x long(d>=3) — '균일 수준이동' vs '장기 무행동 집중' 정식 판별
Panel C   무행동 spell 길이 DiD
Panel D   그룹별 hazard 프로파일 (원자료)

[초판 정정] 초판 Panel A는 이벤트별 층화 DiD였고 버킷마다 4개 셀을 모두 요구해 장기 버킷이
굶었다(d=3에 14건, d>=12에 1건). 그 사양의 '단기 집중' 결론은 estimator 인공물이므로 폐기.

[메모리] 모든 공변량이 이산이므로 기업-월 48,853행을 (이벤트 x 처치 x 사후 x 버킷) 셀로 접어
grouped binomial 로 적합한다. 우도·계수·군집SE 모두 개별관측 적합과 동일하고, 설계행렬이
48,853 x 385 -> 9,096 x 385 (float32) 로 줄어 피크 메모리가 한 자릿수 배 감소한다.
"""
import gc
import numpy as np, pandas as pd
import statsmodels.api as sm
from h30_common import load, deals, build, boot_did_ci, emit, SEED, qci, NB

rng = np.random.default_rng(SEED)
print("[I-02] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
EV, _ = build(G, allt, PE)
Hv, mset = G["Hv"], G["mset"]
print(f"  이벤트 {len(EV)}건")

DCUT = [1, 2, 3, 4, 6, 12]
DLAB = ["d=1", "d=2", "d=3", "d=4-5", "d=6-11", "d>=12"]
NBK = len(DLAB)
LONG_FROM = 2          # 버킷 인덱스 2 = d>=3 을 '장기 무행동'으로 정의


def dbucket(d):
    return int(np.searchsorted(DCUT, d, side="right") - 1)


def spell_obs(row, m0):
    out, d, warm = [], 1, 0
    for k in range(-24, 13):
        j = mset.get(m0 + k)
        if j is None:
            d, warm = 1, 0
            continue
        h = Hv[row, j]
        if not np.isfinite(h):
            d, warm = 1, 0
            continue
        if warm >= 6 and k != 0 and (-12 <= k <= -1 or 1 <= k <= 12):
            out.append((k, 1 if h > 0 else 0, min(d, 24)))
        d = 1 if h > 0 else d + 1
        warm += 1
    return out


rec = []
for ei, e in enumerate(EV):
    for who, row in [(1, e["ti"])] + [(0, c) for c in e["ctrls"]]:
        for k, y, d in spell_obs(row, e["m0"]):
            rec.append((ei, who, 1 if k > 0 else 0, y, dbucket(d), row))
P = pd.DataFrame(rec, columns=["ev", "treated", "post", "hire", "db", "row"])
del rec
NE = P.ev.nunique(); N_FIRM = P.row.nunique(); N_OBS = len(P)
print(f"  기업-월 관측 {N_OBS:,} (이벤트 {NE}, 기업 {N_FIRM:,})")

# ---------- Panel A : 풀링 + 이벤트 부트스트랩 ----------
SUM = np.zeros((len(EV), NBK, 2, 2))
CNT = np.zeros_like(SUM)
ix = (P.ev.values, P.db.values, P.treated.values, P.post.values)
np.add.at(SUM, ix, P.hire.values.astype(float))
np.add.at(CNT, ix, 1.0)
del ix


def stats(sel):
    s, c = SUM[sel].sum(0), CNT[sel].sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = s / np.where(c > 0, c, np.nan)
        absd = (h[:, 1, 1] - h[:, 1, 0]) - (h[:, 0, 1] - h[:, 0, 0])
        rat = np.log(h[:, 1, 1] / h[:, 1, 0]) - np.log(h[:, 0, 1] / h[:, 0, 0])
    return absd, rat, h, c


A_abs, A_rat, H0, C0 = stats(np.arange(len(EV)))
BA = np.empty((NB, NBK)); BR = np.empty((NB, NBK))
for i in range(NB):
    a, r, _, _ = stats(rng.integers(0, len(EV), len(EV)))
    BA[i], BR[i] = a, r
print("\n[Panel A] 지속기간 버킷별 hazard DiD (풀링, 이벤트 부트스트랩 999)")
print(f"  {'버킷':<7} {'관측월':>7} {'처치사전':>7} {'처치사후':>7} {'대조사전':>7} {'대조사후':>7}"
      f" {'DiD(pp)':>9} {'CI95':<19} {'비율DiD':>8} {'CI95':<18}")
PA = {}
for b, lab in enumerate(DLAB):
    n_obs = int(C0[b].sum())
    if C0[b].min() < 30 or not np.isfinite(A_abs[b]):
        print(f"  {lab:<7} {n_obs:>7}   (셀 최소관측 {int(C0[b].min())} < 30)")
        PA[lab] = {"n_obs": n_obs, "min_cell": int(C0[b].min()), "note": "cell<30"}
        continue
    ca = qci(BA[np.isfinite(BA[:, b]), b]); cr = qci(BR[np.isfinite(BR[:, b]), b])
    sa = "✓" if (ca[0] > 0 or ca[1] < 0) else "✗"
    sr = "✓" if (cr[0] > 0 or cr[1] < 0) else "✗"
    rci = [round(float(np.exp(x) - 1), 3) for x in cr]
    print(f"  {lab:<7} {n_obs:>7} {H0[b,1,0]:>7.3f} {H0[b,1,1]:>7.3f} {H0[b,0,0]:>7.3f} {H0[b,0,1]:>7.3f}"
          f" {A_abs[b]:>+9.4f} {str(ca):<19}{sa} {np.exp(A_rat[b])-1:>+7.3f} {str(rci):<18}{sr}")
    PA[lab] = {"n_obs": n_obs, "min_cell": int(C0[b].min()),
               "t_pre": round(float(H0[b, 1, 0]), 4), "t_post": round(float(H0[b, 1, 1]), 4),
               "c_pre": round(float(H0[b, 0, 0]), 4), "c_post": round(float(H0[b, 0, 1]), 4),
               "DiD_pp": round(float(A_abs[b]), 4), "ci_pp": ca, "sig_pp": sa == "✓",
               "DiD_ratio": round(float(np.exp(A_rat[b]) - 1), 4),
               "ci_ratio": rci, "sig_ratio": sr == "✓"}
del BA, BR
gc.collect()

# ---------- 셀 접기 (grouped binomial) ----------
Cl = (P.groupby(["ev", "treated", "post", "db"], as_index=False)
        .agg(succ=("hire", "sum"), n=("hire", "size")))
Cl["fail"] = Cl.n - Cl.succ
nC = len(Cl)
print(f"\n  [메모리] 개별관측 {N_OBS:,}행 -> 셀 {nC:,}행 (등가 grouped binomial)")

# 고정 블록: 상수 + 지속기간더미(d=1 기준) + 이벤트FE(첫 이벤트 기준), float32
evc = pd.Categorical(Cl.ev).codes.astype(np.int32)
dbc = Cl.db.values.astype(np.int32)
BASE = np.zeros((nC, 1 + (NBK - 1) + (NE - 1)), np.float32)
BASE[:, 0] = 1.0
m = dbc > 0
BASE[np.flatnonzero(m), 1 + dbc[m] - 1] = 1.0
m = evc > 0
BASE[np.flatnonzero(m), NBK + evc[m] - 1] = 1.0
ENDOG = np.column_stack([Cl.succ.values, Cl.fail.values]).astype(np.float64)
CLUST = Cl.ev.values
D_OFF = 1                       # BASE 안에서 지속기간더미 시작 위치
del P
gc.collect()


def fit(vary, names):
    """vary: (nC, k) float32 — 관심 변수 블록. BASE 앞에 붙여 적합."""
    X = np.hstack([vary.astype(np.float32), BASE])
    r = sm.GLM(ENDOG, X, family=sm.families.Binomial(sm.families.links.CLogLog())
               ).fit(cov_type="cluster", cov_kwds={"groups": CLUST})
    o = {}
    for i, v in enumerate(names):
        b, se = float(r.params[i]), float(r.bse[i])
        o[v] = {"coef": round(b, 4), "se": round(se, 4), "HR": round(float(np.exp(b)), 4),
                "HR_ci": [round(float(np.exp(b - 1.96 * se)), 4),
                          round(float(np.exp(b + 1.96 * se)), 4)],
                "p": float(f"{r.pvalues[i]:.3g}")}
        print(f"  {v:<9} b={b:+.4f} (se {se:.4f})  HR={np.exp(b):.4f} "
              f"[{np.exp(b-1.96*se):.4f}, {np.exp(b+1.96*se):.4f}]  p={r.pvalues[i]:.4g}")
    par = r.params.copy(); nobs = int(r.nobs)
    del X, r
    gc.collect()
    return o, par, nobs


tr = Cl.treated.values.astype(np.float32)
po = Cl.post.values.astype(np.float32)
tp = tr * po
lng = (dbc >= LONG_FROM).astype(np.float32)

# ---------- Panel B ----------
print("\n[Panel B] cloglog + 이벤트FE + 지속기간더미  (treated x post -> HR)")
PB, parB, nobsB = fit(np.column_stack([tr, po, tp]), ["treated", "post", "tp"])
print(f"  n_cells={nobsB:,}  (원 기업-월 {N_OBS:,})  이벤트FE {NE}개  군집 SE(이벤트)")
PB["n_cells"] = nobsB; PB["n_firm_months"] = N_OBS; PB["n_ev_fe"] = NE
PB["baseline_duration_HR"] = {DLAB[b]: round(float(np.exp(parB[3 + D_OFF + b - 1])), 4)
                              for b in range(1, NBK)}
print("  기저 지속기간 효과(d=1 기준):", PB["baseline_duration_HR"])

# ---------- Panel B2 ----------
print("\n[Panel B2] 삼중교호 treated x post x long(d>=3) — 형태 정식 판별")
PB2, _, _ = fit(np.column_stack([tr, po, tp, tr * lng, po * lng, tp * lng]),
                ["treated", "post", "tp", "t_lng", "p_lng", "tp_lng"])
tri = PB2["tp_lng"]
tri_sig = not (tri["HR_ci"][0] <= 1.0 <= tri["HR_ci"][1])
hr_short = PB2["tp"]["HR"]
hr_long = round(float(np.exp(PB2["tp"]["coef"] + tri["coef"])), 4)
print(f"  -> 단기(d<=2) HR={hr_short}   장기(d>=3) HR={hr_long}   차이 HR={tri['HR']} {tri['HR_ci']}"
      f" {'✓ 유의' if tri_sig else '✗ 비유의'}")
PB2.update({"HR_short_dle2": hr_short, "HR_long_dge3": hr_long,
            "diff_sig": tri_sig, "long_defined_as": "d>=3"})
del BASE, ENDOG
gc.collect()

# ---------- Panel C ----------
print("\n[Panel C] 무행동 spell 길이 DiD (창내 완결 spell 평균)")


def mean_spell(row, m0, lo, hi):
    j = [mset.get(m0 + k) for k in range(lo, hi + 1)]
    if any(x is None for x in j):
        return np.nan
    h = Hv[row, j]
    if not np.isfinite(h).all():
        return np.nan
    sp, run = [], 0
    for x in h:
        if x > 0:
            sp.append(run + 1); run = 0
        else:
            run += 1
    return float(np.mean(sp)) if sp else np.nan


t, c = [], []
for e in EV:
    a = mean_spell(e["ti"], e["m0"], -12, -1); b = mean_spell(e["ti"], e["m0"], 1, 12)
    cd = [mean_spell(k, e["m0"], 1, 12) - mean_spell(k, e["m0"], -12, -1) for k in e["ctrls"]]
    cd = [x for x in cd if np.isfinite(x)]
    if np.isfinite(a) and np.isfinite(b) and cd:
        t.append(b - a); c.append(np.mean(cd))
ptC, ciC, nC2 = boot_did_ci(t, c, rng)
sigC = "✓" if (ciC and (ciC[0] > 0 or ciC[1] < 0)) else "✗"
knifeC = bool(ciC and min(abs(ciC[0]), abs(ciC[1])) < 0.02)
print(f"  평균 spell 길이 DiD  n={nC2}  {ptC:+.4f} {ciC} {sigC}"
      f"{'  (경계 근접 — 배제 주장 금지)' if knifeC and sigC=='✗' else ''}  (개월)")
PC = {"n": nC2, "DiD_months": ptC, "ci": ciC, "sig": sigC == "✓", "knife_edge": knifeC}

# ---------- Panel D ----------
print("\n[Panel D] 그룹별 hazard 프로파일 (원자료)")
PD = {}
for who, wl in [(1, "처치"), (0, "대조")]:
    for pz, pl in [(0, "사전"), (1, "사후")]:
        hz = [round(float(H0[b, who, pz]), 4) if C0[b, who, pz] > 30 else None for b in range(NBK)]
        PD[f"{wl}_{pl}"] = hz
        print(f"  {wl}{pl}: " + "  ".join(f"{DLAB[b]}={hz[b]}" for b in range(NBK) if hz[b] is not None))

# ---------- 판정 ----------
hr = PB["tp"]["HR"]; hrci = PB["tp"]["HR_ci"]
hr_sig = not (hrci[0] <= 1.0 <= hrci[1])
sig_r = [l for l in DLAB if PA.get(l, {}).get("sig_ratio")]
if tri_sig and hr_long > hr_short:
    shape = "장기 무행동 집중 (관성 제거와 정합)"
elif tri_sig and hr_long < hr_short:
    shape = "단기 집중 (관성 제거와 불정합)"
else:
    shape = "균일 수준이동 (지속기간별 차이 비유의)"
status = "GO" if hr_sig else "KILL"
verdict = (f"cloglog treated×post HR={hr} {hrci} p={PB['tp']['p']}{' ✓' if hr_sig else ' ✗'} "
           f"| 형태: {shape} (단기 {hr_short} vs 장기 {hr_long}, 차이 HR {tri['HR']} {tri['HR_ci']}) "
           f"| 비율DiD 유의 버킷 {sig_r or '없음'} | 평균 spell DiD {ptC:+.4f}개월 {sigC}")
emit("I-02", "hazard 전환 (이산시간 cloglog)", status,
     {"panelA_hazard_by_duration": PA, "panelB_cloglog": PB, "panelB2_shape": PB2,
      "panelC_spell_length": PC, "panelD_hazard_profile": PD, "duration_labels": DLAB},
     "채용 hazard의 treated×post HR>1 유의. 지속기간별 형태가 '균일 수준이동'인지 '장기 무행동 집중'인지 판별",
     verdict, kill_met=not hr_sig, n=len(EV),
     extra={"shape": shape, "sig_ratio_buckets": sig_r,
            "estimator_note": "초판 Panel A(이벤트별 층화)는 장기버킷 표본기아로 폐기, 풀링으로 교체",
            "memory_note": f"grouped binomial: {N_OBS:,} 기업-월 -> {nobsB:,} 셀, 설계행렬 float32"})
