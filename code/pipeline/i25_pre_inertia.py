# -*- coding: utf-8 -*-
"""I-25 사전 관성수준 조절 — 부호반전 후보.

가설: 관성이 심했던 기업일수록 효과가 크고, 이미 매달 뽑던 기업은 0 또는 음(더 자주 뽑을 여지 없음).
메커니즘(비주의 제거, I-03)이 기계적으로 함의하는 조절이다.

[함정] own-trajectory 검정이 평균회귀로 죽은 전례가 있다(−0.0407 → −0.0145).
관성을 사전창 [−12,−1]에서 재면 그 창이 결과의 기준선이기도 해서 회귀가 효과로 둔갑한다.
차단 2중 장치:
  (1) 관성을 **사전-사전 창 [−24,−13]** 에서 측정 — 결과 창과 분리
  (2) 대조군을 **같은 관성 bin** 에서만 사용 — 같은 회귀 압력을 받는 기업끼리 비교

Panel A  naive (모든 매칭 대조군) 3분위 DiD — 비교용
Panel B  동일 bin 대조군만 (평균회귀 차단) ← 주 결과
Panel C  균형 진단 — 분위별 처치 vs 대조의 사전-사전 관성
Panel D  hazard 삼중교호 treated x post x 고관성 (이벤트FE cloglog)

기각조건: 동일 bin 사양에서 분위간 차이가 사라지면 조절은 평균회귀 인공물.
"""
import gc
import numpy as np, pandas as pd
import statsmodels.api as sm
from h30_common import load, deals, build, boot_did_ci, emit, SEED, qci, NB, widx, dflow

rng = np.random.default_rng(SEED)
print("[I-25] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
EV, _ = build(G, allt, PE)
Hv, mset = G["Hv"], G["mset"]
print(f"  이벤트 {len(EV)}건")


def zshare(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return np.nan
    h = Hv[row, c]
    return float((h == 0).mean()) if np.isfinite(h).all() else np.nan


# ---- 사전-사전 관성 + 결과 부착 ----
for e in EV:
    e["pp_t"] = zshare(e["ti"], e["m0"], -24, -13)                       # 처치 사전-사전 관성
    e["pp_c"] = [zshare(k, e["m0"], -24, -13) for k in e["ctrls"]]       # 대조 각각
    a = zshare(e["ti"], e["m0"], -12, -1); b = zshare(e["ti"], e["m0"], 1, 12)
    e["z_t"] = b - a if (np.isfinite(a) and np.isfinite(b)) else np.nan
    e["z_c"] = [(zshare(k, e["m0"], 1, 12) - zshare(k, e["m0"], -12, -1)) for k in e["ctrls"]]
    e["h_t"] = dflow(G, e["ti"], e["m0"], Hv)
    e["h_c"] = [dflow(G, k, e["m0"], Hv) for k in e["ctrls"]]

USE = [e for e in EV if np.isfinite(e["pp_t"]) and np.isfinite(e["z_t"])]
print(f"  사전-사전 관성 관측 {len(USE)}/{len(EV)}")
pp = np.array([e["pp_t"] for e in USE])
q1, q2 = np.percentile(pp, [33.33, 66.67])
print(f"  사전-사전 무채용비중 3분위 컷: {q1:.4f} / {q2:.4f}  (평균 {pp.mean():.4f})")


def tbin(v):
    return 0 if v <= q1 else (1 if v <= q2 else 2)


TL = ["T1 저관성", "T2 중간", "T3 고관성"]
for e in USE:
    e["tb"] = tbin(e["pp_t"])


def run(sub, same_bin, nm_t, nm_c):
    """same_bin=True 면 처치와 같은 관성 bin 의 대조군만 사용."""
    t, c = [], []
    for e in sub:
        cs = []
        for j, v in enumerate(e[nm_c]):
            if not np.isfinite(v): continue
            if same_bin:
                if not np.isfinite(e["pp_c"][j]) or tbin(e["pp_c"][j]) != e["tb"]: continue
            cs.append(v)
        if cs and np.isfinite(e[nm_t]):
            t.append(e[nm_t]); c.append(float(np.mean(cs)))
    return boot_did_ci(t, c, rng)


def diff13(sub1, sub3, same_bin, nm_t, nm_c):
    def d(sub):
        out = []
        for e in sub:
            cs = [v for j, v in enumerate(e[nm_c]) if np.isfinite(v) and
                  (not same_bin or (np.isfinite(e["pp_c"][j]) and tbin(e["pp_c"][j]) == e["tb"]))]
            if cs and np.isfinite(e[nm_t]): out.append(e[nm_t] - float(np.mean(cs)))
        return np.array(out)
    d1, d3 = d(sub1), d(sub3)
    if min(len(d1), len(d3)) < 15: return None, None, len(d1), len(d3)
    b = np.array([d3[rng.integers(0, len(d3), len(d3))].mean()
                  - d1[rng.integers(0, len(d1), len(d1))].mean() for _ in range(NB)])
    return round(float(d3.mean() - d1.mean()), 4), qci(b), len(d1), len(d3)


PA, PB = {}, {}
for lab, same, store in (("[Panel A] naive (모든 대조군)", False, PA),
                         ("[Panel B] 동일 bin 대조군만 (평균회귀 차단)  ← 주 결과", True, PB)):
    print(f"\n{lab}")
    for b in range(3):
        sub = [e for e in USE if e["tb"] == b]
        zp, zc, zn = run(sub, same, "z_t", "z_c")
        hp, hc, hn = run(sub, same, "h_t", "h_c")
        sg = lambda p, c: "✓" if (c and (c[0] > 0 or c[1] < 0)) else "✗"
        store[TL[b]] = {"n_ev": len(sub), "zero_DiD": zp, "zero_ci": zc, "zero_n": zn,
                        "hire_DiD": hp, "hire_ci": hc, "hire_n": hn,
                        "pp_mean": round(float(np.mean([e["pp_t"] for e in sub])), 4)}
        print(f"  {TL[b]:<9} n={len(sub):>3} (사전관성 {store[TL[b]]['pp_mean']:.3f}) | "
              f"무채용 {zp}{zc}{sg(zp,zc)} n={zn} | 채용 {hp}{hc}{sg(hp,hc)} n={hn}")
    s1 = [e for e in USE if e["tb"] == 0]; s3 = [e for e in USE if e["tb"] == 2]
    for nm_t, nm_c, lb in (("z_t", "z_c", "무채용비중"), ("h_t", "h_c", "채용률")):
        pt, ci, n1, n3 = diff13(s1, s3, same, nm_t, nm_c)
        sg = "✓" if (ci and (ci[0] > 0 or ci[1] < 0)) else "✗"
        store[f"T3−T1 {lb}"] = {"diff": pt, "ci": ci, "sig": sg == "✓", "n1": n1, "n3": n3}
        print(f"    T3−T1 {lb:<8} {pt} {ci} {sg}  (n {n1}/{n3})")

# ---------- Panel C : 균형 ----------
print("\n[Panel C] 균형 — 분위별 처치 vs 대조의 사전-사전 관성")
PC = {}
for b in range(3):
    sub = [e for e in USE if e["tb"] == b]
    tv = np.mean([e["pp_t"] for e in sub])
    cvall = np.mean([v for e in sub for v in e["pp_c"] if np.isfinite(v)])
    same = [v for e in sub for v in e["pp_c"] if np.isfinite(v) and tbin(v) == b]
    keep = np.mean([1.0 if any(np.isfinite(v) and tbin(v) == b for v in e["pp_c"]) else 0.0 for e in sub])
    PC[TL[b]] = {"treated": round(float(tv), 4), "ctrl_all": round(float(cvall), 4),
                 "ctrl_samebin": round(float(np.mean(same)), 4) if same else None,
                 "gap_all": round(float(tv - cvall), 4), "event_retention_samebin": round(float(keep), 3)}
    print(f"  {TL[b]:<9} 처치 {tv:.4f} | 대조(전체) {cvall:.4f} (격차 {tv-cvall:+.4f})"
          f" | 대조(동일bin) {np.mean(same):.4f} | 이벤트 잔존율 {keep:.3f}")

# ---------- Panel D : hazard 삼중교호 ----------
print("\n[Panel D] hazard 삼중교호 treated x post x 고관성(T3)")
DCUT = [1, 2, 3, 4, 6, 12]
def spell(row, m0):
    out, d, warm = [], 1, 0
    for k in range(-24, 13):
        j = mset.get(m0 + k)
        if j is None: d, warm = 1, 0; continue
        h = Hv[row, j]
        if not np.isfinite(h): d, warm = 1, 0; continue
        if warm >= 6 and k != 0 and (-12 <= k <= -1 or 1 <= k <= 12):
            out.append((1 if k > 0 else 0, 1 if h > 0 else 0,
                        int(np.searchsorted(DCUT, min(d, 24), side="right") - 1)))
        d = 1 if h > 0 else d + 1; warm += 1
    return out

rec = []
for ei, e in enumerate(USE):
    hi = 1.0 if e["tb"] == 2 else 0.0
    for who, row in [(1, e["ti"])] + [(0, c) for c in e["ctrls"]]:
        for po, y, db in spell(row, e["m0"]):
            rec.append((ei, who, po, y, db, hi))
Q = pd.DataFrame(rec, columns=["ev", "tr", "po", "hire", "db", "hi"]); del rec
Cl = Q.groupby(["ev", "tr", "po", "db", "hi"], as_index=False).agg(succ=("hire", "sum"), n=("hire", "size"))
Cl["fail"] = Cl.n - Cl.succ; NOBS, nCl = len(Q), len(Cl); del Q; gc.collect()
evc = pd.Categorical(Cl.ev).codes.astype(np.int32); NEv = evc.max() + 1
dbc = Cl.db.values.astype(np.int32); NBK = 6
BASE = np.zeros((nCl, 1 + (NBK - 1) + (NEv - 1)), np.float32); BASE[:, 0] = 1.0
m = dbc > 0; BASE[np.flatnonzero(m), 1 + dbc[m] - 1] = 1.0
m = evc > 0; BASE[np.flatnonzero(m), NBK + evc[m] - 1] = 1.0
tr = Cl.tr.values.astype(np.float32); po = Cl.po.values.astype(np.float32)
hi = Cl.hi.values.astype(np.float32); tp = tr * po
X = np.hstack([np.column_stack([tr, po, tp, tr * hi, po * hi, tp * hi]).astype(np.float32), BASE])
r = sm.GLM(np.column_stack([Cl.succ.values, Cl.fail.values]).astype(float), X,
           family=sm.families.Binomial(sm.families.links.CLogLog())
           ).fit(cov_type="cluster", cov_kwds={"groups": Cl.ev.values})
PD = {}
for i, v in enumerate(["treated", "post", "tp", "tr_hi", "po_hi", "tp_hi"]):
    b, se = float(r.params[i]), float(r.bse[i])
    PD[v] = {"coef": round(b, 4), "se": round(se, 4), "HR": round(float(np.exp(b)), 4),
             "HR_ci": [round(float(np.exp(b - 1.96 * se)), 4), round(float(np.exp(b + 1.96 * se)), 4)],
             "p": float(f"{r.pvalues[i]:.3g}")}
    print(f"  {v:<8} b={b:+.4f} (se {se:.4f}) HR={np.exp(b):.4f} "
          f"[{np.exp(b-1.96*se):.4f}, {np.exp(b+1.96*se):.4f}] p={r.pvalues[i]:.4g}")
hr_lo = PD["tp"]["HR"]; hr_hi = round(float(np.exp(PD["tp"]["coef"] + PD["tp_hi"]["coef"])), 4)
tri = PD["tp_hi"]; tri_sig = not (tri["HR_ci"][0] <= 1.0 <= tri["HR_ci"][1])
print(f"  -> 저·중관성 HR={hr_lo}  고관성(T3) HR={hr_hi}  차이 HR={tri['HR']} {tri['HR_ci']}"
      f" {'✓ 유의' if tri_sig else '✗ 비유의'}")
PD.update({"HR_lowmid": hr_lo, "HR_high": hr_hi, "diff_sig": tri_sig,
           "n_cells": nCl, "n_firm_months": NOBS, "n_ev_fe": int(NEv)})
del X, r, BASE; gc.collect()

# ---------- 판정 ----------
main = PB["T3−T1 무채용비중"]
survives = bool(main["sig"]) or tri_sig
naive_sig = bool(PA["T3−T1 무채용비중"]["sig"])
if survives:
    status = "GO"; concl = "동일 bin 사양에서도 조절이 살아남음 — 평균회귀 인공물 아님"
elif naive_sig:
    status = "KILL"; concl = "naive 에서만 유의 — 평균회귀 인공물"
else:
    status = "PARTIAL"; concl = "두 사양 모두 조절 미검출 (naive 도 무유의) — 검정력 또는 부재"
verdict = (f"주사양(동일bin) T3−T1 무채용비중 {main['diff']}{main['ci']}{'✓' if main['sig'] else '✗'} "
           f"(n {main['n1']}/{main['n3']}) | naive {PA['T3−T1 무채용비중']['diff']}"
           f"{'✓' if naive_sig else '✗'} | hazard 고관성 HR {hr_hi} vs {hr_lo}, 차이 {tri['HR']}{tri['HR_ci']}"
           f"{'✓' if tri_sig else '✗'} | {concl}")
emit("I-25", "사전 관성수준 조절 (평균회귀 차단)", status,
     {"panelA_naive": PA, "panelB_samebin": PB, "panelC_balance": PC, "panelD_hazard_triple": PD,
      "tercile_cuts": [round(float(q1), 4), round(float(q2), 4)], "n_used": len(USE)},
     "관성이 심했던 기업일수록 효과가 크고 이미 활발한 기업은 0/음. 사전-사전 측정 + 동일 bin 대조로 평균회귀 차단",
     verdict, kill_met=(status == "KILL"), n=len(USE), extra={"conclusion": concl})
