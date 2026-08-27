# -*- coding: utf-8 -*-
"""I-16 딜유형 대비 — 통제권 이전 vs 자본 투입.

I-03이 메커니즘을 '조정비용 하락'이 아니라 '비주의·감시실패 제거'로 판별했다.
감시 채널이라면 효과는 **통제권이 실제로 이전된 거래에서만** 나와야 한다.
처치표본은 Buyout/LBO 와 PE Growth/Expansion 두 유형으로만 구성되며 이것이 정확히 그 대비다.
  Buyout/LBO          = 통제권 이전 (다수지분·경영권)
  PE Growth/Expansion = 소수지분 성장자본 (자본은 들어가되 통제권은 대개 미이전)
자본과 통제권이 같이 움직이는 문제를 pct_acq(취득지분율)로 한 번 더 분리한다.

Panel A  유형 2분할 x 결과 배터리 (채용DiD·P1·rel·무채용비중·최대도달갭) + 차이 검정
Panel B  통제권 dose — pct_acq 다수(>=50%) vs 소수, 3분위
Panel C  하위유형 (Deal Type 2) 서술
Panel D  hazard 삼중교호 treated x post x buyout (이벤트FE cloglog, 셀접기)

기각조건: 두 유형이 동일하면 통제권 서사 KILL (자본유입 또는 선택으로 재해석).
[메모리] I-02 규율 준수 — grouped binomial 셀접기, float32, 고정블록 재사용.
"""
import gc
import numpy as np, pandas as pd
import statsmodels.api as sm
from h30_common import (load, deals, build, attach, summ, boot_did_ci,
                        emit, SEED, qci, NB, widx, dflow)

rng = np.random.default_rng(SEED)
print("[I-16] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
EV, _ = build(G, allt, PE)
Hv, Sv, Ev, mset = G["Hv"], G["Sv"], G["Ev"], G["mset"]

# ---- 딜유형 부착 ----
DT = META["Deal Type"].astype(str)
PCT = pd.to_numeric(META["pct_acq"], errors="coerce")
for e in EV:
    t = DT.get(e["bn"], "NA")
    e["buy"] = 1 if "Buyout" in t else (0 if "Growth" in t else -1)
    e["dt"] = t
    e["pct"] = float(PCT.get(e["bn"], np.nan))
nb_ = sum(1 for e in EV if e["buy"] == 1); ng_ = sum(1 for e in EV if e["buy"] == 0)
nna = sum(1 for e in EV if e["buy"] == -1)
print(f"  이벤트 {len(EV)}  Buyout {nb_}  Growth {ng_}  미분류 {nna}"
      f"  | pct_acq 보유 {sum(1 for e in EV if np.isfinite(e['pct']))}")

# ---- 결과 스칼라 ----
def zshare(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != 12: return np.nan
    h = Hv[row, c]
    return float((h == 0).mean()) if np.isfinite(h).all() else np.nan

def maxgap(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != 12: return np.nan
    h, s, e0 = Hv[row, c], Sv[row, c], Ev[row, c]
    if not (np.isfinite(h).all() and np.isfinite(s).all() and np.isfinite(e0).all()): return np.nan
    cum, Eref, mx = 0.0, e0[0], 0.0
    if not (Eref >= 5): return np.nan
    for i in range(12):
        mx = max(mx, cum / Eref)
        if h[i] > 0: cum, Eref = 0.0, max(e0[i], 5)
        else: cum += s[i]
    return float(mx)

EV = attach(G, EV)
for e in EV:
    for nm, fn in (("z", zshare), ("mg", maxgap)):
        a = fn(e["ti"], e["m0"], -12, -1); b = fn(e["ti"], e["m0"], 1, 12)
        e[f"{nm}_t"] = b - a if (np.isfinite(a) and np.isfinite(b)) else np.nan
        cd = [fn(k, e["m0"], 1, 12) - fn(k, e["m0"], -12, -1) for k in e["ctrls"]]
        cd = [x for x in cd if np.isfinite(x)]
        e[f"{nm}_c"] = float(np.mean(cd)) if cd else np.nan

def did_of(sub, nm):
    return boot_did_ci([e[f"{nm}_t"] for e in sub], [e[f"{nm}_c"] for e in sub], rng)

def diff_test(s1, s2, nm):
    """두 부분표본의 DiD 차이 (s1 - s2) 부트스트랩."""
    d1 = np.array([e[f"{nm}_t"] - e[f"{nm}_c"] for e in s1], float); d1 = d1[np.isfinite(d1)]
    d2 = np.array([e[f"{nm}_t"] - e[f"{nm}_c"] for e in s2], float); d2 = d2[np.isfinite(d2)]
    if len(d1) < 20 or len(d2) < 20: return None, None
    b = np.array([d1[rng.integers(0, len(d1), len(d1))].mean()
                  - d2[rng.integers(0, len(d2), len(d2))].mean() for _ in range(NB)])
    return round(float(d1.mean() - d2.mean()), 4), qci(b)

BUY = [e for e in EV if e["buy"] == 1]; GRW = [e for e in EV if e["buy"] == 0]

print("\n[Panel A] 유형 2분할 x 결과 배터리")
PA = {}
for lab, sub in (("Buyout", BUY), ("Growth", GRW), ("전체", EV)):
    s = summ(sub, rng)
    zp, zc, zn = did_of(sub, "z"); mp, mc, mn = did_of(sub, "mg")
    PA[lab] = {**s, "zero_share_DiD": zp, "zero_ci": zc, "zero_n": zn,
               "maxgap_DiD": mp, "maxgap_ci": mc, "maxgap_n": mn}
    sg = lambda p, c: "✓" if (c and (c[0] > 0 or c[1] < 0)) else "✗"
    print(f"  {lab:<7} n={s['n']:>3} | 채용DiD {s['DiD']:+.4f}{s['DiD_ci']}"
          f" | P1 {s['P1']:+.4f}{s['P1_ci']} | rel {s['rel']}{s['rel_ci']}")
    print(f"  {'':<7}        | 무채용비중 {zp:+.4f}{zc}{sg(zp,zc)} (n={zn})"
          f" | 최대갭 {mp:+.4f}{mc}{sg(mp,mc)} (n={mn})")

print("\n  -- Buyout − Growth 차이 검정 --")
PA["diff"] = {}
for nm, lb in (("t", "채용DiD"), ("z", "무채용비중"), ("mg", "최대갭")):
    if nm == "t":
        d1 = np.array([e["t"] - e["cs"].mean() for e in BUY if np.isfinite(e.get("t", np.nan)) and len(e.get("cs", []))], float)
        d2 = np.array([e["t"] - e["cs"].mean() for e in GRW if np.isfinite(e.get("t", np.nan)) and len(e.get("cs", []))], float)
        b = np.array([d1[rng.integers(0, len(d1), len(d1))].mean()
                      - d2[rng.integers(0, len(d2), len(d2))].mean() for _ in range(NB)])
        pt, ci = round(float(d1.mean() - d2.mean()), 4), qci(b)
    else:
        pt, ci = diff_test(BUY, GRW, nm)
    sg = "✓" if (ci and (ci[0] > 0 or ci[1] < 0)) else "✗"
    print(f"    {lb:<10} 차이 {pt:+.4f} {ci} {sg}")
    PA["diff"][lb] = {"diff": pt, "ci": ci, "sig": sg == "✓"}

# ---------- Panel B : 통제권 dose ----------
print("\n[Panel B] 통제권 dose — pct_acq")
HASP = [e for e in EV if np.isfinite(e["pct"])]
MAJ = [e for e in HASP if e["pct"] >= 50]; MIN_ = [e for e in HASP if e["pct"] < 50]
PB = {"n_with_pct": len(HASP), "n_major": len(MAJ), "n_minor": len(MIN_)}
for lab, sub in (("다수>=50%", MAJ), ("소수<50%", MIN_)):
    if len(sub) < 20:
        print(f"  {lab:<10} n={len(sub)} (<20)"); PB[lab] = {"n": len(sub), "note": "n<20"}; continue
    s = summ(sub, rng); zp, zc, zn = did_of(sub, "z"); mp, mc, mn = did_of(sub, "mg")
    PB[lab] = {**s, "zero_DiD": zp, "zero_ci": zc, "maxgap_DiD": mp, "maxgap_ci": mc}
    print(f"  {lab:<10} n={s['n']:>3} | 채용DiD {s['DiD']:+.4f}{s['DiD_ci']}"
          f" | 무채용 {zp:+.4f}{zc} | 최대갭 {mp:+.4f}{mc}")
if len(MAJ) >= 20 and len(MIN_) >= 20:
    for nm, lb in (("z", "무채용비중"), ("mg", "최대갭")):
        pt, ci = diff_test(MAJ, MIN_, nm)
        sg = "✓" if (ci and (ci[0] > 0 or ci[1] < 0)) else "✗"
        print(f"    다수−소수 {lb:<8} {pt:+.4f} {ci} {sg}")
        PB[f"diff_{lb}"] = {"diff": pt, "ci": ci, "sig": sg == "✓"}

# ---------- Panel C : 하위유형 ----------
print("\n[Panel C] 딜유형 원분포 (매칭 진입 379)")
PC = pd.Series([e["dt"] for e in EV]).value_counts().to_dict()
print("  ", PC)

# ---------- Panel D : hazard 삼중교호 ----------
print("\n[Panel D] hazard 삼중교호 treated x post x buyout (이벤트FE cloglog)")
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
for ei, e in enumerate(EV):
    if e["buy"] < 0: continue
    for who, row in [(1, e["ti"])] + [(0, c) for c in e["ctrls"]]:
        for po, y, db in spell(row, e["m0"]):
            rec.append((ei, who, po, y, db, e["buy"]))
Q = pd.DataFrame(rec, columns=["ev", "tr", "po", "hire", "db", "buy"]); del rec
Cl = Q.groupby(["ev", "tr", "po", "db", "buy"], as_index=False).agg(succ=("hire", "sum"), n=("hire", "size"))
Cl["fail"] = Cl.n - Cl.succ
NOBS, nC = len(Q), len(Cl); del Q; gc.collect()
evc = pd.Categorical(Cl.ev).codes.astype(np.int32); NEv = evc.max() + 1
dbc = Cl.db.values.astype(np.int32); NBK = 6
BASE = np.zeros((nC, 1 + (NBK - 1) + (NEv - 1)), np.float32); BASE[:, 0] = 1.0
m = dbc > 0; BASE[np.flatnonzero(m), 1 + dbc[m] - 1] = 1.0
m = evc > 0; BASE[np.flatnonzero(m), NBK + evc[m] - 1] = 1.0
ENDOG = np.column_stack([Cl.succ.values, Cl.fail.values]).astype(np.float64)
tr = Cl.tr.values.astype(np.float32); po = Cl.po.values.astype(np.float32)
bu = Cl.buy.values.astype(np.float32); tp = tr * po
print(f"  셀 {nC:,} (원 기업-월 {NOBS:,}) 이벤트FE {NEv}")
X = np.hstack([np.column_stack([tr, po, tp, tr * bu, po * bu, tp * bu]).astype(np.float32), BASE])
r = sm.GLM(ENDOG, X, family=sm.families.Binomial(sm.families.links.CLogLog())
           ).fit(cov_type="cluster", cov_kwds={"groups": Cl.ev.values})
PD = {}
for i, v in enumerate(["treated", "post", "tp", "tr_buy", "po_buy", "tp_buy"]):
    b, se = float(r.params[i]), float(r.bse[i])
    PD[v] = {"coef": round(b, 4), "se": round(se, 4), "HR": round(float(np.exp(b)), 4),
             "HR_ci": [round(float(np.exp(b - 1.96 * se)), 4), round(float(np.exp(b + 1.96 * se)), 4)],
             "p": float(f"{r.pvalues[i]:.3g}")}
    print(f"  {v:<9} b={b:+.4f} (se {se:.4f}) HR={np.exp(b):.4f} "
          f"[{np.exp(b-1.96*se):.4f}, {np.exp(b+1.96*se):.4f}] p={r.pvalues[i]:.4g}")
hr_g = PD["tp"]["HR"]
hr_b = round(float(np.exp(PD["tp"]["coef"] + PD["tp_buy"]["coef"])), 4)
tri = PD["tp_buy"]; tri_sig = not (tri["HR_ci"][0] <= 1.0 <= tri["HR_ci"][1])
print(f"  -> Growth HR={hr_g}   Buyout HR={hr_b}   차이 HR={tri['HR']} {tri['HR_ci']}"
      f" {'✓ 유의' if tri_sig else '✗ 비유의'}")
PD.update({"HR_growth": hr_g, "HR_buyout": hr_b, "diff_sig": tri_sig,
           "n_cells": nC, "n_firm_months": NOBS, "n_ev_fe": int(NEv)})
del X, r, BASE, ENDOG; gc.collect()

# ---------- 판정 ----------
any_diff = tri_sig or any(v["sig"] for v in PA["diff"].values() if v["diff"] is not None)
if any_diff and hr_b > hr_g:
    status, concl = "GO", "통제권 이전 거래에 집중 — 감시 채널 지지"
elif any_diff:
    status, concl = "PARTIAL", "유형간 차이는 있으나 방향이 통제권 가설과 불일치"
else:
    status, concl = "KILL", "두 유형이 구별되지 않음 — 통제권 서사 미지지"
verdict = (f"Buyout n={PA['Buyout']['n']} vs Growth n={PA['Growth']['n']} | "
           f"hazard Growth HR={hr_g} vs Buyout HR={hr_b}, 차이 HR {tri['HR']} {tri['HR_ci']} "
           f"{'✓' if tri_sig else '✗'} | 배터리 차이 유의: "
           f"{[k for k,v in PA['diff'].items() if v['sig']] or '없음'} | {concl}")
emit("I-16", "딜유형 대비 — 통제권 vs 자본", status,
     {"panelA_type_battery": PA, "panelB_pct_dose": PB,
      "panelC_type_counts": PC, "panelD_hazard_triple": PD},
     "감시 채널이면 효과는 통제권이 이전된 Buyout/LBO 에 집중되고 소수지분 Growth 에서는 약하거나 없다",
     verdict, kill_met=(status == "KILL"), n=len(EV),
     extra={"conclusion": concl, "n_buyout": nb_, "n_growth": ng_})
