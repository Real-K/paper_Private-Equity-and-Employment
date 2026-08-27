# -*- coding: utf-8 -*-
"""I-04b 실적 앵커 v2 — v1 의 측정 결함 3건 수정.

[v1 실측 결함]
 (1) **분모 오염**: 주 지표 `매출/인원` 의 분모가 처치 대상(고용)이다. 채용이 늘면 기계적으로 하락한다.
     실측: 전체 −0.1207✓ · T3 −0.3232✓ 인데 분자인 `log 매출` 자체는 T3 −0.1426✗ 로 무유의.
     → 분자·분모를 **분해해서 따로** 내고, **고용이 안 들어간 지표**를 주 지표로 삼는다.
 (2) **커버리지**: 재무는 외감 기업만이라 T3 h+1 이 n=32 까지 떨어진다(379 중).
     → NPS 기반 전수 지표(인당임금·총임금·생존)를 병행한다.
 (3) **지평**: PE 가치창출은 3~5년인데 h+1,+2 만 봤다. → h+3 추가.

주 지표 재정의 (고용 무관):
  ROA = 영업이익/자산총계 · 자산회전율 = 매출/자산총계 · log 매출 · log 자산
NPS 전수 지표: log 인당임금 · log 총임금(고지금액 합) · 3년 생존

Panel A  분해 — Δlog매출 · Δlog고용 · Δlog(매출/인원) 을 나란히 (v1 결과의 해부)
Panel B  고용 무관 성과 DiD (ROA · 회전율 · log매출 · log자산), h+1/+2/+3
Panel C  NPS 전수 지표 DiD (인당임금 · 총임금 · 생존), h+1/+2/+3
Panel D  ★ I-25 조절자(사전 관성)로 가르기 — 주 지표 한정
"""
import gc
import numpy as np, pandas as pd
from h30_common import load, deals, build, boot_did_ci, emit, SEED, qci, NB, widx, BASE

rng = np.random.default_rng(SEED)
print("[I-04b] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
EV, _ = build(G, allt, PE)
Ev, Av, Hv, idx, mset = G["Ev"], G["Av"], G["Hv"], G["idx"], G["mset"]
BNV = np.asarray(idx); INPANEL = set(BNV)

NEED = ["사업자등록번호", "회계연도", "분기", "자산총계(천원)", "매출액(천원)", "영업이익(천원)"]
parts = []
for ch in pd.read_csv(f"{BASE}/PI/drops/재무데이터_2009_2025_통합.csv",
                      usecols=NEED, dtype=str, chunksize=200_000):
    ch = ch[ch["분기"].astype(str).str.contains("결산", na=False)]
    ch["bn10"] = ch["사업자등록번호"].str.replace(r"\D", "", regex=True).str.zfill(10)
    parts.append(ch[ch.bn10.isin(INPANEL)])          # 패널 기업만 (메모리)
F = pd.concat(parts, ignore_index=True); del parts; gc.collect()
F["yr"] = pd.to_numeric(F["회계연도"], errors="coerce")
F["asset"] = pd.to_numeric(F["자산총계(천원)"], errors="coerce")
F["rev"] = pd.to_numeric(F["매출액(천원)"], errors="coerce")
F["op"] = pd.to_numeric(F["영업이익(천원)"], errors="coerce")
F = F[F.yr.notna()].drop_duplicates(["bn10", "yr"])
FIN = {(r.bn10, int(r.yr)): (r.asset, r.rev, r.op) for r in F.itertuples()}
print(f"  재무(패널기업 한정) {len(F):,}행 · 기업 {F.bn10.nunique():,}")
del F; gc.collect()

def emp(row, yr):
    j = mset.get(yr * 12 + 12)
    if j is None: return np.nan
    v = Ev[row, j]
    return float(v) if (np.isfinite(v) and v >= 5) else np.nan

def payroll(row, yr):
    js = [mset.get(yr * 12 + m) for m in range(1, 13)]
    if any(j is None for j in js): return np.nan, np.nan
    a, e = Av[row, js], Ev[row, js]
    if not (np.isfinite(a).all() and np.isfinite(e).all()) or np.nanmean(e) < 5: return np.nan, np.nan
    tot = float(np.nansum(a))
    return (tot, tot / float(np.nanmean(e))) if tot > 0 else (np.nan, np.nan)

def alive(row, yr):
    j = mset.get(yr * 12 + 12)
    return np.nan if j is None else float(np.isfinite(Ev[row, j]) and Ev[row, j] >= 1)

L = lambda x: np.log(x) if (np.isfinite(x) and x > 0) else np.nan

def snap(row, yr):
    o = {}
    e = emp(row, yr); o["log_emp"] = L(e)
    f = FIN.get((BNV[row], yr))
    if f:
        a, r, p = f
        o["log_asset"] = L(a); o["log_rev"] = L(r)
        if np.isfinite(a) and a > 0:
            if np.isfinite(p): o["roa"] = float(np.clip(p / a, -1, 1))
            if np.isfinite(r) and r > 0: o["turn"] = float(np.clip(r / a, 0, 20))
        if np.isfinite(r) and r > 0 and np.isfinite(e) and e > 0: o["log_rev_pe"] = np.log(r / e)
    tp, wp = payroll(row, yr)
    o["log_payroll"] = L(tp); o["log_wage"] = L(wp)
    o["alive"] = alive(row, yr)
    return o

GRP = {"분해": ["log_rev", "log_emp", "log_rev_pe"],
       "고용무관": ["roa", "turn", "log_rev", "log_asset"],
       "NPS전수": ["log_wage", "log_payroll", "alive"]}
LAB = {"log_rev": "log 매출", "log_emp": "log 고용", "log_rev_pe": "log 매출/인원",
       "roa": "ROA(영업이익/자산)", "turn": "자산회전율", "log_asset": "log 자산",
       "log_wage": "log 인당임금", "log_payroll": "log 총임금", "alive": "생존"}
HS = (1, 2, 3)

for e in EV:
    y0 = (e["m0"] - 1) // 12; e["y0"] = y0
    pre = snap(e["ti"], y0 - 1)
    e["d"] = {}
    for h in HS:
        po = snap(e["ti"], y0 + h)
        e["d"][h] = {k: po[k] - pre[k] for k in po
                     if k in pre and np.isfinite(pre[k]) and np.isfinite(po[k])}
    e["dc"] = {}
    for h in HS:
        acc = {}
        for k2 in e["ctrls"]:
            pr, po = snap(k2, y0 - 1), snap(k2, y0 + h)
            for kk in po:
                if kk in pr and np.isfinite(pr[kk]) and np.isfinite(po[kk]):
                    acc.setdefault(kk, []).append(po[kk] - pr[kk])
        e["dc"][h] = {kk: float(np.mean(v)) for kk, v in acc.items()}

def zsh(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return np.nan
    x = Hv[row, c]
    return float((x == 0).mean()) if np.isfinite(x).all() else np.nan
_pp = np.array([zsh(e["ti"], e["m0"], -24, -13) for e in EV], float)
_z = np.array([(zsh(e["ti"], e["m0"], 1, 12) - zsh(e["ti"], e["m0"], -12, -1)) for e in EV], float)
_u = np.isfinite(_pp) & np.isfinite(_z)
Q1, Q2 = np.percentile(_pp[_u], [33.33, 66.67])
for e in EV:
    v = zsh(e["ti"], e["m0"], -24, -13)
    e["pb"] = None if not np.isfinite(v) else (0 if v <= Q1 else (1 if v <= Q2 else 2))

def did(sub, k, h):
    t = [x["d"][h][k] for x in sub if k in x["d"][h] and k in x["dc"][h]]
    c = [x["dc"][h][k] for x in sub if k in x["d"][h] and k in x["dc"][h]]
    return boot_did_ci(t, c, rng)

def show(sub, keys, tag):
    o = {}
    for k in keys:
        for h in HS:
            p_, ci, n = did(sub, k, h)
            sg = "✓" if (ci and (ci[0] > 0 or ci[1] < 0)) else ("✗" if ci else "-")
            o[f"{k}|h{h}"] = {"DiD": p_, "ci": ci, "n": n, "sig": sg == "✓"}
            print(f"    {tag:<9} {LAB[k]:<15} h+{h} {str(p_):>9} {str(ci):<21} {sg} (n={n})")
    return o

print("\n[Panel A] 분해 — v1 의 '매출/인원 하락'이 분자인가 분모인가")
PA = show(EV, GRP["분해"], "전체")
print("\n[Panel B] 고용 무관 성과 DiD")
PBv = show(EV, GRP["고용무관"], "전체")
print("\n[Panel C] NPS 전수 지표 DiD")
PCv = show(EV, GRP["NPS전수"], "전체")

print("\n[Panel D] ★ 사전 관성 분위별 (주 지표 한정)")
KEY = ["roa", "turn", "log_rev", "log_wage", "alive"]
PD = {}
for b, bl in ((0, "T1저관성"), (2, "T3고관성")):
    sub = [e for e in EV if e["pb"] == b]
    print(f"  -- {bl} n_ev={len(sub)} --")
    PD[bl] = show(sub, KEY, bl)
print("  -- T3 − T1 --")
PD["diff"] = {}
for k in KEY:
    for h in HS:
        d1 = np.array([e["d"][h][k] - e["dc"][h][k] for e in EV
                       if e["pb"] == 0 and k in e["d"][h] and k in e["dc"][h]])
        d3 = np.array([e["d"][h][k] - e["dc"][h][k] for e in EV
                       if e["pb"] == 2 and k in e["d"][h] and k in e["dc"][h]])
        if min(len(d1), len(d3)) < 15: continue
        bs = np.array([d3[rng.integers(0, len(d3), len(d3))].mean()
                       - d1[rng.integers(0, len(d1), len(d1))].mean() for _ in range(NB)])
        ci = qci(bs); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
        PD["diff"][f"{k}|h{h}"] = {"diff": round(float(d3.mean() - d1.mean()), 4), "ci": ci,
                                   "sig": sg == "✓", "n1": len(d1), "n3": len(d3)}
        print(f"    {LAB[k]:<15} h+{h} {d3.mean()-d1.mean():+.4f} {ci} {sg} (n {len(d1)}/{len(d3)})")

# ---- 판정 ----
mech = (PA.get("log_emp|h1", {}).get("sig") and not PA.get("log_rev|h1", {}).get("sig")
        and PA.get("log_rev_pe|h1", {}).get("sig"))
pos = [k for k, v in {**PBv, **PCv}.items() if v["sig"] and (v["DiD"] or 0) > 0]
neg = [k for k, v in {**PBv, **PCv}.items() if v["sig"] and (v["DiD"] or 0) < 0]
if pos and not neg: status, concl = "GO", "고용 무관 지표에서 성과 개선 — 관성 제거의 가치 지지"
elif neg and not pos: status, concl = "KILL", "고용 무관 지표에서도 성과 악화 — '가치 없음'으로 서술해야 함"
elif pos or neg: status, concl = "PARTIAL", f"혼재: 개선 {pos} / 악화 {neg}"
else: status, concl = "PARTIAL", "고용 무관 지표 전부 무유의 — 성과 효과 미검출(등가성 아님)"
verdict = (f"[분해] log매출 h+1 {PA['log_rev|h1']['DiD']}{'✓' if PA['log_rev|h1']['sig'] else '✗'} · "
           f"log고용 {PA['log_emp|h1']['DiD']}{'✓' if PA['log_emp|h1']['sig'] else '✗'} · "
           f"매출/인원 {PA['log_rev_pe|h1']['DiD']}{'✓' if PA['log_rev_pe|h1']['sig'] else '✗'}"
           f" → v1 하락은 {'분모(고용) 기계효과' if mech else '분자·분모 혼합'} | "
           f"[고용무관] 개선 {pos or '없음'} / 악화 {neg or '없음'} | {concl}")
emit("I-04b", "실적 앵커 v2 (분모오염·커버리지·지평 수정)", status,
     {"panelA_decomposition": PA, "panelB_employment_free": PBv, "panelC_nps_full": PCv,
      "panelD_by_inertia": PD, "labels": LAB, "tercile_cuts": [round(float(Q1),4), round(float(Q2),4)]},
     "고용이 분모에 들어가지 않는 지표로 관성 제거의 가치를 검정하고, 사전 관성으로 가른다",
     verdict, kill_met=(status == "KILL"), n=len(EV),
     extra={"conclusion": concl, "v1_defects_fixed":
            ["분모 오염(매출/인원의 분모가 처치대상)", "재무 커버리지 부족(T3 h+1 n=32) → NPS 전수지표 병행",
             "지평 h+1,+2만 → h+3 추가"],
            "v1_result_kept": "v1 의 매출/인원 하락(전체 -0.1207✓, T3 -0.3232✓)은 폐기가 아니라 "
                              "Panel A 에서 분해해 해석한다."})
