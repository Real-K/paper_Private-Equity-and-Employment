# -*- coding: utf-8 -*-
"""I-04c 실적 앵커 v3 — 부가가치 + 등가성 경계.

[v2 가 확정한 것] '매출/인원 하락'은 분모(고용 +15.2%✓) 기계효과다. 매출 자체는 −0.019✗.
[v2 가 못 한 것] (a) 올바른 생산성 대상인 **부가가치**를 안 썼다 (b) null 에 등가성 경계가 없어
"효과 없음"을 주장할 수 없다 (c) 지평 h+3 까지.

부가가치 = 영업이익 + 총인건비. 재무에 인건비가 없으나 **NPS 고지금액 합(총임금)** 이 그 역할을 한다.
  VA = 영업이익 + NPS총임금          ← 산출측, 고용이 분모에 없음
  VA/인원                            ← 노동생산성 정본
  VA/자산                            ← 자본생산성

Panel A  부가가치 3종 DiD, h+1..h+4
Panel B  등가성 경계 — 핵심 null(매출·ROA·생존·VA)이 어떤 크기까지 배제되는가
Panel C  사전 관성 분위별 VA
Panel D  🚩 T3 생존 음의 추세 정밀 점검 (v2 에서 -0.010/-0.030/-0.030, 전부 무유의)
"""
import gc
import numpy as np, pandas as pd
from h30_common import load, deals, build, boot_did_ci, emit, SEED, qci, NB, widx, BASE

rng = np.random.default_rng(SEED)
print("[I-04c] 로딩...")
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
    parts.append(ch[ch.bn10.isin(INPANEL)])
F = pd.concat(parts, ignore_index=True); del parts; gc.collect()
F["yr"] = pd.to_numeric(F["회계연도"], errors="coerce")
for c, k in (("자산총계(천원)", "asset"), ("매출액(천원)", "rev"), ("영업이익(천원)", "op")):
    F[k] = pd.to_numeric(F[c], errors="coerce")
F = F[F.yr.notna()].drop_duplicates(["bn10", "yr"])
FIN = {(r.bn10, int(r.yr)): (r.asset, r.rev, r.op) for r in F.itertuples()}
print(f"  재무 {len(F):,}행 / 기업 {F.bn10.nunique():,}"); del F; gc.collect()

L = lambda x: np.log(x) if (np.isfinite(x) and x > 0) else np.nan
HS = (1, 2, 3, 4)

def snap(row, yr):
    o = {}
    j = mset.get(yr * 12 + 12)
    e = float(Ev[row, j]) if (j is not None and np.isfinite(Ev[row, j]) and Ev[row, j] >= 5) else np.nan
    o["alive"] = np.nan if j is None else float(np.isfinite(Ev[row, j]) and Ev[row, j] >= 1)
    js = [mset.get(yr * 12 + m) for m in range(1, 13)]
    pay = np.nan
    if all(x is not None for x in js):
        a, ee = Av[row, js], Ev[row, js]
        if np.isfinite(a).all() and np.isfinite(ee).all() and np.nanmean(ee) >= 5:
            pay = float(np.nansum(a))
    f = FIN.get((BNV[row], yr))
    if f:
        A_, R_, P_ = f
        o["log_rev"] = L(R_); o["log_asset"] = L(A_)
        if np.isfinite(A_) and A_ > 0 and np.isfinite(P_): o["roa"] = float(np.clip(P_ / A_, -1, 1))
        if np.isfinite(P_) and np.isfinite(pay):
            va = P_ + pay                                   # 부가가치 = 영업이익 + 총인건비
            o["log_va"] = L(va)
            if np.isfinite(e) and e > 0 and va > 0: o["va_pe"] = np.log(va / e)
            if np.isfinite(A_) and A_ > 0 and va > 0: o["va_pa"] = np.log(va / A_)
    return o

LAB = {"log_va": "log 부가가치", "va_pe": "log VA/인원", "va_pa": "log VA/자산",
       "log_rev": "log 매출", "roa": "ROA", "log_asset": "log 자산", "alive": "생존"}
for e in EV:
    y0 = (e["m0"] - 1) // 12; e["y0"] = y0
    pre = snap(e["ti"], y0 - 1); e["d"] = {}; e["dc"] = {}
    for h in HS:
        po = snap(e["ti"], y0 + h)
        e["d"][h] = {k: po[k] - pre[k] for k in po if k in pre
                     and np.isfinite(pre[k]) and np.isfinite(po[k])}
        acc = {}
        for c in e["ctrls"]:
            pr2, po2 = snap(c, y0 - 1), snap(c, y0 + h)
            for kk in po2:
                if kk in pr2 and np.isfinite(pr2[kk]) and np.isfinite(po2[kk]):
                    acc.setdefault(kk, []).append(po2[kk] - pr2[kk])
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

def show(sub, keys, tag, hs=HS):
    o = {}
    for k in keys:
        for h in hs:
            p_, ci, n = did(sub, k, h)
            sg = "✓" if (ci and (ci[0] > 0 or ci[1] < 0)) else ("✗" if ci else "-")
            o[f"{k}|h{h}"] = {"DiD": p_, "ci": ci, "n": n, "sig": sg == "✓"}
            print(f"    {tag:<9} {LAB[k]:<13} h+{h} {str(p_):>9} {str(ci):<21} {sg} (n={n})")
    return o

print("\n[Panel A] 부가가치 (= 영업이익 + NPS 총임금)")
PA = show(EV, ["log_va", "va_pe", "va_pa"], "전체")
print("\n  참고: 매출·ROA·자산·생존")
PA.update(show(EV, ["log_rev", "roa", "log_asset", "alive"], "전체"))

print("\n[Panel B] 등가성 경계 — null 을 '효과 없음'으로 쓸 수 있는가 (규칙 11)")
BENCH = {"log_va": 0.15, "va_pe": 0.10, "log_rev": 0.15, "roa": 0.03, "alive": 0.05}
PB = {}
for k, S in BENCH.items():
    for h in HS:
        v = PA.get(f"{k}|h{h}")
        if not v or not v["ci"] or v["sig"]: continue
        lo, hi = v["ci"]; m = [round(lo + S, 4), round(S - hi, 4)]
        ok = bool(lo > -S and hi < S); kn = bool(min(m) < 0.001)
        PB[f"{k}|h{h}"] = {"SESOI": S, "ci": v["ci"], "holds": ok, "margin": m, "knife": kn, "n": v["n"]}
        print(f"    {LAB[k]:<13} h+{h} δ={S}  CI{v['ci']}  "
              f"{'✓ 등가성 성립' if (ok and not kn) else '✗ 미성립'} 여유 {m}")

print("\n[Panel C] 사전 관성 분위별 부가가치")
PC = {}
for b, bl in ((0, "T1저관성"), (2, "T3고관성")):
    sub = [e for e in EV if e["pb"] == b]
    print(f"  -- {bl} n_ev={len(sub)} --")
    PC[bl] = show(sub, ["log_va", "va_pe"], bl, hs=(1, 2, 3))

print("\n[Panel D] 🚩 생존 — T3 음의 추세 정밀 점검")
PD = {}
for b, bl in ((0, "T1"), (2, "T3")):
    sub = [e for e in EV if e["pb"] == b]
    PD[bl] = show(sub, ["alive"], bl)
for h in HS:
    d1 = np.array([e["d"][h]["alive"] - e["dc"][h]["alive"] for e in EV
                   if e["pb"] == 0 and "alive" in e["d"][h] and "alive" in e["dc"][h]])
    d3 = np.array([e["d"][h]["alive"] - e["dc"][h]["alive"] for e in EV
                   if e["pb"] == 2 and "alive" in e["d"][h] and "alive" in e["dc"][h]])
    if min(len(d1), len(d3)) < 15: continue
    bs = np.array([d3[rng.integers(0, len(d3), len(d3))].mean()
                   - d1[rng.integers(0, len(d1), len(d1))].mean() for _ in range(NB)])
    ci = qci(bs); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
    PD[f"T3-T1|h{h}"] = {"diff": round(float(d3.mean() - d1.mean()), 4), "ci": ci, "sig": sg == "✓"}
    print(f"    생존 T3−T1 h+{h} {d3.mean()-d1.mean():+.4f} {ci} {sg} (n {len(d1)}/{len(d3)})")

va_sig = [k for k in PA if k.startswith(("log_va", "va_pe", "va_pa")) and PA[k]["sig"]]
va_pos = [k for k in va_sig if (PA[k]["DiD"] or 0) > 0]
va_neg = [k for k in va_sig if (PA[k]["DiD"] or 0) < 0]
if va_pos and not va_neg: status, concl = "GO", "부가가치 증가 — 관성 제거의 가치 지지"
elif va_neg and not va_pos: status, concl = "KILL", "부가가치 감소 — 가치 파괴로 서술"
elif va_sig: status, concl = "PARTIAL", f"부가가치 혼재: +{va_pos} / −{va_neg}"
else: status, concl = "PARTIAL", "부가가치 효과 미검출"
verdict = (f"log VA: " + " ".join(f"h+{h} {PA[f'log_va|h{h}']['DiD']}"
           f"{'✓' if PA[f'log_va|h{h}']['sig'] else '✗'}" for h in HS) +
           f" | VA/인원: " + " ".join(f"h+{h} {PA[f'va_pe|h{h}']['DiD']}"
           f"{'✓' if PA[f'va_pe|h{h}']['sig'] else '✗'}" for h in HS) +
           f" | 등가성 성립 {[k for k,v in PB.items() if v['holds'] and not v['knife']] or '없음'} | {concl}")
emit("I-04c", "실적 앵커 v3 (부가가치 + 등가성)", status,
     {"panelA_value_added": PA, "panelB_equivalence": PB, "panelC_by_inertia": PC,
      "panelD_survival": PD, "labels": LAB, "va_def": "영업이익 + NPS 고지금액 연합(총인건비)"},
     "부가가치(영업이익+인건비)로 관성 제거의 가치를 검정하고, null 에는 등가성 경계를 붙인다",
     verdict, kill_met=(status == "KILL"), n=len(EV), extra={"conclusion": concl})
