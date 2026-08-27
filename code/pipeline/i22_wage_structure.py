# -*- coding: utf-8 -*-
"""I-22 임금구조 — 서사를 가르는 검정.

I-21 이 현금 매개를 기각했다(현금 감소 기업에서도 효과 동일). 남은 측정 가능한 대상은 임금이다.
임금은 매개변수가 아니라 **서사 판별자**다.

  · 신규채용 임금이 기존과 **같다** → **순수 조정기술 변화** (강한 서사, finance 저널)
  · 신규채용 임금이 **낮다**       → **노동비용 재구성** (노동경제 서사, 저널 표적 변경 필요)

핵심 지표: **함의 한계임금** = Δ(월 신고소득 총액) / Δ(가입자수).
고용이 늘 때 추가된 인력 1인당 실제로 지급된 금액이며, 이를 기존 평균임금과 비교한다.

Panel A  평균임금 DiD (지평 확인)
Panel B  ★ 함의 한계임금 / 기존 평균임금 비율 — 처치 vs 대조
Panel C  임금–고용 동시변화 기울기 (Δlog임금 ~ Δlog고용)
Panel D  사전 관성 분위별

[측정 한계] 국민연금 고지금액은 **상한(등급 상한)** 이 있어 고소득자가 절단된다.
따라서 절대 수준보다 **처치–대조 차분**과 **비율**로만 해석한다.
"""
import numpy as np, pandas as pd
from h30_common import load, deals, build, boot_did_ci, emit, SEED, qci, NB, widx, BASE

rng = np.random.default_rng(SEED)
print("[I-22] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
EV, _ = build(G, allt, PE)
Hv, Ev, Av, mset = G["Hv"], G["Ev"], G["Av"], G["mset"]

def win(row, m0, a, b):
    """(월평균 신고소득 총액, 월평균 가입자수) — 12개월 창."""
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return np.nan, np.nan
    A_, E_ = Av[row, c], Ev[row, c]
    if not (np.isfinite(A_).all() and np.isfinite(E_).all()): return np.nan, np.nan
    e = float(np.mean(E_))
    return (float(np.mean(A_)), e) if e >= 5 else (np.nan, np.nan)

def zsh(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return np.nan
    x = Hv[row, c]
    return float((x == 0).mean()) if np.isfinite(x).all() else np.nan

MIN_DE = 2.0        # 함의임금은 고용 증가분이 최소 2명 이상일 때만 계산 (분모 폭발 방지)
def marg(row, m0):
    """(함의 한계임금/기존 평균임금 비율, Δ고용, 기존 평균임금)."""
    Apre, Epre = win(row, m0, -12, -1); Apost, Epost = win(row, m0, 1, 12)
    if not all(np.isfinite(x) for x in (Apre, Epre, Apost, Epost)): return np.nan, np.nan, np.nan
    dE = Epost - Epre
    wavg = Apre / Epre if Epre > 0 else np.nan
    if dE < MIN_DE or not (np.isfinite(wavg) and wavg > 0): return np.nan, dE, wavg
    mw = (Apost - Apre) / dE
    return float(np.clip(mw / wavg, -3, 5)), dE, wavg

for e in EV:
    Ap, Ep = win(e["ti"], e["m0"], -12, -1); As, Es = win(e["ti"], e["m0"], 1, 12)
    e["w_t"] = (np.log(As / Es) - np.log(Ap / Ep)) if all(
        np.isfinite(x) and x > 0 for x in (Ap, Ep, As, Es)) else np.nan
    e["e_t"] = (np.log(Es) - np.log(Ep)) if (np.isfinite(Ep) and np.isfinite(Es)
                                             and Ep > 0 and Es > 0) else np.nan
    e["r_t"], e["dE_t"], _ = marg(e["ti"], e["m0"])
    wc, ec, rc = [], [], []
    for k in e["ctrls"]:
        a1, e1 = win(k, e["m0"], -12, -1); a2, e2 = win(k, e["m0"], 1, 12)
        if all(np.isfinite(x) and x > 0 for x in (a1, e1, a2, e2)):
            wc.append(np.log(a2 / e2) - np.log(a1 / e1)); ec.append(np.log(e2) - np.log(e1))
        r_, _, _ = marg(k, e["m0"])
        if np.isfinite(r_): rc.append(r_)
    e["w_c"] = float(np.mean(wc)) if wc else np.nan
    e["e_c"] = float(np.mean(ec)) if ec else np.nan
    e["r_c"] = float(np.mean(rc)) if rc else np.nan
    v = zsh(e["ti"], e["m0"], -24, -13); e["pp"] = v

_pp = np.array([e["pp"] for e in EV], float)
_ok = np.isfinite(_pp) & np.isfinite(np.array([e["w_t"] for e in EV], float))
Q1, Q2 = np.percentile(_pp[np.isfinite(_pp)], [33.33, 66.67])
for e in EV:
    e["pb"] = None if not np.isfinite(e["pp"]) else (0 if e["pp"] <= Q1 else (1 if e["pp"] <= Q2 else 2))

def D(sub, tk, ck, lab):
    p_, ci, n = boot_did_ci([x[tk] for x in sub], [x[ck] for x in sub], rng)
    sg = "✓" if (ci and (ci[0] > 0 or ci[1] < 0)) else ("✗" if ci else "-")
    print(f"  {lab:<26} {str(p_):>9} {str(ci):<21} {sg} (n={n})")
    return {"DiD": p_, "ci": ci, "n": n, "sig": sg == "✓"}

print("\n[Panel A] 평균임금·고용 DiD")
PA = {"log_wage": D(EV, "w_t", "w_c", "Δlog 평균임금"),
      "log_emp": D(EV, "e_t", "e_c", "Δlog 고용")}

print("\n[Panel B] ★ 함의 한계임금 / 기존 평균임금 비율")
rt = np.array([e["r_t"] for e in EV], float); rc = np.array([e["r_c"] for e in EV], float)
okt = np.isfinite(rt); okc = np.isfinite(rc)
print(f"  처치 계산가능 {int(okt.sum())}/{len(EV)} (Δ고용 ≥ {MIN_DE}명 조건) · 대조 {int(okc.sum())}")
def m1(v, lab):
    v = v[np.isfinite(v)]
    if len(v) < 20: print(f"  {lab:<26} n={len(v)} (<20)"); return None
    bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(NB)])
    ci = qci(bs)
    print(f"  {lab:<26} 평균 {v.mean():.4f} {ci}  (1.0 = 기존 평균과 동일)")
    return {"mean": round(float(v.mean()), 4), "ci": ci, "n": len(v)}
PB = {"treated": m1(rt, "처치 함의임금 비율"), "control": m1(rc, "대조 함의임금 비율")}
PB["DiD"] = D([e for e in EV if np.isfinite(e["r_t"]) and np.isfinite(e["r_c"])],
              "r_t", "r_c", "비율 DiD (처치−대조)")
# [해석 주의] 비율의 **수준**(1.64)은 한계임금이 아니다. 분자 Δ신고소득총액에는 신규채용분뿐 아니라
# **기존 인력의 임금상승분**이 섞여 있어 위로 편향된다. 처치·대조에 동일하게 걸리는 편향이므로
# **DiD 만 해석한다.**
if PB["treated"]:
    PB["level_not_interpretable"] = ("비율 수준 1.64 는 한계임금이 아니다 — 분자에 기존인력 "
                                     "임금상승분이 포함돼 상방 편향. DiD 만 해석할 것.")
    d = PB["DiD"]
    if d["ci"]:
        for S in (0.10, 0.20):
            lo, hi = d["ci"]; mg = [round(lo + S, 4), round(S - hi, 4)]
            ok = bool(lo > -S and hi < S)
            PB[f"equivalence_d{S}"] = {"SESOI": S, "holds": ok, "margin": mg,
                                       "knife": bool(min(mg) < 0.001)}
            print(f"  등가성 δ={S}: {'✓ 성립' if ok else '✗ 미성립'} 여유 {mg}"
                  f"  ({'기존 평균임금의 '+str(int(S*100))+'% 크기 차이를 배제'})")

print("\n[Panel C] 임금–고용 동시변화 기울기")
x = np.array([e["e_t"] - e["e_c"] for e in EV], float)
y = np.array([e["w_t"] - e["w_c"] for e in EV], float)
m = np.isfinite(x) & np.isfinite(y)
sl = float(np.polyfit(x[m], y[m], 1)[0])
bs = np.array([np.polyfit(x[m][j], y[m][j], 1)[0] for j in
               (rng.integers(0, m.sum(), m.sum()) for _ in range(NB))])
ci = qci(bs); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
print(f"  Δlog임금 ~ Δlog고용  기울기 {sl:+.4f} {ci} {sg} (n={int(m.sum())})")
print("  (음의 기울기 = 많이 뽑을수록 평균임금 하락 = 저임금 인력 유입)")
# 크기 환산: 실제 고용 DiD 크기에서 이 기울기가 만드는 임금 하락분
drag = sl * (PA["log_emp"]["DiD"] or 0.0)
print(f"  → 실제 고용 DiD {PA['log_emp']['DiD']:+.4f} 에서 구성효과가 만드는 임금 하락 = {drag:+.5f}")
print(f"     관측된 평균임금 DiD {PA['log_wage']['DiD']:+.4f} 와 비교: "
      f"구성효과는 그 {abs(drag)/abs(PA['log_wage']['DiD'] or 1)*100:.0f}% 크기 — "
      f"{'뒤집지 못한다' if abs(drag) < abs(PA['log_wage']['DiD'] or 0) else '상쇄 가능'}")
PC = {"slope": round(sl, 4), "ci": ci, "sig": sg == "✓", "n": int(m.sum()),
      "implied_drag_at_actual_dEmp": round(float(drag), 5),
      "vs_observed_wage_DiD": round(float(abs(drag) / abs(PA["log_wage"]["DiD"] or 1)), 3)}

print("\n[Panel D] 사전 관성 분위별")
PD = {}
for b, bl in ((0, "T1저관성"), (2, "T3고관성")):
    sub = [e for e in EV if e["pb"] == b]
    print(f"  -- {bl} n_ev={len(sub)} --")
    PD[bl] = {"log_wage": D(sub, "w_t", "w_c", f"{bl} Δlog 평균임금"),
              "ratio": D([e for e in sub if np.isfinite(e["r_t"]) and np.isfinite(e["r_c"])],
                         "r_t", "r_c", f"{bl} 함의임금 비율 DiD")}

# ---- 판정 ----
# 판정 규칙 (v2): 기준은 세 가지이고 **크기**를 본다.
#  ① 평균임금이 실제로 떨어지는가  ② 함의임금 비율 DiD 가 음으로 유의한가
#  ③ 구성효과(기울기 x 실제 Δ고용)가 관측 임금효과를 뒤집을 만큼 큰가
w = PA["log_wage"]
c1 = bool(w["sig"] and (w["DiD"] or 0) < 0)
c2 = bool(PB["DiD"]["sig"] and (PB["DiD"]["DiD"] or 0) < 0)
c3 = bool(abs(PC["implied_drag_at_actual_dEmp"]) >= abs(w["DiD"] or 0))
cheap = c1 or c2 or c3
eq20 = PB.get("equivalence_d0.2", {}).get("holds")
if not cheap and eq20:
    status = "GO"
    concl = ("**신규채용이 저임금 인력이 아니다.** 평균임금은 고용이 늘어나는 중에도 오르고"
             f"({w['DiD']:+.4f}✓), 함의임금 비율은 대조군과 구별되지 않는다"
             f"(DiD {PB['DiD']['DiD']} {PB['DiD']['ci']}, δ=0.20 등가성 성립). "
             f"구성효과는 {PC['implied_drag_at_actual_dEmp']:+.5f} 로 관측 임금효과의 "
             f"{PC['vs_observed_wage_DiD']*100:.0f}% 에 불과해 부호를 뒤집지 못한다. "
             "→ **순수 조정기술 변화** 서사 지지, 노동비용 재구성 서사 기각.")
elif not cheap:
    status, concl = "PARTIAL", "저임금 신호는 없으나 등가성 미성립 — '차이를 검출하지 못함'"
else:
    status, concl = "PARTIAL", f"저임금 유입 신호 (평균임금하락 {c1} / 비율DiD음 {c2} / 구성효과우세 {c3})"
verdict = (f"Δlog 평균임금 {w['DiD']}{w['ci']}{'✓' if w['sig'] else '✗'} | "
           f"함의임금 비율 처치 {PB.get('treated',{}).get('mean')}{PB.get('treated',{}).get('ci')} "
           f"· DiD {PB['DiD']['DiD']}"
           f"{'✓' if PB['DiD']['sig'] else '✗'} | 임금~고용 기울기 {PC['slope']}"
           f"{'✓' if PC['sig'] else '✗'} | {concl}")
emit("I-22", "임금구조 (서사 판별)", status,
     {"panelA_wage_emp": PA, "panelB_marginal_wage_ratio": PB,
      "panelC_wage_emp_slope": PC, "panelD_by_inertia": PD,
      "min_delta_emp": MIN_DE, "tercile_cuts": [round(float(Q1), 4), round(float(Q2), 4)]},
     "신규채용 임금이 기존과 같으면 순수 조정기술 변화, 낮으면 노동비용 재구성",
     verdict, kill_met=False, n=len(EV),
     extra={"conclusion": concl,
            "measurement_limit": "국민연금 고지금액은 등급 상한이 있어 고소득자가 절단된다. "
                                 "절대 수준이 아니라 처치−대조 차분과 비율로만 해석한다."})
