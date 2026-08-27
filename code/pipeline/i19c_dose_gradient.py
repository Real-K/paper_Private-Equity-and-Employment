# -*- coding: utf-8 -*-
"""I-19c 지배권 이전 dose 기울기 — 다중검정 없는 단일 검정 + PE 위치.

[I-19b 문제] 하위유형 6종 중 ①과반취득 −0.0178✓, ⑤지분변화상위 −0.0167✓, ②법인 조절 −0.0471✓ 이
나왔으나 **6개 검정의 다중성**을 보정하지 않았다. 보정하면 살아남는지 먼저 계산하고,
그와 별개로 **다중성이 없는 단일 검정**으로 다시 친다.

단일 검정: **신규 최대주주 지분(= 이전된 지배권의 크기)에 대한 효과의 연속 기울기.**
 · 지배권 이전이 메커니즘이면 기울기가 **음(−)** 이어야 한다 (많이 넘어갈수록 효과 큼)
 · 그리고 **PE 를 같은 기울기 위에 올려** PE 가 그 선 위에 있는지(= 같은 현상) 위에 있는지
   (= PE 고유 프리미엄) 본다.

Panel A  I-19b 6종의 Šidák 보정
Panel B  ★ OWN 연속 dose 기울기 (단일 검정)
Panel C  ★ PE 를 OWN 기울기 위에 배치 — 예측치 vs 실측치
"""
import gc, re, json
import numpy as np, pandas as pd
from math import erfc, sqrt
from difflib import SequenceMatcher
from h30_common import (load, deals, build, attach, boot_did_ci, emit,
                        SEED, qci, NB, widx, BASE)

rng = np.random.default_rng(SEED)
H = f"{BASE}/P014_upgrade_package/harness30"

print("[Panel A] I-19b 6종 Šidák 보정")
B = json.load(open(f"{H}/out/I19b.json", encoding="utf-8"))["estimates"]["panelA_subtypes"]
PA = {}
K = len(B)
for k, v in B.items():
    if not v.get("ci"): continue
    lo, hi = v["ci"]; se = (hi - lo) / (2 * 1.96)
    z = abs(v["DiD"]) / se if se > 0 else 0
    p = erfc(z / sqrt(2)); sid = 1 - (1 - p) ** K
    PA[k] = {"DiD": v["DiD"], "p": round(p, 4), "sidak_p": round(sid, 4), "survives": bool(sid < 0.05)}
    print(f"  {k:<26} {v['DiD']:+.4f}  p={p:.4f}  Šidák({K}) p={sid:.4f} "
          f"{'✓ 생존' if sid < 0.05 else '✗ 소멸'}")
    m = v.get("T3_T1")
    if m and m.get("ci"):
        lo, hi = m["ci"]; se = (hi - lo) / (2 * 1.96); z = abs(m["diff"]) / se if se > 0 else 0
        p2 = erfc(z / sqrt(2)); s2 = 1 - (1 - p2) ** K
        PA[k]["T3T1_sidak_p"] = round(s2, 4); PA[k]["T3T1_survives"] = bool(s2 < 0.05)
        print(f"      └ T3−T1 {m['diff']:+.4f} p={p2:.4f} Šidák p={s2:.4f} {'✓' if s2 < 0.05 else '✗'}")

print("\n로딩...")
G = load()
orig, allt, PE, META = deals(G)
Hv, Ev, idx, mset = G["Hv"], G["Ev"], G["idx"], G["mset"]
INP = set(np.asarray(idx))
cols = ["business_number", "기준일", "주주명", "보통주_지분율"]
keep = []
for ch in pd.read_csv(f"{BASE}/PI/drops/외감_주주_시계열_2009plus.csv",
                      usecols=cols, dtype=str, chunksize=400_000):
    ch["bn10"] = ch.business_number.str.replace(r"\D", "", regex=True).str.zfill(10)
    ch = ch[ch.bn10.isin(INP)]
    ch["pct"] = pd.to_numeric(ch["보통주_지분율"], errors="coerce")
    keep.append(ch.loc[ch.pct >= 15, ["bn10", "기준일", "주주명", "pct"]])
S = pd.concat(keep, ignore_index=True); del keep; gc.collect()
S["dt"] = pd.to_datetime(S["기준일"], format="%Y%m%d", errors="coerce"); S = S[S.dt.notna()]
S["yr"] = S.dt.dt.year
S = S[S.dt == S.groupby(["bn10", "yr"])["dt"].transform("max")]
def nz(x):
    x = str(x).lower()
    x = re.sub(r"주식회사|유한회사|유한책임회사|합자회사|\(주\)|\(유\)|㈜|limited|ltd|inc|corp|company|co\b", "", x)
    return re.sub(r"[^0-9a-z가-힣]", "", x.replace("홀딩즈", "홀딩스"))
S["nm"] = S["주주명"].map(nz)
CL = {}
for bn, g in S.groupby("bn10"):
    reps = []
    for v in sorted(g.nm.unique(), key=len, reverse=True):
        if not v: CL[(bn, v)] = v; continue
        hit = next((r for r in reps if v == r or (len(v) >= 5 and len(r) >= 5 and
                    (v in r or r in v or SequenceMatcher(None, v, r).ratio() >= 0.85))), None)
        if hit is None: reps.append(v); hit = v
        CL[(bn, v)] = hit
S["key"] = [CL[(b, v)] for b, v in zip(S.bn10, S.nm)]
T = S.sort_values("pct").groupby(["bn10", "yr"]).tail(1).sort_values(["bn10", "yr"]).copy()
T["prevkey"] = T.groupby("bn10")["key"].shift(1); T["pyr"] = T.groupby("bn10")["yr"].shift(1)
CHG = T[(T.prevkey.notna()) & (T.key != T.prevkey) & (T.yr - T.pyr <= 2)]
CHANGED = set(CHG.bn10)
PEPAT = r"투자|인베스트|캐피탈|사모|펀드|조합|파트너스|에쿼티|벤처|PEF|Capital|Invest|Partner|Equity|Fund"
OWN = CHG[(~CHG.bn10.isin(PE)) & (~CHG["주주명"].fillna("").str.contains(PEPAT, case=False, regex=True))]
OWN = OWN.drop_duplicates("bn10").copy()
OWN["mi"] = OWN.yr * 12 + 6; OWN["src"] = "own"
del S, T; gc.collect()
EXCL = CHANGED | set(PE)

def zsh(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return np.nan
    x = Hv[row, c]
    return float((x == 0).mean()) if np.isfinite(x).all() else np.nan
def prep(L):
    for e in L:
        a = zsh(e["ti"], e["m0"], -12, -1); b = zsh(e["ti"], e["m0"], 1, 12)
        e["z_t"] = b - a if (np.isfinite(a) and np.isfinite(b)) else np.nan
        cd = [zsh(k, e["m0"], 1, 12) - zsh(k, e["m0"], -12, -1) for k in e["ctrls"]]
        cd = [x for x in cd if np.isfinite(x)]
        e["z_c"] = float(np.mean(cd)) if cd else np.nan
        e["eff"] = e["z_t"] - e["z_c"] if (np.isfinite(e["z_t"]) and np.isfinite(e["z_c"])) else np.nan
    return L

CAP = 2500
sub = OWN.sample(min(CAP, len(OWN)), random_state=42)
L, _ = build(G, sub[["bn10", "mi", "src"]], PE, ctrl_extra_exclude=EXCL); attach(G, L); prep(L)
ST = dict(zip(sub.bn10, sub.pct))
x = np.array([ST.get(e["bn"], np.nan) for e in L], float)
y = np.array([e["eff"] for e in L], float)
m = np.isfinite(x) & np.isfinite(y)
sl, ic = np.polyfit(x[m], y[m], 1)
bs = np.array([np.polyfit(x[m][j], y[m][j], 1) for j in
               (rng.integers(0, m.sum(), m.sum()) for _ in range(NB))])
ci = qci(bs[:, 0]); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
print(f"\n[Panel B] ★ OWN 연속 dose 기울기 (단일 검정, n={int(m.sum())})")
print(f"  신규 최대주주 지분(%) → 효과   기울기 {sl:+.6f}/%p {ci} {sg}")
print(f"  절편 {ic:+.4f} · 지분 중앙값 {np.median(x[m]):.1f}%")
for q in (25, 50, 75, 90):
    v = np.percentile(x[m], q)
    print(f"    지분 {v:>5.1f}% (p{q}) 에서 예측 효과 {sl*v+ic:+.4f}")
PB = {"slope_per_pct": round(float(sl), 6), "ci": ci, "sig": sg == "✓",
      "intercept": round(float(ic), 4), "n": int(m.sum()),
      "stake_median": round(float(np.median(x[m])), 1)}

print("\n[Panel C] ★ PE 를 OWN 기울기 위에 배치")
EVpe, _ = build(G, allt, PE)
PEy = pd.DataFrame({"bn10": [e["bn"] for e in EVpe],
                    "mi": [((e["m0"] - 1) // 12) * 12 + 6 for e in EVpe], "src": "pe"})
LP, _ = build(G, PEy, PE, ctrl_extra_exclude=EXCL); attach(G, LP); prep(LP)
pe_eff = np.array([e["eff"] for e in LP], float); pe_eff = pe_eff[np.isfinite(pe_eff)]
D14 = json.load(open(f"{H}/out/I14.json", encoding="utf-8"))["estimates"]["panelC_dose"]
pe_stake = D14.get("dose_median")
pred = sl * pe_stake + ic
bsp = np.array([pe_eff[j].mean() for j in (rng.integers(0, len(pe_eff), len(pe_eff)) for _ in range(NB))])
gap = pe_eff.mean() - pred
bg = np.array([bsp[i] - (bs[i, 0] * pe_stake + bs[i, 1]) for i in range(NB)])
cig = qci(bg); sgg = "✓" if (cig[0] > 0 or cig[1] < 0) else "✗"
print(f"  PE 지분 중앙값 {pe_stake}% (I-14) → OWN 기울기 예측 효과 {pred:+.4f}")
print(f"  PE 실측 효과 {pe_eff.mean():+.4f} {qci(bsp)} (n={len(pe_eff)})")
print(f"  **PE 프리미엄(실측−예측) {gap:+.4f} {cig} {sgg}**")
PC = {"pe_stake_median": pe_stake, "predicted_from_OWN": round(float(pred), 4),
      "pe_actual": round(float(pe_eff.mean()), 4), "pe_ci": qci(bsp),
      "premium": round(float(gap), 4), "premium_ci": cig, "premium_sig": sgg == "✓",
      "n_pe": len(pe_eff)}

surv = [k for k, v in PA.items() if v.get("survives") or v.get("T3T1_survives")]
if PB["sig"] and PB["slope_per_pct"] < 0 and not PC["premium_sig"]:
    status = "GO"; concl = ("**지배권 이전의 크기에 대한 연속 기울기가 음으로 유의**하고 "
                            "**PE 는 그 선 위에 있다(프리미엄 무유의)** → 메커니즘은 PE 고유가 아니라 "
                            "**지배권 이전 일반**이며 PE 는 이전 규모가 큰 극단 사례다.")
elif PB["sig"] and PB["slope_per_pct"] < 0 and PC["premium_sig"]:
    status = "GO"; concl = ("연속 기울기 음으로 유의하나 **PE 가 그 선 위에 유의하게 얹혀 있다** → "
                            "지배권 이전이 공통 기제이되 **PE 고유 프리미엄이 존재**한다.")
elif not PB["sig"]:
    status = "PARTIAL"; concl = (f"연속 기울기 무유의 → 지배권 이전 크기로 효과가 설명되지 않는다. "
                                 f"하위유형 중 Šidák 생존: {surv or '없음'}.")
else:
    status = "PARTIAL"; concl = "기울기 부호가 예측과 반대"
verdict = (f"Šidák 생존 {surv or '없음'} | OWN dose 기울기 {PB['slope_per_pct']}/%p {ci}{sg} "
           f"(n={PB['n']}) | PE 예측 {PC['predicted_from_OWN']} vs 실측 {PC['pe_actual']}, "
           f"프리미엄 {PC['premium']}{PC['premium_ci']}{'✓' if PC['premium_sig'] else '✗'} | {concl}")
emit("I-19c", "지배권 이전 dose 기울기 + PE 위치", status,
     {"panelA_sidak": PA, "panelB_own_dose": PB, "panelC_pe_on_gradient": PC},
     "지배권 이전 크기에 대한 연속 기울기(단일 검정)와 그 위에서의 PE 위치로 "
     "'PE 고유' 대 '지배권 이전 일반'을 가른다",
     verdict, kill_met=False, n=PB["n"] + PC["n_pe"],
     extra={"conclusion": concl,
            "why_single_test": "I-19b 하위유형 6종은 다중검정 문제가 있어 Šidák 보정 결과를 함께 낸다. "
                               "연속 기울기는 단일 검정이라 그 문제가 없다."})
