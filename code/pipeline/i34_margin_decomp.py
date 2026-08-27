# -*- coding: utf-8 -*-
"""I-34 외연/내연 마진 — '71%' 대체. 비율이 아니라 성분별 DiD 로.

[문제] 기존 기록의 "외연 비중 71%" 는 집계 비율 Σ(i0·Δp)/Σ(Δ(p·i)) 이다. I-33 에서 재계산하니
**−4.35 [−23.6, +19.2]** 로 폭발했다. 분모(총변화)가 이벤트 간 상쇄로 0 근처가 되기 때문이다.
**비율 통계량 자체가 불안정하다. 원고에 쓸 수 없다.**

[대체] 채용률 = 12 · p · i  (p = 채용이 있는 달의 비율, i = 활동월의 평균 채용강도/고용).
비율 대신 **p 와 i 각각의 DiD** 를 낸다. 주장은 "외연이 몇 % 다" 가 아니라
**"외연은 움직이고 내연은 움직이지 않는다"** 이며, 후자는 등가성으로 뒷받침한다.

Panel A  p(외연) · i(내연) DiD + 등가성
Panel B  log 분해 — Δlog(rate) = Δlog(p) + Δlog(i) (상쇄 없는 가법 분해)
Panel C  사전 관성 분위별
"""
import numpy as np
from h30_common import load, deals, build, boot_did_ci, emit, SEED, qci, NB, widx, pi_parts

rng = np.random.default_rng(SEED)
print("[I-34] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
EV, _ = build(G, allt, PE)
Hv = G["Hv"]

def zsh(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return np.nan
    x = Hv[row, c]
    return float((x == 0).mean()) if np.isfinite(x).all() else np.nan

def parts(row, m0):
    a = pi_parts(G, row, m0, -12, -1); b = pi_parts(G, row, m0, 1, 12)
    if a is None or b is None: return None
    return a, b

for e in EV:
    o = {}
    r = parts(e["ti"], e["m0"])
    if r:
        (p0, i0), (p1, i1) = r
        o["dp"] = p1 - p0; o["di"] = i1 - i0
        o["dlp"] = np.log(p1 / p0) if (p0 > 0 and p1 > 0) else np.nan
        o["dli"] = np.log(i1 / i0) if (i0 > 0 and i1 > 0) else np.nan
        o["p0"] = p0; o["i0"] = i0
    e["t"] = o
    acc = {}
    for k in e["ctrls"]:
        r2 = parts(k, e["m0"])
        if not r2: continue
        (p0, i0), (p1, i1) = r2
        acc.setdefault("dp", []).append(p1 - p0)
        acc.setdefault("di", []).append(i1 - i0)
        if p0 > 0 and p1 > 0: acc.setdefault("dlp", []).append(np.log(p1 / p0))
        if i0 > 0 and i1 > 0: acc.setdefault("dli", []).append(np.log(i1 / i0))
    e["c"] = {k: float(np.mean(v)) for k, v in acc.items() if v}
    e["pp"] = zsh(e["ti"], e["m0"], -24, -13)
_pp = np.array([e["pp"] for e in EV], float)
Q1, Q2 = np.percentile(_pp[np.isfinite(_pp)], [33.33, 66.67])
for e in EV:
    v = e["pp"]; e["pb"] = None if not np.isfinite(v) else (0 if v <= Q1 else (1 if v <= Q2 else 2))

def D(sub, k, lab, sesoi=None):
    t = [x["t"].get(k) for x in sub if x["t"].get(k) is not None and k in x["c"]]
    c = [x["c"][k] for x in sub if x["t"].get(k) is not None and k in x["c"]]
    t = [v for v in t if np.isfinite(v)]; c = c[:len(t)]
    p_, ci, n = boot_did_ci(t, c, rng)
    sg = "✓" if (ci and (ci[0] > 0 or ci[1] < 0)) else ("✗" if ci else "-")
    o = {"DiD": p_, "ci": ci, "n": n, "sig": sg == "✓"}
    extra = ""
    if sesoi and ci and not o["sig"]:
        mg = [round(ci[0] + sesoi, 4), round(sesoi - ci[1], 4)]
        ok = bool(ci[0] > -sesoi and ci[1] < sesoi); kn = bool(min(mg) < 0.001)
        o["equivalence"] = {"SESOI": sesoi, "holds": ok, "margin": mg, "knife": kn}
        extra = f"  등가성(δ={sesoi}) {'✓성립' if (ok and not kn) else '✗미성립'} 여유{mg}"
    print(f"  {lab:<28} {str(p_):>9} {str(ci):<21} {sg} (n={n}){extra}")
    return o

print("\n[Panel A] 수준 분해 — p(외연) · i(내연)")
base_p = float(np.mean([e["t"]["p0"] for e in EV if "p0" in e["t"]]))
base_i = float(np.mean([e["t"]["i0"] for e in EV if "i0" in e["t"]]))
print(f"  처치 사전 평균: p={base_p:.4f} (채용월 비율) · i={base_i:.4f} (활동월 강도)")
PA = {"pre_p": round(base_p, 4), "pre_i": round(base_i, 4),
      "p_extensive": D(EV, "dp", "Δp (외연)"),
      "i_intensive": D(EV, "di", "Δi (내연)", sesoi=round(0.10 * base_i, 4))}

print("\n[Panel B] 로그 가법 분해 — Δlog(rate) = Δlog(p) + Δlog(i)")
PB = {"dlog_p": D(EV, "dlp", "Δlog p (외연)"), "dlog_i": D(EV, "dli", "Δlog i (내연)", sesoi=0.10)}
a, b = PB["dlog_p"]["DiD"], PB["dlog_i"]["DiD"]
if a is not None and b is not None:
    tot = a + b
    PB["dlog_total"] = round(tot, 4)
    PB["extensive_share_log"] = round(a / tot, 3) if abs(tot) > 1e-6 else None
    print(f"  → Δlog(rate) ≈ {tot:+.4f} = 외연 {a:+.4f} + 내연 {b:+.4f}"
          f"  (외연 비중 {PB['extensive_share_log']})")
    print("     * 로그 가법 분해는 상쇄가 없어 안정적이나, 비중은 여전히 총합이 작으면 민감하다.")

print("\n[Panel C] 사전 관성 분위별")
PC = {}
for bq, bl in ((0, "T1저관성"), (2, "T3고관성")):
    sub = [e for e in EV if e["pb"] == bq]
    print(f"  -- {bl} n={len(sub)} --")
    PC[bl] = {"p": D(sub, "dp", f"{bl} Δp"), "i": D(sub, "di", f"{bl} Δi", sesoi=round(0.10*base_i, 4))}

pe = PA["p_extensive"]; ie = PA["i_intensive"]
eq = ie.get("equivalence", {})
if pe["sig"] and not ie["sig"] and eq.get("holds") and not eq.get("knife"):
    status, concl = "GO", ("외연은 유의하게 움직이고 내연은 등가성 범위 안에서 불변 — "
                           "**'외연 마진' 주장이 비율 없이 성립한다.**")
elif pe["sig"] and not ie["sig"]:
    status, concl = "PARTIAL", "외연 유의·내연 무유의이나 등가성 미성립 — '내연 불변'은 못 쓰고 '미검출'만"
else:
    status, concl = "PARTIAL", "분해 판별 불가"
verdict = (f"Δp {pe['DiD']}{pe['ci']}{'✓' if pe['sig'] else '✗'} (사전 p={base_p:.3f}) | "
           f"Δi {ie['DiD']}{ie['ci']}{'✓' if ie['sig'] else '✗'} 등가성 "
           f"{'성립' if eq.get('holds') else '미성립'} | log 분해 외연 {PB['dlog_p']['DiD']} "
           f"내연 {PB['dlog_i']['DiD']} | {concl}")
emit("I-34", "외연/내연 마진 분해 (비율 대체)", status,
     {"panelA_levels": PA, "panelB_log": PB, "panelC_by_inertia": PC,
      "retired_statistic": {"name": "외연 비중 71%",
                            "reason": "집계 비율 Σ(i0·Δp)/Σ(Δ(p·i)) 가 분모 상쇄로 폭발 "
                                      "(I-33 재계산 −4.35 [−23.6, +19.2]). 원고 사용 금지."}},
     "비율 대신 성분별 DiD 로 '외연은 움직이고 내연은 불변'을 보인다",
     verdict, kill_met=False, n=len(EV), extra={"conclusion": concl})
