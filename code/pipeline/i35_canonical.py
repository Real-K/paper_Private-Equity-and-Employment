# -*- coding: utf-8 -*-
"""I-35 정본 수치 전수 재계산 — 원고에 들어갈 모든 숫자를 하나의 정의로.

[왜] 식별 결과는 반복 검증됐으나 기술통계는 한 번 계산된 뒤 재검증되지 않았다. 그 결과
'외연 71%'(비율 폭발) · '내연 null'(재계산 불일치) · '수준=선택'(인공물 arm 근거) 이 차례로 무너졌다.
경로별로 하나씩 터뜨리는 대신 **원고 수치 전부를 한 정의로 재계산하고 기존 기록과 전수 대조**한다.

[정본 정의 — 이 파일이 유일한 출처]
 · 표본       p014_treated_sample_v2_expanded.csv → build(allt, PE), 매칭 진입 379
 · 창         사전 [−12,−1] · 사후 [1,12] (연차분석만 [13,24],[25,36])
 · 대조       셀(산업2×규모×성장×연령) 5NN, never-PE(752 제외)
 · 관성분위   사전-사전 [−24,−13] 무채용비중의 33.33/66.67 백분위 (finite p & z 표본에서)
 · **동일bin** 조절 분석은 처치와 같은 관성 bin 의 대조군만 사용 (I-25 사양). 전체 대조 사양도 병기.
 · 부트스트랩 이벤트 재표본 999회, seed 42

출력: out/I35_CANONICAL.json + 기존 기록 대조표
"""
import numpy as np
from h30_common import (load, deals, build, attach, summ, boot_did_ci, emit,
                        SEED, qci, NB, widx, dflow, rel_log, pi_parts)

rng = np.random.default_rng(SEED)
print("[I-35] 정본 재계산 로딩...")
G = load()
orig, allt, PE, META = deals(G)
EV, _ = build(G, allt, PE); attach(G, EV)
Hv, Sv = G["Hv"], G["Sv"]
print(f"  매칭 진입 {len(EV)}")

def zsh(r, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return np.nan
    x = Hv[r, c]
    return float((x == 0).mean()) if np.isfinite(x).all() else np.nan

for e in EV:
    a, b = zsh(e["ti"], e["m0"], -12, -1), zsh(e["ti"], e["m0"], 1, 12)
    e["z_t"] = b - a if (np.isfinite(a) and np.isfinite(b)) else np.nan
    cs = [zsh(k, e["m0"], 1, 12) - zsh(k, e["m0"], -12, -1) for k in e["ctrls"]]
    e["z_all"] = float(np.mean([x for x in cs if np.isfinite(x)])) if any(np.isfinite(cs)) else np.nan
    e["pp"] = zsh(e["ti"], e["m0"], -24, -13)
    e["ppc"] = [zsh(k, e["m0"], -24, -13) for k in e["ctrls"]]
    e["zc_list"] = cs
    e["s_t"] = dflow(G, e["ti"], e["m0"], Sv)
    sc = [dflow(G, k, e["m0"], Sv) for k in e["ctrls"]]
    e["s_c"] = float(np.mean([x for x in sc if np.isfinite(x)])) if any(np.isfinite(sc)) else np.nan

_p = np.array([e["pp"] for e in EV], float); _z = np.array([e["z_t"] for e in EV], float)
u = np.isfinite(_p) & np.isfinite(_z)
Q1, Q2 = np.percentile(_p[u], [33.33, 66.67])
tb = lambda v: None if not np.isfinite(v) else (0 if v <= Q1 else (1 if v <= Q2 else 2))
for e in EV: e["pb"] = tb(e["pp"])
print(f"  관성 3분위 컷 {Q1:.6f} / {Q2:.6f} (n={int(u.sum())})")

def did(t, c, lab, sesoi=None):
    p_, ci, n = boot_did_ci(t, c, rng)
    sg = "✓" if (ci and (ci[0] > 0 or ci[1] < 0)) else ("✗" if ci else "-")
    o = {"est": p_, "ci": ci, "n": n, "sig": sg == "✓"}
    ex = ""
    if sesoi and ci and not o["sig"]:
        mg = [round(ci[0] + sesoi, 4), round(sesoi - ci[1], 4)]
        o["equiv"] = {"SESOI": sesoi, "holds": bool(ci[0] > -sesoi and ci[1] < sesoi),
                      "margin": mg, "knife": bool(min(mg) < 0.001)}
        ex = f"  등가성δ={sesoi} {'✓' if o['equiv']['holds'] and not o['equiv']['knife'] else '✗'}"
    print(f"  {lab:<34} {str(p_):>9} {str(ci):<21} {sg} (n={n}){ex}")
    return o

C = {}
print("\n── A. 헤드라인 배터리 ──")
s = summ(EV, rng)
C["A1_hire_DiD"] = {"est": s["DiD"], "ci": s["DiD_ci"], "n": s["n"]}
C["A2_rel12"] = {"est": s["rel"], "ci": s["rel_ci"], "n": s["n"]}
C["A3_P1"] = {"est": s["P1"], "ci": s["P1_ci"], "n": s["n"]}
print(f"  {'채용률 DiD':<34} {s['DiD']} {s['DiD_ci']} (n={s['n']})")
print(f"  {'rel12 (log 고용)':<34} {s['rel']} {s['rel_ci']}")
print(f"  {'P1 축소확률':<34} {s['P1']} {s['P1_ci']}")
C["A4_zero_share"] = did([e["z_t"] for e in EV], [e["z_all"] for e in EV], "무채용비중 DiD")
C["A5_separation"] = did([e["s_t"] for e in EV], [e["s_c"] for e in EV], "이직률 DiD", sesoi=0.046)

print("\n── B. 마진 분해 (I-34 정본) ──")
for e in EV:
    a = pi_parts(G, e["ti"], e["m0"], -12, -1); b = pi_parts(G, e["ti"], e["m0"], 1, 12)
    e["m"] = {}
    if a and b:
        (p0, i0), (p1, i1) = a, b
        e["m"] = {"dp": p1 - p0, "di": i1 - i0,
                  "dlp": np.log(p1 / p0) if p0 > 0 and p1 > 0 else np.nan,
                  "dli": np.log(i1 / i0) if i0 > 0 and i1 > 0 else np.nan, "p0": p0, "i0": i0}
    acc = {}
    for k in e["ctrls"]:
        a2 = pi_parts(G, k, e["m0"], -12, -1); b2 = pi_parts(G, k, e["m0"], 1, 12)
        if not (a2 and b2): continue
        (p0, i0), (p1, i1) = a2, b2
        acc.setdefault("dp", []).append(p1 - p0); acc.setdefault("di", []).append(i1 - i0)
        if p0 > 0 and p1 > 0: acc.setdefault("dlp", []).append(np.log(p1 / p0))
        if i0 > 0 and i1 > 0: acc.setdefault("dli", []).append(np.log(i1 / i0))
    e["mc"] = {k: float(np.mean(v)) for k, v in acc.items() if v}
def mdid(k, lab, sesoi=None):
    t = [e["m"].get(k) for e in EV if e["m"].get(k) is not None and k in e["mc"]]
    c = [e["mc"][k] for e in EV if e["m"].get(k) is not None and k in e["mc"]]
    ok = [i for i, v in enumerate(t) if np.isfinite(v)]
    return did([t[i] for i in ok], [c[i] for i in ok], lab, sesoi)
C["B1_dp"] = mdid("dp", "Δp 외연 (수준)")
C["B2_di"] = mdid("di", "Δi 내연 (수준)", sesoi=0.0055)
C["B3_dlogp"] = mdid("dlp", "Δlog p 외연")
C["B4_dlogi"] = mdid("dli", "Δlog i 내연", sesoi=0.10)
if C["B3_dlogp"]["est"] and C["B4_dlogi"]["est"]:
    tt = C["B3_dlogp"]["est"] + C["B4_dlogi"]["est"]
    C["B5_extensive_share_log"] = round(C["B3_dlogp"]["est"] / tt, 3)
    print(f"  → Δlog(rate) {tt:+.4f} = 외연 {C['B3_dlogp']['est']:+.4f} + 내연 "
          f"{C['B4_dlogi']['est']:+.4f} · 외연비중 {C['B5_extensive_share_log']}")

print("\n── C. 관성 조절 — 두 사양 병기 (표류 원인 확정) ──")
def modspec(same_bin):
    out = {}
    for bq, bl in ((0, "T1"), (2, "T3")):
        t, c = [], []
        for e in EV:
            if e["pb"] != bq or not np.isfinite(e["z_t"]): continue
            cs = []
            for j, v in enumerate(e["zc_list"]):
                if not np.isfinite(v): continue
                if same_bin and tb(e["ppc"][j]) != bq: continue
                cs.append(v)
            if cs: t.append(e["z_t"]); c.append(float(np.mean(cs)))
        out[bl] = did(t, c, f"{'동일bin' if same_bin else '전체대조'} {bl}")
        out[bl + "_raw"] = (np.array(t), np.array(c))
    d1 = out["T1_raw"][0] - out["T1_raw"][1]; d3 = out["T3_raw"][0] - out["T3_raw"][1]
    bs = np.array([d3[rng.integers(0, len(d3), len(d3))].mean()
                   - d1[rng.integers(0, len(d1), len(d1))].mean() for _ in range(NB)])
    ci = qci(bs); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
    print(f"  {'  └ T3−T1':<34} {d3.mean()-d1.mean():>+9.4f} {str(ci):<21} {sg}")
    for k in ("T1_raw", "T3_raw"): out.pop(k)
    out["T3_T1"] = {"est": round(float(d3.mean() - d1.mean()), 4), "ci": ci, "sig": sg == "✓"}
    return out
C["C1_samebin"] = modspec(True)
C["C2_allctrl"] = modspec(False)
print(f"  → 표류 원인: 동일bin {C['C1_samebin']['T3_T1']['est']} vs "
      f"전체대조 {C['C2_allctrl']['T3_T1']['est']} — **원고는 동일bin 을 정본으로 쓴다**")

# ---------- 기존 기록 대조 ----------
LEDGER = {
    "A1_hire_DiD": 0.0485, "A2_rel12": 0.0878, "A3_P1": -0.1049, "A4_zero_share": -0.0460,
    "B5_extensive_share_log": 0.71, "C1_samebin.T3_T1": -0.0958,
}
print("\n── D. 기존 기록 대조 ──")
D = {}
for k, old in LEDGER.items():
    cur = C
    for part in k.split("."): cur = cur[part]
    new = cur["est"] if isinstance(cur, dict) and "est" in cur else cur
    if new is None: continue
    diff = abs(new - old); rel = diff / abs(old) if old else np.inf
    ok = rel < 0.05
    D[k] = {"ledger": old, "canonical": new, "abs_diff": round(float(diff), 4),
            "rel_diff": round(float(rel), 3), "match_5pct": bool(ok)}
    print(f"  {k:<28} 기록 {old:>+8.4f} → 정본 {new:>+8.4f}  차이 {rel*100:>6.1f}% "
          f"{'✓ 일치' if ok else '🔴 불일치'}")
bad = [k for k, v in D.items() if not v["match_5pct"]]
emit("I-35", "정본 수치 전수 재계산", "GO" if not bad else "PARTIAL",
     {"canonical": C, "ledger_crosscheck": D, "tercile_cuts": [round(float(Q1), 6), round(float(Q2), 6)]},
     "원고에 들어갈 모든 수치를 하나의 정의로 재계산하고 기존 기록과 전수 대조한다",
     f"대조 {len(D)}건 중 불일치 {len(bad)}건: {bad or '없음'}. "
     f"관성 조절 표류 원인 확정 — 동일bin {C['C1_samebin']['T3_T1']['est']} vs "
     f"전체대조 {C['C2_allctrl']['T3_T1']['est']}",
     kill_met=False, n=len(EV),
     extra={"canonical_definitions": "표본 379 · 창 [−12,−1]v[1,12] · 셀5NN never-PE · "
                                     "관성컷 사전-사전 33.33/66.67 백분위 · 부트 999 seed 42 · "
                                     "조절은 동일bin 정본",
            "mismatches": bad})
