# -*- coding: utf-8 -*-
"""I-39 최장 spell·집중도에도 동일 벤치마크 — I-38 의 정직한 완성.

I-38 에서 초과 무채용은 0 이었으나 최장 spell(−0.357✓)·HHI(−0.025✓)·최다2개월(−0.025✓)이
남았다. **그러나 이들도 총채용 N 이 늘면 기계적으로 줄어든다.** 같은 총량-고정 벤치마크를 걸지 않으면
증거로 쓸 수 없다.

벤치마크: 각 기업-창에서 실제 N 건을 12개월에 **고용 비중 가중 무작위 배분**(다항)해 R=100 회
모의하고, 기대 최장 spell·기대 HHI·기대 최다2개월 비중을 구한다. 초과 = 실제 − 기대.
"""
import numpy as np
from h30_common import load, deals, build, boot_did_ci, emit, SEED, qci, NB, widx

rng = np.random.default_rng(SEED)
R = 100
print("[I-39] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Ev = G["Hv"], G["Ev"]

def maxspell(h):
    run = mx = 0
    for x in h:
        run = run + 1 if x == 0 else 0
        if run > mx: mx = run
    return mx

def stats(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != 12: return None
    h, e = Hv[row, c].astype(float), Ev[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.nanmean(e) < 5: return None
    N = int(h.sum())
    wj = e / e.sum() if e.sum() > 0 else np.full(12, 1/12)
    act = dict(ms=float(maxspell(h)),
               hhi=float(np.sum((h/N)**2)) if N > 0 else np.nan,
               t2=float(np.sort(h)[-2:].sum()/N) if N > 0 else np.nan,
               zero=float((h == 0).sum()))
    if N == 0:
        exp = dict(ms=12.0, hhi=np.nan, t2=np.nan, zero=12.0)
    else:
        sim = rng.multinomial(N, wj, size=R).astype(float)          # R x 12
        exp = dict(ms=float(np.mean([maxspell(s) for s in sim])),
                   hhi=float(np.mean(np.sum((sim/N)**2, axis=1))),
                   t2=float(np.mean(np.sort(sim, axis=1)[:, -2:].sum(axis=1)/N)),
                   zero=float(np.mean((sim == 0).sum(axis=1))))
    return dict(N=float(N),
                **{f"a_{k}": v for k, v in act.items()},
                **{f"e_{k}": v for k, v in exp.items()},
                **{f"x_{k}": act[k]-exp[k] for k in act})

K = ["N"] + [f"{p}_{k}" for k in ("ms", "hhi", "t2", "zero") for p in ("a", "e", "x")]
print(f"  모의 {R}회/창 · 이벤트 {len(EV)}")
for e in EV:
    a, b = stats(e["ti"], e["m0"], -12, -1), stats(e["ti"], e["m0"], 1, 12)
    e["t"] = {k: b[k]-a[k] for k in K} if (a and b) else {}
    e["pre"] = a
    acc = {}
    for c in e["ctrls"]:
        a2, b2 = stats(c, e["m0"], -12, -1), stats(c, e["m0"], 1, 12)
        if a2 and b2:
            for k in K: acc.setdefault(k, []).append(b2[k]-a2[k])
    e["c"] = {k: float(np.nanmean(v)) for k, v in acc.items() if v}

def D(k, lab, sesoi=None):
    t = [x["t"].get(k) for x in EV if k in x["t"] and k in x["c"]]
    c = [x["c"][k] for x in EV if k in x["t"] and k in x["c"]]
    ok = [i for i, v in enumerate(t) if np.isfinite(v) and np.isfinite(c[i])]
    p_, ci, n = boot_did_ci([t[i] for i in ok], [c[i] for i in ok], rng)
    sg = "✓" if (ci and (ci[0] > 0 or ci[1] < 0)) else ("✗" if ci else "-")
    o = {"DiD": p_, "ci": ci, "n": n, "sig": sg == "✓"}
    ex = ""
    if sesoi and ci and not o["sig"]:
        mg = [round(ci[0]+sesoi, 4), round(sesoi-ci[1], 4)]
        o["equiv"] = {"SESOI": sesoi, "holds": bool(ci[0] > -sesoi and ci[1] < sesoi), "margin": mg}
        ex = f"  등가성δ={sesoi} {'✓ 성립' if o['equiv']['holds'] else '✗'}"
    print(f"  {lab:<42} {str(p_):>9} {str(ci):<21} {sg} (n={n}){ex}")
    return o

pre = {k: float(np.nanmean([e["pre"][k] for e in EV if e["pre"]])) for k in K}
print(f"\n  처치 사전: 최장spell 실제 {pre['a_ms']:.2f} vs 기대 {pre['e_ms']:.2f} (초과 {pre['x_ms']:+.2f}개월) · "
      f"HHI 실제 {pre['a_hhi']:.3f} vs 기대 {pre['e_hhi']:.3f}")
print("\n[결과] 실제 · 기대 · 초과 DiD")
OUT = {"pre_levels": {k: round(pre[k], 4) for k in K}}
for k, lab, s in [("ms", "최장 무채용 spell (개월)", 0.36), ("hhi", "채용 집중도 HHI", 0.025),
                  ("t2", "최다 2개월 채용 비중", 0.025), ("zero", "무채용 월 수", 0.55)]:
    print(f"  -- {lab} --")
    OUT[k] = {"actual": D(f"a_{k}", "   실제"), "expected": D(f"e_{k}", "   기대(총량고정 벤치마크)"),
              "excess": D(f"x_{k}", "   ★ 초과", sesoi=s)}

surv = [k for k in ("ms", "hhi", "t2", "zero") if OUT[k]["excess"]["sig"]]
eqv = [k for k in ("ms", "hhi", "t2", "zero")
       if OUT[k]["excess"].get("equiv", {}).get("holds")]
if surv:
    status = "PARTIAL"; concl = f"총량 고정 후에도 남는 지표: {surv} — 시간적 재배치의 부분 증거"
else:
    status = "KILL"; concl = ("**총량을 고정하면 네 지표 모두 초과분이 사라진다.** 최장 spell·집중도 감소는 "
                              f"전부 채용량 증가의 기계적 결과다. 등가성 성립: {eqv}. "
                              "'조정 시점/cadence 변화' 주장은 자료가 지지하지 않는다.")
verdict = " | ".join(f"{k} 실제 {OUT[k]['actual']['DiD']}{'✓' if OUT[k]['actual']['sig'] else '✗'} "
                     f"기대 {OUT[k]['expected']['DiD']} 초과 {OUT[k]['excess']['DiD']}"
                     f"{'✓' if OUT[k]['excess']['sig'] else '✗'}"
                     for k in ("ms", "hhi", "t2")) + f" || {concl}"
emit("I-39", "spell·집중도 벤치마크 (I-38 완성)", status, OUT,
     "최장 spell 과 집중도에도 총량 고정 벤치마크를 걸어 기계적 성분을 제거한다",
     verdict, kill_met=(status == "KILL"), n=len(EV),
     extra={"conclusion": concl, "R_sim": R,
            "benchmark": "실제 N 건을 고용비중 가중 다항분포로 12개월에 배분, R=100 회 모의"})
