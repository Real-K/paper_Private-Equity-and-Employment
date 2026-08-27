# -*- coding: utf-8 -*-
"""I-40 남은 것을 지킬 수 있는가 — 두 결정적 재검정.

[A] **조절자가 천장효과 인공물인가.**
무채용비중은 T1 에서 p≈0.97 로 천장에 붙어 있어 더 오를 여지가 없다. 그러면 T3>T1 은 인공물이다.
반대로 **비례확장(log 일정)이면 수준 채용률 변화는 사전 수준이 높은 T1 에서 더 커야 한다.**
관측은 T3 +0.1369 vs T1 −0.0017 로 반대다. **로그 결과대상에서 조절자가 살아남는지**가 판별한다.
  · Δlog(채용률) · Δlog(총채용건수) · Δlog(고용) 의 T3−T1
  · 로그에서도 T3 가 크면 → 비례를 넘어선 진짜 이질성 (기계적 아님)
  · 로그에서 사라지면 → 비례확장 + 천장효과. 조절자도 철회

[B] **집중도 null 의 검정력.** 12개월 창은 clustering 추정에 짧다. 36개월 창으로 재검정한다.
표본은 줄지만 기업당 측정 정밀도가 크게 오른다. δ 도 재산정한다.
"""
import numpy as np
from h30_common import load, deals, build, boot_did_ci, emit, SEED, qci, NB, widx

rng = np.random.default_rng(SEED)
rng_fig = np.random.default_rng(SEED + 1)   # 그림용 집단 CI 전용 (주 스트림 불변)
print("[I-40] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Ev = G["Hv"], G["Ev"]

def W(row, m0, a, b):
    c = widx(G, m0, a, b)
    n = b - a + 1
    if len(c) != n: return None
    h, e = Hv[row, c].astype(float), Ev[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.nanmean(e) < 5: return None
    return h, e

def lv(row, m0, a, b):
    """로그 결과대상 3종 + 수준 채용률."""
    w = W(row, m0, a, b)
    if w is None: return None
    h, e = w
    N, E = float(h.sum()), float(np.mean(e))
    rate = N / E
    return dict(rate=rate, lrate=np.log(rate) if rate > 0 else np.nan,
                lN=np.log(N) if N > 0 else np.nan, lE=np.log(E),
                p=float((h > 0).mean()))

def zsh(row, m0, a, b):
    w = W(row, m0, a, b)
    return float((w[0] == 0).mean()) if w else np.nan

for e in EV:
    a, b = lv(e["ti"], e["m0"], -12, -1), lv(e["ti"], e["m0"], 1, 12)
    e["t"] = {k: b[k] - a[k] for k in a} if (a and b) else {}
    e["pre"] = a
    acc = {}
    for c in e["ctrls"]:
        a2, b2 = lv(c, e["m0"], -12, -1), lv(c, e["m0"], 1, 12)
        if a2 and b2:
            for k in a2: acc.setdefault(k, []).append(b2[k] - a2[k])
    e["c"] = {k: float(np.nanmean(v)) for k, v in acc.items() if np.isfinite(v).any()}
    e["pp"] = zsh(e["ti"], e["m0"], -24, -13)
_p = np.array([e["pp"] for e in EV], float)
Q1, Q2 = np.percentile(_p[np.isfinite(_p)], [33.33, 66.67])
for e in EV:
    v = e["pp"]; e["pb"] = None if not np.isfinite(v) else (0 if v <= Q1 else (1 if v <= Q2 else 2))

def D(sub, k):
    t = [x["t"].get(k) for x in sub if k in x["t"] and k in x["c"]]
    c = [x["c"][k] for x in sub if k in x["t"] and k in x["c"]]
    ok = [i for i, v in enumerate(t) if np.isfinite(v) and np.isfinite(c[i])]
    return boot_did_ci([t[i] for i in ok], [c[i] for i in ok], rng)
def diff13(k):
    def d(b):
        return np.array([x["t"][k] - x["c"][k] for x in EV if x["pb"] == b
                         and k in x["t"] and k in x["c"]
                         and np.isfinite(x["t"][k]) and np.isfinite(x["c"][k])], float)
    d1, d3 = d(0), d(2)
    if min(len(d1), len(d3)) < 15: return None
    bs = np.array([d3[rng.integers(0, len(d3), len(d3))].mean()
                   - d1[rng.integers(0, len(d1), len(d1))].mean() for _ in range(NB)])
    ci = qci(bs)
    # 집단별 평균 CI 는 **별도 난수원**을 쓴다 — 주 부트스트랩 스트림을 소비하면 이미 보고한
    # 대비(T3−T1) CI 가 흔들린다. 그림에서 양쪽 패널에 동등하게 오차막대를 붙이기 위한 것.
    gci = lambda v: qci(np.array([v[rng_fig.integers(0, len(v), len(v))].mean()
                                  for _ in range(NB)]))
    return dict(T1=round(float(d1.mean()), 4), T3=round(float(d3.mean()), 4),
                diff=round(float(d3.mean() - d1.mean()), 4), ci=ci,
                ci_T1=gci(d1), ci_T3=gci(d3),
                sig=bool(ci[0] > 0 or ci[1] < 0), n1=len(d1), n3=len(d3))

print("\n[Panel A] ★ 조절자가 천장효과 인공물인가 — 수준 vs 로그")
pre1 = np.mean([e["pre"]["rate"] for e in EV if e["pre"] and e["pb"] == 0])
pre3 = np.mean([e["pre"]["rate"] for e in EV if e["pre"] and e["pb"] == 2])
pp1 = np.mean([e["pre"]["p"] for e in EV if e["pre"] and e["pb"] == 0])
pp3 = np.mean([e["pre"]["p"] for e in EV if e["pre"] and e["pb"] == 2])
print(f"  사전 수준: 채용률 T1 {pre1:.3f} vs T3 {pre3:.3f} · 활동월비중 T1 {pp1:.3f} vs T3 {pp3:.3f}")
print(f"  → 비례확장 가설이면 **수준** 변화는 T1 이 커야 한다 (사전이 {pre1/pre3:.1f}배)")
PA = {"pre_rate_T1": round(float(pre1), 4), "pre_rate_T3": round(float(pre3), 4),
      "pre_p_T1": round(float(pp1), 4), "pre_p_T3": round(float(pp3), 4)}
for k, lab in (("rate", "수준 채용률"), ("lrate", "★ Δlog 채용률"),
               ("lN", "★ Δlog 총채용건수"), ("lE", "★ Δlog 고용"), ("p", "활동월 비중")):
    r = diff13(k)
    if r is None: print(f"  {lab:<20} (표본부족)"); continue
    PA[k] = r
    print(f"  {lab:<20} T1 {r['T1']:>+8.4f} · T3 {r['T3']:>+8.4f} · "
          f"T3−T1 {r['diff']:>+8.4f} {r['ci']} {'✓' if r['sig'] else '✗'} (n {r['n1']}/{r['n3']})")

print("\n[Panel B] 집중도 null 의 검정력 — 36개월 창")
def exc(row, m0, a, b):
    w = W(row, m0, a, b)
    if w is None: return None
    h, e = w; L = b - a + 1
    N = float(h.sum())
    if N == 0: return dict(z=1.0, x=1.0 - 1.0, ms=float(L), xms=0.0)
    wj = e / e.sum()
    z = float((h == 0).mean())
    ez = float(np.sum((1.0 - wj) ** N)) / L
    run = mx = 0
    for x in h:
        run = run + 1 if x == 0 else 0
        mx = max(mx, run)
    sim = rng.multinomial(int(N), wj, size=60).astype(float)
    ems = float(np.mean([max((lambda s: [max(r) for r in [[len(list(g)) for k, g in
        __import__('itertools').groupby(s == 0) if k] or [0]]])(s)) for s in sim]))
    return dict(z=z, x=z - ez, ms=float(mx), xms=float(mx) - ems)
KB = ["z", "x", "ms", "xms"]
n36 = 0
for e in EV:
    a, b = exc(e["ti"], e["m0"], -36, -1), exc(e["ti"], e["m0"], 1, 36)
    e["t36"] = {k: b[k] - a[k] for k in KB} if (a and b) else {}
    if e["t36"]: n36 += 1
    acc = {}
    for c in e["ctrls"]:
        a2, b2 = exc(c, e["m0"], -36, -1), exc(c, e["m0"], 1, 36)
        if a2 and b2:
            for k in KB: acc.setdefault(k, []).append(b2[k] - a2[k])
    e["c36"] = {k: float(np.mean(v)) for k, v in acc.items() if v}
print(f"  36개월 창 가용 이벤트 {n36}/{len(EV)}")
PB = {}
for k, lab, s in (("z", "무채용비중 (실제)", None), ("x", "★ 초과 무채용비중", 0.046),
                  ("ms", "최장 spell (실제)", None), ("xms", "★ 초과 최장 spell", 1.0)):
    t = [x["t36"].get(k) for x in EV if k in x["t36"] and k in x["c36"]]
    c = [x["c36"][k] for x in EV if k in x["t36"] and k in x["c36"]]
    ok = [i for i, v in enumerate(t) if np.isfinite(v) and np.isfinite(c[i])]
    p_, ci, n = boot_did_ci([t[i] for i in ok], [c[i] for i in ok], rng)
    sg = "✓" if (ci and (ci[0] > 0 or ci[1] < 0)) else ("✗" if ci else "-")
    o = {"DiD": p_, "ci": ci, "n": n, "sig": sg == "✓"}
    ex = ""
    if s and ci and not o["sig"]:
        mg = [round(ci[0]+s, 4), round(s-ci[1], 4)]
        o["equiv"] = {"SESOI": s, "holds": bool(ci[0] > -s and ci[1] < s), "margin": mg}
        ex = f"  등가성δ={s} {'✓' if o['equiv']['holds'] else '✗'}"
    PB[k] = o
    print(f"  {lab:<26} {str(p_):>9} {str(ci):<21} {sg} (n={n}){ex}")

lr = PA.get("lrate", {}); ln_ = PA.get("lN", {})
survives = bool(lr.get("sig") and (lr.get("diff") or 0) > 0)
if survives:
    concl = ("**조절자는 천장효과 인공물이 아니다.** 로그 채용률에서도 T3 가 유의하게 크다 — "
             "비례확장을 넘어선 진짜 이질성이다. 논문의 중심을 '상태의존적 채용량 증가'로 옮기면 살아남는다.")
    status = "GO"
else:
    concl = ("로그 결과대상에서 조절자가 유의하지 않다 — 수준 조절자는 사전 수준 차이로 설명될 "
             "가능성이 있다. 조절자 주장도 하향해야 한다.")
    status = "PARTIAL"
verdict = (f"[A] Δlog채용률 T3−T1 {lr.get('diff')}{lr.get('ci')}{'✓' if lr.get('sig') else '✗'} · "
           f"Δlog총채용 {ln_.get('diff')}{'✓' if ln_.get('sig') else '✗'} · "
           f"Δlog고용 {PA.get('lE',{}).get('diff')}{'✓' if PA.get('lE',{}).get('sig') else '✗'} | "
           f"[B] 36개월 초과무채용 {PB['x']['DiD']}{PB['x']['ci']}"
           f"{'✓' if PB['x']['sig'] else '✗'} · 초과최장spell {PB['xms']['DiD']}"
           f"{'✓' if PB['xms']['sig'] else '✗'} (n={PB['x']['n']}) | {concl}")
emit("I-40", "조절자 천장효과 검정 + 36개월 집중도", status,
     {"panelA_log_moderator": PA, "panelB_36m_clustering": PB, "n_36m": n36},
     "조절자가 로그 결과대상에서도 살아남으면 천장효과 인공물이 아니다. 36개월 창으로 집중도 null 의 검정력을 올린다",
     verdict, kill_met=False, n=len(EV), extra={"conclusion": concl})
