# -*- coding: utf-8 -*-
"""I-68 분기 채용률 이벤트 경로 (기준 매칭 설계).

Figure 1(a) 는 지금까지 '월중 채용 확률'(I-32 Panel B)을 그렸는데, 논문 §7 스스로
월별 extensive-margin 지표는 물량 증가의 산술적 귀결이라 독립적 timing 반응으로
읽으면 안 된다고 결론짓는다. 첫 그림의 첫 패널이 논문이 나중에 할인하는 변수를 쓰는 셈이다.
같은 기준 매칭 설계(379)에서 **주 결과변수 계열인 채용률**의 분기 경로를 낸다.

y_q = (분기 q 채용수 / 분기 q 평균고용) 의 처치 − 대조평균,
      사전 4분기 평균을 뺀 값 (사전기준 정규화).
CI 는 이벤트 부트스트랩 2,000회.
"""
import numpy as np
from h30_common import load, deals, build, emit, SEED, qci, widx
rng = np.random.default_rng(SEED); NB = 2000
print("[I-68] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Ev = G["Hv"], G["Ev"]
def qrate(row, m0, q):
    """분기 q 의 채용률. q=-4..-1 은 사전, q=1..12 는 사후."""
    a = (q-1)*3 + 1 if q > 0 else q*3 + 1
    b = a + 2
    c = widx(G, m0, a, b)
    if len(c) != 3: return np.nan
    h = Hv[row, c].astype(float); e = Ev[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.mean(e) < 5: return np.nan
    return float(h.sum()/np.mean(e))
QS = list(range(-4, 0)) + list(range(1, 13))
rows = []
for ev in EV:
    ti, ctr, m0 = ev["ti"], list(ev["ctrls"]), ev["m0"]
    if not len(ctr): continue
    d = {}
    for q in QS:
        t = qrate(ti, m0, q)
        cs = [qrate(int(c), m0, q) for c in ctr]
        cs = [v for v in cs if np.isfinite(v)]
        d[q] = (t - float(np.mean(cs))) if (np.isfinite(t) and cs) else np.nan
    pre = [d[q] for q in (-4, -3, -2, -1) if np.isfinite(d[q])]
    if len(pre) < 4: continue
    base = float(np.mean(pre))
    rows.append({q: (d[q]-base) for q in QS})
print(f"  이벤트 {len(rows)}건")
M = np.array([[r[q] for q in QS] for r in rows], float)
def curve(idx):
    sub = M[idx]
    return np.array([np.nanmean(sub[:, j]) if np.isfinite(sub[:, j]).sum() >= 20 else np.nan
                     for j in range(len(QS))])
obs = curve(np.arange(len(M)))
B = np.empty((NB, len(QS)))
for i in range(NB):
    B[i] = curve(rng.integers(0, len(M), len(M)))
beta = {}
for j, q in enumerate(QS):
    col = B[:, j][np.isfinite(B[:, j])]
    if not np.isfinite(obs[j]) or len(col) < 100: continue
    lo, hi = qci(col)
    beta[f"q{q}"] = {"b": round(float(obs[j]), 4), "ci": [lo, hi],
                     "n": int(np.isfinite(M[:, j]).sum()), "sig": bool(lo > 0 or hi < 0)}
    print(f"    q{q:>3}  {obs[j]:+.4f}  [{lo:+.4f}, {hi:+.4f}]  n={beta[f'q{q}']['n']}"
          f"  {'✓' if beta[f'q{q}']['sig'] else ''}")
pre_max = max(abs(beta[f"q{q}"]["b"]) for q in (-4, -3, -2, -1) if f"q{q}" in beta)
post = [beta[f"q{q}"]["b"] for q in range(1, 13) if f"q{q}" in beta]
first_sig = next((q for q in range(1, 13) if beta.get(f"q{q}", {}).get("sig")), None)
S = {"beta": beta, "n_ev": len(rows), "n_boot": NB,
     "pre_max_abs": round(float(pre_max), 4),
     "post_min": round(float(min(post)), 4), "post_max": round(float(max(post)), 4),
     "first_significant_quarter": first_sig,
     "outcome": "quarterly hires / mean quarterly employment, treated minus control mean, "
                "normalized to the mean of quarters -4..-1",
     "design": "기준 매칭 설계 (build()), 상태균형 아님 — 평균효과용"}
emit("I-68", "분기 채용률 이벤트 경로 (Figure 1a 용)", "GO" if first_sig else "PARTIAL", S,
     "주 결과변수 계열에서도 사전 평탄 · 사후 상승 형태가 나오는가",
     f"사전 최대 |{pre_max:.4f}| · 사후 {min(post):+.4f}~{max(post):+.4f} · "
     f"최초 유의 분기 q{first_sig}. n={len(rows)}.",
     kill_met=False, n=len(rows))
