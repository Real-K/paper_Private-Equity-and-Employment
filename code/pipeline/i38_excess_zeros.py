# -*- coding: utf-8 -*-
"""I-38 ★ 빈도가 기계적 부산물인가 — 리뷰 Major Comment 1 에 대한 결정적 검정.

[지적] 고용이 8.8% 늘고 이직이 불변이면 총채용은 산술적으로 는다. 월별 채용건수가 Poisson 이면
무채용 확률은 e^{-λ} 이므로, **λ 가 커지는 것만으로 무채용 월은 줄어든다.** 따라서 외연마진 반응이
'조정 시점의 변화'가 아니라 '규모 증가의 기계적 결과'일 수 있다.

[검정] 각 기업-창에서 **그 기업의 실제 총채용건수를 12개월에 무작위 배분**했을 때 기대되는
무채용 월 수를 계산하고, 실제와의 차이(excess zeros)를 본다. 총량을 고정했으므로 excess 는
**시간적 집중도(clustering)만** 측정한다.

  E[무채용 월 | N건을 12개월에 균등 무작위 배분] = 12 · (11/12)^N          … 균등 벤치마크
  고용 가중: 각 월의 배분확률을 그 달 고용 비중으로            … 노출 보정 벤치마크

Panel A  actual · expected · excess 의 사전/사후 수준과 DiD (균등·고용가중)
Panel B  최장 무채용 spell · 상위분위 spell (리뷰 제안 결과대상)
Panel C  채용 집중도 (HHI over months) · 최다 2개월 집중비중
Panel D  사전 관성 분위별 excess DiD — 조절이 excess 에서도 살아남는가

기각조건: excess zeros DiD 가 0 이면 외연마진 반응은 규모 증가의 기계적 결과이고,
논문의 'cadence' 주장은 성립하지 않는다. 그 경우 정직하게 서술을 바꾼다.
"""
import numpy as np
from h30_common import load, deals, build, boot_did_ci, emit, SEED, qci, NB, widx

rng = np.random.default_rng(SEED)
print("[I-38] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Ev = G["Hv"], G["Ev"]

def win(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != 12: return None
    h, e = Hv[row, c], Ev[row, c]
    if not (np.isfinite(h).all() and np.isfinite(e).all()): return None
    if np.nanmean(e) < 5: return None
    return h.astype(float), e.astype(float)

def stats(row, m0, a, b):
    w = win(row, m0, a, b)
    if w is None: return None
    h, e = w
    N = float(h.sum()); zero = float((h == 0).sum())
    # 균등 벤치마크
    exp_u = 12.0 * (11.0 / 12.0) ** N
    # 고용가중 벤치마크: 월 j 에 한 건도 안 갈 확률 = (1 - w_j)^N
    wj = e / e.sum() if e.sum() > 0 else np.full(12, 1/12)
    exp_w = float(np.sum((1.0 - wj) ** N))
    # 최장 spell
    run = mx = 0
    for x in h:
        run = run + 1 if x == 0 else 0
        mx = max(mx, run)
    # 집중도
    hhi = float(np.sum((h / N) ** 2)) if N > 0 else np.nan
    top2 = float(np.sort(h)[-2:].sum() / N) if N > 0 else np.nan
    return dict(N=N, zero=zero / 12, exp_u=exp_u / 12, exp_w=exp_w / 12,
                exc_u=(zero - exp_u) / 12, exc_w=(zero - exp_w) / 12,
                maxspell=float(mx), hhi=hhi, top2=top2)

K = ["zero", "exp_u", "exp_w", "exc_u", "exc_w", "maxspell", "hhi", "top2", "N"]
for e in EV:
    a, b = stats(e["ti"], e["m0"], -12, -1), stats(e["ti"], e["m0"], 1, 12)
    e["t"] = {k: (b[k] - a[k]) for k in K} if (a and b) else {}
    e["tpre"] = a
    acc = {}
    for c in e["ctrls"]:
        a2, b2 = stats(c, e["m0"], -12, -1), stats(c, e["m0"], 1, 12)
        if a2 and b2:
            for k in K: acc.setdefault(k, []).append(b2[k] - a2[k])
    e["c"] = {k: float(np.nanmean(v)) for k, v in acc.items() if v}
    z = stats(e["ti"], e["m0"], -24, -13)
    e["pp"] = z["zero"] if z else np.nan
_p = np.array([e["pp"] for e in EV], float)
Q1, Q2 = np.percentile(_p[np.isfinite(_p)], [33.33, 66.67])
for e in EV:
    v = e["pp"]; e["pb"] = None if not np.isfinite(v) else (0 if v <= Q1 else (1 if v <= Q2 else 2))

def D(sub, k, lab, sesoi=None):
    t = [x["t"].get(k) for x in sub if k in x["t"] and k in x["c"]]
    c = [x["c"][k] for x in sub if k in x["t"] and k in x["c"]]
    ok = [i for i, v in enumerate(t) if np.isfinite(v) and np.isfinite(c[i])]
    p_, ci, n = boot_did_ci([t[i] for i in ok], [c[i] for i in ok], rng)
    sg = "✓" if (ci and (ci[0] > 0 or ci[1] < 0)) else ("✗" if ci else "-")
    o = {"DiD": p_, "ci": ci, "n": n, "sig": sg == "✓"}
    ex = ""
    if sesoi and ci and not o["sig"]:
        mg = [round(ci[0]+sesoi, 4), round(sesoi-ci[1], 4)]
        o["equiv"] = {"SESOI": sesoi, "holds": bool(ci[0] > -sesoi and ci[1] < sesoi), "margin": mg}
        ex = f"  등가성δ={sesoi} {'✓' if o['equiv']['holds'] else '✗'}"
    print(f"  {lab:<40} {str(p_):>9} {str(ci):<21} {sg} (n={n}){ex}")
    return o

pre_lv = {k: float(np.nanmean([e["tpre"][k] for e in EV if e["tpre"]])) for k in K}
print(f"\n  처치 사전 수준: 무채용비중 {pre_lv['zero']:.4f} · 균등기대 {pre_lv['exp_u']:.4f} "
      f"· 고용가중기대 {pre_lv['exp_w']:.4f} · excess(균등) {pre_lv['exc_u']:+.4f}")
print(f"  → 실제 무채용이 무작위배분 기대보다 {pre_lv['exc_u']*12:+.2f}개월 많다 = 사전 clustering 존재")

print("\n[Panel A] ★ 실제 · 기대 · 초과 무채용비중 DiD")
PA = {"pre_levels": {k: round(pre_lv[k], 4) for k in K}}
PA["actual"] = D(EV, "zero", "실제 무채용비중 (기존 헤드라인)")
PA["expected_uniform"] = D(EV, "exp_u", "기대 무채용비중 — 균등 벤치마크")
PA["expected_wgt"] = D(EV, "exp_w", "기대 무채용비중 — 고용가중 벤치마크")
PA["excess_uniform"] = D(EV, "exc_u", "★ 초과 무채용비중 (균등)", sesoi=0.046)
PA["excess_wgt"] = D(EV, "exc_w", "★ 초과 무채용비중 (고용가중)", sesoi=0.046)
PA["total_hires"] = D(EV, "N", "총 채용건수 (12개월)")

print("\n[Panel B] spell 과 집중도")
PB = {"maxspell": D(EV, "maxspell", "최장 무채용 spell (개월)"),
      "hhi": D(EV, "hhi", "채용 월별 집중도 HHI", sesoi=0.05),
      "top2": D(EV, "top2", "최다 2개월 채용 비중", sesoi=0.05)}

print("\n[Panel C] 사전 관성 분위별 초과 무채용")
PC = {}
for b, bl in ((0, "T1 저관성"), (2, "T3 고관성")):
    sub = [e for e in EV if e["pb"] == b]
    PC[bl] = {"actual": D(sub, "zero", f"{bl} 실제"), "excess": D(sub, "exc_u", f"{bl} ★초과")}
d1 = np.array([e["t"]["exc_u"] - e["c"]["exc_u"] for e in EV
               if e["pb"] == 0 and "exc_u" in e["t"] and "exc_u" in e["c"]], float)
d3 = np.array([e["t"]["exc_u"] - e["c"]["exc_u"] for e in EV
               if e["pb"] == 2 and "exc_u" in e["t"] and "exc_u" in e["c"]], float)
d1, d3 = d1[np.isfinite(d1)], d3[np.isfinite(d3)]
bs = np.array([d3[rng.integers(0, len(d3), len(d3))].mean()
               - d1[rng.integers(0, len(d1), len(d1))].mean() for _ in range(NB)])
ci = qci(bs); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
PC["T3_T1_excess"] = {"diff": round(float(d3.mean()-d1.mean()), 4), "ci": ci, "sig": sg == "✓"}
print(f"  {'T3−T1 초과 무채용':<40} {d3.mean()-d1.mean():>+9.4f} {str(ci):<21} {sg}")

# ---- 판정 ----
ex = PA["excess_uniform"]; exw = PA["excess_wgt"]
mech = (not ex["sig"]) and (not exw["sig"])
if ex["sig"] and ex["DiD"] < 0:
    status = "GO"; concl = ("**빈도 효과는 기계적이지 않다.** 총채용을 고정한 무작위배분 벤치마크 대비 "
                            "초과 무채용이 유의하게 줄었다 — 시간적 집중도 자체가 바뀌었다.")
elif mech and ex.get("equiv", {}).get("holds"):
    status = "KILL"; concl = ("**빈도 효과는 규모 증가로 설명된다.** 총채용 고정 후 초과 무채용이 "
                              "등가성 범위 안이다. 'cadence' 주장을 철회하고 외연마진을 "
                              "'규모 증가의 발현 형태'로 서술해야 한다.")
else:
    status = "PARTIAL"; concl = ("초과 무채용 변화를 검출하지 못했으나 등가성도 미성립 — "
                                 "기계적 설명을 배제하지 못한다. 주장 하향 필요.")
verdict = (f"실제 {PA['actual']['DiD']} vs 기대(균등) {PA['expected_uniform']['DiD']} → "
           f"★초과 {ex['DiD']}{ex['ci']}{'✓' if ex['sig'] else '✗'} · "
           f"고용가중 초과 {exw['DiD']}{exw['ci']}{'✓' if exw['sig'] else '✗'} | "
           f"최장spell {PB['maxspell']['DiD']}{'✓' if PB['maxspell']['sig'] else '✗'} | "
           f"총채용 {PA['total_hires']['DiD']} | {concl}")
emit("I-38", "빈도의 기계성 검정 (리뷰 Major Comment 1)", status,
     {"panelA_excess": PA, "panelB_spell_concentration": PB, "panelC_by_inertia": PC,
      "benchmark": "E[zero months] = 12·(11/12)^N (균등) 또는 Σ(1−w_j)^N (고용가중). "
                   "총채용 N 을 고정하므로 초과분은 시간적 집중도만 측정한다."},
     "총채용을 고정한 무작위배분 벤치마크 대비 초과 무채용이 줄면 빈도 효과는 기계적이지 않다",
     verdict, kill_met=(status == "KILL"), n=len(EV), extra={"conclusion": concl})
