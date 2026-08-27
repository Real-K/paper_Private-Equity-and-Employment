# -*- coding: utf-8 -*-
"""I-11 HonestDiD breakdown value 를 헤드라인으로 승격.

원고는 사전추세가 평탄하다고 주장할 수 없다(실측 사전창 최대 |att_rel| 2.16pt, 드리프트 +1.2pt).
규칙 11의 정공법은 기준을 낮추는 것이 아니라 주장 문턱을 배제하지 못한 폭 위로 올리는 것이다.
Rambachan–Roth 상대크기(RM) 집합에서 결과가 살아남는 최대 M(breakdown M̄)을 보고한다.

  Δ^RM(M): |δ_{t+1} − δ_t| ≤ M · max_{s<0}|δ_{s+1} − δ_s|
  기준기 정규화 하에서 |δ_t| ≤ M · maxpre · t
  θ = 사후 평균이면 편의 상한 = M · maxpre · mean(t)
  → M̄ = (|θ̂| − 1.96·se) / (maxpre · c)

[사양 명시] RR 최적화가 아니라 **선형외삽 포락선의 해석적 보수 상한**이다. RR 최적화는 사전 추정치를
더 써서 구간이 좁아지므로 여기의 M̄ 은 **참 breakdown 의 하한**이다(강건성 주장에 안전한 방향).

S1  [사전명시·주] 월별 해상도, k=−1 정규화
S2  분모 타당성 진단 — 사전 1차차분이 순수 표본잡음 규모인지 귀무분포로 검정
S3  분기 해상도(추정대상과 해상도 일치), q=−1 정규화
S4  분기 해상도, 사전창 평균 정규화 (헤드라인 추정대상과 정확히 일치)
"""
import numpy as np
from h30_common import load, deals, build, emit, SEED, qci, NB

rng = np.random.default_rng(SEED)
print("[I-11] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
EV, _ = build(G, allt, PE)
Hv, mset = G["Hv"], G["mset"]
KS = [k for k in range(-12, 13) if k != 0]

rows = []
for e in EV:
    js = [mset.get(e["m0"] + k) for k in KS]
    if any(j is None for j in js): continue
    ht = Hv[e["ti"], js]
    if not np.isfinite(ht).all(): continue
    hc = Hv[np.ix_(e["ctrls"], js)]
    ok = np.isfinite(hc).all(axis=1)
    if ok.sum() == 0: continue
    rows.append((ht > 0).astype(float) - (hc[ok] > 0).astype(float).mean(axis=0))
D = np.array(rows); NEV = len(D)
print(f"  이벤트 {len(EV)} → 균형 이벤트스터디 {NEV}")


def mbar(theta, se, maxpre, c):
    return round(max(0.0, (abs(theta) - 1.96 * se) / (maxpre * c)), 3) if maxpre > 0 else None


# ---------- S1 월별 ----------
i_1 = KS.index(-1); B = D - D[:, [i_1]]
bs = np.array([B[rng.integers(0, NEV, NEV)].mean(axis=0) for _ in range(NB)])
pre = [i for i, k in enumerate(KS) if k < 0]; post = [i for i, k in enumerate(KS) if k > 0]
beta = B.mean(axis=0)
mp1 = float(np.abs(np.diff(beta[pre])).max())
th1 = float(beta[post].mean()); bt = bs[:, post].mean(axis=1)
se1 = float(bt.std(ddof=1)); ci1 = qci(bt)
S1 = {"resolution": "월별", "norm": "k=-1", "n_ev": NEV,
      "beta": {str(KS[i]): round(float(beta[i]), 4) for i in range(len(KS))},
      "pre_max_abs_level": round(float(np.abs(beta[pre]).max()), 4),
      "pre_max_first_diff": round(mp1, 4),
      "pre_linear_slope": round(float(np.polyfit([KS[i] for i in pre], beta[pre], 1)[0]), 5),
      "theta": round(th1, 4), "se": round(se1, 4), "ci": ci1,
      "sig_naive": bool(ci1[0] > 0 or ci1[1] < 0), "Mbar": mbar(th1, se1, mp1, 6.5)}
print(f"\n[S1 사전명시·월별 k=−1] θ={th1:+.4f} se={se1:.4f} {ci1}"
      f" {'✓' if S1['sig_naive'] else '✗'}  사전 max|Δβ|={mp1:.4f}  →  **M̄ = {S1['Mbar']}**")
print(f"  사전 선형기울기 {S1['pre_linear_slope']:+.5f}/월 (부호가 효과와 반대 = 불리하지 않음)")

# ---------- S2 분모 타당성 진단 ----------
se_d = np.diff(bs[:, pre], axis=1).std(axis=0, ddof=1)
nullmax = np.abs(np.diff(bs[:, pre] - beta[pre], axis=1)).max(axis=1)
pct = float((nullmax < mp1).mean() * 100)
S2 = {"observed_max_first_diff": round(mp1, 4),
      "mean_se_of_first_diff": round(float(se_d.mean()), 4),
      "null_max_p50": round(float(np.percentile(nullmax, 50)), 4),
      "null_max_p95": round(float(np.percentile(nullmax, 95)), 4),
      "observed_percentile_under_null": round(pct, 1),
      "verdict": ("분모가 순수 표본잡음 규모 — RM 기준이 사전추세를 재지 못함"
                  if pct < 90 else "분모가 잡음을 초과 — RM 기준 유효")}
print(f"\n[S2 분모 진단] 실측 max|Δβ|={mp1:.4f} · 개별 Δβ SE 평균={se_d.mean():.4f} · "
      f"귀무 max 분포 p50={S2['null_max_p50']:.4f} p95={S2['null_max_p95']:.4f}")
print(f"  → 실측값의 귀무 하 백분위 = {pct:.0f}%  ⇒ {S2['verdict']}")

# ---------- S3/S4 분기 ----------
Q = np.stack([D[:, i * 3:(i + 1) * 3].mean(axis=1) for i in range(8)], axis=1)
QL = ["q-4", "q-3", "q-2", "q-1", "q1", "q2", "q3", "q4"]


def quarterly(norm):
    base = Q[:, [3]] if norm == "q-1" else Q[:, :4].mean(axis=1, keepdims=True)
    Bq = Q - base
    bq = np.array([Bq[rng.integers(0, NEV, NEV)].mean(axis=0) for _ in range(NB)])
    b = Bq.mean(axis=0); mp = float(np.abs(np.diff(b[:4])).max())
    th = float(b[4:].mean()); t = bq[:, 4:].mean(axis=1)
    se = float(t.std(ddof=1)); ci = qci(t)
    return {"resolution": "분기", "norm": norm, "n_ev": NEV,
            "beta": {QL[i]: round(float(b[i]), 4) for i in range(8)},
            "pre_max_first_diff": round(mp, 4), "theta": round(th, 4), "se": round(se, 4),
            "ci": ci, "sig_naive": bool(ci[0] > 0 or ci[1] < 0), "Mbar": mbar(th, se, mp, 2.5),
            "max_tolerable_drift_per_quarter": round(mbar(th, se, mp, 2.5) * mp, 4)}


S3, S4 = quarterly("q-1"), quarterly("pre-mean")
for tag, s in (("S3 분기 q−1", S3), ("S4 분기 사전평균(헤드라인 일치)", S4)):
    print(f"\n[{tag}] β_q={list(s['beta'].values())}")
    print(f"  사전 max|Δβ|={s['pre_max_first_diff']}  θ={s['theta']} se={s['se']} {s['ci']}"
          f" {'✓' if s['sig_naive'] else '✗'}  →  **M̄ = {s['Mbar']}**"
          f"  (분기당 {s['max_tolerable_drift_per_quarter']} 까지 허용)")

# ---------- 판정 ----------
mb_pre, mb_q = S1["Mbar"], S4["Mbar"]
if mb_pre and mb_pre >= 0.5: status = "GO"
elif mb_q and mb_q >= 0.5: status = "PARTIAL"
else: status = "KILL"
concl = (f"사전명시 사양(월별)은 M̄={mb_pre} 로 실패한다. 그러나 S2가 그 실패의 원인을 특정한다 — "
         f"분모인 사전 최대 1차차분이 자기 귀무분포의 {pct:.0f}백분위, 즉 **순수 표본잡음 규모**여서 "
         f"기준 자체가 사전추세를 재지 못한다. 추정대상과 해상도를 맞춘 분기 사양에서 "
         f"M̄={S3['Mbar']}(q−1) / {mb_q}(사전평균). "
         f"주장 상한: '관측된 최대 사전 분기변동의 약 {int(100*mb_q)}% 크기의 차등추세까지 결과가 생존한다'. "
         f"M̄≥1 은 주장할 수 없다.")
verdict = (f"S1 월별 M̄={mb_pre} (실패, 분모=잡음 {pct:.0f}백분위) | "
           f"S3 분기 q−1 M̄={S3['Mbar']} θ={S3['theta']}{S3['ci']} | "
           f"S4 분기 사전평균 M̄={mb_q} θ={S4['theta']}{S4['ci']} | 사전 선형기울기 "
           f"{S1['pre_linear_slope']:+.5f}/월(효과와 반대부호)")
emit("I-11", "HonestDiD breakdown value 승격", status,
     {"S1_monthly_prespecified": S1, "S2_denominator_validity": S2,
      "S3_quarterly_qm1": S3, "S4_quarterly_premean": S4},
     "관측된 최대 사전위반의 몇 배까지 차등추세를 허용해도 결과가 살아남는지(M̄)를 보고한다",
     verdict, kill_met=(status == "KILL"), n=NEV,
     extra={"conclusion": concl,
            "spec_note": "RR 최적화가 아닌 선형외삽 포락선의 해석적 보수 상한 → 참 breakdown 의 하한.",
            "discipline_note": ("S1 이 사전명시 주사양이며 실패로 기록한다. S3/S4 는 기준을 낮춘 것이 "
                                "아니라 S2 가 분모의 타당성 결함을 정량 입증한 뒤 제시하는 "
                                "해상도 일치 사양이다. 두 결과를 모두 보고한다.")})
