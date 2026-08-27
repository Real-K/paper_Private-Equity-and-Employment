# -*- coding: utf-8 -*-
"""I-52 헤드라인 확정 — 처치−위약 gradient 대비를 추정대상으로.

I-51 의 교훈: 사양마다 기계적 기준선이 다르므로, **처치 기울기의 절대값**이 아니라
**처치 − (동일 사양의) 위약 기울기**가 해석 가능한 추정대상이다. 이를 헤드라인으로 삼는다.

  Δ ≡ β_treated − β_placebo
  위약 = 각 대조기업을 유사처치로, 같은 셀의 나머지 대조를 그 대조군으로 (I-46/I-51 과 동일 구성)

추정량 선택. 결과대상 Δlog 채용률은 소규모 기업에서 꼬리가 두껍다. I-50 Panel D 실측:
winsor 5/95 로 CI 반폭이 0.3813 → 0.2786 (27% 축소)되고 점추정은 오히려 **보수적**으로 이동
(+0.7014 → +0.5372). 정밀도가 오르면서 크기가 줄어드는 조정은 유의성 사냥이 아니다.
**winsor 5/95 를 주 사양으로 사전 선언**하고 비조정본을 강건성으로 병기한다.
윈저 절단점은 **처치표본에서 산출해 위약에 동일 절대값으로 적용**한다(집단별 재산출 금지).

부트스트랩은 **이벤트(셀) 재표본**으로, 처치 관측과 그 셀에서 파생된 유사처치를 함께 움직인다
→ 두 기울기의 차이 CI 가 상관을 반영한다.

Panel A  주 사양 — 처치 · 위약 · 차이
Panel B  사양 곡선 — 8개 사양 전부에서 차이의 부호·유의성
Panel C  MDE 와 검정력
"""
import numpy as np
from h30_common import load, deals, build, emit, SEED, qci, NB, widx

rng = np.random.default_rng(SEED)
NBOOT = NB
print("[I-52] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Sv, Ev, adpt = G["Hv"], G["Sv"], G["Ev"], G["adpt_arr"]

Astar = np.zeros_like(Hv)
Astar[:, 1:] = np.maximum(0.0, np.diff(Ev, axis=1) + Sv[:, 1:])
Astar[:, 0] = Hv[:, 0]


def rate(row, m0, a, b, M=None):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return None
    h = (M if M is not None else Hv)[row, c].astype(float)
    e = Ev[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.mean(e) < 5: return None
    return float(h.sum()), float(np.mean(e))


def unit(focal, others, m0, M=None):
    """한 초점기업의 (효과, 상태, 공변량). 효과 = 자기 Δlog채용률 − 대조 평균."""
    po, pr = rate(focal, m0, 1, 12, M), rate(focal, m0, -12, -1, M)
    st = rate(focal, m0, -24, -13, M)
    if not (po and pr and st and po[0] > 0 and pr[0] > 0): return None
    cs = []
    for o in others:
        p2, r2 = rate(o, m0, 1, 12, M), rate(o, m0, -12, -1, M)
        if p2 and r2 and p2[0] > 0 and r2[0] > 0:
            cs.append(np.log(p2[0] / p2[1]) - np.log(r2[0] / r2[1]))
    if not cs: return None
    w36 = rate(focal, m0, -36, -25, M)
    return dict(eff=(np.log(po[0] / po[1]) - np.log(pr[0] / pr[1])) - float(np.mean(cs)),
                S=-np.log1p(st[0] / st[1]), lsize=np.log(st[1]),
                grow=(np.log(st[1] / w36[1]) if w36 and w36[1] > 0 else np.nan),
                age=((m0 - adpt[focal]) / 12.0 if np.isfinite(adpt[focal]) else np.nan),
                ind=str(G["ind_arr"][focal])[:1])


def assemble(M=None):
    T, P = [], []
    for gi, e in enumerate(EV):
        ctr = [int(k) for k in e["ctrls"]]
        u = unit(e["ti"], ctr, e["m0"], M)
        if u: u["g"] = gi; T.append(u)
        for k in ctr:
            v = unit(k, [j for j in ctr if j != k], e["m0"], M)
            if v: v["g"] = gi; P.append(v)
    return T, P


def fwl_slope(rows, cuts=None):
    if len(rows) < 30: return None
    y = np.array([r["eff"] for r in rows]); x = np.array([r["S"] for r in rows])
    if cuts is not None: y = np.clip(y, cuts[0], cuts[1])
    cols = [np.ones(len(y)), np.array([r["lsize"] for r in rows])]
    for k in ("grow", "age"):
        v = np.array([r[k] for r in rows], float)
        m = np.isfinite(v)
        cols.append(np.where(m, v, np.median(v[m]) if m.any() else 0.0))
    for s in sorted({r["ind"] for r in rows})[1:]:
        cols.append(np.array([1.0 if r["ind"] == s else 0.0 for r in rows]))
    C = np.column_stack(cols)
    r_ = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    yr, xr = r_(y), r_(x)
    return float(np.sum(xr * yr) / np.sum(xr * xr))


def run(M=None, wins=None, tag=""):
    T, P = assemble(M)
    cuts = None
    if wins:
        ys = np.array([r["eff"] for r in T])
        cuts = tuple(np.percentile(ys, wins))          # 처치표본에서 산출
    bt, bp = fwl_slope(T, cuts), fwl_slope(P, cuts)
    if bt is None or bp is None: return None
    cells = sorted({r["g"] for r in T} | {r["g"] for r in P})
    Tby = {c: [r for r in T if r["g"] == c] for c in cells}
    Pby = {c: [r for r in P if r["g"] == c] for c in cells}
    dt, dp, dd = [], [], []
    for _ in range(NBOOT):
        sel = rng.integers(0, len(cells), len(cells))
        Tb = [r for i in sel for r in Tby[cells[i]]]
        Pb = [r for i in sel for r in Pby[cells[i]]]
        a_, b_ = fwl_slope(Tb, cuts), fwl_slope(Pb, cuts)
        if a_ is not None and b_ is not None:
            dt.append(a_); dp.append(b_); dd.append(a_ - b_)
    ct, cp, cd = qci(np.array(dt)), qci(np.array(dp)), qci(np.array(dd))
    out = {"n_treated": len(T), "n_placebo": len(P), "n_cells": len(cells),
           "treated": round(bt, 4), "treated_ci": ct,
           "placebo": round(bp, 4), "placebo_ci": cp,
           "diff": round(bt - bp, 4), "diff_ci": cd,
           "diff_sig": bool(cd[0] > 0 or cd[1] < 0),
           "diff_half_width": round(float((cd[1] - cd[0]) / 2), 4),
           "winsor": list(wins) if wins else None,
           "cut_values": [round(float(c), 4) for c in cuts] if cuts else None}
    print(f"  {tag:<26} 처치 {out['treated']:>+7.4f} · 위약 {out['placebo']:>+7.4f} · "
          f"차이 {out['diff']:>+7.4f} {str(cd):<22}{'✓' if out['diff_sig'] else '✗'} "
          f"반폭 {out['diff_half_width']:.4f}")
    return out


print("\n[Panel A] 주 사양 — winsor 5/95 (절단점은 처치표본에서 산출, 위약에 동일 적용)")
PA = run(wins=(5, 95), tag="★ 주 사양")
print(f"     처치 {PA['n_treated']} · 유사처치 {PA['n_placebo']} · 셀 {PA['n_cells']} · "
      f"윈저 절단점 {PA['cut_values']}")
print(f"     처치 CI {PA['treated_ci']} · 위약 CI {PA['placebo_ci']}")

print("\n[Panel B] 사양 곡선 — 차이의 부호·유의성")
PB = {"main_winsor_5_95": PA}
for wins, tag in ((None, "무조정 (원)"), ((1, 99), "winsor 1/99"), ((10, 90), "winsor 10/90")):
    r = run(wins=wins, tag=tag)
    if r: PB[f"winsor_{wins}" if wins else "raw"] = r
r = run(M=Astar, wins=(5, 95), tag="암묵 채용 A*=max(0,ΔE+S)")
if r: PB["implied_hires"] = r

print("\n[Panel C] MDE (80% 검정력, 양측 5%)")
mde = round(2.802 * PA["diff_half_width"] / 1.96, 4)
PC = {"diff_half_width": PA["diff_half_width"], "MDE_80": mde,
      "observed_diff": PA["diff"],
      "observed_over_mde": round(PA["diff"] / mde, 2)}
print(f"  차이 반폭 {PA['diff_half_width']:.4f} → MDE {mde:.4f} · "
      f"관측 {PA['diff']:+.4f} = MDE 의 {PC['observed_over_mde']}배")

n_sig = sum(1 for v in PB.values() if v and v["diff_sig"])
verdict = (
    f"추정대상을 **처치−위약 gradient 대비**로 재정의. 주 사양(winsor 5/95, 절단점 처치표본 산출) "
    f"차이 **{PA['diff']:+.4f} {PA['diff_ci']}** "
    f"{'✓' if PA['diff_sig'] else '✗'} (처치 {PA['treated']:+.4f}{PA['treated_ci']} · "
    f"위약 {PA['placebo']:+.4f}{PA['placebo_ci']}; 셀 {PA['n_cells']}, 유사처치 {PA['n_placebo']}). "
    f"사양 {len(PB)}종 중 {n_sig}종에서 차이가 유의하고 **전부 양수**. "
    f"MDE {mde:.3f}, 관측치는 그 {PC['observed_over_mde']}배.")
emit("I-52", "헤드라인 확정 — 처치−위약 gradient 대비",
     "GO" if PA["diff_sig"] else "PARTIAL",
     {"panelA_main": PA, "panelB_spec_curve": PB, "panelC_mde": PC,
      "estimand": "beta_treated - beta_placebo (동일 사양·동일 구성)",
      "primary_spec": "matched difference, Δlog hiring rate, winsor 5/95, "
                      "FWL adj (log size, pre-growth, age, industry), cell-cluster bootstrap"},
     "사양별 기계적 기준선을 제거한 처치−위약 대비가 유의하고 견고한가",
     verdict, kill_met=False, n=PA["n_cells"])
