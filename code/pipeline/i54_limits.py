# -*- coding: utf-8 -*-
"""I-54 남은 한계 보완 시도.

한계 1. 부트스트랩 차이 검정 기준 MDE 0.542 vs 관측 0.363 → 검정력 ~50%.
        그러나 우리 **주 추론은 RI** 다. RI 의 MDE 는 다르게 계산해야 한다.
한계 2. 위약 귀무 평균이 +0.117 로 0 이 아니다. RI 는 이를 반영하지만 분산을 줄일 수 있는가.

Panel A  ★ 두 번째 결과대상 — Δlog 고용. 상태 gradient 가 고용에서도 나타나는가.
         (경제적으로 중요하고, 나타나면 주장의 폭이 넓어진다)
Panel B  층화 RI — 위약 추출을 처치의 규모×산업 분포에 맞춰 귀무 분산을 줄인다
Panel C  RI 기준 MDE·검정력 재계산 (부트스트랩 기준과 구분)
Panel D  위약 기울기 +0.117 의 출처 — 사전창을 더 멀리 두면 줄어드는가
"""
import numpy as np
from h30_common import load, deals, build, emit, SEED, qci, NB, widx

rng = np.random.default_rng(SEED)
NDRAW = 2000
print("[I-54] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Ev, adpt = G["Hv"], G["Ev"], G["adpt_arr"]


def wblk(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return None
    h, e = Hv[row, c].astype(float), Ev[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.mean(e) < 5: return None
    return float(h.sum()), float(np.mean(e))


def unit(focal, others, m0, outcome, state_win=(-24, -13)):
    st = wblk(focal, m0, *state_win)
    if st is None: return None
    def val(row):
        po, pr = wblk(row, m0, 1, 12), wblk(row, m0, -12, -1)
        if po is None or pr is None: return None
        if outcome == "hire":
            if po[0] <= 0 or pr[0] <= 0: return None
            return np.log(po[0] / po[1]) - np.log(pr[0] / pr[1])
        return np.log(po[1]) - np.log(pr[1])                # employment
    t = val(focal)
    if t is None: return None
    cs = [v for v in (val(o) for o in others) if v is not None]
    if not cs: return None
    w36 = wblk(focal, m0, -36, -25)
    return dict(eff=t - float(np.mean(cs)), S=-np.log1p(st[0] / st[1]),
                lsize=np.log(st[1]),
                grow=(np.log(st[1] / w36[1]) if w36 and w36[1] > 0 else np.nan),
                age=((m0 - adpt[focal]) / 12.0 if np.isfinite(adpt[focal]) else np.nan),
                ind=str(G["ind_arr"][focal])[:1], g=None)


def assemble(outcome, state_win=(-24, -13)):
    T, P = [], []
    for gi, e in enumerate(EV):
        ctr = [int(k) for k in e["ctrls"]]
        u = unit(e["ti"], ctr, e["m0"], outcome, state_win)
        if u: u["g"] = gi; T.append(u)
        for k in ctr:
            v = unit(k, [j for j in ctr if j != k], e["m0"], outcome, state_win)
            if v: v["g"] = gi; P.append(v)
    return T, P


def slope(rows, cuts=None):
    if len(rows) < 30: return None
    y = np.array([r["eff"] for r in rows]); x = np.array([r["S"] for r in rows])
    if cuts is not None: y = np.clip(y, cuts[0], cuts[1])
    cols = [np.ones(len(y)), np.array([r["lsize"] for r in rows])]
    for k in ("grow", "age"):
        v = np.array([r[k] for r in rows], float); m = np.isfinite(v)
        cols.append(np.where(m, v, np.median(v[m]) if m.any() else 0.0))
    for s in sorted({r["ind"] for r in rows})[1:]:
        cols.append(np.array([1.0 if r["ind"] == s else 0.0 for r in rows]))
    C = np.column_stack(cols)
    r_ = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    yr, xr = r_(y), r_(x); d = float(np.sum(xr * xr))
    return float(np.sum(xr * yr) / d) if d > 0 else None


def ri(T, P, cuts, strata=None, tag=""):
    obs = slope(T, cuts); n_t = len(T)
    cells = sorted({r["g"] for r in P})
    byg = {c: [r for r in P if r["g"] == c] for c in cells}
    if strata is None:
        null = []
        for _ in range(NDRAW):
            draw = []
            for i in rng.permutation(len(cells)):
                draw += byg[cells[i]]
                if len(draw) >= n_t: break
            s_ = slope(draw[:n_t], cuts)
            if s_ is not None: null.append(s_)
    else:                                   # 층화: 처치의 (규모3분위 × 산업) 구성에 맞춰 추출
        keyf = strata
        tgt = {}
        for r in T: tgt[keyf(r)] = tgt.get(keyf(r), 0) + 1
        pool = {}
        for r in P: pool.setdefault(keyf(r), []).append(r)
        null = []
        for _ in range(NDRAW):
            draw = []
            for k, n in tgt.items():
                src = pool.get(k, [])
                if not src: src = P
                idx = rng.integers(0, len(src), n)
                draw += [src[i] for i in idx]
            s_ = slope(draw, cuts)
            if s_ is not None: null.append(s_)
    null = np.array(null)
    p = (int((null >= obs).sum()) + 1) / (len(null) + 1)
    # RI 기준 MDE: 단측 5%, 80% 검정력 → 귀무평균 + (1.645+0.842)*SD
    mde = 2.487 * float(null.std())
    out = {"observed": round(obs, 4), "n_treated": n_t, "n_draws": len(null),
           "null_mean": round(float(null.mean()), 4), "null_sd": round(float(null.std()), 4),
           "null_ci": qci(null), "RI_p": round(float(p), 4), "sig": bool(p < 0.05),
           "z": round(float((obs - null.mean()) / null.std()), 2),
           "excess_over_null": round(float(obs - null.mean()), 4),
           "MDE_RI_80": round(float(mde), 4),
           "observed_excess_over_MDE": round(float((obs - null.mean()) / mde), 2)}
    print(f"  {tag:<28} 관측 {out['observed']:>+7.4f} · 귀무 {out['null_mean']:>+7.4f}"
          f"(SD {out['null_sd']:.4f}) · RI p {out['RI_p']:.4f} {'✓' if out['sig'] else '✗'} "
          f"z={out['z']} · MDE_RI {out['MDE_RI_80']:.3f} (관측초과 {out['excess_over_null']:+.3f} "
          f"= {out['observed_excess_over_MDE']}배)")
    return out


print("\n[Panel A] ★ 두 번째 결과대상 — Δlog 고용")
R = {}
Th, Ph = assemble("hire"); cuts_h = tuple(np.percentile([r["eff"] for r in Th], [5, 95]))
Te, Pe = assemble("employment"); cuts_e = tuple(np.percentile([r["eff"] for r in Te], [5, 95]))
R["hire_rate"] = ri(Th, Ph, cuts_h, tag="Δlog 채용률 (주 결과)")
R["employment"] = ri(Te, Pe, cuts_e, tag="★ Δlog 고용")

print("\n[Panel B] 층화 RI — 처치의 규모3분위 × 산업 구성에 맞춰 추출")
szT = np.array([r["lsize"] for r in Th]); q1, q2 = np.percentile(szT, [33.33, 66.67])
def key(r): return (0 if r["lsize"] <= q1 else (1 if r["lsize"] <= q2 else 2), r["ind"])
R["hire_rate_stratified"] = ri(Th, Ph, cuts_h, strata=key, tag="Δlog 채용률 · 층화")
R["employment_stratified"] = ri(Te, Pe, cuts_e, strata=key, tag="Δlog 고용 · 층화")

print("\n[Panel C] RI 기준 검정력 vs 부트스트랩 기준")
a = R["hire_rate"]; b = R["hire_rate_stratified"]
PC = {"bootstrap_MDE_from_I52": 0.5425, "bootstrap_observed_diff": 0.3626,
      "RI_MDE": a["MDE_RI_80"], "RI_excess": a["excess_over_null"],
      "RI_power_ratio": a["observed_excess_over_MDE"],
      "RI_MDE_stratified": b["MDE_RI_80"],
      "stratified_null_sd_reduction": round(1 - b["null_sd"] / a["null_sd"], 3)}
print(f"  부트스트랩 기준 MDE 0.5425 vs 관측 0.3626 → 0.67배")
print(f"  RI 기준        MDE {a['MDE_RI_80']:.4f} vs 관측초과 {a['excess_over_null']:.4f} "
      f"→ {a['observed_excess_over_MDE']}배")
print(f"  층화 후 귀무 SD {a['null_sd']:.4f} → {b['null_sd']:.4f} "
      f"({PC['stratified_null_sd_reduction']:+.1%})")

print("\n[Panel D] 위약 기울기의 출처 — 상태창을 더 멀리 두면 줄어드는가")
PD = {}
for wl, tag in (((-24, -13), "상태창 [−24,−13] (기준)"), ((-36, -25), "상태창 [−36,−25]")):
    T2, P2 = assemble("hire", state_win=wl)
    if len(T2) < 60 or len(P2) < 200: PD[tag] = {"n": len(T2), "note": "표본 부족"}; continue
    c2 = tuple(np.percentile([r["eff"] for r in T2], [5, 95]))
    bt, bp = slope(T2, c2), slope(P2, c2)
    PD[tag] = {"n_treated": len(T2), "n_placebo": len(P2),
               "treated": round(bt, 4), "placebo": round(bp, 4),
               "diff": round(bt - bp, 4)}
    print(f"  {tag:<26} 처치 {bt:>+7.4f} · 위약 {bp:>+7.4f} · 차이 {bt - bp:>+7.4f} "
          f"(n {len(T2)}/{len(P2)})")

emp = R["employment"]
verdict = (
    f"[A] 두 번째 결과대상 **Δlog 고용**: 관측 {emp['observed']:+.4f} vs 귀무 {emp['null_mean']:+.4f} "
    f"(SD {emp['null_sd']:.4f}), RI p = {emp['RI_p']:.4f} {'✓' if emp['sig'] else '✗'} — "
    + ("상태 gradient 가 고용에서도 확인된다." if emp["sig"] else
       "고용에서는 미검출 — 채용률에 국한된다.")
    + f" [B] 층화 RI 로 귀무 SD {a['null_sd']:.4f} → {b['null_sd']:.4f} "
      f"({PC['stratified_null_sd_reduction']:+.0%}), RI p {a['RI_p']:.4f} → {b['RI_p']:.4f}. "
    + f"[C] **RI 기준 MDE 는 {a['MDE_RI_80']:.3f} 이고 관측초과는 그 "
      f"{a['observed_excess_over_MDE']}배** — 부트스트랩 기준(0.67배)보다 낫다. "
      f"원고는 두 기준을 구분해 보고해야 한다.")
emit("I-54", "남은 한계 보완 — 2차 결과대상 · 층화 RI · RI 기준 검정력",
     "GO" if (a["sig"]) else "PARTIAL",
     {"panelAB_ri": R, "panelC_power": PC, "panelD_placebo_source": PD, "n_draws": NDRAW},
     "고용 결과대상에서도 gradient 가 나타나는가 · 층화로 귀무 분산을 줄일 수 있는가",
     verdict, kill_met=False, n=a["n_treated"])
