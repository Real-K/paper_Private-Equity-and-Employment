# -*- coding: utf-8 -*-
"""I-51 사양 판별 — 차분 강제 vs 자유 lag, **처치와 위약을 같은 사양으로**.

I-50 Panel A/H 에서 풀링 사양(자유 lag)은 gradient 를 검출하지 못했다(+0.316 ✗, S 셀상수판 −0.052 ✗).
반면 매칭차분은 +0.7014 ✓ 다. 두 사양은 **추정대상이 다르다**: 차분은 사전 수준 계수를 1 로 강제하고,
자유 lag 사양은 사전 수준의 일부(1−γ)를 결과에 남긴다. 상태변수는 사전 수준과 강하게 상관되므로
남은 성분이 계수를 끌어내린다.

따라서 "자유 lag 에서 null" 이 **효과 없음**인지 **추정대상이 달라서**인지는, 같은 사양을
**아무 일도 없었던 기업**에 돌려봐야만 알 수 있다. 사양마다 자기 위약이 필요하다.

설계. 각 이벤트에서
  · 처치 셀: 처치기업 + 대조 5 (S = 처치기업 상태, 셀 상수 → 셀 FE 가 S 주효과 흡수)
  · 위약 셀: 대조 k 를 유사처치로, 같은 셀의 나머지 대조를 그 대조군으로 (S = k 의 상태)
두 집합에 **동일한 회귀**를 돌려 T×S 를 비교한다.
"""
import numpy as np, pandas as pd
from h30_common import load, deals, build, emit, SEED, qci, NB, widx

rng = np.random.default_rng(SEED)
print("[I-51] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Ev = G["Hv"], G["Ev"]


def rate(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return None
    h, e = Hv[row, c].astype(float), Ev[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.mean(e) < 5: return None
    return float(h.sum()), float(np.mean(e))


def cell_rows(focal, others, m0, cid):
    st = rate(focal, m0, -24, -13)
    if st is None: return []
    S = -np.log1p(st[0] / st[1]); lsz = np.log(st[1])
    out = []
    for row, T in [(focal, 1)] + [(o, 0) for o in others]:
        po, pr = rate(row, m0, 1, 12), rate(row, m0, -12, -1)
        if po is None or pr is None or po[0] <= 0 or pr[0] <= 0: continue
        out.append(dict(cell=cid, T=T, y=np.log(po[0] / po[1]),
                        lag=np.log(pr[0] / pr[1]), S=S, lsize=lsz))
    return out if (sum(r["T"] for r in out) == 1 and len(out) >= 3) else []


TR, PL = [], []
cid = 0
for e in EV:
    ctr = [int(k) for k in e["ctrls"]]
    TR += cell_rows(e["ti"], ctr, e["m0"], cid); cid += 1
    for k in ctr:                                   # 위약: 대조 k 를 유사처치로
        PL += cell_rows(k, [j for j in ctr if j != k], e["m0"], cid); cid += 1
DT = pd.DataFrame(TR); DP = pd.DataFrame(PL)
for D in (DT, DP): D["TS"] = D["T"] * D["S"]
print(f"  처치 셀 {DT.cell.nunique()} / 관측 {len(DT)}"
      f" · 위약 셀 {DP.cell.nunique()} / 관측 {len(DP)}")


def within(df, cols):
    o = df.copy()
    for c in cols: o[c] = df[c] - df.groupby("cell")[c].transform("mean")
    return o


def est(D, cols, outcome, R=NB):
    d = D.copy(); d["yy"] = outcome(d)
    w = within(d, cols + ["yy"])
    X = w[cols].to_numpy(); b = np.linalg.lstsq(X, w["yy"].to_numpy(), rcond=None)[0]
    j = cols.index("TS")
    cells = d.cell.unique(); byg = {c: d.index[d.cell == c].to_numpy() for c in cells}
    bb = []
    for _ in range(R):
        sel = rng.integers(0, len(cells), len(cells)); parts, labs = [], []
        for r_, i in enumerate(sel):
            ix = byg[cells[i]]; parts.append(ix); labs.append(np.full(len(ix), r_))
        d2 = d.loc[np.concatenate(parts)].copy(); d2["cell"] = np.concatenate(labs)
        w2 = within(d2, cols + ["yy"])
        try: bb.append(np.linalg.lstsq(w2[cols].to_numpy(), w2["yy"].to_numpy(),
                                       rcond=None)[0][j])
        except Exception: pass
    ci = qci(np.array(bb))
    lagc = (round(float(b[cols.index("lag")]), 4) if "lag" in cols else 1.0)
    return {"coef": round(float(b[j]), 4), "ci": ci, "sig": bool(ci[0] > 0 or ci[1] < 0),
            "half_width": round(float((ci[1] - ci[0]) / 2), 4), "lag_coef": lagc,
            "n_obs": len(d), "n_cells": int(d.cell.nunique())}


SPECS = (("차분 강제 (γ=1)", ["T", "TS"], lambda d: d["y"] - d["lag"]),
         ("자유 lag (ANCOVA)", ["T", "TS", "lag"], lambda d: d["y"]))
print("\n[사양별] 처치 vs 위약 — 같은 회귀, 같은 대비")
R = {}
for tag, cols, outc in SPECS:
    t_ = est(DT, cols, outc); p_ = est(DP, cols, outc)
    # 처치−위약 차이의 CI (독립 부트 근사: 반폭 제곱합)
    diff = round(t_["coef"] - p_["coef"], 4)
    hw = float(np.hypot(t_["half_width"], p_["half_width"]))
    R[tag] = {"treated": t_, "placebo": p_, "diff": diff,
              "diff_ci": [round(diff - hw, 4), round(diff + hw, 4)],
              "diff_sig": bool(abs(diff) > hw)}
    print(f"  {tag:<20} 처치 {t_['coef']:>+7.4f} {str(t_['ci']):<22}"
          f"{'✓' if t_['sig'] else '✗'} (lag {t_['lag_coef']})")
    print(f"  {'':<20} 위약 {p_['coef']:>+7.4f} {str(p_['ci']):<22}"
          f"{'✓' if p_['sig'] else '✗'} (셀 {p_['n_cells']})")
    print(f"  {'':<20} 처치−위약 {diff:>+7.4f} {R[tag]['diff_ci']} "
          f"{'✓' if R[tag]['diff_sig'] else '✗'}")

key = "자유 lag (ANCOVA)"
verdict = (
    f"차분 강제: 처치 {R['차분 강제 (γ=1)']['treated']['coef']:+.4f}"
    f"{R['차분 강제 (γ=1)']['treated']['ci']} vs 위약 "
    f"{R['차분 강제 (γ=1)']['placebo']['coef']:+.4f}{R['차분 강제 (γ=1)']['placebo']['ci']} → "
    f"차이 {R['차분 강제 (γ=1)']['diff']:+.4f} "
    f"{'유의' if R['차분 강제 (γ=1)']['diff_sig'] else '미검출'}. "
    f"자유 lag: 처치 {R[key]['treated']['coef']:+.4f}{R[key]['treated']['ci']} vs 위약 "
    f"{R[key]['placebo']['coef']:+.4f}{R[key]['placebo']['ci']} → 차이 {R[key]['diff']:+.4f} "
    f"{'유의' if R[key]['diff_sig'] else '미검출'} (lag 계수 {R[key]['treated']['lag_coef']}). "
    "사양마다 자기 위약을 붙여야 null 의 의미가 정해진다.")
emit("I-51", "사양 판별 — 차분 강제 vs 자유 lag, 처치와 위약을 같은 사양으로",
     "GO" if R["차분 강제 (γ=1)"]["diff_sig"] else "PARTIAL",
     {"specs": R, "design": "S = 초점기업 상태(셀 상수) → 셀 FE 가 S 주효과 흡수",
      "n_treated_cells": int(DT.cell.nunique()), "n_placebo_cells": int(DP.cell.nunique())},
     "자유 lag 사양의 null 이 '효과 없음'인가 '추정대상 차이'인가",
     verdict, kill_met=False, n=int(DT.cell.nunique()))
