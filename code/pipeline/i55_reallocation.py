# -*- coding: utf-8 -*-
"""I-55 재배치 가설 — 채용은 늘고 고용은 안 느는데, 이직은?

I-54 실측: 상태 gradient 가 **Δlog 채용률**에는 있고(+0.4827, RI p 0.018)
**Δlog 고용**에는 없다(−0.0292, RI p 0.983).

회계 항등식상 고용 변화 ≈ 채용 − 이직 이므로, 저활동 기업에서 채용만 크게 늘고 고용이 안 늘었다면
**이직도 같이 늘었어야 한다.** 이는 단순 확장이 아니라 **노동 재배치(turnover) 재가동** 가설이다.
Davis et al. (2014) 의 총유량·재배치 서사와 직접 연결된다.

Panel A  Δlog 이직률의 상태 gradient (RI)
Panel B  총유량(채용+이직) / 순증(채용−이직) 의 상태 gradient
Panel C  회계 정합 — 세 gradient 가 항등식과 맞는가
Panel D  사전 이직률을 상태변수로 썼을 때 (I-47 Panel E 확장)
"""
import numpy as np
from h30_common import load, deals, build, emit, SEED, qci, widx

rng = np.random.default_rng(SEED)
NDRAW = 2000
print("[I-55] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Sv, Ev, adpt = G["Hv"], G["Sv"], G["Ev"], G["adpt_arr"]


def blk(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return None
    h, s, e = Hv[row, c].astype(float), Sv[row, c].astype(float), Ev[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(s).all() and np.isfinite(e).all()): return None
    if np.mean(e) < 5: return None
    return float(h.sum()), float(s.sum()), float(np.mean(e))


OUT = {"hire": lambda H, S, E: (np.log(H / E) if H > 0 else None),
       "sep": lambda H, S, E: (np.log(S / E) if S > 0 else None),
       "gross": lambda H, S, E: (np.log((H + S) / E) if (H + S) > 0 else None),
       "net": lambda H, S, E: (H - S) / E,
       "emp": lambda H, S, E: np.log(E)}


def unit(focal, others, m0, key):
    st = blk(focal, m0, -24, -13)
    if st is None: return None
    f = OUT[key]
    def val(row):
        po, pr = blk(row, m0, 1, 12), blk(row, m0, -12, -1)
        if po is None or pr is None: return None
        a, b = f(*po), f(*pr)
        return None if (a is None or b is None) else a - b
    t = val(focal)
    if t is None: return None
    cs = [v for v in (val(o) for o in others) if v is not None]
    if not cs: return None
    w36 = blk(focal, m0, -36, -25)
    return dict(eff=t - float(np.mean(cs)), S=-np.log1p(st[0] / st[2]),
                lsize=np.log(st[2]),
                grow=(np.log(st[2] / w36[2]) if w36 and w36[2] > 0 else np.nan),
                age=((m0 - adpt[focal]) / 12.0 if np.isfinite(adpt[focal]) else np.nan),
                ind=str(G["ind_arr"][focal])[:1], g=None)


def assemble(key):
    T, P = [], []
    for gi, e in enumerate(EV):
        ctr = [int(k) for k in e["ctrls"]]
        u = unit(e["ti"], ctr, e["m0"], key)
        if u: u["g"] = gi; T.append(u)
        for k in ctr:
            v = unit(k, [j for j in ctr if j != k], e["m0"], key)
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


def ri(key, tag):
    T, P = assemble(key)
    if len(T) < 60: print(f"  {tag}: 표본 부족 {len(T)}"); return None
    cuts = tuple(np.percentile([r["eff"] for r in T], [5, 95]))
    obs = slope(T, cuts); n_t = len(T)
    cells = sorted({r["g"] for r in P}); byg = {c: [r for r in P if r["g"] == c] for c in cells}
    null = []
    for _ in range(NDRAW):
        draw = []
        for i in rng.permutation(len(cells)):
            draw += byg[cells[i]]
            if len(draw) >= n_t: break
        s_ = slope(draw[:n_t], cuts)
        if s_ is not None: null.append(s_)
    null = np.array(null)
    p = (int((null >= obs).sum()) + 1) / (len(null) + 1)
    p2 = (int((null <= obs).sum()) + 1) / (len(null) + 1)
    out = {"observed": round(obs, 4), "n_treated": n_t, "n_placebo": len(P),
           "null_mean": round(float(null.mean()), 4), "null_sd": round(float(null.std()), 4),
           "null_ci": qci(null), "RI_p_upper": round(float(p), 4),
           "RI_p_lower": round(float(p2), 4),
           "excess": round(float(obs - null.mean()), 4),
           "z": round(float((obs - null.mean()) / null.std()), 2),
           "sig_positive": bool(p < 0.05), "sig_negative": bool(p2 < 0.05)}
    mark = "✓+" if out["sig_positive"] else ("✓−" if out["sig_negative"] else "✗")
    print(f"  {tag:<26} 관측 {out['observed']:>+7.4f} · 귀무 {out['null_mean']:>+7.4f}"
          f"(SD {out['null_sd']:.4f}) · 초과 {out['excess']:>+7.4f} · z={out['z']:>5.2f} "
          f"· RI p(상){out['RI_p_upper']:.3f} {mark}  n={n_t}")
    return out


print("\n[Panel A·B] 결과대상별 상태 gradient (winsor 5/95, FWL 조정, RI 2000회)")
R = {}
for key, tag in (("hire", "Δlog 채용률"), ("sep", "★ Δlog 이직률"),
                 ("gross", "Δlog 총유량 (채용+이직)"), ("net", "순증률 (채용−이직)/E"),
                 ("emp", "Δlog 고용")):
    r = ri(key, tag)
    if r: R[key] = r

print("\n[Panel C] 회계 정합 점검")
PC = {}
if all(k in R for k in ("hire", "sep", "net", "emp")):
    PC = {"hire_gradient": R["hire"]["observed"], "sep_gradient": R["sep"]["observed"],
          "net_gradient": R["net"]["observed"], "emp_gradient": R["emp"]["observed"],
          "hire_minus_sep": round(R["hire"]["observed"] - R["sep"]["observed"], 4)}
    print(f"  채용 {PC['hire_gradient']:+.4f} · 이직 {PC['sep_gradient']:+.4f} → "
          f"차 {PC['hire_minus_sep']:+.4f} · 순증 {PC['net_gradient']:+.4f} · "
          f"고용 {PC['emp_gradient']:+.4f}")
    print("  (로그 채용률·이직률의 차는 순증과 단위가 다르므로 부호·상대크기만 해석)")

print("\n[Panel D] 사전 **이직률**을 상태변수로 (I-47 Panel E 확장)")
PD = {}
def unit_sep_state(focal, others, m0, key):
    st = blk(focal, m0, -24, -13)
    if st is None or st[1] <= 0: return None
    u = unit(focal, others, m0, key)
    if u is None: return None
    u["S"] = -np.log1p(st[1] / st[2])              # 상태 = −log(1+사전 이직률)
    return u
for key, tag in (("hire", "결과 Δlog 채용률 · 상태 사전이직률"),
                 ("emp", "결과 Δlog 고용 · 상태 사전이직률")):
    T, P = [], []
    for gi, e in enumerate(EV):
        ctr = [int(k) for k in e["ctrls"]]
        u = unit_sep_state(e["ti"], ctr, e["m0"], key)
        if u: u["g"] = gi; T.append(u)
        for k in ctr:
            v = unit_sep_state(k, [j for j in ctr if j != k], e["m0"], key)
            if v: v["g"] = gi; P.append(v)
    if len(T) < 60: continue
    cuts = tuple(np.percentile([r["eff"] for r in T], [5, 95]))
    bt, bp = slope(T, cuts), slope(P, cuts)
    PD[tag] = {"treated": round(bt, 4), "placebo": round(bp, 4),
               "diff": round(bt - bp, 4), "n_treated": len(T), "n_placebo": len(P)}
    print(f"  {tag:<34} 처치 {bt:>+7.4f} · 위약 {bp:>+7.4f} · 차이 {bt - bp:>+7.4f} (n={len(T)})")

sep = R.get("sep"); gross = R.get("gross")
verdict = (
    f"채용률 gradient +{R['hire']['observed']:.4f} (RI p {R['hire']['RI_p_upper']:.4f} ✓) 인데 "
    f"고용 gradient {R['emp']['observed']:+.4f} (RI p 상 {R['emp']['RI_p_upper']:.3f}) 로 부재. "
    + (f"**이직률 gradient {sep['observed']:+.4f}** (귀무 {sep['null_mean']:+.4f}, z {sep['z']}, "
       f"RI p 상 {sep['RI_p_upper']:.4f}) "
       + ("→ 저활동 기업에서 채용과 이직이 **함께** 늘어난다: 순증이 아니라 **재배치 재가동**."
          if sep["sig_positive"] else
          "→ 이직에서는 미검출. 채용 증가가 고용으로 이어지지 않는 이유는 이 자료로 특정되지 않는다.")
       if sep else "이직 추정 불가. ")
    + (f" 총유량 gradient {gross['observed']:+.4f} (RI p 상 {gross['RI_p_upper']:.4f})."
       if gross else ""))
emit("I-55", "재배치 가설 — 채용·이직·총유량의 상태 gradient",
     "GO" if (sep and sep["sig_positive"]) else "PARTIAL",
     {"panelAB_by_outcome": R, "panelC_accounting": PC, "panelD_separation_state": PD,
      "n_draws": NDRAW},
     "저활동 기업에서 채용만 늘고 고용이 안 느는 이유가 이직 동반 증가(재배치)인가",
     verdict, kill_met=False, n=R["hire"]["n_treated"])
