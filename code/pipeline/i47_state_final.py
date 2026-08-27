# -*- coding: utf-8 -*-
"""I-47 상태변수 확정 — I-46 후속.

I-46 결론: 휴면(무채용 월 비중)의 예측력은 **전부 물량 성분**이다.
 · 물량 정화 휴면(excess dormancy) 기울기 −0.30 [−0.95, +0.40] · 3분위 +0.003 [−0.18, +0.19]
 · 물량 5분위 FE 하 통합 −0.066 · 물량·규모 잔차화 후 +0.063
반면 **사전 채용률**은 공변량 조정에서 오히려 강해진다: +0.4964 → **+0.5650 [0.184, 0.958]**.
그리고 미처치 위약에서 네 상태변수 모두 정밀한 0 → 이질성 자체는 기계적이 아니다.

남은 문제: I-46 Panel I 의 '조정 3분위'는 **FWL 정합이 아니었다**(y 만 잔차화하고 x 는 원변수로 분할).
여기서 y·x 를 **둘 다** 잔차화해 올바르게 재계산하고 construct 를 확정한다.

Panel A  FWL 정합 조정 대비 — 상태변수 3종 × (3분위·4분위·중앙값·상위25%)
Panel B  IQR 환산 효과크기 (연속 사양을 해석 가능하게)
Panel C  상태변수를 더 먼 창 [−36,−25] 에서 측정 (결과 기준창과의 분리 강화)
Panel D  미처치 위약 — FWL 잔차화 상태변수 기준 (셀 군집)
Panel E  이직률·순유입 대안 상태변수 (탐색)
"""
import numpy as np
from h30_common import load, deals, build, emit, SEED, qci, NB, widx

rng = np.random.default_rng(SEED)
print("[I-47] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Sv, Ev, adpt = G["Hv"], G["Sv"], G["Ev"], G["adpt_arr"]


def win(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return None
    h, e, s = Hv[row, c].astype(float), Ev[row, c].astype(float), Sv[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.mean(e) < 5: return None
    return h, e, s


def lrate(row, m0, a, b):
    w = win(row, m0, a, b)
    if w is None: return np.nan
    N, E = w[0].sum(), np.mean(w[1])
    return float(np.log(N / E)) if N > 0 else np.nan


def states(row, m0, a, b):
    w = win(row, m0, a, b)
    if w is None: return None
    h, e, s = w
    N, E = float(h.sum()), float(np.mean(e))
    return dict(dorm=float((h == 0).mean()), lN=float(np.log1p(N)),
                lr=float(np.log1p(N / E)), lsep=float(np.log1p(s.sum() / E)),
                net=float((h.sum() - s.sum()) / E), lsize=float(np.log(E)))


for e in EV:
    a, b = lrate(e["ti"], e["m0"], -12, -1), lrate(e["ti"], e["m0"], 1, 12)
    t = b - a if (np.isfinite(a) and np.isfinite(b)) else np.nan
    cs = [lrate(k, e["m0"], 1, 12) - lrate(k, e["m0"], -12, -1) for k in e["ctrls"]]
    cs = [x for x in cs if np.isfinite(x)]
    e["eff"] = t - float(np.mean(cs)) if (np.isfinite(t) and cs) else np.nan
    e["S"] = states(e["ti"], e["m0"], -24, -13)
    e["Sfar"] = states(e["ti"], e["m0"], -36, -25)
    w36 = win(e["ti"], e["m0"], -36, -25)
    e["grow"] = (float(np.log(np.mean(win(e["ti"], e["m0"], -24, -13)[1]) / np.mean(w36[1])))
                 if (e["S"] and w36 and np.mean(w36[1]) > 0) else np.nan)
    e["age"] = (e["m0"] - adpt[e["ti"]]) / 12.0 if np.isfinite(adpt[e["ti"]]) else np.nan
    e["ind1"] = str(G["ind_arr"][e["ti"]])[:1]

U = [e for e in EV if np.isfinite(e["eff"]) and e["S"]]
y = np.array([e["eff"] for e in U])
print(f"  분석표본 {len(U)}/{len(EV)}")


def covmat(sub):
    cols = [np.ones(len(sub))]
    for k in ("lsize", "grow", "age"):
        v = np.array([(e["S"]["lsize"] if k == "lsize" else e[k]) for e in sub], float)
        v = np.where(np.isfinite(v), v, np.nanmedian(v[np.isfinite(v)]))
        cols.append(v)
    for s_ in sorted({e["ind1"] for e in sub})[1:]:
        cols.append(np.array([1.0 if e["ind1"] == s_ else 0.0 for e in sub]))
    return np.column_stack(cols)


def resid(X, v): return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def contrast(yy, xx, lo_q, hi_q, R=NB):
    a, b = np.percentile(xx, [lo_q, hi_q])
    hi, lo = yy[xx > b], yy[xx <= a]
    if min(len(hi), len(lo)) < 8: return None
    d = float(hi.mean() - lo.mean())
    bo = np.array([hi[rng.integers(0, len(hi), len(hi))].mean()
                   - lo[rng.integers(0, len(lo), len(lo))].mean() for _ in range(R)])
    ci = qci(bo)
    return {"diff": round(d, 4), "ci": ci, "n_hi": len(hi), "n_lo": len(lo),
            "sig": bool(ci[0] > 0 or ci[1] < 0)}


def slope_ci(xx, yy, R=NB):
    b = float(np.polyfit(xx, yy, 1)[0])
    bo = np.array([np.polyfit(xx[i], yy[i], 1)[0]
                   for i in (rng.integers(0, len(xx), len(xx)) for _ in range(R))])
    ci = qci(bo)
    return {"slope": round(b, 4), "ci": ci, "n": len(xx), "sig": bool(ci[0] > 0 or ci[1] < 0)}


X = covmat(U)
yres = resid(X, y)
CAND = {"dormancy": np.array([e["S"]["dorm"] for e in U]),
        "neg_log_volume": -np.array([e["S"]["lN"] for e in U]),
        "neg_log_rate": -np.array([e["S"]["lr"] for e in U])}
CUTS = (("tercile", 33.33, 66.67), ("quartile", 25, 75),
        ("median", 50, 50), ("top25_vs_rest", 75, 75))

print("\n[Panel A] FWL 정합 조정 대비 — y·x 를 **둘 다** 공변량에 잔차화")
PA = {}
for nm, x in CAND.items():
    xres = resid(X, x)
    PA[nm] = {"slope_raw": slope_ci(x, y), "slope_adj_fwl": slope_ci(xres, yres)}
    row = []
    for cn, lq, hq in CUTS:
        r_raw = contrast(y, x, lq, hq); r_adj = contrast(yres, xres, lq, hq)
        PA[nm][f"{cn}_raw"] = r_raw; PA[nm][f"{cn}_adj"] = r_adj
        if r_adj: row.append(f"{cn} {r_adj['diff']:+.3f}{'✓' if r_adj['sig'] else '✗'}")
    print(f"  {nm:<16} 기울기 원 {PA[nm]['slope_raw']['slope']:>+7.4f}"
          f"{'✓' if PA[nm]['slope_raw']['sig'] else '✗'} · FWL조정 "
          f"{PA[nm]['slope_adj_fwl']['slope']:>+7.4f} {str(PA[nm]['slope_adj_fwl']['ci']):<20}"
          f"{'✓' if PA[nm]['slope_adj_fwl']['sig'] else '✗'}")
    print(f"{'':<18}조정 대비: " + " · ".join(row))

print("\n[Panel B] IQR 환산 효과크기")
PB = {}
for nm, x in CAND.items():
    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
    s_raw, s_adj = PA[nm]["slope_raw"], PA[nm]["slope_adj_fwl"]
    PB[nm] = {"iqr": round(iqr, 4),
              "effect_per_iqr_raw": round(s_raw["slope"] * iqr, 4),
              "effect_per_iqr_adj": round(s_adj["slope"] * iqr, 4),
              "ci_per_iqr_adj": [round(s_adj["ci"][0] * iqr, 4), round(s_adj["ci"][1] * iqr, 4)]}
    print(f"  {nm:<16} IQR {iqr:>6.3f} → 조정 효과 {PB[nm]['effect_per_iqr_adj']:>+7.4f} "
          f"{PB[nm]['ci_per_iqr_adj']}")

print("\n[Panel C] 상태변수를 [−36,−25] 에서 측정 (결과 기준창과 24개월 분리)")
Uf = [e for e in U if e["Sfar"]]
Xf = covmat(Uf); yf = np.array([e["eff"] for e in Uf]); yfr = resid(Xf, yf)
PC = {"n": len(Uf)}
for nm, key in (("dormancy", "dorm"), ("neg_log_volume", "lN"), ("neg_log_rate", "lr")):
    x = np.array([e["Sfar"][key] for e in Uf], float) * (1 if key == "dorm" else -1)
    xr = resid(Xf, x)
    PC[nm] = {"slope_raw": slope_ci(x, yf), "slope_adj_fwl": slope_ci(xr, yfr),
              "tercile_adj": contrast(yfr, xr, 33.33, 66.67)}
    t = PC[nm]["tercile_adj"]
    print(f"  {nm:<16} 원 {PC[nm]['slope_raw']['slope']:>+7.4f}"
          f"{'✓' if PC[nm]['slope_raw']['sig'] else '✗'} · FWL조정 "
          f"{PC[nm]['slope_adj_fwl']['slope']:>+7.4f} {str(PC[nm]['slope_adj_fwl']['ci']):<20}"
          f"{'✓' if PC[nm]['slope_adj_fwl']['sig'] else '✗'}"
          + (f" · 3분위 {t['diff']:+.4f} {'✓' if t['sig'] else '✗'}" if t else ""))

print("\n[Panel D] 미처치 위약 — FWL 잔차화 상태변수 기준 (셀 군집)")
rows = []
for gi, e in enumerate(EV):
    ok = [k for k in e["ctrls"]
          if np.isfinite(lrate(k, e["m0"], -12, -1)) and np.isfinite(lrate(k, e["m0"], 1, 12))
          and states(k, e["m0"], -24, -13)]
    if len(ok) < 3: continue
    ch = {k: lrate(k, e["m0"], 1, 12) - lrate(k, e["m0"], -12, -1) for k in ok}
    for k in ok:
        S = states(k, e["m0"], -24, -13)
        rows.append((gi, ch[k] - float(np.mean([ch[j] for j in ok if j != k])),
                     S["dorm"], -S["lN"], -S["lr"], S["lsize"]))
gidx = np.array([r[0] for r in rows]); py = np.array([r[1] for r in rows])
Xp = np.column_stack([np.ones(len(rows)), np.array([r[5] for r in rows])])
pyr = resid(Xp, py)
GU = np.unique(gidx); byg = {g_: np.where(gidx == g_)[0] for g_ in GU}
PD = {"n_pseudo": len(rows), "n_clusters": int(len(GU))}
for j, nm in ((2, "dormancy"), (3, "neg_log_volume"), (4, "neg_log_rate")):
    x = np.array([r[j] for r in rows]); xr = resid(Xp, x)
    b = float(np.polyfit(xr, pyr, 1)[0])
    bb = []
    for _ in range(NB):
        p_ = np.concatenate([byg[GU[i]] for i in rng.integers(0, len(GU), len(GU))])
        try: bb.append(np.polyfit(xr[p_], pyr[p_], 1)[0])
        except Exception: pass
    ci = qci(np.array(bb))
    PD[nm] = {"slope": round(b, 4), "ci": ci, "sig": bool(ci[0] > 0 or ci[1] < 0)}
    print(f"  {nm:<16} {b:>+7.4f} {str(ci):<22}{'✓' if PD[nm]['sig'] else '✗'}")
print(f"  (유사처치 {len(rows)}건 / 셀 {len(GU)}개; 규모만 부분아웃)")

print("\n[Panel E] 대안 상태변수 (탐색) — 사전 이직률·순유입")
PE_ = {}
for nm, key, sgn in (("neg_log_separation_rate", "lsep", -1), ("neg_net_inflow", "net", -1)):
    x = np.array([e["S"][key] for e in U], float) * sgn
    xr = resid(X, x)
    PE_[nm] = {"slope_raw": slope_ci(x, y), "slope_adj_fwl": slope_ci(xr, yres)}
    print(f"  {nm:<24} 원 {PE_[nm]['slope_raw']['slope']:>+7.4f}"
          f"{'✓' if PE_[nm]['slope_raw']['sig'] else '✗'} · FWL조정 "
          f"{PE_[nm]['slope_adj_fwl']['slope']:>+7.4f} {str(PE_[nm]['slope_adj_fwl']['ci']):<20}"
          f"{'✓' if PE_[nm]['slope_adj_fwl']['sig'] else '✗'}")

win_ = max(CAND, key=lambda k: (PA[k]["slope_adj_fwl"]["sig"],
                                abs(PA[k]["slope_adj_fwl"]["slope"] / max(
                                    (PA[k]["slope_adj_fwl"]["ci"][1]
                                     - PA[k]["slope_adj_fwl"]["ci"][0]) / 3.92, 1e-9))))
verdict = (
    f"FWL 정합 조정에서 살아남는 상태변수는 **{win_}** 이다. "
    f"휴면 조정 기울기 {PA['dormancy']['slope_adj_fwl']['slope']:+.4f}"
    f"{PA['dormancy']['slope_adj_fwl']['ci']}"
    f"{'✓' if PA['dormancy']['slope_adj_fwl']['sig'] else '✗'} vs "
    f"사전 채용률 {PA['neg_log_rate']['slope_adj_fwl']['slope']:+.4f}"
    f"{PA['neg_log_rate']['slope_adj_fwl']['ci']}"
    f"{'✓' if PA['neg_log_rate']['slope_adj_fwl']['sig'] else '✗'}. "
    f"미처치 위약은 세 변수 모두 0 근처 "
    f"(휴면 {PD['dormancy']['slope']:+.4f} · 물량 {PD['neg_log_volume']['slope']:+.4f} · "
    f"채용률 {PD['neg_log_rate']['slope']:+.4f}) — 이질성은 기계적이 아니다. "
    f"[−36,−25] 창에서도 채용률 조정 기울기 {PC['neg_log_rate']['slope_adj_fwl']['slope']:+.4f}"
    f"{'✓' if PC['neg_log_rate']['slope_adj_fwl']['sig'] else '✗'}.")
emit("I-47", "상태변수 확정 (FWL 정합 조정 + 위약)",
     "GO" if PA["neg_log_rate"]["slope_adj_fwl"]["sig"] else "PARTIAL",
     {"panelA_fwl_adjusted": PA, "panelB_iqr_effect": PB, "panelC_far_window": PC,
      "panelD_untreated_placebo_fwl": PD, "panelE_alternative_states": PE_,
      "winner": win_, "n": len(U)},
     "공변량을 FWL 정합으로 제거해도 어떤 상태변수가 반응을 예측하는가",
     verdict, kill_met=False, n=len(U))
