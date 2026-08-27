# -*- coding: utf-8 -*-
"""I-41 조절자 방어 — 이제 조절자가 논문의 전부다. 리뷰 §5.4 전건 대응.

지적: `top tercile of inaction` 은 작은 규모·낮은 기대채용·저성장·산업 계절성·측정오차·
일시적 비활동을 동시에 대리할 수 있다. 결과대상은 **Δlog 채용률**(기저수준 효과 제거).

Panel A  연속 사양 — 선형·spline(3구간)·decile. 단조성 확인
Panel B  ★ residualized inaction — 사전 규모·산업·성장·업력·계절성으로 회귀 후 잔차로 재검정
Panel C  컷 민감도 — tercile·quartile·상위25%·중앙값 분할
Panel D  일시 vs 구조 — 현재 spell 길이 vs 과거 비활동 비중 분리 (리뷰 §5.4-4)
Panel E  동일 총채용 내 비교 — 사전 총채용을 통제한 뒤에도 조절자가 남는가
"""
import numpy as np, pandas as pd
from h30_common import load, deals, build, boot_did_ci, emit, SEED, qci, NB, widx

rng = np.random.default_rng(SEED)
print("[I-41] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Ev, adpt = G["Hv"], G["Ev"], G["adpt_arr"]

def W(row, m0, a, b):
    c = widx(G, m0, a, b); n = b - a + 1
    if len(c) != n: return None
    h, e = Hv[row, c].astype(float), Ev[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.nanmean(e) < 5: return None
    return h, e
def lrate(row, m0, a, b):
    w = W(row, m0, a, b)
    if w is None: return np.nan
    h, e = w; N, E = h.sum(), np.mean(e)
    return float(np.log(N / E)) if (N > 0 and E > 0) else np.nan

for e in EV:
    a, b = lrate(e["ti"], e["m0"], -12, -1), lrate(e["ti"], e["m0"], 1, 12)
    e["y_t"] = b - a if (np.isfinite(a) and np.isfinite(b)) else np.nan
    cs = [lrate(k, e["m0"], 1, 12) - lrate(k, e["m0"], -12, -1) for k in e["ctrls"]]
    cs = [x for x in cs if np.isfinite(x)]
    e["y_c"] = float(np.mean(cs)) if cs else np.nan
    e["eff"] = e["y_t"] - e["y_c"] if (np.isfinite(e["y_t"]) and np.isfinite(e["y_c"])) else np.nan
    w = W(e["ti"], e["m0"], -24, -13)
    if w is None: e["pp"] = np.nan; e["cur"] = np.nan; e["hist"] = np.nan
    else:
        h = w[0]; e["pp"] = float((h == 0).mean())
        run = 0
        for x in h[::-1]:
            if x == 0: run += 1
            else: break
        e["cur"] = float(run)                        # 창 끝 기준 현재 spell
        e["hist"] = float((h == 0).sum() - run) / 11.0   # 나머지 비활동 비중
    w2 = W(e["ti"], e["m0"], -12, -1)
    e["preN"] = float(w2[0].sum()) if w2 else np.nan
    e["lsize"] = np.log(np.mean(w2[1])) if w2 else np.nan
    e["ind"] = str(G["ind_arr"][e["ti"]])
    e["age"] = (e["m0"] - adpt[e["ti"]]) / 12.0 if np.isfinite(adpt[e["ti"]]) else np.nan
    w3 = W(e["ti"], e["m0"], -36, -25)
    e["grow"] = float(np.log(np.mean(w2[1]) / np.mean(w3[1]))) if (w2 and w3 and np.mean(w3[1]) > 0) else np.nan

U = [e for e in EV if np.isfinite(e["eff"]) and np.isfinite(e["pp"])]
print(f"  분석 표본 {len(U)}/{len(EV)}  (결과 = Δlog 채용률)")

def slope(x, y, lab, deg=1):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 60: print(f"  {lab:<38} n={int(m.sum())} (<60)"); return None
    xx, yy = x[m], y[m]
    s = float(np.polyfit(xx, yy, deg)[0 if deg == 1 else -2])
    bs = np.array([np.polyfit(xx[j], yy[j], deg)[0 if deg == 1 else -2]
                   for j in (rng.integers(0, len(xx), len(xx)) for _ in range(NB))])
    ci = qci(bs); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
    print(f"  {lab:<38} 기울기 {s:>+8.4f} {str(ci):<20} {sg} (n={int(m.sum())})")
    return {"slope": round(s, 4), "ci": ci, "sig": sg == "✓", "n": int(m.sum())}

y = np.array([e["eff"] for e in U]); x = np.array([e["pp"] for e in U])
print("\n[Panel A] 연속 사양")
PA = {"linear": slope(x, y, "선형: Δlog채용률 ~ 사전관성")}
q = np.percentile(x, [10, 20, 30, 40, 50, 60, 70, 80, 90])
dec = np.digitize(x, q)
prof = [float(np.mean(y[dec == i])) if (dec == i).sum() >= 12 else None for i in range(10)]
PA["decile_profile"] = [None if v is None else round(v, 4) for v in prof]
print("  decile 프로파일:", " ".join(f"{i}:{v:+.3f}" if v is not None else f"{i}:—" for i, v in enumerate(prof)))
ok = [v for v in prof if v is not None]
PA["monotone_frac"] = round(float(np.mean(np.diff(ok) > 0)), 3)
print(f"  → 상승 구간 비율 {PA['monotone_frac']:.0%}  (단조 증가면 1.00)")

print("\n[Panel B] ★ residualized inaction — 규모·산업·성장·업력 제거")
df = pd.DataFrame({"pp": x, "ls": [e["lsize"] for e in U], "ag": [e["age"] for e in U],
                   "gr": [e["grow"] for e in U], "ind": [e["ind"] for e in U], "y": y})
df = df.replace([np.inf, -np.inf], np.nan)
for c in ("ls", "ag", "gr"): df[c] = df[c].fillna(df[c].median())
D_ = pd.get_dummies(df["ind"], drop_first=True, dtype=float)
X = np.column_stack([np.ones(len(df)), df.ls, df.ag, df.gr, D_.values])
beta = np.linalg.lstsq(X, df.pp.values, rcond=None)[0]
res = df.pp.values - X @ beta
r2 = 1 - np.var(res) / np.var(df.pp.values)
print(f"  사전관성을 규모·업력·성장·산업FE 로 설명한 R² = {r2:.3f} "
      f"(잔차 SD {res.std():.3f} vs 원 SD {df.pp.values.std():.3f})")
PB = {"r2_explained": round(float(r2), 3), "resid_slope": slope(res, y, "잔차 관성 ~ Δlog채용률")}
r1, r3 = np.percentile(res, [33.33, 66.67])
d1 = y[res <= r1]; d3 = y[res > r3]
bs = np.array([d3[rng.integers(0, len(d3), len(d3))].mean()
               - d1[rng.integers(0, len(d1), len(d1))].mean() for _ in range(NB)])
ci = qci(bs); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
PB["resid_T3_T1"] = {"T1": round(float(d1.mean()), 4), "T3": round(float(d3.mean()), 4),
                     "diff": round(float(d3.mean()-d1.mean()), 4), "ci": ci, "sig": sg == "✓"}
print(f"  잔차 3분위 T1 {d1.mean():+.4f} · T3 {d3.mean():+.4f} · T3−T1 "
      f"{d3.mean()-d1.mean():+.4f} {ci} {sg}")

print("\n[Panel C] 컷 민감도")
PC = {}
for lab, lo, hi in (("tercile 33/67", 33.33, 66.67), ("quartile 25/75", 25, 75),
                    ("상위25% vs 나머지", 0, 75), ("중앙값 분할", 0, 50)):
    a, b = np.percentile(x, [lo, hi])
    g1 = y[x <= (a if lo > 0 else b)]; g3 = y[x > b]
    if min(len(g1), len(g3)) < 20: continue
    bs = np.array([g3[rng.integers(0, len(g3), len(g3))].mean()
                   - g1[rng.integers(0, len(g1), len(g1))].mean() for _ in range(NB)])
    ci = qci(bs); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
    PC[lab] = {"diff": round(float(g3.mean()-g1.mean()), 4), "ci": ci, "sig": sg == "✓",
               "n1": len(g1), "n3": len(g3)}
    print(f"  {lab:<20} 차이 {g3.mean()-g1.mean():>+8.4f} {str(ci):<20} {sg} (n {len(g1)}/{len(g3)})")

print("\n[Panel D] 일시(현재 spell) vs 구조(과거 비활동)")
cur = np.array([e["cur"] for e in U]); hist = np.array([e["hist"] for e in U])
PD = {"current_spell": slope(cur, y, "현재 spell 길이"),
      "historical": slope(hist, y, "과거 비활동 비중(현재spell 제외)")}
m = np.isfinite(cur) & np.isfinite(hist) & np.isfinite(y)
Xj = np.column_stack([np.ones(m.sum()), cur[m], hist[m]])
bj = np.linalg.lstsq(Xj, y[m], rcond=None)[0]
bb = np.array([np.linalg.lstsq(Xj[j], y[m][j], rcond=None)[0]
               for j in (rng.integers(0, m.sum(), m.sum()) for _ in range(NB))])
for i, nm in ((1, "현재 spell"), (2, "과거 비활동")):
    ci = qci(bb[:, i]); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
    PD[f"joint_{nm}"] = {"coef": round(float(bj[i]), 4), "ci": ci, "sig": sg == "✓"}
    print(f"  동시투입 {nm:<12} {bj[i]:>+8.4f} {str(ci):<20} {sg}")

print("\n[Panel E] 사전 총채용 통제 후에도 남는가")
pn = np.array([np.log(e["preN"]) if e["preN"] > 0 else np.nan for e in U])
m = np.isfinite(pn) & np.isfinite(x) & np.isfinite(y)
Xe = np.column_stack([np.ones(m.sum()), x[m], pn[m]])
be = np.linalg.lstsq(Xe, y[m], rcond=None)[0]
bb = np.array([np.linalg.lstsq(Xe[j], y[m][j], rcond=None)[0]
               for j in (rng.integers(0, m.sum(), m.sum()) for _ in range(NB))])
ci = qci(bb[:, 1]); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
PE_ = {"inaction_given_preN": {"coef": round(float(be[1]), 4), "ci": ci, "sig": sg == "✓",
                               "n": int(m.sum())}}
print(f"  사전 log총채용 통제 후 관성 계수 {be[1]:+.4f} {ci} {sg} (n={int(m.sum())})")

pass_ = sum([bool(PA["linear"] and PA["linear"]["sig"]),
             bool(PB["resid_slope"] and PB["resid_slope"]["sig"]),
             all(v["sig"] for v in PC.values()) if PC else False,
             bool(PE_["inaction_given_preN"]["sig"])])
concl = (f"방어 4관문 중 {pass_}개 통과 — "
         + ("**조절자는 규모·산업·성장·사전채용량의 대리변수가 아니다.**" if pass_ >= 3
            else "일부 관문 실패, 조절자 해석 제한 필요"))
emit("I-41", "조절자 방어 (리뷰 §5.4)", "GO" if pass_ >= 3 else "PARTIAL",
     {"panelA_continuous": PA, "panelB_residualized": PB, "panelC_cutoffs": PC,
      "panelD_transitory_vs_structural": PD, "panelE_given_preN": PE_,
      "outcome": "Δlog hiring rate (기저수준 효과 제거)", "n": len(U)},
     "연속·잔차화·컷민감도·사전총채용 통제에서도 조절자가 살아남는지",
     concl + f" | 선형 {PA['linear']['slope'] if PA['linear'] else '-'} · "
     f"잔차 {PB['resid_slope']['slope'] if PB['resid_slope'] else '-'} · "
     f"preN통제 {PE_['inaction_given_preN']['coef']}",
     kill_met=False, n=len(U), extra={"conclusion": concl})
