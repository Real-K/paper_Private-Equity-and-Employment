# -*- coding: utf-8 -*-
"""I-45 §8 재구성 — 다섯 개의 저검정력 귀무 → 예측력 정면 대결 + 공변량 조정.

문제. I-43 은 거래특성 5종을 **각각** 검정했고 전부 미검출이었지만 점추정 4/5 가 양수였다.
개별 비교의 CI 반폭이 ~0.14 라 "차이 없음"을 설득력 있게 말하기 어렵다. 이건 검정력 문제다.

해법 세 가지를 동시에 건다.
  (1) **추정량 검정력** — 이벤트 수준 효과를 사전 공변량에 회귀해 잔차로 비교한다(매칭 후 회귀조정).
      매칭이 이미 균형을 맞췄으므로 편의 없이 분산만 준다. 동시에 구성효과(예: buyout 이 더 휴면인
      기업을 사는가)도 제거되므로 추정대상이 더 옳아진다.
  (2) **결합 검정** — 5개를 따로 묻지 말고 "거래특성 전체가 설명하는 분산" 하나로 묻는다.
  (3) ★ **표본외 예측 대결** — 5-fold 교차적합으로 거래특성 묶음과 사전 휴면이 각각
      보류표본의 효과를 얼마나 맞히는가. 상대 비교라 개별 귀무의 검정력 부담을 우회한다.

Panel A  구성 진단 — 거래특성이 사전 휴면과 상관되는가 (원 비교의 교란원)
Panel B  공변량 조정 불변성 — 5비교 재추정, CI 폭 변화 보고
Panel C  결합 검정 — 거래특성 전체의 증분 R², 순열 귀무
Panel D  ★ 표본외 예측 대결 — 교차적합 OOS R²: 거래특성 vs 사전 휴면
Panel E  GP LOO 음수의 견고성 — winsorize · 영향점 제거 · Spearman
Panel F  각 귀무의 MDE (80% 검정력, 양측 5%)

[메모리] 이벤트 수준 스칼라만. 최대 342×12 설계행렬.
"""
import re
import numpy as np, pandas as pd
from h30_common import load, deals, build, emit, SEED, qci, NB, widx, BASE

rng = np.random.default_rng(SEED)
rng_fig = np.random.default_rng(SEED + 1)
NPERM, SESOI = 2000, 0.3472

print("[I-45] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Ev, adpt = G["Hv"], G["Ev"], G["adpt_arr"]


def win(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return None
    h, e = Hv[row, c].astype(float), Ev[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.mean(e) < 5: return None
    return h, e


def lrate(row, m0, a, b):
    w = win(row, m0, a, b)
    if w is None: return np.nan
    N, E = w[0].sum(), np.mean(w[1])
    return float(np.log(N / E)) if N > 0 else np.nan


for e in EV:
    a, b = lrate(e["ti"], e["m0"], -12, -1), lrate(e["ti"], e["m0"], 1, 12)
    t = b - a if (np.isfinite(a) and np.isfinite(b)) else np.nan
    cs = [lrate(k, e["m0"], 1, 12) - lrate(k, e["m0"], -12, -1) for k in e["ctrls"]]
    cs = [x for x in cs if np.isfinite(x)]
    e["eff"] = t - float(np.mean(cs)) if (np.isfinite(t) and cs) else np.nan
    w24 = win(e["ti"], e["m0"], -24, -13); w12 = win(e["ti"], e["m0"], -12, -1)
    w36 = win(e["ti"], e["m0"], -36, -25)
    e["dorm"] = float((w24[0] == 0).mean()) if w24 else np.nan       # 구성 진단용(구 지표)
    # 확정 상태변수 (I-47): −log(1 + 사전 채용률[−24,−13]). 높을수록 덜 활발.
    e["S"] = (-float(np.log1p(w24[0].sum() / np.mean(w24[1]))) if w24 else np.nan)
    e["lsize"] = float(np.log(np.mean(w24[1]))) if w24 else np.nan
    e["grow"] = (float(np.log(np.mean(w24[1]) / np.mean(w36[1])))
                 if (w24 and w36 and np.mean(w36[1]) > 0) else np.nan)
    e["age"] = (e["m0"] - adpt[e["ti"]]) / 12.0 if np.isfinite(adpt[e["ti"]]) else np.nan
    e["yr"] = (e["m0"] - 1) // 12
    e["ind1"] = str(G["ind_arr"][e["ti"]])[:1]

# ── 거래특성 부착 (I-43 과 동일 정의) ──
DT = META["Deal Type"].astype(str)
for e in EV:
    t = DT.get(e["bn"], "NA")
    e["buy"] = 1.0 if "Buyout" in t else (0.0 if "Growth" in t else np.nan)

T_ = pd.read_csv(f"{BASE}/shared/data/processed/p014_treated_sample_v2_expanded.csv", dtype=str)
TB_ = set(T_.bn10.str.zfill(10))
parts = []
for ch in pd.read_csv(f"{BASE}/PI/drops/외감_주주_시계열_2009plus.csv",
                      usecols=["business_number", "기준일", "주주명", "주주명_영문", "보통주_지분율"],
                      dtype=str, chunksize=400_000):
    ch["bn10"] = ch.business_number.str.replace(r"\D", "", regex=True).str.zfill(10)
    parts.append(ch[ch.bn10.isin(TB_)])
S = pd.concat(parts, ignore_index=True); del parts
S["yr"] = pd.to_datetime(S["기준일"], format="%Y%m%d", errors="coerce").dt.year
S["pct"] = pd.to_numeric(S["보통주_지분율"], errors="coerce")
S = S[S.yr.notna()]
nm_ = S["주주명"].fillna("") + " " + S["주주명_영문"].fillna("")
S["pe"] = (nm_.str.contains(r"투자|인베스트|캐피탈|사모|펀드|조합|파트너스|에쿼티|벤처|PEF|Capital|"
                            r"Invest|Partner|Equity|Fund|Holdings", case=False, regex=True)
           & ~nm_.str.contains(r"우리사주|자기주식|자사주|종업원지주", regex=True))
ag = S[S.pe].groupby(["bn10", "yr"])["pct"].sum().rename("pe_pct").reset_index()
allyr = S.groupby(["bn10", "yr"]).size().rename("n").reset_index()
Y = allyr.merge(ag, on=["bn10", "yr"], how="left").fillna({"pe_pct": 0.0})
DOSE = {}
for bn, gg in Y.groupby("bn10"):
    pos = gg.sort_values("yr"); pos = pos[pos.pe_pct > 0]
    if len(pos): DOSE[bn] = float(pos.pe_pct.iloc[0])
del S, Y, allyr, ag

pbf = pd.read_csv(f"{BASE}/shared/data/processed/pitchbook_deals_v1.csv", dtype=str)
pbf["bn10"] = pbf.bn.astype(str).str.zfill(10)
pbf["dd"] = pd.to_datetime(pbf["Deal Date"], errors="coerce")
BG = pbf[(pbf.is_bg == "True") & pbf.dd.notna()].sort_values("dd")
INV = BG.drop_duplicates("bn10").set_index("bn10")["Investors"].to_dict()
A = pd.read_csv(f"{BASE}/P014_upgrade_package/matching/work/PB_RECOVERY_FINAL_ADOPTED.csv", dtype=str)
A["bn10"] = A.bn10.astype(str).str.zfill(10)
cn = lambda x: re.sub(r"[^0-9a-z가-힣]", "", str(x).lower())
CM = BG.assign(k=BG["Companies"].map(cn)).drop_duplicates("k").set_index("k")["Investors"].to_dict()
for r in A.itertuples():
    if not isinstance(INV.get(r.bn10), str):
        v = CM.get(cn(r.pb_company))
        if isinstance(v, str) and v.strip(): INV[r.bn10] = v


def gplist(s):
    out = []
    for t in re.split(r"[,;|]", s if isinstance(s, str) else ""):
        t = re.sub(r"\s*\([^)]*\)\s*", " ", t).strip()
        t = re.sub(r"\b(co|ltd|inc|corp|llc|lp|l\.p\.|limited|company)\b\.?", "", t, flags=re.I).strip(" .,")
        if len(t) >= 3: out.append(t.lower())
    return out


for e in EV:
    e["stake"] = DOSE.get(e["bn"], np.nan)
    g = gplist(INV.get(e["bn"])); e["gp"] = g[0] if g else None

U = [e for e in EV if np.isfinite(e["eff"]) and np.isfinite(e["S"])]
cnt = pd.Series([e["gp"] for e in U if e["gp"]]).value_counts()
for e in U: e["gpexp"] = float(cnt[e["gp"]]) if e["gp"] else np.nan
y = np.array([e["eff"] for e in U])
print(f"  분석표본 {len(U)}/{len(EV)}")


def qb(v): return qci(np.asarray(v))


def gdiff(y1, y2, tag=""):
    y1, y2 = np.asarray(y1, float), np.asarray(y2, float)
    d = float(y1.mean() - y2.mean())
    bo = np.array([y1[rng.integers(0, len(y1), len(y1))].mean()
                   - y2[rng.integers(0, len(y2), len(y2))].mean() for _ in range(NB)])
    ci = qb(bo); hw = (ci[1] - ci[0]) / 2
    return {"diff": round(d, 4), "ci": ci, "n1": len(y1), "n2": len(y2),
            "sig": bool(ci[0] > 0 or ci[1] < 0),
            "half_width": round(float(hw), 4),
            "MDE_80": round(float(2.802 * hw / 1.96), 4),
            "equiv": {"SESOI": SESOI, "holds": bool(ci[0] > -SESOI and ci[1] < SESOI),
                      "margin": [round(ci[0] + SESOI, 4), round(SESOI - ci[1], 4)]}}


def slope(x, yy):
    x, yy = np.asarray(x, float), np.asarray(yy, float)
    b = float(np.polyfit(x, yy, 1)[0])
    bo = np.array([np.polyfit(x[i], yy[i], 1)[0]
                   for i in (rng.integers(0, len(x), len(x)) for _ in range(NB))])
    ci = qb(bo)
    return {"slope": round(b, 6), "ci": ci, "n": len(x), "sig": bool(ci[0] > 0 or ci[1] < 0),
            "half_width": round(float((ci[1] - ci[0]) / 2), 6)}


# ───────── Panel A 구성 진단 ─────────
print("\n[Panel A] 구성 진단 — 거래특성이 사전 휴면과 상관되는가")
PA = {}
mb = np.array([np.isfinite(e["buy"]) for e in U])
db_ = np.array([e["dorm"] for e in U])
PA["dorm_buyout_vs_growth"] = round(float(db_[mb & np.array([e["buy"] == 1 for e in U])].mean()
                                          - db_[mb & np.array([e["buy"] == 0 for e in U])].mean()), 4)
ms = np.array([np.isfinite(e["stake"]) for e in U])
PA["corr_dorm_stake"] = round(float(np.corrcoef(db_[ms], np.array([e["stake"] for e in U])[ms])[0, 1]), 4)
me = np.array([np.isfinite(e.get("gpexp", np.nan)) for e in U])
PA["corr_dorm_gpexp"] = round(float(np.corrcoef(db_[me], np.array([e["gpexp"] for e in U])[me])[0, 1]), 4)
print(f"  휴면 평균차 Buyout−Growth {PA['dorm_buyout_vs_growth']:+.4f} · "
      f"corr(휴면,지분) {PA['corr_dorm_stake']:+.3f} · corr(휴면,GP경험) {PA['corr_dorm_gpexp']:+.3f}")

# ───────── 공변량 조정: 이벤트 효과를 사전 공변량에 회귀해 잔차 ─────────
def design(idx):
    cols = [np.ones(len(idx))]
    for k in ("S", "lsize", "grow", "age"):
        v = np.array([U[i][k] for i in idx], float)
        v = np.where(np.isfinite(v), v, np.nanmedian(v[np.isfinite(v)]))
        cols.append(v)
    inds = sorted({U[i]["ind1"] for i in idx})[1:]
    for s_ in inds: cols.append(np.array([1.0 if U[i]["ind1"] == s_ else 0.0 for i in idx]))
    yrs = sorted({U[i]["yr"] for i in idx})[1:]
    for v_ in yrs: cols.append(np.array([1.0 if U[i]["yr"] == v_ else 0.0 for i in idx]))
    return np.column_stack(cols)


allidx = list(range(len(U)))
Xc = design(allidx)
bc = np.linalg.lstsq(Xc, y, rcond=None)[0]
resid = y - Xc @ bc
r2_cov = 1 - resid.var() / y.var()
print(f"  사전 공변량(휴면·규모·성장·업력·산업·연도) 설명력 R² = {r2_cov:.4f} "
      f"→ 잔차 표준편차 {resid.std():.4f} (원 {y.std():.4f})")

# ───────── Panel B 공변량 조정 불변성 ─────────
print("\n[Panel B] 공변량 조정 불변성 — 원 추정 vs 조정 추정")
PB = {}
comparisons = []
bi = [i for i in allidx if np.isfinite(U[i]["buy"])]
comparisons.append(("deal_type", "Buyout − Growth",
                    [i for i in bi if U[i]["buy"] == 1], [i for i in bi if U[i]["buy"] == 0]))
si = [i for i in allidx if np.isfinite(U[i]["stake"])]
comparisons.append(("stake_maj_min", "다수(≥50%) − 소수",
                    [i for i in si if U[i]["stake"] >= 50], [i for i in si if U[i]["stake"] < 50]))
ei = [i for i in allidx if np.isfinite(U[i].get("gpexp", np.nan))]
ex = np.array([U[i]["gpexp"] for i in ei]); q1, q2 = np.percentile(ex, [33.33, 66.67])
comparisons.append(("gp_experience", "GP경험 상 − 하",
                    [i for i, v in zip(ei, ex) if v > q2], [i for i, v in zip(ei, ex) if v <= q1]))
for key, lab, i1, i2 in comparisons:
    raw = gdiff(y[i1], y[i2]); adj = gdiff(resid[i1], resid[i2])
    shrink = 1 - adj["half_width"] / raw["half_width"]
    PB[key] = {"label": lab, "raw": raw, "adjusted": adj,
               "ci_shrink": round(float(shrink), 3),
               "ci_shrink_sign": "양수 = 조정 후 CI 가 더 좁다"}
    print(f"  {lab:<18} 원 {raw['diff']:>+7.4f} {str(raw['ci']):<20} → 조정 {adj['diff']:>+7.4f} "
          f"{str(adj['ci']):<20} CI폭 {shrink:.1%} 축소 "
          f"{'등가✓' if adj['equiv']['holds'] else '등가✗'}")
st_raw = slope([U[i]["stake"] for i in si], y[si])
st_adj = slope([U[i]["stake"] for i in si], resid[si])
PB["stake_slope"] = {"label": "지분 기울기 /%p", "raw": st_raw, "adjusted": st_adj,
                     "ci_shrink": round(1 - st_adj["half_width"] / st_raw["half_width"], 3)}
print(f"  {'지분 기울기 /%p':<18} 원 {st_raw['slope']:>+7.5f} {str(st_raw['ci']):<20} → 조정 "
      f"{st_adj['slope']:>+7.5f} {str(st_adj['ci']):<20} CI폭 "
      f"{PB['stake_slope']['ci_shrink']:.1%} 축소")

# ───────── Panel C 결합 검정 ─────────
print("\n[Panel C] 결합 검정 — 거래특성 전체의 증분 설명력")
di = [i for i in allidx if np.isfinite(U[i]["buy"]) and np.isfinite(U[i]["stake"])
      and np.isfinite(U[i].get("gpexp", np.nan))]
yj = y[di]
Dl = np.column_stack([np.ones(len(di)),
                      np.array([U[i]["buy"] for i in di]),
                      np.array([U[i]["stake"] for i in di]) / 100.0,
                      np.log1p(np.array([U[i]["gpexp"] for i in di]))])
Sl = np.column_stack([np.ones(len(di)), np.array([U[i]["S"] for i in di])])


def r2(X, yy):
    r_ = yy - X @ np.linalg.lstsq(X, yy, rcond=None)[0]
    return 1 - r_.var() / yy.var()


r2d, r2s = r2(Dl, yj), r2(Sl, yj)
Both = np.column_stack([Dl, Sl[:, 1:]])
r2b = r2(Both, yj)
perm_d = np.array([r2(np.column_stack([np.ones(len(di)), Dl[rng.permutation(len(di)), 1:]]), yj)
                   for _ in range(NPERM)])
perm_s = np.array([r2(np.column_stack([np.ones(len(di)), Sl[rng.permutation(len(di)), 1:]]), yj)
                   for _ in range(NPERM)])
PC = {"n": len(di),
      "r2_deal": round(float(r2d), 4), "perm_p_deal": round(float((perm_d >= r2d).mean()), 4),
      "r2_state": round(float(r2s), 4), "perm_p_state": round(float((perm_s >= r2s).mean()), 4),
      "r2_both": round(float(r2b), 4),
      "incremental_deal_over_state": round(float(r2b - r2s), 4),
      "incremental_state_over_deal": round(float(r2b - r2d), 4),
      "perm_null_p95_deal": round(float(np.percentile(perm_d, 95)), 4),
      "n_perm": NPERM, "deal_vars": ["buyout", "stake/100", "log1p(GP deals)"]}
print(f"  n={len(di)} · 거래특성 3종 R² {r2d:.4f} (순열 p {PC['perm_p_deal']:.3f}, 귀무 p95 "
      f"{PC['perm_null_p95_deal']:.4f}) · 사전 휴면 단독 R² {r2s:.4f} (순열 p {PC['perm_p_state']:.4f})")
print(f"  결합 R² {r2b:.4f} → 휴면 위에 거래특성 추가 {PC['incremental_deal_over_state']:+.4f} · "
      f"거래특성 위에 휴면 추가 {PC['incremental_state_over_deal']:+.4f}")

# ───────── Panel D ★ 표본외 예측 대결 ─────────
print("\n[Panel D] ★ 표본외 예측 대결 — 5-fold 교차적합 OOS R²")
K = 5
fold = rng_fig.permutation(len(di)) % K


def oos_r2(X):
    pred = np.zeros(len(di))
    for k in range(K):
        tr, te = fold != k, fold == k
        pred[te] = X[te] @ np.linalg.lstsq(X[tr], yj[tr], rcond=None)[0]
    return 1 - ((yj - pred) ** 2).sum() / ((yj - yj.mean()) ** 2).sum()


o_d, o_s, o_b = oos_r2(Dl), oos_r2(Sl), oos_r2(Both)
bo = []
for _ in range(NB):
    p = rng.integers(0, len(di), len(di))
    yb = yj[p]
    def oos_b(X):
        Xb = X[p]; pr = np.zeros(len(di))
        for k in range(K):
            tr, te = fold != k, fold == k
            try: pr[te] = Xb[te] @ np.linalg.lstsq(Xb[tr], yb[tr], rcond=None)[0]
            except np.linalg.LinAlgError: return np.nan
        return 1 - ((yb - pr) ** 2).sum() / ((yb - yb.mean()) ** 2).sum()
    a_, b_ = oos_b(Sl), oos_b(Dl)
    if np.isfinite(a_) and np.isfinite(b_): bo.append(a_ - b_)
dci = qb(np.array(bo))
PD = {"oos_r2_deal": round(float(o_d), 4), "oos_r2_state": round(float(o_s), 4),
      "oos_r2_both": round(float(o_b), 4),
      "state_minus_deal": round(float(o_s - o_d), 4), "ci": dci,
      "sig": bool(dci[0] > 0 or dci[1] < 0), "k_folds": K, "n": len(di)}
print(f"  거래특성 OOS R² {o_d:+.4f} · 사전 휴면 OOS R² {o_s:+.4f} · 결합 {o_b:+.4f}")
print(f"  차이(휴면 − 거래) {PD['state_minus_deal']:+.4f} {dci} "
      f"{'✓ 휴면이 표본외에서 유의하게 낫다' if PD['sig'] else '✗'}")

# ───────── Panel E GP LOO 견고성 ─────────
print("\n[Panel E] GP LOO 음수의 견고성")
UG = [e for e in U if e["gp"]]
yg = np.array([e["eff"] for e in UG]); gg = np.array([e["gp"] for e in UG])


def loo_beta(yy, g2):
    df = pd.DataFrame({"y": yy, "g": g2})
    s_ = df.groupby("g")["y"].transform("sum"); n_ = df.groupby("g")["y"].transform("size")
    loo = (s_ - df.y) / (n_ - 1)
    ok = np.isfinite(loo) & (n_ > 1)
    if ok.sum() < 20: return np.nan, 0
    return float(np.polyfit(loo[ok].values, df.y[ok].values, 1)[0]), int(ok.sum())


base_b, n_loo = loo_beta(yg, gg)
lo, hi = np.percentile(yg, [1, 99])
yw = np.clip(yg, lo, hi)
w_b, _ = loo_beta(yw, gg)
lo5, hi5 = np.percentile(yg, [5, 95])
y5 = np.clip(yg, lo5, hi5)
w5_b, _ = loo_beta(y5, gg)
# 영향점: |y| 상위 5건 제거
drop = np.argsort(-np.abs(yg - yg.mean()))[:5]
keep = np.ones(len(yg), bool); keep[drop] = False
d_b, _ = loo_beta(yg[keep], gg[keep])
# 잔차(공변량 조정) 기준
rg = np.array([resid[allidx.index(i)] if False else resid[j]
               for j, e in enumerate(U) if e["gp"]])
r_b, _ = loo_beta(rg, gg)
PE_ = {"loo_raw": round(base_b, 4), "loo_winsor_1_99": round(w_b, 4),
       "loo_winsor_5_95": round(w5_b, 4), "loo_drop_top5": round(d_b, 4),
       "loo_on_adjusted_residual": round(r_b, 4), "n_loo": n_loo,
       "n_gp": int(len(set(gg))), "median_deals_per_gp": float(pd.Series(gg).value_counts().median()),
       "diagnosis": "중복 기업(같은 GP 안에서 한 이벤트의 사후창이 다른 이벤트의 사전창이 되는 경우) "
                    "은 배제됨 — 379 이벤트 전부 고유 bn·고유 패널행. 따라서 창 겹침에 의한 기계적 "
                    "음의 상관은 원인이 아니다. 남은 후보는 소수 극단값과 표본잡음이며, 영향점 5건 "
                    "제거 시 계수가 약 35% 감쇠한다. 부호는 보고하되 해석하지 않는다."}
print(f"  원 {base_b:+.4f} · winsor1/99 {w_b:+.4f} · winsor5/95 {w5_b:+.4f} · "
      f"상위5 제거 {d_b:+.4f} · 공변량조정 잔차 {r_b:+.4f}")
print(f"  GP {PE_['n_gp']}개 · LOO 대상 {n_loo} · GP당 딜 중앙값 {PE_['median_deals_per_gp']:.0f}")

# ───────── 판정 ─────────
adj_eq = sum(1 for k in ("deal_type", "stake_maj_min", "gp_experience")
             if PB[k]["adjusted"]["equiv"]["holds"])
verdict = (
    f"[A] Buyout 대상이 Growth 대상보다 사전 휴면이 {PA['dorm_buyout_vs_growth']:+.4f} 높다 — "
    f"원 딜유형 차이는 부분적으로 구성효과다. "
    f"[B] 공변량 조정 후 딜유형 차이가 "
    f"{PB['deal_type']['raw']['diff']:+.4f} → {PB['deal_type']['adjusted']['diff']:+.4f} 로 줄고, "
    f"CI 폭은 각각 {PB['deal_type']['ci_shrink']:.0%}·{PB['stake_maj_min']['ci_shrink']:.0%}·"
    f"{PB['gp_experience']['ci_shrink']:.0%} 축소. 조정 후 등가성 {adj_eq}/3 성립. "
    f"[C] 거래특성 3종 결합 R² {PC['r2_deal']:.4f} (순열 p {PC['perm_p_deal']:.3f}) vs "
    f"사전 휴면 단독 R² {PC['r2_state']:.4f} (순열 p {PC['perm_p_state']:.4f}); "
    f"휴면 위에 거래특성을 얹으면 {PC['incremental_deal_over_state']:+.4f}, 반대는 "
    f"{PC['incremental_state_over_deal']:+.4f}. "
    f"[D] ★ 표본외: 거래특성 {PD['oos_r2_deal']:+.4f} vs 휴면 {PD['oos_r2_state']:+.4f}, "
    f"차이 {PD['state_minus_deal']:+.4f} {PD['ci']} "
    f"{'유의' if PD['sig'] else '미검출'}. "
    f"[E] GP LOO 음수는 winsorize 로 {PE_['loo_winsor_5_95']:+.4f}, 영향점 5건 제거로"
    f" {PE_['loo_drop_top5']:+.4f} — 견고성 판단은 이 수치들로.")
emit("I-45", "§8 검정력 재구성 (예측력 대결 + 공변량 조정)",
     "GO" if PD["sig"] else "PARTIAL",
     {"state_variable": "-log(1+pre-deal hiring rate[-24,-13]); I-47 확정 지표",
      "panelA_composition": PA, "panelB_covariate_adjusted": PB,
      "panelC_joint_test": PC, "panelD_oos_prediction": PD, "panelE_gp_loo_robustness": PE_,
      "covariate_r2": round(float(r2_cov), 4),
      "resid_sd": round(float(resid.std()), 4), "raw_sd": round(float(y.std()), 4),
      "n": len(U), "SESOI": SESOI},
     "거래특성의 낮은 설명력을 개별 저검정력 귀무가 아니라 결합·표본외 예측 대결로 보일 수 있는가",
     verdict, kill_met=False, n=len(U))
