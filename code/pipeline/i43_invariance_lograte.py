# -*- coding: utf-8 -*-
"""I-43 불변성 재계산 — 결과대상을 Δlog 채용률로 교체.

경위: I-38/I-39 가 무채용월 비중(및 spell·집중도)의 변화는 **총채용량 증가가 산술적으로 함의하는
만큼만** 움직인다는 것을 보였다. 그 결과 원고의 헤드라인은 Δlog 채용률로 이동했다.
그런데 §8 '거래 특성은 반응을 예측하지 못한다'의 네 비교(I-16 딜유형 · I-14 지분 dose ·
I-17 GP LOO · GP 경험)는 전부 **강등된 지표 위에서** 계산돼 있었다. 같은 대비를 새 결과대상에서
다시 계산하지 않으면 §8 은 헤드라인과 다른 변수에 대한 진술이 된다.

Panel A  딜유형 Buyout vs Growth
Panel B  취득지분율 dose (연속 기울기 + 3분위)
Panel C  GP leave-one-out 예측 (GP군집 부트 + GP라벨 순열)
Panel D  GP 경험(딜 수) 3분위

주의: '차이 없음'은 등가성으로만 주장한다(규칙 11). SESOI = 헤드라인 기울기 0.3472.
[메모리] float32·고정블록 없음(회귀 미사용, 이벤트 수준 스칼라만). 셀접기 불필요.
"""
import re
import numpy as np, pandas as pd
from h30_common import load, deals, build, emit, SEED, qci, NB, widx, BASE

rng = np.random.default_rng(SEED)
NPERM = 2000
SESOI = 0.3472                      # 헤드라인 T3−T1 기울기. 이보다 작은 차이는 '효과보다 작다'.

print("[I-43] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Ev = G["Hv"], G["Ev"]


def lrate(row, m0, a, b):
    """log(연환산 채용률) = log(창 내 총채용 / 평균고용). 고용 5인 미만 창은 제외."""
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return np.nan
    h, e = Hv[row, c].astype(float), Ev[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()): return np.nan
    if np.mean(e) < 5 or h.sum() <= 0: return np.nan
    return float(np.log(h.sum() / np.mean(e)))


for e in EV:
    a, b = lrate(e["ti"], e["m0"], -12, -1), lrate(e["ti"], e["m0"], 1, 12)
    t = b - a if (np.isfinite(a) and np.isfinite(b)) else np.nan
    cd = [lrate(k, e["m0"], 1, 12) - lrate(k, e["m0"], -12, -1) for k in e["ctrls"]]
    cd = [x for x in cd if np.isfinite(x)]
    e["eff"] = t - float(np.mean(cd)) if (np.isfinite(t) and cd) else np.nan

USE = [e for e in EV if np.isfinite(e["eff"])]
y_all = np.array([e["eff"] for e in USE])
print(f"  Δlog 채용률 산출 가능 이벤트 {len(USE)}/{len(EV)} · 평균 {y_all.mean():+.4f}")


def equiv(ci, d=SESOI):
    if ci is None: return None
    holds = bool(ci[0] > -d and ci[1] < d)
    margin = [round(ci[0] + d, 4), round(d - ci[1], 4)]
    return {"SESOI": d, "holds": holds, "margin": margin,
            "knife_edge": bool(min(margin) < 0.001)}


# 집단별 평균 CI 는 **별도 난수원**을 쓴다. 주 추정치의 부트스트랩 스트림을 소비하면
# 진단을 하나 추가할 때마다 이미 보고한 CI 가 흔들린다 (2026-08-25 실측: 0.0025 → 0.0026).
rng_fig = np.random.default_rng(SEED + 1)


def gmean_ci(v):
    v = np.asarray(v, float)
    return qci(np.array([v[rng_fig.integers(0, len(v), len(v))].mean() for _ in range(NB)]))


def gdiff(y1, y2):
    """두 집단 평균효과의 차이 + 집단별 독립 재표본 부트 CI."""
    y1, y2 = np.asarray(y1, float), np.asarray(y2, float)
    d = float(y1.mean() - y2.mean())
    bo = np.array([y1[rng.integers(0, len(y1), len(y1))].mean()
                   - y2[rng.integers(0, len(y2), len(y2))].mean() for _ in range(NB)])
    ci = qci(bo)
    return {"diff": round(d, 4), "ci": ci, "n1": len(y1), "n2": len(y2),
            "sig": bool(ci[0] > 0 or ci[1] < 0), "equiv": equiv(ci),
            "m1": round(float(y1.mean()), 4), "m2": round(float(y2.mean()), 4),
            "ci1": gmean_ci(y1), "ci2": gmean_ci(y2)}


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b = float(np.polyfit(x, y, 1)[0])
    bo = np.array([np.polyfit(x[i], y[i], 1)[0]
                   for i in (rng.integers(0, len(x), len(x)) for _ in range(NB))])
    ci = qci(bo)
    return {"slope": round(b, 6), "ci": ci, "n": len(x),
            "sig": bool(ci[0] > 0 or ci[1] < 0)}


# ══════════ Panel A 딜유형 ══════════
DT = META["Deal Type"].astype(str)
PCT = pd.to_numeric(META["pct_acq"], errors="coerce")
for e in USE:
    t = DT.get(e["bn"], "NA")
    e["buy"] = 1 if "Buyout" in t else (0 if "Growth" in t else -1)
    e["pct"] = float(PCT.get(e["bn"], np.nan))
yb = [e["eff"] for e in USE if e["buy"] == 1]
yg = [e["eff"] for e in USE if e["buy"] == 0]
PA = gdiff(yb, yg)
print(f"\n[Panel A] Buyout {PA['m1']:+.4f} (n={PA['n1']}) vs Growth {PA['m2']:+.4f} (n={PA['n2']})"
      f"  차이 {PA['diff']:+.4f} {PA['ci']} "
      f"{'✓유의' if PA['sig'] else ('등가✓' if PA['equiv']['holds'] else '미검출·등가✗')}")

# ══════════ Panel B 지분 dose (주주명부 기준 — I-14 와 동일 구성) ══════════
# PitchBook pct_acq 는 100% 에 질량이 몰려 3분위가 붕괴한다(중앙값 100.0, D3 공집합).
# 원고가 서술하는 dose 는 '진입연도 주주명부상 PE 보통주 지분율 합'이므로 그것을 재구성한다.
import gc
T_ = pd.read_csv(f"{BASE}/shared/data/processed/p014_treated_sample_v2_expanded.csv", dtype=str)
TB_ = set(T_.bn10.str.zfill(10))
cols_ = ["business_number", "기준일", "주주명", "주주명_영문", "보통주_지분율"]
parts = []
for ch in pd.read_csv(f"{BASE}/PI/drops/외감_주주_시계열_2009plus.csv",
                      usecols=cols_, dtype=str, chunksize=400_000):
    ch["bn10"] = ch.business_number.str.replace(r"\D", "", regex=True).str.zfill(10)
    parts.append(ch[ch.bn10.isin(TB_)])
S = pd.concat(parts, ignore_index=True); del parts; gc.collect()
S["yr"] = pd.to_datetime(S["기준일"], format="%Y%m%d", errors="coerce").dt.year
S["pct"] = pd.to_numeric(S["보통주_지분율"], errors="coerce")
S = S[S.yr.notna()]
EXCL = r"우리사주|자기주식|자사주|종업원지주"
PAT = r"투자|인베스트|캐피탈|사모|펀드|조합|파트너스|에쿼티|벤처|PEF|Capital|Invest|Partner|Equity|Fund|Holdings"
nm_ = S["주주명"].fillna("") + " " + S["주주명_영문"].fillna("")
S["pe"] = nm_.str.contains(PAT, case=False, regex=True) & ~nm_.str.contains(EXCL, regex=True)
ag = S[S.pe].groupby(["bn10", "yr"])["pct"].sum().rename("pe_pct").reset_index()
allyr = S.groupby(["bn10", "yr"]).size().rename("n").reset_index()
Y = allyr.merge(ag, on=["bn10", "yr"], how="left").fillna({"pe_pct": 0.0})
DOSE = {}
for bn, gg in Y.groupby("bn10"):
    pos = gg.sort_values("yr")
    pos = pos[pos.pe_pct > 0]
    if len(pos): DOSE[bn] = float(pos.pe_pct.iloc[0])
del S, Y, allyr, ag; gc.collect()

D = [(DOSE[e["bn"]], e["eff"]) for e in USE if e["bn"] in DOSE]
xs, ys = np.array([d[0] for d in D]), np.array([d[1] for d in D])
PB = {"source": "shareholder register, PE common-share stake at entry year",
      "n_with_dose": len(D), "median_pct": round(float(np.median(xs)), 2),
      "linear": slope(xs, ys)}
maj, mino = ys[xs >= 50], ys[xs < 50]
PB["majority_vs_minority"] = gdiff(maj, mino)
c1, c2 = np.percentile(xs, [33.33, 66.67])
ter = [ys[xs <= c1], ys[(xs > c1) & (xs <= c2)], ys[xs > c2]]
PB["cuts"] = [round(float(c1), 2), round(float(c2), 2)]
PB["terciles"] = {f"D{i+1}": {"n": int(len(v)),
                              "mean": (round(float(v.mean()), 4) if len(v) else None),
                              "ci": (gmean_ci(v) if len(v) else None)}
                  for i, v in enumerate(ter)}
PB["D3_D1"] = gdiff(ter[2], ter[0]) if (len(ter[0]) and len(ter[2])) else None
print(f"[Panel B] 명부 dose n={len(D)} 중앙값 {PB['median_pct']}% · 기울기 "
      f"{PB['linear']['slope']:+.6f}/%p {PB['linear']['ci']} · "
      f"다수(≥50%, n={len(maj)}) {PB['majority_vs_minority']['m1']:+.4f} vs "
      f"소수(n={len(mino)}) {PB['majority_vs_minority']['m2']:+.4f} → "
      f"{PB['majority_vs_minority']['diff']:+.4f} {PB['majority_vs_minority']['ci']} "
      f"{'등가✓' if PB['majority_vs_minority']['equiv']['holds'] else '등가✗'}")

# ══════════ Panel C GP LOO ══════════
pbf = pd.read_csv(f"{BASE}/shared/data/processed/pitchbook_deals_v1.csv", dtype=str)
pbf["bn10"] = pbf.bn.astype(str).str.zfill(10)
pbf["dd"] = pd.to_datetime(pbf["Deal Date"], errors="coerce")
BG = pbf[(pbf.is_bg == "True") & pbf.dd.notna()].sort_values("dd")
INV = BG.drop_duplicates("bn10").set_index("bn10")["Investors"].to_dict()
A = pd.read_csv(f"{BASE}/P014_upgrade_package/matching/work/PB_RECOVERY_FINAL_ADOPTED.csv", dtype=str)
A["bn10"] = A.bn10.astype(str).str.zfill(10)
cnorm = lambda x: re.sub(r"[^0-9a-z가-힣]", "", str(x).lower())
CMAP = BG.assign(k=BG["Companies"].map(cnorm)).drop_duplicates("k").set_index("k")["Investors"].to_dict()
for r in A.itertuples():
    if not isinstance(INV.get(r.bn10), str):
        v = CMAP.get(cnorm(r.pb_company))
        if isinstance(v, str) and v.strip(): INV[r.bn10] = v


def gplist(s):
    if not isinstance(s, str): return []
    out = []
    for t in re.split(r"[,;|]", s):
        t = re.sub(r"\s*\([^)]*\)\s*", " ", t).strip()
        t = re.sub(r"\b(co|ltd|inc|corp|llc|lp|l\.p\.|limited|company)\b\.?", "", t, flags=re.I).strip(" .,")
        if len(t) >= 3: out.append(t.lower())
    return out


for e in USE:
    g = gplist(INV.get(e["bn"])); e["gp"] = g[0] if g else None
UG = [e for e in USE if e["gp"]]
y = np.array([e["eff"] for e in UG]); g = np.array([e["gp"] for e in UG])
cnt = pd.Series(g).value_counts()


def loo_beta(yy, gg):
    df = pd.DataFrame({"y": yy, "g": gg})
    s = df.groupby("g")["y"].transform("sum"); n = df.groupby("g")["y"].transform("size")
    loo = (s - df.y) / (n - 1)
    ok = np.isfinite(loo) & (n > 1)
    if ok.sum() < 20: return np.nan, 0
    return float(np.polyfit(loo[ok].values, df.y[ok].values, 1)[0]), int(ok.sum())


obs, n_loo = loo_beta(y, g)
nullb = np.array([loo_beta(y, rng.permutation(g))[0] for _ in range(NPERM)])
nullb = nullb[np.isfinite(nullb)]
p_perm = float((nullb >= obs).mean())
GPS = np.array(sorted(set(g)))
bo = []
for _ in range(NB):
    pick = rng.integers(0, len(GPS), len(GPS)); yy, gg = [], []
    for r, k in enumerate(pick):
        m = g == GPS[k]; yy.append(y[m]); gg.append(np.full(m.sum(), f"{GPS[k]}#{r}"))
    b_, _n = loo_beta(np.concatenate(yy), np.concatenate(gg))
    if np.isfinite(b_): bo.append(b_)
ci_loo = qci(np.array(bo))
PC = {"loo_beta": round(obs, 4), "ci": ci_loo, "n_loo": n_loo, "perm_p": round(p_perm, 4),
      "n_with_gp": len(UG), "n_unique_gp": int(len(cnt)),
      "sig": bool(ci_loo[0] > 0 or ci_loo[1] < 0), "equiv": equiv(ci_loo), "n_perm": len(nullb)}
print(f"[Panel C] GP LOO β {obs:+.4f} {ci_loo} · 순열 p {p_perm:.4f} "
      f"(GP {len(cnt)}개, LOO 대상 {n_loo}) {'등가✓' if PC['equiv']['holds'] else '등가✗'}")

# ══════════ Panel D GP 경험 ══════════
exp = np.array([cnt[e["gp"]] for e in UG], float)
q1, q2 = np.percentile(exp, [33.33, 66.67])
bins = [y[exp <= q1], y[(exp > q1) & (exp <= q2)], y[exp > q2]]
PD = {f"E{i+1}": {"n": len(v), "mean": round(float(v.mean()), 4), "ci": gmean_ci(v)}
      for i, v in enumerate(bins)}
PD["E3_E1"] = gdiff(bins[2], bins[0]); PD["cuts"] = [round(float(q1), 1), round(float(q2), 1)]
mono = (PD["E1"]["mean"] <= PD["E2"]["mean"] <= PD["E3"]["mean"]) or \
       (PD["E1"]["mean"] >= PD["E2"]["mean"] >= PD["E3"]["mean"])
PD["monotone"] = bool(mono)
print(f"[Panel D] 경험 3분위 {[PD[f'E{i}']['mean'] for i in (1,2,3)]} "
      f"단조 {'✓' if mono else '✗'} · E3−E1 {PD['E3_E1']['diff']:+.4f} {PD['E3_E1']['ci']}")

# ══════════ 판정 ══════════
tests = {"딜유형": PA, "지분 다수vs소수": PB["majority_vs_minority"], "GP LOO": PC, "GP경험 E3−E1": PD["E3_E1"]}
n_sig = sum(1 for v in tests.values() if v["sig"])
n_eq = sum(1 for v in tests.values() if v["equiv"] and v["equiv"]["holds"])
verdict = (f"Δlog 채용률에서 4개 거래특성 비교 중 유의 {n_sig}건 · 등가성 성립 {n_eq}건. "
           f"딜유형 {PA['diff']:+.4f}{PA['ci']} · 지분기울기 {PB['linear']['slope']:+.6f}/%p · "
           f"GP LOO {PC['loo_beta']:+.4f}(순열 p {PC['perm_p']:.2f}) · "
           f"경험 E3−E1 {PD['E3_E1']['diff']:+.4f}. "
           + ("거래 특성은 반응을 조직하지 않는다 — 강등된 지표가 아니라 헤드라인 결과대상에서도 동일."
              if n_sig == 0 else
              f"{n_sig}건이 유의 — §8 불변성 주장은 그만큼 축소해야 한다."))
emit("I-43", "불변성 재계산 (Δlog 채용률)", "GO" if n_sig == 0 else "PARTIAL",
     {"panelA_deal_type": PA, "panelB_stake_dose": PB, "panelC_gp_loo": PC,
      "panelD_gp_experience": PD, "outcome": "Δlog hiring rate", "SESOI": SESOI,
      "n_events_usable": len(USE)},
     "강등된 무채용월 지표가 아니라 헤드라인 Δlog 채용률에서도 거래특성이 반응을 예측하지 못하는가",
     verdict, kill_met=bool(n_sig >= 2), n=len(USE),
     extra={"reason": "I-38/I-39 로 결과대상 이동 후 §8 을 같은 대상에서 재계산",
            "supersedes_for_manuscript": ["I-16 PanelA", "I-14 PanelC", "I-17 PanelC"]})
