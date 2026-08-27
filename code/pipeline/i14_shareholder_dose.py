# -*- coding: utf-8 -*-
"""I-14 주주명부 기반 진입시점 검증 + 지분율 dose.

세 문제를 동시에 겨냥한다.
  (1) I-01: 딜 월분포 1월 몰림 2.66배 — PitchBook 날짜 대치 의심
  (2) I-16: pct_acq 소수지분 26건뿐 — 통제권 dose 검정력 부재
  (3) I-05: exit 관측 가능성 (선행조건)

자료: PI/drops/외감_주주_시계열_2009plus.csv (239MB, 2009+, 기업당 중앙값 11 기준일).
[해상도 한계] 기준일이 사실상 연 1회이므로 **월 단위 시점 확정은 불가**하다.
가능한 것은 (a) 진입 '구간' 확정과 PitchBook 연도의 독립 검증, (b) 지분율 dose, (c) 소멸(exit) 탐지.

Panel A  커버리지 + PE 주주 탐지 (정밀=PitchBook Investors 대조 / 광의=패턴, 우리사주조합 등 제외)
Panel B  PitchBook 딜일자 vs 주주명부 진입구간 — 검증률, 1월 딜 별도
Panel C  지분율 dose → 결과 (연속 3분위 · 다수/소수)
Panel D  exit 탐지 — PE 지분 소멸 건수 (I-05 게이트)
Panel E  검증된 부분표본으로 헤드라인 재추정
"""
import gc, re
import numpy as np, pandas as pd
from h30_common import (load, deals, build, attach, boot_did_ci, summ, emit,
                        SEED, qci, NB, widx, dflow, BASE)

rng = np.random.default_rng(SEED)
print("[I-14] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
EV, _ = build(G, allt, PE)
Hv, mset, idx = G["Hv"], G["mset"], G["idx"]
EV = attach(G, EV)          # t / cs / rel 부착 (summ 전제)
EVBN = {e["bn"]: e for e in EV}
print(f"  이벤트 {len(EV)}")

# ---- 주주 시계열 (필요 bn 만) ----
T = pd.read_csv(f"{BASE}/shared/data/processed/p014_treated_sample_v2_expanded.csv", dtype=str)
TB = set(T.bn10.str.zfill(10))
cols = ["business_number", "기준일", "주주명", "주주명_영문", "관계", "보통주_지분율"]
parts = []
for ch in pd.read_csv(f"{BASE}/PI/drops/외감_주주_시계열_2009plus.csv",
                      usecols=cols, dtype=str, chunksize=400_000):
    ch["bn10"] = ch.business_number.str.replace(r"\D", "", regex=True).str.zfill(10)
    parts.append(ch[ch.bn10.isin(TB)])
S = pd.concat(parts, ignore_index=True); del parts; gc.collect()
S["dt"] = pd.to_datetime(S["기준일"], format="%Y%m%d", errors="coerce")
S["yr"] = S.dt.dt.year
S["pct"] = pd.to_numeric(S["보통주_지분율"], errors="coerce")
S = S[S.dt.notna()]
print(f"  주주 시계열 매칭 {len(S):,}행 · 기업 {S.bn10.nunique()} / 처치표본 {len(TB)}"
      f" ({S.bn10.nunique()/len(TB):.1%}) · 매칭설계 379 중 {len(set(S.bn10)&set(EVBN))}")

# ---- PE 주주 탐지 ----
EXCL = r"우리사주|자기주식|자사주|종업원지주"
PAT = r"투자|인베스트|캐피탈|사모|펀드|조합|파트너스|에쿼티|벤처|PEF|Capital|Invest|Partner|Equity|Fund|Holdings"
nm = S["주주명"].fillna("") + " " + S["주주명_영문"].fillna("")
S["pe_broad"] = nm.str.contains(PAT, case=False, regex=True) & ~nm.str.contains(EXCL, regex=True)

# 정밀: 해당 딜의 PitchBook Investors 와 대조
pbf = pd.read_csv(f"{BASE}/shared/data/processed/pitchbook_deals_v1.csv", dtype=str)
pbf["bn10"] = pbf.bn.astype(str).str.zfill(10)
INV = (pbf[pbf.is_bg == "True"].dropna(subset=["Investors"])
       .groupby("bn10")["Investors"].apply(lambda x: " | ".join(x)).to_dict())
def toks(s):
    return {t for t in re.split(r"[,|;()]+", str(s).lower()) for t in [t.strip()] if len(t) >= 4}
INVT = {k: toks(v) for k, v in INV.items()}
def precise(r):
    ts = INVT.get(r.bn10)
    if not ts: return False
    e = str(r.주주명_영문 or "").lower()
    return bool(e) and any(t in e or e in t for t in ts)
S["pe_prec"] = [precise(r) for r in S.itertuples()]
print(f"\n[Panel A] PE 주주 탐지  광의 {int(S.pe_broad.sum()):,}행 / 정밀 {int(S.pe_prec.sum()):,}행"
      f"  일치율(광의∩정밀/정밀) {(S.pe_broad & S.pe_prec).sum()/max(S.pe_prec.sum(),1):.1%}")
for tag, c in (("광의", "pe_broad"), ("정밀", "pe_prec")):
    print(f"  {tag}: 한 번이라도 보유한 기업 {S.groupby('bn10')[c].any().sum()} / {S.bn10.nunique()}")
PA = {"n_rows": len(S), "n_firms_sh": int(S.bn10.nunique()), "n_treated_sample": len(TB),
      "coverage": round(S.bn10.nunique() / len(TB), 3),
      "n_in_matched_design": len(set(S.bn10) & set(EVBN)),
      "broad_firms": int(S.groupby("bn10")["pe_broad"].any().sum()),
      "precise_firms": int(S.groupby("bn10")["pe_prec"].any().sum())}

# ---- 기업별 PE 지분 연도별 합 ----
FLAG = "pe_broad"
ag = (S[S[FLAG]].groupby(["bn10", "yr"])["pct"].sum().rename("pe_pct").reset_index())
allyr = S.groupby(["bn10", "yr"]).size().rename("n").reset_index()
Y = allyr.merge(ag, on=["bn10", "yr"], how="left").fillna({"pe_pct": 0.0})

# ---- Panel B : 진입구간 vs PitchBook ----
print("\n[Panel B] PitchBook 딜연도 vs 주주명부 진입구간")
rows = []
for bn, g in Y.groupby("bn10"):
    if bn not in EVBN: continue
    g = g.sort_values("yr")
    pos = g[g.pe_pct > 0]
    e = EVBN[bn]; dy = (e["m0"] - 1) // 12; dm = ((e["m0"] - 1) % 12) + 1
    if pos.empty:
        rows.append((bn, dy, dm, None, None, np.nan, "no_pe_observed")); continue
    fy = int(pos.yr.iloc[0])
    prev = g[g.yr < fy]
    ly = int(prev.yr.iloc[-1]) if len(prev) else None       # 직전 무보유 기준일
    stake = float(pos.pe_pct.iloc[0])
    if ly is None: st = "left_censored" if fy <= dy else "late_first_obs"
    elif ly < dy <= fy: st = "confirmed"
    elif dy <= ly: st = "pb_late"      # 명부상 이미 보유 → PB 날짜가 늦다
    else: st = "pb_early"              # 명부 진입이 PB보다 늦다
    rows.append((bn, dy, dm, ly, fy, stake, st))
V = pd.DataFrame(rows, columns=["bn10", "pb_year", "pb_month", "last_zero_yr", "first_pos_yr", "stake0", "status"])
print("  판정 분포:", V.status.value_counts().to_dict())
conf = V.status == "confirmed"
print(f"  검증률(confirmed) {conf.mean():.1%}  n={len(V)}")
jan = V.pb_month == 1
print(f"  1월 딜 {int(jan.sum())}건 중 confirmed {conf[jan].mean():.1%}"
      f" | 비1월 {int((~jan).sum())}건 중 confirmed {conf[~jan].mean():.1%}")
gapyr = (V.first_pos_yr - V.last_zero_yr).dropna()
print(f"  진입구간 폭(년): 중앙값 {gapyr.median():.0f}  p75 {gapyr.quantile(.75):.0f}  =1년 비중 {(gapyr==1).mean():.1%}")
PB_ = {"status_counts": V.status.value_counts().to_dict(), "confirm_rate": round(float(conf.mean()), 3),
       "n": len(V), "jan_confirm": round(float(conf[jan].mean()), 3) if jan.any() else None,
       "nonjan_confirm": round(float(conf[~jan].mean()), 3),
       "n_jan": int(jan.sum()), "interval_median_yr": float(gapyr.median()) if len(gapyr) else None,
       "interval_eq1_share": round(float((gapyr == 1).mean()), 3) if len(gapyr) else None}

# ---- 결과 스칼라 ----
def zsh(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != 12: return np.nan
    h = Hv[row, c]
    return float((h == 0).mean()) if np.isfinite(h).all() else np.nan
for e in EV:
    a = zsh(e["ti"], e["m0"], -12, -1); b = zsh(e["ti"], e["m0"], 1, 12)
    e["z_t"] = b - a if (np.isfinite(a) and np.isfinite(b)) else np.nan
    cd = [zsh(k, e["m0"], 1, 12) - zsh(k, e["m0"], -12, -1) for k in e["ctrls"]]
    cd = [x for x in cd if np.isfinite(x)]
    e["z_c"] = float(np.mean(cd)) if cd else np.nan
def dtest(s1, s2, lab):
    """두 부분표본의 무채용비중 DiD 차이 (s1 − s2)."""
    f = lambda ss: np.array([e["z_t"] - e["z_c"] for e in ss
                             if np.isfinite(e["z_t"]) and np.isfinite(e["z_c"])])
    d1, d2 = f(s1), f(s2)
    if min(len(d1), len(d2)) < 20: return None
    b = np.array([d1[rng.integers(0, len(d1), len(d1))].mean()
                  - d2[rng.integers(0, len(d2), len(d2))].mean() for _ in range(NB)])
    ci = qci(b); pt = round(float(d1.mean() - d2.mean()), 4)
    sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
    S_ = 0.0460; eqm = [round(ci[0] + S_, 4), round(S_ - ci[1], 4)]
    eq = bool(ci[0] > -S_ and ci[1] < S_)
    print(f"      {lab:<22} 차이 {pt:+.4f} {ci} {sg}"
          f"  등가성(δ=0.046) {'성립' if eq else '미성립'} 여유 {eqm}")
    return {"diff": pt, "ci": ci, "sig": sg == "✓", "n1": len(d1), "n2": len(d2),
            "equivalence": {"SESOI": S_, "holds": eq, "margin": eqm,
                            "knife_edge": bool(min(eqm) < 0.001)}}


def batt(sub, lab):
    s = summ(sub, rng)
    zp, zc, zn = boot_did_ci([e["z_t"] for e in sub], [e["z_c"] for e in sub], rng)
    sg = "✓" if (zc and (zc[0] > 0 or zc[1] < 0)) else "✗"
    print(f"    {lab:<16} n={s.get('n',0):>3} 채용DiD {s.get('DiD')}{s.get('DiD_ci')} | "
          f"무채용 {zp}{zc}{sg} (n={zn})")
    return {**s, "zero_DiD": zp, "zero_ci": zc, "zero_n": zn, "zero_sig": sg == "✓"}

# ---- Panel C : 지분율 dose ----
print("\n[Panel C] 주주명부 지분율 dose (진입 시점 PE 보통주 지분율 합)")
D0 = V[V.stake0.notna() & (V.stake0 > 0)].set_index("bn10")["stake0"].to_dict()
print(f"  dose 부여 가능 {len(D0)} / 매칭설계 {len(EV)}  (PitchBook pct_acq 는 255)")
PC = {"n_with_dose": len(D0), "dose_median": round(float(np.median(list(D0.values()))), 2) if D0 else None}
if len(D0) >= 60:
    v = np.array(list(D0.values())); c1, c2 = np.percentile(v, [33.33, 66.67])
    print(f"  3분위 컷 {c1:.1f}% / {c2:.1f}%  중앙값 {np.median(v):.1f}%")
    PC["cuts"] = [round(float(c1), 2), round(float(c2), 2)]
    for lab, sel in (("D1 저지분", lambda x: x <= c1), ("D2 중간", lambda x: c1 < x <= c2),
                     ("D3 고지분", lambda x: x > c2)):
        sub = [e for e in EV if e["bn"] in D0 and sel(D0[e["bn"]])]
        if len(sub) >= 20: PC[lab] = batt(sub, lab)
    for lab, sel in (("다수>=50%", lambda x: x >= 50), ("소수<50%", lambda x: x < 50)):
        sub = [e for e in EV if e["bn"] in D0 and sel(D0[e["bn"]])]
        PC[f"n_{lab}"] = len(sub)
        if len(sub) >= 20: PC[lab] = batt(sub, lab)
    print(f"  → 다수 {PC.get('n_다수>=50%')}건 / 소수 {PC.get('n_소수<50%')}건 "
          f"(I-16 의 pct_acq 기준 소수는 26건이었다)")
    MAJ = [e for e in EV if e["bn"] in D0 and D0[e["bn"]] >= 50]
    MIN_ = [e for e in EV if e["bn"] in D0 and D0[e["bn"]] < 50]
    PC["diff_major_minor"] = dtest(MAJ, MIN_, "다수 − 소수")
    T1 = [e for e in EV if e["bn"] in D0 and D0[e["bn"]] <= c1]
    T3 = [e for e in EV if e["bn"] in D0 and D0[e["bn"]] > c2]
    PC["diff_D3_D1"] = dtest(T3, T1, "D3 − D1")
    # 연속 dose 단조성: 지분율에 대한 가중 OLS 기울기
    xs = np.array([D0[e["bn"]] for e in EV if e["bn"] in D0
                   and np.isfinite(e["z_t"]) and np.isfinite(e["z_c"])])
    ys = np.array([e["z_t"] - e["z_c"] for e in EV if e["bn"] in D0
                   and np.isfinite(e["z_t"]) and np.isfinite(e["z_c"])])
    sl = np.polyfit(xs, ys, 1)[0]
    bsl = np.array([np.polyfit(xs[j], ys[j], 1)[0]
                    for j in (rng.integers(0, len(xs), len(xs)) for _ in range(NB))])
    cis = qci(bsl); sgs = "✓" if (cis[0] > 0 or cis[1] < 0) else "✗"
    print(f"      연속 dose 기울기 {sl:+.6f}/%p {cis} {sgs}  (n={len(xs)})")
    PC["continuous_slope"] = {"slope_per_pct": round(float(sl), 6), "ci": cis,
                              "sig": sgs == "✓", "n": int(len(xs))}

# ---- Panel D : exit 탐지 ----
print("\n[Panel D] exit 탐지 — PE 지분 소멸 (I-05 게이트)")
ex = []
for bn, g in Y.groupby("bn10"):
    if bn not in EVBN: continue
    g = g.sort_values("yr"); pos = g[g.pe_pct > 0]
    if pos.empty: continue
    ly = int(pos.yr.iloc[-1]); after = g[g.yr > ly]
    if len(after) and (after.pe_pct == 0).all():
        ex.append((bn, ly, int(after.yr.iloc[0]), (EVBN[bn]["m0"] - 1) // 12))
E = pd.DataFrame(ex, columns=["bn10", "last_pos_yr", "first_zero_yr", "deal_yr"])
E["hold_yr"] = E.last_pos_yr - E.deal_yr
E = E[E.hold_yr >= 0]
print(f"  지분 소멸 관측 {len(E)}건 / 매칭설계 {len(EV)}  보유기간 중앙값 {E.hold_yr.median() if len(E) else np.nan}년")
if len(E): print("  소멸연도 분포:", E.first_zero_yr.value_counts().sort_index().to_dict())
gate = len(E) >= 40
print(f"  → I-05 게이트(>=40건): {'✅ 통과' if gate else '❌ BLOCKED'}")
PD = {"n_exit": len(E), "gate_40": bool(gate),
      "hold_median_yr": float(E.hold_yr.median()) if len(E) else None,
      "exit_year_counts": {int(k): int(v) for k, v in E.first_zero_yr.value_counts().sort_index().items()} if len(E) else {}}

# ---- Panel E : 검증 부분표본 재추정 ----
print("\n[Panel E] 헤드라인 재추정 — 주주명부 검증 여부별")
PE_ = {}
CB = set(V.loc[conf, "bn10"])
for lab, sub in (("전체", EV), ("confirmed", [e for e in EV if e["bn"] in CB]),
                 ("non-confirmed", [e for e in EV if e["bn"] in set(V.bn10) and e["bn"] not in CB])):
    if len(sub) >= 20: PE_[lab] = batt(sub, lab)
PE_["diff_conf_vs_nonconf"] = dtest([e for e in EV if e["bn"] in CB],
    [e for e in EV if e["bn"] in set(V.bn10) and e["bn"] not in CB], "confirmed − non-conf")

# ---- 판정 ----
ok_dose = PC.get("n_소수<50%", 0) >= 20
status = "GO" if (PB_["confirm_rate"] >= 0.5 or ok_dose or gate) else "PARTIAL"
verdict = (f"커버리지 {PA['coverage']:.0%} ({PA['n_in_matched_design']}/379 매칭설계) | "
           f"PitchBook 연도 검증률 {PB_['confirm_rate']:.0%} (1월 {PB_['jan_confirm']} vs 비1월 {PB_['nonjan_confirm']}) | "
           f"dose 부여 {PC['n_with_dose']}건, 소수지분 {PC.get('n_소수<50%','-')}건 | "
           f"exit 소멸 {PD['n_exit']}건 → I-05 {'통과' if gate else 'BLOCKED'}")
emit("I-14", "주주명부 진입시점 검증 + 지분율 dose", status,
     {"panelA_coverage": PA, "panelB_timing_validation": PB_, "panelC_dose": PC,
      "panelD_exit_gate": PD, "panelE_validated_subsample": PE_},
     "주주명부로 (1) PitchBook 딜연도 독립검증 (2) 전표본 지분율 dose (3) exit 탐지",
     verdict, kill_met=False, n=len(EV),
     extra={"resolution_limit": "기준일이 사실상 연 1회 — 월 단위 시점 확정 불가. 연도 검증과 구간 확정만 가능.",
            "pe_detection": "광의=패턴(우리사주 등 제외), 정밀=PitchBook Investors 영문명 대조"})
