# -*- coding: utf-8 -*-
"""I-19 비-PE 지배구조 변화 대조 — 메커니즘 서사를 확정하는 검정.

[질문] 효과가 지분율·통제권·GP정체·경험·현금 어디에도 불변이고 사전 관성에만 의존한다면,
**PE 여부 자체도 무관한가?** 비-PE 최대주주 변경이 같은 효과를 내면 '주인 교체 일반' 이고,
PE 만 내면 'PE 고유' 다. 어느 쪽이든 서사가 확정된다.

[자료 판정]
 · 대표이사 변경 — **사용 불가.** 재무데이터 `대표이사` 비결측률 2018–19 **0.5%**, 2020 19.2%, 이후 2~6%.
   2020 급증은 형식 변경 인공물(변경건의 8.7%가 공백 차이뿐: '조 관 호'→'조관호    ').
   두 연속연도 관측이 드물어 탐지 불가. IROS 는 PDF 11장뿐. **CEO arm 기각.**
 · 최대주주 변경 — **사용 가능.** 주주 시계열 2.3M행, 연말 스냅샷 579,537행/46,070사.
   정규화+퍼지 클러스터로 34,522 → 28,493 (오탐 17% 제거). 비-PE **27,747건 / 14,037사**,
   연도 분포 안정(연 1,900~2,200).

[설계] PE 와 OWN 을 **동일 파이프라인·동일 연간 타이밍·동일 대조풀**로 돌린다.
 · 타이밍: 양쪽 모두 연도 → 6월 (PE 의 월 정밀도 이점을 제거해 공정 비교)
 · 대조풀: PE(752) ∪ 최대주주 변경 경험 기업(14,037) 전부 제외 → 오염 없는 무변화 기업만
 · OWN 표본: PE 의 (딜연도 × 규모bin) 결합분포에 맞춰 층화추출
 · OWN 에서 신규 최대주주가 **PE 패턴이면 제외** (미기록 PE 딜 오염 차단)

[사전 명시 판정 규칙 — 결과 보기 전 작성]
 GENERIC : OWN 효과가 PE 와 등가(δ=0.046) 이고 OWN 에서도 관성 조절(T3−T1)이 유의
 PE_SPEC : OWN 효과가 무유의이고 PE−OWN 차이가 유의
 PARTIAL : 그 외

Panel A  표본 구축·특성 비교   Panel B  유형별 효과   Panel C  관성 조절 재현   Panel D  차이·등가성
"""
import gc, re
import numpy as np, pandas as pd
from difflib import SequenceMatcher
from h30_common import (load, deals, build, attach, summ, boot_did_ci,
                        emit, SEED, qci, NB, widx, BASE)

rng = np.random.default_rng(SEED)
PRESPEC = ("GENERIC: OWN 효과가 PE 와 등가(δ=0.046) & OWN 에서도 T3−T1 유의 | "
           "PE_SPEC: OWN 무유의 & PE−OWN 차이 유의 | PARTIAL: 그 외")
print("[I-19] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
Hv, idx, mset = G["Hv"], G["idx"], G["mset"]
INP = set(np.asarray(idx))

# ---------- 최대주주 변경 이벤트 ----------
cols = ["business_number", "기준일", "주주명", "보통주_지분율"]
keep = []
for ch in pd.read_csv(f"{BASE}/PI/drops/외감_주주_시계열_2009plus.csv",
                      usecols=cols, dtype=str, chunksize=400_000):
    ch["bn10"] = ch.business_number.str.replace(r"\D", "", regex=True).str.zfill(10)
    ch = ch[ch.bn10.isin(INP)]
    ch["pct"] = pd.to_numeric(ch["보통주_지분율"], errors="coerce")
    keep.append(ch.loc[ch.pct >= 15, ["bn10", "기준일", "주주명", "pct"]])
S = pd.concat(keep, ignore_index=True); del keep; gc.collect()
S["dt"] = pd.to_datetime(S["기준일"], format="%Y%m%d", errors="coerce"); S = S[S.dt.notna()]
S["yr"] = S.dt.dt.year
S = S[S.dt == S.groupby(["bn10", "yr"])["dt"].transform("max")]
def nz(x):
    x = str(x).lower()
    x = re.sub(r"주식회사|유한회사|유한책임회사|합자회사|\(주\)|\(유\)|㈜|limited|ltd|inc|corp|company|co\b", "", x)
    return re.sub(r"[^0-9a-z가-힣]", "", x.replace("홀딩즈", "홀딩스"))
S["nm"] = S["주주명"].map(nz)
CL = {}
for bn, g in S.groupby("bn10"):
    reps = []
    for v in sorted(g.nm.unique(), key=len, reverse=True):
        if not v: CL[(bn, v)] = v; continue
        hit = next((r for r in reps if v == r or (len(v) >= 5 and len(r) >= 5 and
                    (v in r or r in v or SequenceMatcher(None, v, r).ratio() >= 0.85))), None)
        if hit is None: reps.append(v); hit = v
        CL[(bn, v)] = hit
S["key"] = [CL[(b, v)] for b, v in zip(S.bn10, S.nm)]
T = S.sort_values("pct").groupby(["bn10", "yr"]).tail(1).sort_values(["bn10", "yr"]).copy()
T["prev"] = T.groupby("bn10")["key"].shift(1); T["pyr"] = T.groupby("bn10")["yr"].shift(1)
CHG = T[(T.prev.notna()) & (T.key != T.prev) & (T.yr - T.pyr <= 2)]
CHANGED = set(CHG.bn10)                       # 대조풀 제외용 (PE 포함 전체)
PEPAT = (r"투자|인베스트|캐피탈|사모|펀드|조합|파트너스|에쿼티|벤처|PEF|"
         r"Capital|Invest|Partner|Equity|Fund")
OWN = CHG[(~CHG.bn10.isin(PE)) & (~CHG["주주명"].fillna("").str.contains(PEPAT, case=False, regex=True))]
print(f"  최대주주 변경 {len(CHG):,} · 비-PE {len(CHG[~CHG.bn10.isin(PE)]):,} "
      f"· PE패턴 주주 제외 후 **{len(OWN):,}** ({OWN.bn10.nunique():,}사)")
del S, T; gc.collect()

# ---------- PE 이벤트 (연간 타이밍) ----------
EVx, _ = build(G, allt, PE)                   # 월 정밀 (참고용)
PEy = pd.DataFrame({"bn10": [e["bn"] for e in EVx],
                    "mi": [((e["m0"] - 1) // 12) * 12 + 6, ][0] if False else
                          [((e["m0"] - 1) // 12) * 12 + 6 for e in EVx], "src": "pe"})
SIZE_B = [5, 10, 20, 50, 100, 250, np.inf]
pe_key = {}
for e in EVx:
    y = (e["m0"] - 1) // 12; sb = int(np.digitize(e["Epre"], SIZE_B, right=False))
    pe_key[(y, sb)] = pe_key.get((y, sb), 0) + 1
print(f"  PE 이벤트 {len(EVx)} · (연도×규모) 셀 {len(pe_key)}")

# ---------- OWN 층화추출 (PE 의 연도×규모 분포에 4배) ----------
Ev = G["Ev"]; pos = {b: i for i, b in enumerate(np.asarray(idx))}
def epre_of(bn, y):
    i = pos.get(bn)
    if i is None: return np.nan
    js = [mset.get(y * 12 + m) for m in range(1, 7)]
    js = [j for j in js if j is not None]
    if not js: return np.nan
    v = np.nanmean(Ev[i, js])
    return float(v) if np.isfinite(v) and v >= 5 else np.nan
own_rows = []
for r in OWN.itertuples():
    ep = epre_of(r.bn10, int(r.yr))
    if np.isfinite(ep):
        own_rows.append((r.bn10, int(r.yr), int(np.digitize(ep, SIZE_B, right=False))))
OW = pd.DataFrame(own_rows, columns=["bn10", "yr", "sb"]).drop_duplicates("bn10")
MULT = 4
pick = []
for (y, sb), n in pe_key.items():
    c = OW[(OW.yr == y) & (OW.sb == sb)]
    if len(c) == 0: continue
    k = min(len(c), n * MULT)
    pick.append(c.sample(k, random_state=int(rng.integers(1e6))))
OWNs = pd.concat(pick, ignore_index=True) if pick else OW.head(0)
OWNs["mi"] = OWNs.yr * 12 + 6; OWNs["src"] = "own"
print(f"  OWN 층화추출 {len(OWNs)} (PE 셀 {len(pe_key)} 중 매칭 {len(pick)}, 배수 {MULT})")

# ---------- 공통 대조풀로 재구축 ----------
EXCL = CHANGED | set(PE)
print(f"  대조 제외집합 {len(EXCL):,} → 잔여 대조풀 {len(INP) - len(EXCL):,}")
EV_PE, _ = build(G, PEy, PE, ctrl_extra_exclude=EXCL)
EV_OW, _ = build(G, OWNs[["bn10", "mi", "src"]], PE, ctrl_extra_exclude=EXCL)
EV_PEm, _ = build(G, allt, PE, ctrl_extra_exclude=EXCL)     # PE 월정밀 (타이밍 비교용)
for L in (EV_PE, EV_OW, EV_PEm): attach(G, L)
print(f"  구축: PE(연간) {len(EV_PE)} · OWN {len(EV_OW)} · PE(월정밀) {len(EV_PEm)}")

def zsh(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return np.nan
    x = Hv[row, c]
    return float((x == 0).mean()) if np.isfinite(x).all() else np.nan
for L in (EV_PE, EV_OW, EV_PEm):
    for e in L:
        a = zsh(e["ti"], e["m0"], -12, -1); b = zsh(e["ti"], e["m0"], 1, 12)
        e["z_t"] = b - a if (np.isfinite(a) and np.isfinite(b)) else np.nan
        cd = [zsh(k, e["m0"], 1, 12) - zsh(k, e["m0"], -12, -1) for k in e["ctrls"]]
        cd = [x for x in cd if np.isfinite(x)]
        e["z_c"] = float(np.mean(cd)) if cd else np.nan
        e["pp"] = zsh(e["ti"], e["m0"], -24, -13)
_pp = np.array([e["pp"] for e in EV_PEm], float)
Q1, Q2 = np.percentile(_pp[np.isfinite(_pp)], [33.33, 66.67])
tb = lambda v: None if not np.isfinite(v) else (0 if v <= Q1 else (1 if v <= Q2 else 2))
for L in (EV_PE, EV_OW, EV_PEm):
    for e in L: e["pb"] = tb(e["pp"])
print(f"  관성 컷(PE 월정밀 기준) {Q1:.4f}/{Q2:.4f}")

print("\n[Panel A] 표본 특성")
PA = {}
for lab, L in (("PE(연간)", EV_PE), ("OWN", EV_OW), ("PE(월정밀)", EV_PEm)):
    yy = [e["year"] for e in L]; ee = [e["Epre"] for e in L]
    pp2 = [e["pp"] for e in L if np.isfinite(e["pp"])]
    PA[lab] = {"n": len(L), "year_median": float(np.median(yy)),
               "size_median": round(float(np.median(ee)), 1),
               "pre_inertia_mean": round(float(np.mean(pp2)), 4),
               "T3_share": round(float(np.mean([e["pb"] == 2 for e in L if e["pb"] is not None])), 3)}
    print(f"  {lab:<10} n={len(L):>4} 연도중앙 {np.median(yy):.0f} 규모중앙 {np.median(ee):.0f} "
          f"사전관성 {np.mean(pp2):.3f} T3비중 {PA[lab]['T3_share']:.3f}")

def D(L, lab):
    p_, ci, n = boot_did_ci([e["z_t"] for e in L], [e["z_c"] for e in L], rng)
    sg = "✓" if (ci and (ci[0] > 0 or ci[1] < 0)) else ("✗" if ci else "-")
    print(f"  {lab:<22} {str(p_):>9} {str(ci):<21} {sg} (n={n})")
    return {"DiD": p_, "ci": ci, "n": n, "sig": sg == "✓"}

print("\n[Panel B] 유형별 무채용비중 DiD")
PB = {lab: D(L, lab) for lab, L in (("PE(연간)", EV_PE), ("OWN", EV_OW), ("PE(월정밀)", EV_PEm))}
print("\n  채용률 DiD (summ)")
for lab, L in (("PE(연간)", EV_PE), ("OWN", EV_OW)):
    s = summ(L, rng); PB[lab + "|hire"] = s
    print(f"  {lab:<22} DiD {s.get('DiD')}{s.get('DiD_ci')} n={s.get('n')}")

print("\n[Panel C] ★ 관성 조절 재현 — OWN 에서도 나타나는가")
PC = {}
for lab, L in (("PE(연간)", EV_PE), ("OWN", EV_OW), ("PE(월정밀)", EV_PEm)):
    sub1 = [e for e in L if e["pb"] == 0]; sub3 = [e for e in L if e["pb"] == 2]
    r1 = D(sub1, f"{lab} T1저관성"); r3 = D(sub3, f"{lab} T3고관성")
    d1 = np.array([e["z_t"] - e["z_c"] for e in sub1 if np.isfinite(e["z_t"]) and np.isfinite(e["z_c"])])
    d3 = np.array([e["z_t"] - e["z_c"] for e in sub3 if np.isfinite(e["z_t"]) and np.isfinite(e["z_c"])])
    o = {"T1": r1, "T3": r3}
    if min(len(d1), len(d3)) >= 15:
        bs = np.array([d3[rng.integers(0, len(d3), len(d3))].mean()
                       - d1[rng.integers(0, len(d1), len(d1))].mean() for _ in range(NB)])
        ci = qci(bs); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
        o["T3-T1"] = {"diff": round(float(d3.mean() - d1.mean()), 4), "ci": ci, "sig": sg == "✓"}
        print(f"    {lab} T3−T1 {d3.mean()-d1.mean():+.4f} {ci} {sg}")
    PC[lab] = o

print("\n[Panel D] PE(연간) − OWN 차이 + 등가성")
dp = np.array([e["z_t"] - e["z_c"] for e in EV_PE if np.isfinite(e["z_t"]) and np.isfinite(e["z_c"])])
do = np.array([e["z_t"] - e["z_c"] for e in EV_OW if np.isfinite(e["z_t"]) and np.isfinite(e["z_c"])])
bs = np.array([dp[rng.integers(0, len(dp), len(dp))].mean()
               - do[rng.integers(0, len(do), len(do))].mean() for _ in range(NB)])
ci = qci(bs); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
Sd = 0.046; mg = [round(ci[0] + Sd, 4), round(Sd - ci[1], 4)]
eq = bool(ci[0] > -Sd and ci[1] < Sd); kn = bool(min(mg) < 0.001)
PD = {"diff": round(float(dp.mean() - do.mean()), 4), "ci": ci, "sig": sg == "✓",
      "n_pe": len(dp), "n_own": len(do),
      "equivalence": {"SESOI": Sd, "holds": eq, "margin": mg, "knife": kn}}
print(f"  PE−OWN {dp.mean()-do.mean():+.4f} {ci} {sg}  등가성(δ=0.046) "
      f"{'성립' if (eq and not kn) else '미성립'} 여유 {mg}")

# ---------- 판정 (사전 규칙) ----------
own_sig = PB["OWN"]["sig"]; own_mod = PC["OWN"].get("T3-T1", {}).get("sig")
if own_sig and eq and own_mod: pre = "GENERIC"
elif (not own_sig) and PD["sig"]: pre = "PE_SPEC"
else: pre = "PARTIAL"
concl = {"GENERIC": "비-PE 최대주주 변경도 같은 효과·같은 조절 — **주인 교체 일반** 메커니즘",
         "PE_SPEC": "OWN 무효과 + PE−OWN 유의 — **PE 고유** 메커니즘",
         "PARTIAL": "혼재 — 아래 수치로 개별 판단"}[pre]
status = {"GENERIC": "GO", "PE_SPEC": "GO", "PARTIAL": "PARTIAL"}[pre]
verdict = (f"PE(연간) {PB['PE(연간)']['DiD']}{PB['PE(연간)']['ci']}{'✓' if PB['PE(연간)']['sig'] else '✗'} "
           f"vs OWN {PB['OWN']['DiD']}{PB['OWN']['ci']}{'✓' if own_sig else '✗'} | "
           f"PE−OWN {PD['diff']}{PD['ci']}{'✓' if PD['sig'] else '✗'} 등가성 {'성립' if eq else '미성립'} | "
           f"T3−T1: PE {PC['PE(연간)'].get('T3-T1',{}).get('diff')} vs OWN "
           f"{PC['OWN'].get('T3-T1',{}).get('diff')}{'✓' if own_mod else '✗'} | {pre}: {concl}")
emit("I-19", "비-PE 지배구조 변화 대조 (최대주주 변경)", status,
     {"panelA_samples": PA, "panelB_by_type": PB, "panelC_inertia_moderator": PC,
      "panelD_pe_minus_own": PD, "tercile_cuts": [round(float(Q1), 4), round(float(Q2), 4)],
      "own_universe": {"all_changes": int(len(CHG)), "nonPE": int(len(CHG[~CHG.bn10.isin(PE)])),
                       "after_PEpattern_filter": int(len(OWN)), "firms": int(OWN.bn10.nunique()),
                       "sampled": int(len(OWNs)), "strat_multiple": MULT},
      "control_pool_excluded": int(len(EXCL))},
     "비-PE 최대주주 변경이 같은 효과와 같은 관성 조절을 내면 '주인 교체 일반', 아니면 'PE 고유'",
     verdict, kill_met=False, n=len(EV_PE) + len(EV_OW),
     extra={"prespecified_verdict": PRESPEC, "prespec_outcome": pre, "conclusion": concl,
            "ceo_arm_killed": "재무 대표이사 비결측 2018-19 0.5%/2020 19.2%(형식변경 인공물, "
                              "변경건 8.7%가 공백차이)/이후 2~6% → 탐지 불가. IROS 는 PDF 11장.",
            "timing_note": "PE·OWN 모두 연도→6월 로 통일해 타이밍 정밀도 이점을 제거. "
                           "PE(월정밀)은 참고용."})
