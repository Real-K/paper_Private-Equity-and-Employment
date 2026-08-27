# -*- coding: utf-8 -*-
"""I-19b OWN 무효과가 '가짜 이벤트' 때문인가 — 하위분할 + positive control.

[I-19 결과] 비-PE 최대주주 변경(n=1,191) 효과 **+0.0050 [−0.007, +0.018] ✗**, 관성 조절도 부재
(T3−T1 +0.0017 ✗). PE 는 −0.0335✓ / T3−T1 −0.0764✓. 사전규칙대로 PE_SPEC.

[치명적 취약점] OWN 이 **진짜 지배권 변경이 아니면** 영이 나오는 건 당연하다. 한국 외감기업의
최대주주 변경 상당수가 가족 간 이전·계열사 재편일 수 있다. 이를 깨지 못하면 결론이 서지 않는다.

검정 2종:
 A. **하위분할** — 명백한 지배권 이전만 남겨도 여전히 0 인가
    · 신규 최대주주 지분 ≥50% (직전 <50%) = 명백한 통제권 취득
    · 신규 최대주주가 **법인** (전략적 인수) vs **개인**
    · 직전 보유자와 **성(姓)이 다른 개인** = 가족승계 아님
    · 지분 변화폭 (신규 − 직전) 상위 3분위
 B. **positive control** — OWN 이벤트가 다른 결과는 움직이는가
    (움직이는 게 하나도 없으면 '이벤트 자체가 없는 것'이고, 움직이면 '실재하나 채용빈도만 불변')
"""
import gc, re
import numpy as np, pandas as pd
from difflib import SequenceMatcher
from h30_common import (load, deals, build, attach, summ, boot_did_ci,
                        emit, SEED, qci, NB, widx, dflow, BASE)

rng = np.random.default_rng(SEED)
print("[I-19b] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
Hv, Sv, Ev, idx, mset = G["Hv"], G["Sv"], G["Ev"], G["idx"], G["mset"]
INP = set(np.asarray(idx))

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
for c, s in (("key", "prevkey"), ("주주명", "prevnm"), ("pct", "prevpct"), ("yr", "pyr")):
    T[s] = T.groupby("bn10")[c].shift(1)
CHG = T[(T.prevkey.notna()) & (T.key != T.prevkey) & (T.yr - T.pyr <= 2)]
CHANGED = set(CHG.bn10)
PEPAT = r"투자|인베스트|캐피탈|사모|펀드|조합|파트너스|에쿼티|벤처|PEF|Capital|Invest|Partner|Equity|Fund"
OWN = CHG[(~CHG.bn10.isin(PE)) & (~CHG["주주명"].fillna("").str.contains(PEPAT, case=False, regex=True))].copy()
del S, T; gc.collect()

# ---- 하위유형 분류 ----
CORP = r"주식회사|\(주\)|\(유\)|㈜|유한|법인|홀딩스|산업|전자|화학|건설|은행|증권|보험|Co|Ltd|Inc|Corp|Group"
is_corp = OWN["주주명"].fillna("").str.contains(CORP, case=False, regex=True)
def indiv(x):
    s = re.sub(r"[^가-힣]", "", str(x))
    return s if 2 <= len(s) <= 4 else ""
OWN["new_ind"] = OWN["주주명"].map(indiv); OWN["old_ind"] = OWN["prevnm"].map(indiv)
OWN["is_corp"] = is_corp
OWN["both_ind"] = (OWN.new_ind != "") & (OWN.old_ind != "")
OWN["same_sur"] = OWN.both_ind & (OWN.new_ind.str[0] == OWN.old_ind.str[0])
OWN["maj_gain"] = (OWN.pct >= 50) & (OWN.prevpct < 50)
OWN["dpct"] = OWN.pct - OWN.prevpct
print(f"  OWN {len(OWN):,}  법인 {int(is_corp.sum()):,} · 개인쌍 {int(OWN.both_ind.sum()):,} "
      f"(동성 {int(OWN.same_sur.sum()):,}) · 과반취득 {int(OWN.maj_gain.sum()):,}")

SIZE_B = [5, 10, 20, 50, 100, 250, np.inf]
pos = {b: i for i, b in enumerate(np.asarray(idx))}
def epre_of(bn, y):
    i = pos.get(bn)
    if i is None: return np.nan
    js = [mset.get(y * 12 + m) for m in range(1, 7)]; js = [j for j in js if j is not None]
    if not js: return np.nan
    v = np.nanmean(Ev[i, js])
    return float(v) if np.isfinite(v) and v >= 5 else np.nan
OWN["ep"] = [epre_of(r.bn10, int(r.yr)) for r in OWN.itertuples()]
OWN = OWN[np.isfinite(OWN.ep)].drop_duplicates("bn10")
OWN["mi"] = OWN.yr * 12 + 6; OWN["src"] = "own"
EXCL = CHANGED | set(PE)

EVpe, _ = build(G, allt, PE)
PEy = pd.DataFrame({"bn10": [e["bn"] for e in EVpe],
                    "mi": [((e["m0"] - 1) // 12) * 12 + 6 for e in EVpe], "src": "pe"})
EV_PE, _ = build(G, PEy, PE, ctrl_extra_exclude=EXCL); attach(G, EV_PE)

def zsh(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return np.nan
    x = Hv[row, c]
    return float((x == 0).mean()) if np.isfinite(x).all() else np.nan
def prep(L):
    for e in L:
        a = zsh(e["ti"], e["m0"], -12, -1); b = zsh(e["ti"], e["m0"], 1, 12)
        e["z_t"] = b - a if (np.isfinite(a) and np.isfinite(b)) else np.nan
        cd = [zsh(k, e["m0"], 1, 12) - zsh(k, e["m0"], -12, -1) for k in e["ctrls"]]
        cd = [x for x in cd if np.isfinite(x)]
        e["z_c"] = float(np.mean(cd)) if cd else np.nan
        e["pp"] = zsh(e["ti"], e["m0"], -24, -13)
        e["s_t"] = dflow(G, e["ti"], e["m0"], Sv)
        sc = [dflow(G, k, e["m0"], Sv) for k in e["ctrls"]]
        sc = [x for x in sc if np.isfinite(x)]
        e["s_c"] = float(np.mean(sc)) if sc else np.nan
    return L
prep(EV_PE)

def D(L, tk, ck, lab):
    p_, ci, n = boot_did_ci([e[tk] for e in L], [e[ck] for e in L], rng)
    sg = "✓" if (ci and (ci[0] > 0 or ci[1] < 0)) else ("✗" if ci else "-")
    print(f"  {lab:<30} {str(p_):>9} {str(ci):<21} {sg} (n={n})")
    return {"DiD": p_, "ci": ci, "n": n, "sig": sg == "✓"}

SUB = [("전체", np.ones(len(OWN), bool)),
       ("① 과반취득(<50→≥50)", OWN.maj_gain.values),
       ("② 법인 인수", OWN.is_corp.values),
       ("③ 개인·異姓(비가족)", (OWN.both_ind & ~OWN.same_sur).values),
       ("④ 개인·同姓(가족승계)", OWN.same_sur.values),
       ("⑤ 지분변화 상위3분위", (OWN.dpct >= np.nanpercentile(OWN.dpct, 66.67)).values)]
print("\n[Panel A] OWN 하위유형별 — 명백한 지배권 이전만 남겨도 0 인가")
PA = {}
CAP = 1600
for lab, m in SUB:
    sub = OWN[m]
    if len(sub) > CAP: sub = sub.sample(CAP, random_state=42)
    if len(sub) < 60: print(f"  {lab:<30} n={len(sub)} (<60)"); PA[lab] = {"n": len(sub), "note": "n<60"}; continue
    L, _ = build(G, sub[["bn10", "mi", "src"]], PE, ctrl_extra_exclude=EXCL)
    attach(G, L); prep(L)
    r = D(L, "z_t", "z_c", lab)
    p3 = [e for e in L if np.isfinite(e["pp"])]
    q1, q2 = np.percentile([e["pp"] for e in p3], [33.33, 66.67]) if len(p3) > 60 else (np.nan, np.nan)
    t1 = [e for e in L if np.isfinite(e["pp"]) and e["pp"] <= q1]
    t3 = [e for e in L if np.isfinite(e["pp"]) and e["pp"] > q2]
    r["n_events_built"] = len(L)
    if min(len(t1), len(t3)) >= 20:
        d1 = np.array([e["z_t"] - e["z_c"] for e in t1 if np.isfinite(e["z_t"]) and np.isfinite(e["z_c"])])
        d3 = np.array([e["z_t"] - e["z_c"] for e in t3 if np.isfinite(e["z_t"]) and np.isfinite(e["z_c"])])
        if min(len(d1), len(d3)) >= 15:
            bs = np.array([d3[rng.integers(0, len(d3), len(d3))].mean()
                           - d1[rng.integers(0, len(d1), len(d1))].mean() for _ in range(NB)])
            ci = qci(bs); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
            r["T3_T1"] = {"diff": round(float(d3.mean() - d1.mean()), 4), "ci": ci, "sig": sg == "✓"}
            print(f"      └ T3−T1 {d3.mean()-d1.mean():+.4f} {ci} {sg}")
    PA[lab] = r
    if lab == "전체": L_ALL = L
    del L; gc.collect()

print("\n[Panel B] positive control — OWN 이벤트가 다른 결과는 움직이는가")
PB = {}
for lab, L in (("PE", EV_PE), ("OWN 전체", L_ALL)):
    s = summ(L, rng)
    PB[lab] = {"hire_DiD": s.get("DiD"), "hire_ci": s.get("DiD_ci"),
               "rel_logemp": s.get("rel"), "rel_ci": s.get("rel_ci"), "n": s.get("n")}
    sep = D(L, "s_t", "s_c", f"{lab} 이직률 DiD")
    PB[lab]["sep_DiD"] = sep
    print(f"  {lab:<12} 채용률 {s.get('DiD')}{s.get('DiD_ci')} · "
          f"고용수준 rel {s.get('rel')}{s.get('rel_ci')} (n={s.get('n')})")

# ---- 판정 ----
key = [k for k in PA if k.startswith(("①", "②", "③", "⑤"))]
any_sig = [k for k in key if PA[k].get("sig")]
any_mod = [k for k in key if PA[k].get("T3_T1", {}).get("sig")]
rel_moves = bool(PB["OWN 전체"]["rel_logemp"] is not None and PB["OWN 전체"]["rel_ci"]
                 and (PB["OWN 전체"]["rel_ci"][0] > 0 or PB["OWN 전체"]["rel_ci"][1] < 0))
if any_sig or any_mod:
    status = "PARTIAL"; concl = f"일부 하위유형에서 효과 발현 — PE 고유 주장 약화 (유의 {any_sig} / 조절 {any_mod})"
elif rel_moves:
    status = "GO"; concl = ("명백한 지배권 이전(과반취득·법인인수·비가족)에서도 효과 0 이고, "
                            "OWN 이벤트는 고용수준은 실제로 움직인다 → **이벤트는 실재하며 "
                            "채용빈도만 불변. PE 고유 결론 유지.**")
else:
    status = "PARTIAL"; concl = ("OWN 이 어떤 결과도 움직이지 않음 — 이벤트 실재성 미확인. "
                                 "PE 고유 결론을 이 검정만으로 지지할 수 없다.")
verdict = (" | ".join(f"{k} {PA[k].get('DiD')}{'✓' if PA[k].get('sig') else '✗'}(n={PA[k].get('n')})"
                      for k in PA) +
           f" || positive control: OWN 고용수준 rel {PB['OWN 전체']['rel_logemp']}"
           f"{PB['OWN 전체']['rel_ci']} {'✓' if rel_moves else '✗'} | {concl}")
emit("I-19b", "OWN 하위유형 + positive control", status,
     {"panelA_subtypes": PA, "panelB_positive_control": PB,
      "own_universe": {"n": int(len(OWN)), "corp": int(OWN.is_corp.sum()),
                       "indiv_pair": int(OWN.both_ind.sum()), "same_surname": int(OWN.same_sur.sum()),
                       "majority_gain": int(OWN.maj_gain.sum())}, "cap_per_subtype": CAP},
     "명백한 지배권 이전만 남겨도 OWN 효과가 0 이고, OWN 이 다른 결과는 움직이면 PE 고유 결론이 선다",
     verdict, kill_met=False, n=int(len(OWN)), extra={"conclusion": concl})
