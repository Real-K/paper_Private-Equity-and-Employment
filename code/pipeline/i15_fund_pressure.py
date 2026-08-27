# -*- coding: utf-8 -*-
"""I-15 GP 펀드 소진압력 — 인과 갭의 마지막 검정.

[남은 위협] I-31 이 '관성 수준으로 예측 가능한 반등'은 배제했으나,
**'PE 만 아는 사적 정보에 기반한 시점 선택'** 은 배제하지 못했다. IV 8종은 전멸했다.

[설계 전환] shift-share IV 는 F≈2.0 으로 죽었다. **선택을 도구화하지 않는다.**
대신 **이미 선택된 딜들 사이에서 GP 가 얼마나 선별할 여유가 있었는가**의 변동을 쓴다.

  소진압력 = 딜 시점의 GP 최근 펀드 경과연수. PE 펀드 투자기간은 통상 5년이므로
  경과가 짧으면 GP 는 **고를 여유**가 있고, 길면 **기한 내 집행 압력**을 받는다.

  · 선택 가설 예측 → 효과는 **압력이 낮을수록 크다** (여유 있을 때 잘 골랐으므로)
  · 인과 가설 예측 → 효과는 **압력과 무관하게 평탄**

이는 IV 가 아니라 **반증 가능한 예측을 만드는 변동**이다. 그렇게만 서술한다.

Panel A  펀드-GP-딜 연결 + 압력 분포
Panel B  관련성 — 압력이 관측가능한 선별도(사전 성장·규모·관성)와 상관되는가
Panel C  ★ 압력 3분위별 효과 (무채용비중 DiD) + T3−T1 차이
Panel D  hazard 삼중교호 treated x post x 고압력
"""
import gc, re
import numpy as np, pandas as pd, glob
import statsmodels.api as sm
from h30_common import (load, deals, build, attach, boot_did_ci, emit,
                        SEED, qci, NB, widx, BASE)

rng = np.random.default_rng(SEED)
print("[I-15] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
EV, _ = build(G, allt, PE); EV = attach(G, EV)
Hv, Ev, mset = G["Hv"], G["Ev"], G["mset"]

def norm(x):
    x = re.sub(r"\s*\([^)]*\)\s*", " ", str(x)).lower()
    x = re.sub(r"\b(co|ltd|inc|corp|llc|lp|l\.p\.|limited|company|partners?|capital|group)\b\.?", " ", x)
    return re.sub(r"[^0-9a-z가-힣]", "", x)

# ---- 펀드 ----
ff = glob.glob(f"{BASE}/PI/drops/Pitchbook_fund data 0814/*.xlsx")[0]
FD = pd.read_excel(ff, usecols=["Fund ID", "Investor", "Vintage", "Fund Status", "Fund Size",
                                "Close Date", "Fund Strategy", "Fund Location"])
FD["cd"] = pd.to_datetime(FD["Close Date"], errors="coerce")
FD["vt"] = pd.to_numeric(FD["Vintage"], errors="coerce")
FD["cmi"] = np.where(FD.cd.notna(), FD.cd.dt.year * 12 + FD.cd.dt.month,
                     np.where(FD.vt.notna(), FD.vt * 12 + 6, np.nan))   # 마감일 없으면 vintage 중반
FD = FD[FD.cmi.notna()].copy()
FD["k"] = FD["Investor"].map(norm)
PEF = FD[FD["Fund Strategy"].astype(str).str.contains(
    "Buyout|Growth|Mezzanine|Special Situations|Turnaround|PE", case=False, na=False)]
print(f"  펀드 {len(FD):,} · PE계열 {len(PEF):,} · 고유 GP키 {PEF.k.nunique():,}")
FUNDS = {}
for k, g in PEF.groupby("k"):
    FUNDS[k] = np.sort(g.cmi.values.astype(float))

# ---- 딜 GP ----
pbf = pd.read_csv(f"{BASE}/shared/data/processed/pitchbook_deals_v1.csv", dtype=str)
pbf["bn10"] = pbf.bn.astype(str).str.zfill(10)
pbf["dd"] = pd.to_datetime(pbf["Deal Date"], errors="coerce")
BG = pbf[(pbf.is_bg == "True") & pbf.dd.notna()].sort_values("dd")
INV = BG.drop_duplicates("bn10").set_index("bn10")["Investors"].to_dict()
A = pd.read_csv(f"{BASE}/P014_upgrade_package/matching/work/PB_RECOVERY_FINAL_ADOPTED.csv", dtype=str)
A["bn10"] = A.bn10.astype(str).str.zfill(10)
CM = BG.assign(c=BG["Companies"].map(lambda x: re.sub(r"[^0-9a-z가-힣]", "", str(x).lower()))) \
       .drop_duplicates("c").set_index("c")["Investors"].to_dict()
for r in A.itertuples():
    if not isinstance(INV.get(r.bn10), str):
        v = CM.get(re.sub(r"[^0-9a-z가-힣]", "", str(r.pb_company).lower()))
        if isinstance(v, str): INV[r.bn10] = v

def gps(s):
    if not isinstance(s, str): return []
    return [norm(t) for t in re.split(r"[,;|]", s) if len(norm(t)) >= 3]

hit = 0
for e in EV:
    e["gp"] = None; e["press"] = np.nan; e["nfund"] = 0
    best = None
    for g in gps(INV.get(e["bn"])):
        arr = FUNDS.get(g)
        if arr is None: continue
        prior = arr[arr <= e["m0"]]
        if not len(prior): continue
        age = (e["m0"] - prior[-1]) / 12.0
        if best is None or age < best[1]: best = (g, age, len(prior))
    if best:
        e["gp"], e["press"], e["nfund"] = best[0], float(best[1]), int(best[2]); hit += 1
U = [e for e in EV if np.isfinite(e["press"])]
pr = np.array([e["press"] for e in U])
print(f"\n[Panel A] 펀드-딜 연결 {hit}/{len(EV)}")
print(f"  소진압력(최근 펀드 경과연수): 중앙값 {np.median(pr):.1f} · p25 {np.percentile(pr,25):.1f}"
      f" · p75 {np.percentile(pr,75):.1f} · 최대 {pr.max():.1f}")
p1, p2 = np.percentile(pr, [33.33, 66.67])
print(f"  3분위 컷 {p1:.2f} / {p2:.2f} 년")
PA = {"linked": hit, "n_ev": len(EV), "press_median": round(float(np.median(pr)), 2),
      "cuts": [round(float(p1), 2), round(float(p2), 2)]}

def zsh(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return np.nan
    x = Hv[row, c]
    return float((x == 0).mean()) if np.isfinite(x).all() else np.nan
for e in EV:
    a = zsh(e["ti"], e["m0"], -12, -1); b = zsh(e["ti"], e["m0"], 1, 12)
    e["z_t"] = b - a if (np.isfinite(a) and np.isfinite(b)) else np.nan
    cd = [zsh(k, e["m0"], 1, 12) - zsh(k, e["m0"], -12, -1) for k in e["ctrls"]]
    cd = [x for x in cd if np.isfinite(x)]
    e["z_c"] = float(np.mean(cd)) if cd else np.nan
    e["pp"] = zsh(e["ti"], e["m0"], -24, -13)

# ---- Panel B 관련성 ----
print("\n[Panel B] 관련성 — 압력이 관측가능한 선별도와 상관되는가")
PB = {}
for nm, vals in (("사전-사전 관성", [e["pp"] for e in U]),
                 ("사전 규모 log", [np.log(e["Epre"]) for e in U]),
                 ("사전 성장", [e["g"] for e in U])):
    y = np.array(vals, float); m = np.isfinite(y) & np.isfinite(pr)
    if m.sum() < 40: continue
    sl = float(np.polyfit(pr[m], y[m], 1)[0])
    bs = np.array([np.polyfit(pr[m][j], y[m][j], 1)[0] for j in
                   (rng.integers(0, m.sum(), m.sum()) for _ in range(NB))])
    ci = qci(bs); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
    PB[nm] = {"slope_per_yr": round(sl, 5), "ci": ci, "sig": sg == "✓", "n": int(m.sum())}
    print(f"  {nm:<14} 기울기 {sl:+.5f}/년 {ci} {sg} (n={int(m.sum())})")

# ---- Panel C ★ 압력별 효과 ----
print("\n[Panel C] ★ 소진압력 3분위별 효과 — 선택 가설은 감소, 인과 가설은 평탄")
PC = {}
def did(sub):
    return boot_did_ci([e["z_t"] for e in sub], [e["z_c"] for e in sub], rng)
grp = {}
for lab, m in (("P1 저압력(신선)", pr <= p1), ("P2 중간", (pr > p1) & (pr <= p2)),
               ("P3 고압력(노후)", pr > p2)):
    sub = [U[i] for i in np.flatnonzero(m)]; grp[lab] = sub
    p_, ci, n = did(sub)
    sg = "✓" if (ci and (ci[0] > 0 or ci[1] < 0)) else "✗"
    PC[lab] = {"DiD": p_, "ci": ci, "n": n, "sig": sg == "✓",
               "press_mean": round(float(np.mean([e["press"] for e in sub])), 2)}
    print(f"  {lab:<14} n={n:>3} (압력 {PC[lab]['press_mean']:.1f}년) DiD {p_} {ci} {sg}")
d1 = np.array([e["z_t"] - e["z_c"] for e in grp["P1 저압력(신선)"]
               if np.isfinite(e["z_t"]) and np.isfinite(e["z_c"])])
d3 = np.array([e["z_t"] - e["z_c"] for e in grp["P3 고압력(노후)"]
               if np.isfinite(e["z_t"]) and np.isfinite(e["z_c"])])
bs = np.array([d3[rng.integers(0, len(d3), len(d3))].mean()
               - d1[rng.integers(0, len(d1), len(d1))].mean() for _ in range(NB)])
ci = qci(bs); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
S = 0.0460; m_ = [round(ci[0] + S, 4), round(S - ci[1], 4)]
eq = bool(ci[0] > -S and ci[1] < S); kn = bool(min(m_) < 0.001)
PC["P3−P1"] = {"diff": round(float(d3.mean() - d1.mean()), 4), "ci": ci, "sig": sg == "✓",
               "equivalence": {"SESOI": S, "holds": eq, "margin": m_, "knife": kn}}
print(f"  P3−P1 차이 {d3.mean()-d1.mean():+.4f} {ci} {sg}  "
      f"등가성(δ=0.046) {'성립' if (eq and not kn) else '미성립'} 여유 {m_}")
print(f"  → 선택 가설이면 P1 이 크고 P3 가 작아야 한다(차이 > 0). 관측: {d3.mean()-d1.mean():+.4f}")

# ---- Panel D hazard 삼중교호 ----
print("\n[Panel D] hazard 삼중교호 treated x post x 고압력")
DC = [1, 2, 3, 4, 6, 12]
def spell(row, m0):
    o, d, w = [], 1, 0
    for k in range(-24, 13):
        j = mset.get(m0 + k)
        if j is None: d, w = 1, 0; continue
        h = Hv[row, j]
        if not np.isfinite(h): d, w = 1, 0; continue
        if w >= 6 and k != 0 and (-12 <= k <= -1 or 1 <= k <= 12):
            o.append((1 if k > 0 else 0, 1 if h > 0 else 0,
                      int(np.searchsorted(DC, min(d, 24), side="right") - 1)))
        d = 1 if h > 0 else d + 1; w += 1
    return o
rec = []
for ei, e in enumerate(U):
    hi = 1.0 if e["press"] > p2 else 0.0
    for who, row in [(1, e["ti"])] + [(0, c) for c in e["ctrls"]]:
        for po, y, db in spell(row, e["m0"]): rec.append((ei, who, po, y, db, hi))
Q = pd.DataFrame(rec, columns=["ev", "tr", "po", "hire", "db", "hi"]); del rec
Cl = Q.groupby(["ev", "tr", "po", "db", "hi"], as_index=False).agg(s=("hire", "sum"), n=("hire", "size"))
Cl["f"] = Cl.n - Cl.s; nC = len(Cl); NOBS = len(Q); del Q; gc.collect()
evc = pd.Categorical(Cl.ev).codes.astype(np.int32); NE = evc.max() + 1
dbc = Cl.db.values.astype(np.int32)
B = np.zeros((nC, 1 + 5 + (NE - 1)), np.float32); B[:, 0] = 1.0
m = dbc > 0; B[np.flatnonzero(m), 1 + dbc[m] - 1] = 1.0
m = evc > 0; B[np.flatnonzero(m), 6 + evc[m] - 1] = 1.0
tr = Cl.tr.values.astype(np.float32); po = Cl.po.values.astype(np.float32)
hi = Cl.hi.values.astype(np.float32); tp = tr * po
X = np.hstack([np.column_stack([tr, po, tp, tr * hi, po * hi, tp * hi]).astype(np.float32), B])
r = sm.GLM(np.column_stack([Cl.s.values, Cl.f.values]).astype(float), X,
           family=sm.families.Binomial(sm.families.links.CLogLog())
           ).fit(cov_type="cluster", cov_kwds={"groups": Cl.ev.values})
PD = {}
for i, v in enumerate(["treated", "post", "tp", "tr_hi", "po_hi", "tp_hi"]):
    b, se = float(r.params[i]), float(r.bse[i])
    PD[v] = {"coef": round(b, 4), "HR": round(float(np.exp(b)), 4),
             "HR_ci": [round(float(np.exp(b - 1.96 * se)), 4), round(float(np.exp(b + 1.96 * se)), 4)],
             "p": float(f"{r.pvalues[i]:.3g}")}
    print(f"  {v:<8} HR={np.exp(b):.4f} [{np.exp(b-1.96*se):.4f}, {np.exp(b+1.96*se):.4f}] p={r.pvalues[i]:.4g}")
hr_lo = PD["tp"]["HR"]; hr_hi = round(float(np.exp(PD["tp"]["coef"] + PD["tp_hi"]["coef"])), 4)
tri = PD["tp_hi"]; ts = not (tri["HR_ci"][0] <= 1.0 <= tri["HR_ci"][1])
print(f"  -> 저·중압력 HR={hr_lo}  고압력 HR={hr_hi}  차이 {tri['HR']} {tri['HR_ci']} {'✓' if ts else '✗'}")
PD.update({"HR_lowmid": hr_lo, "HR_highpress": hr_hi, "diff_sig": ts, "n_cells": nC, "n_ev": int(NE)})
del X, r, B; gc.collect()

# ---- 판정 ----
flat = (not PC["P3−P1"]["sig"]) and (not ts)
if flat and PC["P3−P1"]["equivalence"]["holds"] and not PC["P3−P1"]["equivalence"]["knife"]:
    status, concl = "GO", ("효과가 소진압력과 무관하다(등가성 성립). 선택 가설의 예측 — "
                           "'여유 있을 때 잘 골라서 효과가 크다' — 이 기각된다.")
elif flat:
    status, concl = "PARTIAL", "압력별 차이 미검출이나 등가성 미성립 — 배제가 아니라 '검출 못 함'"
elif PC["P3−P1"]["diff"] > 0 or (ts and hr_hi < hr_lo):
    status, concl = "KILL", "고압력에서 효과가 작다 — 선택 가설과 정합, 인과 해석 하향 필요"
else:
    status, concl = "PARTIAL", "고압력에서 효과가 오히려 크다 — 선택 가설과 불정합하나 해석 불명"
verdict = (f"연결 {hit}/{len(EV)} · 압력 중앙 {PA['press_median']}년 | "
           f"P1(신선) {PC['P1 저압력(신선)']['DiD']} vs P3(노후) {PC['P3 고압력(노후)']['DiD']}, "
           f"P3−P1 {PC['P3−P1']['diff']}{PC['P3−P1']['ci']}{'✓' if PC['P3−P1']['sig'] else '✗'} "
           f"등가성 {'성립' if PC['P3−P1']['equivalence']['holds'] else '미성립'} | "
           f"hazard 고압력 {hr_hi} vs {hr_lo} 차이 {tri['HR']}{'✓' if ts else '✗'} | {concl}")
emit("I-15", "GP 펀드 소진압력 (선택 가설 반증 검정)", status,
     {"panelA_linkage": PA, "panelB_relevance": PB, "panelC_by_pressure": PC,
      "panelD_hazard_triple": PD},
     "선택 가설은 '펀드가 신선할수록(선별 여유) 효과가 크다'를 예측한다. 인과 가설은 평탄을 예측한다.",
     verdict, kill_met=(status == "KILL"), n=len(U),
     extra={"conclusion": concl,
            "design_note": "IV 가 아니다. 소진압력은 GP 의 선별 여유를 바꾸는 변동이며, "
                           "선택 가설에 반증가능한 예측을 부여하는 용도로만 쓴다."})
