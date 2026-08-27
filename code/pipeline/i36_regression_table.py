# -*- coding: utf-8 -*-
"""I-36 hazard 회귀표 — 저널 게재 포맷용 중첩 사양 + 적합통계.

기존 I-02/I-25/I-15 는 단일 사양만 보고했다. 저널 표는 **중첩 사양을 열로** 놓고
계수·군집SE·FE 행·관측수·클러스터수·pseudo-R² 를 함께 낸다. 그 자료를 여기서 만든다.

사양 (전부 이산시간 cloglog, grouped binomial, 이벤트 군집 SE)
 (1) treated × post,  지속기간 FE
 (2) (1) + 이벤트 FE                                    ← 주 사양
 (3) (2) + × 사전 고관성(T3)
 (4) (2) + × 스폰서 소진압력(상위 3분위)
"""
import gc
import numpy as np, pandas as pd
import statsmodels.api as sm
from h30_common import load, deals, build, emit, SEED, widx, BASE
import re, glob

rng = np.random.default_rng(SEED)
print("[I-36] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, mset, idx = G["Hv"], G["mset"], G["idx"]
DC = [1, 2, 3, 4, 6, 12]

def zsh(r, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return np.nan
    x = Hv[r, c]
    return float((x == 0).mean()) if np.isfinite(x).all() else np.nan
_pp = np.array([zsh(e["ti"], e["m0"], -24, -13) for e in EV], float)
_z = np.array([(zsh(e["ti"], e["m0"], 1, 12) - zsh(e["ti"], e["m0"], -12, -1)) for e in EV], float)
Q1, Q2 = np.percentile(_pp[np.isfinite(_pp) & np.isfinite(_z)], [33.33, 66.67])
for e, v in zip(EV, _pp): e["hi"] = 1.0 if (np.isfinite(v) and v > Q2) else 0.0

# 소진압력 (I-15 절차)
def nrm(x):
    x = re.sub(r"\s*\([^)]*\)\s*", " ", str(x)).lower()
    x = re.sub(r"\b(co|ltd|inc|corp|llc|lp|limited|company|partners?|capital|group)\b\.?", " ", x)
    return re.sub(r"[^0-9a-z가-힣]", "", x)
FD = pd.read_excel(glob.glob(f"{BASE}/PI/drops/Pitchbook_fund data 0814/*.xlsx")[0],
                   usecols=["Investor", "Vintage", "Close Date", "Fund Strategy"])
FD["cd"] = pd.to_datetime(FD["Close Date"], errors="coerce"); FD["vt"] = pd.to_numeric(FD["Vintage"], errors="coerce")
FD["cmi"] = np.where(FD.cd.notna(), FD.cd.dt.year * 12 + FD.cd.dt.month, np.where(FD.vt.notna(), FD.vt * 12 + 6, np.nan))
FD = FD[FD.cmi.notna()]; FD["k"] = FD["Investor"].map(nrm)
PEF = FD[FD["Fund Strategy"].astype(str).str.contains("Buyout|Growth|Mezzanine|Special|Turnaround|PE", case=False, na=False)]
FUNDS = {k: np.sort(g.cmi.values.astype(float)) for k, g in PEF.groupby("k")}
pbf = pd.read_csv(f"{BASE}/shared/data/processed/pitchbook_deals_v1.csv", dtype=str)
pbf["bn10"] = pbf.bn.astype(str).str.zfill(10); pbf["dd"] = pd.to_datetime(pbf["Deal Date"], errors="coerce")
INV = pbf[(pbf.is_bg == "True") & pbf.dd.notna()].sort_values("dd").drop_duplicates("bn10").set_index("bn10")["Investors"].to_dict()
prs = []
for e in EV:
    best = None
    for g in [nrm(t) for t in re.split(r"[,;|]", str(INV.get(e["bn"], ""))) if len(nrm(t)) >= 3]:
        a = FUNDS.get(g)
        if a is None: continue
        pri = a[a <= e["m0"]]
        if len(pri):
            age = (e["m0"] - pri[-1]) / 12.0
            if best is None or age < best: best = age
    e["press"] = best if best is not None else np.nan
    prs.append(e["press"])
P2 = np.nanpercentile(np.array(prs, float), 66.67)
for e in EV: e["hp"] = 1.0 if (np.isfinite(e["press"]) and e["press"] > P2) else 0.0
print(f"  고관성 {sum(e['hi'] for e in EV):.0f} · 고압력 {sum(e['hp'] for e in EV):.0f} (컷 {P2:.2f}년)")

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
for ei, e in enumerate(EV):
    for who, row in [(1, e["ti"])] + [(0, c) for c in e["ctrls"]]:
        for po, y, db in spell(row, e["m0"]):
            rec.append((ei, who, po, y, db, e["hi"], e["hp"]))
Q = pd.DataFrame(rec, columns=["ev", "tr", "po", "hire", "db", "hi", "hp"]); del rec
NFM = len(Q)
Cl = Q.groupby(["ev", "tr", "po", "db", "hi", "hp"], as_index=False).agg(s=("hire", "sum"), n=("hire", "size"))
Cl["f"] = Cl.n - Cl.s; nC = len(Cl); del Q; gc.collect()
evc = pd.Categorical(Cl.ev).codes.astype(np.int32); NE = int(evc.max() + 1)
dbc = Cl.db.values.astype(np.int32)
tr = Cl.tr.values.astype(np.float32); po = Cl.po.values.astype(np.float32)
hi = Cl.hi.values.astype(np.float32); hp = Cl.hp.values.astype(np.float32); tp = tr * po
ENDOG = np.column_stack([Cl.s.values, Cl.f.values]).astype(float)
GRP = Cl.ev.values
DFE = np.zeros((nC, 5), np.float32)
m = dbc > 0; DFE[np.flatnonzero(m), dbc[m] - 1] = 1.0
EFE = np.zeros((nC, NE - 1), np.float32)
m = evc > 0; EFE[np.flatnonzero(m), evc[m] - 1] = 1.0
ONE = np.ones((nC, 1), np.float32)
FAM = lambda: sm.families.Binomial(sm.families.links.CLogLog())
r0 = sm.GLM(ENDOG, ONE, family=FAM()).fit()           # 절편만 (McFadden 분모)

SPECS = [
 ("(1)", np.column_stack([tr, po, tp]), ["treated", "post", "treated x post"], False),
 ("(2)", np.column_stack([tr, po, tp]), ["treated", "post", "treated x post"], True),
 ("(3)", np.column_stack([tr, po, tp, tr*hi, po*hi, tp*hi]),
  ["treated", "post", "treated x post", "treated x high inaction",
   "post x high inaction", "treated x post x high inaction"], True),
 ("(4)", np.column_stack([tr, po, tp, tr*hp, po*hp, tp*hp]),
  ["treated", "post", "treated x post", "treated x pressure",
   "post x pressure", "treated x post x pressure"], True),
]
OUT = {"n_firm_months": NFM, "n_cells": nC, "n_events": NE,
       "tercile_cut_inaction": round(float(Q2), 4), "pressure_cut_years": round(float(P2), 2),
       "specs": {}}
print(f"\n  기업-월 {NFM:,} → 셀 {nC:,} · 이벤트 {NE}")
for tag, V, nm, efe in SPECS:
    X = np.hstack([V.astype(np.float32), ONE, DFE] + ([EFE] if efe else []))
    r = sm.GLM(ENDOG, X, family=FAM()).fit(cov_type="cluster", cov_kwds={"groups": GRP})
    mcf = 1.0 - float(r.llf) / float(r0.llf)
    o = {"fe_event": efe, "fe_duration": True, "cluster": "event", "n_clusters": NE,
         "n_cells": nC, "n_firm_months": NFM,
         "pseudo_r2_mcfadden": round(mcf, 4), "loglik": round(float(r.llf), 1),
         "deviance": round(float(r.deviance), 1), "df_model": int(r.df_model), "terms": {}}
    for i, v in enumerate(nm):
        b, se = float(r.params[i]), float(r.bse[i])
        o["terms"][v] = {"coef": round(b, 4), "se": round(se, 4), "z": round(b/se, 2),
                         "p": float(f"{r.pvalues[i]:.3g}"),
                         "HR": round(float(np.exp(b)), 4),
                         "ci": [round(float(np.exp(b-1.96*se)), 4), round(float(np.exp(b+1.96*se)), 4)]}
    OUT["specs"][tag] = o
    key = "treated x post"
    print(f"  {tag} FE(event)={'Y' if efe else 'N'}  {key}: b={o['terms'][key]['coef']:+.4f} "
          f"({o['terms'][key]['se']:.4f})  HR={o['terms'][key]['HR']}  "
          f"pseudo-R²={mcf:.4f}  clusters={NE}")
    for k in nm:
        if "x post x" in k:
            t = o["terms"][k]; print(f"      {k}: b={t['coef']:+.4f} ({t['se']:.4f}) HR={t['HR']} p={t['p']}")
    del X, r; gc.collect()
emit("I-36", "hazard 회귀표 (중첩 사양 + 적합통계)", "GO", OUT,
     "저널 표 형식에 필요한 중첩 사양별 계수·군집SE·FE·N·클러스터·pseudo-R² 를 산출한다",
     f"4개 사양. 주 사양(2) treated×post HR={OUT['specs']['(2)']['terms']['treated x post']['HR']}, "
     f"pseudo-R²={OUT['specs']['(2)']['pseudo_r2_mcfadden']}, clusters={NE}",
     kill_met=False, n=NE)
