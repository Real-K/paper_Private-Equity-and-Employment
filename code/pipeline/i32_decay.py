# -*- coding: utf-8 -*-
"""I-32 효과 감쇠 — '1회성 리뷰' vs '영구 체제변화' 판별.

[왜] I-16·I-14·I-17·I-21 이 지분율·통제권·GP정체·GP경험·현금 **어디에도 효과가 의존하지 않음**을
보였고, I-25·I-31 은 **사전 관성에만** 의존함을 보였다. 이 불변성 서명에 맞는 메커니즘은
**"주인 교체가 무행동으로 굳은 결정을 강제 재검토시킨다"** 이다. 그 모형은 검정 가능한 예측을 낳는다.

  · 1회성 리뷰(기본값 깨기) → 효과가 **1년차에 몰리고 이후 감쇠**한다. 밀린 채용을 처리하면 끝난다.
  · 영구 체제변화(관리방식 교체) → 효과가 **지속**된다.

추가 예측: 백로그 해소 모형이면 **고관성(T3) 기업일수록 1년차 효과가 크고 감쇠도 빠르다.**

Panel A  연차별 무채용비중 DiD — Y1[1,12] · Y2[13,24] · Y3[25,36] (기준 사전 [−12,−1])
Panel B  분기 이벤트스터디 q1..q12 (36개월) — 감쇠 형태
Panel C  hazard 연차별 (이벤트FE cloglog, post 를 Y1/Y2/Y3 로 분할)
Panel D  ★ 사전 관성 분위 x 연차 — 백로그 해소 예측
"""
import gc
import numpy as np, pandas as pd
import statsmodels.api as sm
from h30_common import load, deals, build, boot_did_ci, emit, SEED, qci, NB, widx, BASE

rng = np.random.default_rng(SEED)
print("[I-32] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
EV, _ = build(G, allt, PE)
Hv, mset = G["Hv"], G["mset"]

def zsh(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return np.nan
    x = Hv[row, c]
    return float((x == 0).mean()) if np.isfinite(x).all() else np.nan

YRS = [("Y1", 1, 12), ("Y2", 13, 24), ("Y3", 25, 36)]
for e in EV:
    pre = zsh(e["ti"], e["m0"], -12, -1)
    e["z"] = {}; e["zc"] = {}
    for lab, a, b in YRS:
        po = zsh(e["ti"], e["m0"], a, b)
        e["z"][lab] = po - pre if (np.isfinite(pre) and np.isfinite(po)) else np.nan
        v = []
        for k in e["ctrls"]:
            p2 = zsh(k, e["m0"], -12, -1); q2 = zsh(k, e["m0"], a, b)
            if np.isfinite(p2) and np.isfinite(q2): v.append(q2 - p2)
        e["zc"][lab] = float(np.mean(v)) if v else np.nan
    e["pp"] = zsh(e["ti"], e["m0"], -24, -13)
_pp = np.array([e["pp"] for e in EV], float)
Q1, Q2 = np.percentile(_pp[np.isfinite(_pp)], [33.33, 66.67])
for e in EV:
    e["pb"] = None if not np.isfinite(e["pp"]) else (0 if e["pp"] <= Q1 else (1 if e["pp"] <= Q2 else 2))

def D(sub, lab, tag):
    p_, ci, n = boot_did_ci([x["z"][lab] for x in sub], [x["zc"][lab] for x in sub], rng)
    sg = "✓" if (ci and (ci[0] > 0 or ci[1] < 0)) else ("✗" if ci else "-")
    print(f"  {tag:<12} {lab}  {str(p_):>9} {str(ci):<21} {sg} (n={n})")
    return {"DiD": p_, "ci": ci, "n": n, "sig": sg == "✓"}

print("\n[Panel A] 연차별 무채용비중 DiD")
PA = {lab: D(EV, lab, "전체") for lab, _, _ in YRS}

print("\n[Panel B] 분기 이벤트스터디 q1..q12 (사전창 평균 정규화)")
KS = list(range(-12, 37))
rows = []
for e in EV:
    js = [mset.get(e["m0"] + k) for k in KS if k != 0]
    if any(j is None for j in js): continue
    ht = Hv[e["ti"], js]
    if not np.isfinite(ht).all(): continue
    hc = Hv[np.ix_(e["ctrls"], js)]
    ok = np.isfinite(hc).all(axis=1)
    if ok.sum() == 0: continue
    rows.append((ht > 0).astype(float) - (hc[ok] > 0).astype(float).mean(axis=0))
Dm = np.array(rows); NB2 = len(Dm)
Q = np.stack([Dm[:, i*3:(i+1)*3].mean(axis=1) for i in range(16)], axis=1)   # q-4..q-1, q1..q12
base = Q[:, :4].mean(axis=1, keepdims=True); Bq = Q - base
bq = np.array([Bq[rng.integers(0, NB2, NB2)].mean(axis=0) for _ in range(NB)])
b = Bq.mean(axis=0)
QL = [f"q{i-4}" if i < 4 else f"q{i-3}" for i in range(16)]
PB = {"n_ev": NB2, "beta": {}}
print(f"  균형 이벤트 {NB2}/{len(EV)} (36개월 사후 필요)")
for i in range(16):
    ci = qci(bq[:, i]); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
    PB["beta"][QL[i]] = {"b": round(float(b[i]), 4), "ci": ci, "sig": sg == "✓"}
print("  사전:", "  ".join(f"{QL[i]}:{b[i]:+.3f}" for i in range(4)))
for s, t in ((4, 8), (8, 12), (12, 16)):
    print(f"  Y{s//4}:  " + "  ".join(f"{QL[i]}:{b[i]:+.3f}{'✓' if PB['beta'][QL[i]]['sig'] else ''}"
                                      for i in range(s, t)))
y1 = float(np.mean(b[4:8])); y2 = float(np.mean(b[8:12])); y3 = float(np.mean(b[12:16]))
by1 = bq[:, 4:8].mean(axis=1); by2 = bq[:, 8:12].mean(axis=1); by3 = bq[:, 12:16].mean(axis=1)
PB["year_means"] = {"Y1": {"b": round(y1, 4), "ci": qci(by1)},
                    "Y2": {"b": round(y2, 4), "ci": qci(by2)},
                    "Y3": {"b": round(y3, 4), "ci": qci(by3)}}
dd = by1 - by3
PB["Y1_minus_Y3"] = {"diff": round(float(y1 - y3), 4), "ci": qci(dd),
                     "sig": bool(qci(dd)[0] > 0 or qci(dd)[1] < 0)}
print(f"  연평균: Y1 {y1:+.4f}{qci(by1)} · Y2 {y2:+.4f}{qci(by2)} · Y3 {y3:+.4f}{qci(by3)}")
print(f"  Y1−Y3 {y1-y3:+.4f} {qci(dd)} {'✓ 감쇠' if PB['Y1_minus_Y3']['sig'] else '✗ 감쇠 미검출'}")

print("\n[Panel C] hazard 연차별 (이벤트FE cloglog)")
DC = [1, 2, 3, 4, 6, 12]
def spell(row, m0):
    o, d, w = [], 1, 0
    for k in range(-24, 37):
        j = mset.get(m0 + k)
        if j is None: d, w = 1, 0; continue
        h = Hv[row, j]
        if not np.isfinite(h): d, w = 1, 0; continue
        per = 0 if -12 <= k <= -1 else (1 if 1 <= k <= 12 else (2 if 13 <= k <= 24 else (3 if 25 <= k <= 36 else -1)))
        if w >= 6 and per >= 0:
            o.append((per, 1 if h > 0 else 0, int(np.searchsorted(DC, min(d, 24), side="right") - 1)))
        d = 1 if h > 0 else d + 1; w += 1
    return o
rec = []
for ei, e in enumerate(EV):
    for who, row in [(1, e["ti"])] + [(0, c) for c in e["ctrls"]]:
        for per, y, db in spell(row, e["m0"]): rec.append((ei, who, per, y, db))
Qd = pd.DataFrame(rec, columns=["ev", "tr", "per", "hire", "db"]); del rec
Cl = Qd.groupby(["ev", "tr", "per", "db"], as_index=False).agg(s=("hire", "sum"), n=("hire", "size"))
Cl["f"] = Cl.n - Cl.s; nC = len(Cl); NOBS = len(Qd); del Qd; gc.collect()
evc = pd.Categorical(Cl.ev).codes.astype(np.int32); NE = evc.max() + 1
dbc = Cl.db.values.astype(np.int32); per = Cl.per.values.astype(np.int32)
tr = Cl.tr.values.astype(np.float32)
Xv = np.column_stack([tr] + [(per == p).astype(np.float32) for p in (1, 2, 3)]
                     + [tr * (per == p).astype(np.float32) for p in (1, 2, 3)])
BASEm = np.zeros((nC, 1 + 5 + (NE - 1)), np.float32); BASEm[:, 0] = 1.0
m = dbc > 0; BASEm[np.flatnonzero(m), 1 + dbc[m] - 1] = 1.0
m = evc > 0; BASEm[np.flatnonzero(m), 6 + evc[m] - 1] = 1.0
X = np.hstack([Xv.astype(np.float32), BASEm])
r = sm.GLM(np.column_stack([Cl.s.values, Cl.f.values]).astype(float), X,
           family=sm.families.Binomial(sm.families.links.CLogLog())
           ).fit(cov_type="cluster", cov_kwds={"groups": Cl.ev.values})
NM = ["treated", "postY1", "postY2", "postY3", "tr_Y1", "tr_Y2", "tr_Y3"]
PC = {}
for i, v in enumerate(NM):
    bb, se = float(r.params[i]), float(r.bse[i])
    PC[v] = {"HR": round(float(np.exp(bb)), 4),
             "HR_ci": [round(float(np.exp(bb - 1.96*se)), 4), round(float(np.exp(bb + 1.96*se)), 4)],
             "p": float(f"{r.pvalues[i]:.3g}")}
    if v.startswith("tr_"):
        print(f"  {v:<8} HR={np.exp(bb):.4f} [{np.exp(bb-1.96*se):.4f}, {np.exp(bb+1.96*se):.4f}] p={r.pvalues[i]:.4g}")
PC["n_cells"] = nC; PC["n_firm_months"] = NOBS
del X, r, BASEm; gc.collect()

print("\n[Panel D] ★ 사전 관성 분위 x 연차 — 백로그 해소 예측")
PD = {}
for bq_, bl in ((0, "T1저관성"), (2, "T3고관성")):
    sub = [e for e in EV if e["pb"] == bq_]
    PD[bl] = {lab: D(sub, lab, bl) for lab, _, _ in YRS}

# ---- 판정 ----
d13 = PB["Y1_minus_Y3"]
a1, a3 = PA["Y1"], PA["Y3"]
decay = bool(d13["sig"] and (d13["diff"] or 0) > 0) or bool(a1["sig"] and not a3["sig"])
persist = bool(a1["sig"] and a3["sig"] and not d13["sig"])
if decay and not persist: status, concl = "GO", "효과가 감쇠한다 — **1회성 리뷰(기본값 깨기)** 모형 지지"
elif persist: status, concl = "PARTIAL", "효과가 지속된다 — **영구 체제변화** 모형 지지, 1회성 리뷰 기각"
else: status, concl = "PARTIAL", "감쇠·지속 판별 불가"
t3 = PD.get("T3고관성", {}); t1 = PD.get("T1저관성", {})
verdict = (f"연차 DiD Y1 {a1['DiD']}{'✓' if a1['sig'] else '✗'} · Y2 {PA['Y2']['DiD']}"
           f"{'✓' if PA['Y2']['sig'] else '✗'} · Y3 {a3['DiD']}{'✓' if a3['sig'] else '✗'} | "
           f"분기ES 연평균 Y1 {PB['year_means']['Y1']['b']} Y3 {PB['year_means']['Y3']['b']}, "
           f"Y1−Y3 {d13['diff']}{d13['ci']}{'✓' if d13['sig'] else '✗'} | "
           f"hazard 교호 Y1 {PC['tr_Y1']['HR']} Y2 {PC['tr_Y2']['HR']} Y3 {PC['tr_Y3']['HR']} | "
           f"T3 Y1 {t3.get('Y1',{}).get('DiD')} → Y3 {t3.get('Y3',{}).get('DiD')} | {concl}")
emit("I-32", "효과 감쇠 (1회성 리뷰 vs 영구 체제변화)", status,
     {"panelA_by_year": PA, "panelB_quarterly_ES": PB, "panelC_hazard_by_year": PC,
      "panelD_by_inertia_year": PD, "tercile_cuts": [round(float(Q1), 4), round(float(Q2), 4)]},
     "1회성 리뷰 모형은 Y1 집중·이후 감쇠를, 영구 체제변화 모형은 지속을 예측한다",
     verdict, kill_met=False, n=len(EV),
     extra={"conclusion": concl,
            "prespecified_rule": "decay = (Y1−Y3 유의&양) or (Y1 유의 & Y3 무유의); "
                                 "persist = (Y1,Y3 모두 유의 & Y1−Y3 무유의)"})
