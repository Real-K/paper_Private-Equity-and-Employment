# -*- coding: utf-8 -*-
"""I-06 not-yet-treated 붕괴 해부.

§34 진단: 283→379 확장에서 not-yet-treated P1 이 −0.0863 ✓ → −0.0577 ✗ 로 무너졌다.
**n 은 186→256 으로 늘었는데** 계수가 줄었으므로 검정력 문제가 아니라 구성 문제다.

구조적 사실: 이 설계의 대조군은 **처치군 자체에서** 뽑힌다(gap 개월 뒤에 처치되는 기업).
따라서 회수 96사를 넣으면 **처치군과 대조풀이 동시에** 바뀐다. 두 경로를 분리한다.

  DiD_final − DiD_orig,oldpool
    = [DiD_orig,newpool − DiD_orig,oldpool]                  ... (A) 대조풀 경로 (쌍대응)
    + w_rec × [DiD_rec,newpool − DiD_orig,newpool]           ... (B) 구성 경로
  (DiD 는 이벤트 평균이므로 이 분해는 항등식이다.)

Panel A  4개 사양 S1~S4 + 회수전용 S5
Panel B  정확 분해 (A)/(B) 기여도
Panel C  '붕괴'가 통계적으로 확립되는가 — S1 vs S4 차이 검정 (공통 283 쌍대응)
Panel D  회수 96사 대 기존 283사 구성 진단 (딜연도·규모·사전 hazard·매칭거리)

기각조건: 차이가 유의하지 않으면 '붕괴'는 확립되지 않은 것이며 원고 서술을 바꿔야 한다.
(유의→비유의 전환은 그 자체로 '변화의 증거'가 아니다.)
"""
import numpy as np, pandas as pd
from h30_common import load, deals, build, summ, emit, SEED, qci, NB, dflow, rel_log

rng = np.random.default_rng(SEED)
print("[I-06] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
_, CACHE = build(G, allt, PE)
idx, Ev, Hv = G["idx"], G["Ev"], G["Hv"]
GAPS = (24, 36)

FIRST_A = [(r.bn10, int(r.mi), idx.get_loc(r.bn10)) for r in allt.itertuples() if r.bn10 in idx]
ORIGBN = set(orig.bn10)
FIRST_O = [x for x in FIRST_A if x[0] in ORIGBN]
FIRST_R = [x for x in FIRST_A if x[0] not in ORIGBN]
print(f"  처치 후보  전체 {len(FIRST_A)}  기존 {len(FIRST_O)}  회수 {len(FIRST_R)}")


def nyt(treat_list, pool_list, gap):
    """not-yet-treated 설계. 대조 후보 = pool 중 m0+gap 이후 처치. V4 캘리퍼 |dlogsize|<=0.25, 5NN."""
    recs = []
    for bn, m0, ti in treat_list:
        c = CACHE.get(m0)
        if c is None: continue
        Ep, g, sb, gb, ageb = c
        if not (np.isfinite(Ep[ti]) and Ep[ti] >= 5): continue
        cand = np.array([tj for (bj, mj, tj) in pool_list
                         if mj >= m0 + gap and tj != ti and np.isfinite(Ep[tj]) and Ep[tj] >= 5])
        if len(cand) < 3: continue
        dls = np.log(Ep[cand]) - np.log(Ep[ti])
        keep = np.abs(dls) <= 0.25; cand, dls = cand[keep], dls[keep]
        if len(cand) < 3: continue
        gt = g[ti] if np.isfinite(g[ti]) else 0.0
        gc = np.where(np.isfinite(g[cand]), g[cand], 0.0)
        dist = (dls / 0.9) ** 2 + ((np.clip(gc, -1, 2) - np.clip(gt, -1, 2)) / 0.35) ** 2
        o = np.argsort(dist)[:5]; ctrls = cand[o]
        t = dflow(G, ti, m0, Hv)
        cs = np.array([dflow(G, c_, m0, Hv) for c_ in ctrls], float); cs = cs[np.isfinite(cs)]
        if np.isfinite(t) and len(cs):
            recs.append({"bn": bn, "t": t, "cs": cs, "rel": rel_log(G, ti, ctrls, m0),
                         "d": float(dist[o].mean()), "n_cand": int(len(cand))})
    return recs


SPEC = {"S1 기존처치·기존풀": (FIRST_O, FIRST_O), "S2 기존처치·확장풀": (FIRST_O, FIRST_A),
        "S3 확장처치·기존풀": (FIRST_A, FIRST_O), "S4 확장처치·확장풀": (FIRST_A, FIRST_A),
        "S5 회수전용·확장풀": (FIRST_R, FIRST_A)}
print("\n[Panel A] 사양별 not-yet-treated")
PA, REC = {}, {}
for gap in GAPS:
    print(f"  -- G={gap} --")
    for lab, (tl, pl) in SPEC.items():
        r = nyt(tl, pl, gap); REC[(gap, lab)] = r
        s = summ(r, rng)
        s["mean_dist"] = round(float(np.mean([x["d"] for x in r])), 4) if r else None
        s["mean_ncand"] = round(float(np.mean([x["n_cand"] for x in r])), 1) if r else None
        PA[f"G{gap}|{lab}"] = s
        if s.get("n", 0) >= 20:
            print(f"    {lab:<18} n={s['n']:>3} DiD {s['DiD']:+.4f}{s['DiD_ci']} "
                  f"P1 {s['P1']:+.4f}{s['P1_ci']} 매칭거리 {s['mean_dist']} 후보 {s['mean_ncand']}")
        else:
            print(f"    {lab:<18} n={s.get('n',0)} (<20)")

# ---------- Panel B : 정확 분해 ----------
print("\n[Panel B] DiD 정확 분해  Δ = (A)대조풀 + (B)구성")
PB = {}
for gap in GAPS:
    d = lambda lab: np.array([x["t"] - x["cs"].mean() for x in REC[(gap, lab)]], float)
    o_old, o_new = d("S1 기존처치·기존풀"), d("S2 기존처치·확장풀")
    a_new, r_new = d("S4 확장처치·확장풀"), d("S5 회수전용·확장풀")
    if min(len(o_old), len(o_new), len(a_new), len(r_new)) < 5:
        PB[f"G{gap}"] = {"note": "표본부족"}; continue
    # 공통 bn 쌍대응 (대조풀 경로)
    bo = {x["bn"]: x for x in REC[(gap, "S1 기존처치·기존풀")]}
    bn2 = {x["bn"]: x for x in REC[(gap, "S2 기존처치·확장풀")]}
    com = sorted(set(bo) & set(bn2))
    pair = np.array([(bn2[b]["t"] - bn2[b]["cs"].mean()) - (bo[b]["t"] - bo[b]["cs"].mean()) for b in com])
    bp = np.array([pair[rng.integers(0, len(pair), len(pair))].mean() for _ in range(NB)])
    w_rec = len(r_new) / len(a_new)
    chB = w_rec * (r_new.mean() - o_new.mean())
    tot = a_new.mean() - o_old.mean()
    PB[f"G{gap}"] = {"total": round(float(tot), 4),
                     "A_pool_paired": round(float(pair.mean()), 4), "A_ci": qci(bp),
                     "A_n_common": len(com),
                     "B_composition": round(float(chB), 4), "w_rec": round(w_rec, 3),
                     "DiD_rec": round(float(r_new.mean()), 4), "DiD_orig_newpool": round(float(o_new.mean()), 4),
                     "residual": round(float(tot - pair.mean() - chB), 4)}
    p = PB[f"G{gap}"]
    print(f"  G={gap}: 총변화 {p['total']:+.4f} = 대조풀 {p['A_pool_paired']:+.4f}{p['A_ci']}"
          f"(공통 {p['A_n_common']}) + 구성 {p['B_composition']:+.4f}"
          f"(w_rec {p['w_rec']}, 회수 DiD {p['DiD_rec']:+.4f} vs 기존 {p['DiD_orig_newpool']:+.4f})"
          f" + 잔차 {p['residual']:+.4f}")

# ---------- Panel C : '붕괴'가 확립되는가 ----------
print("\n[Panel C] S1 vs S4 차이 검정 — 유의→비유의 전환은 변화의 증거가 아니다")
PC = {}
for gap in GAPS:
    r1, r4 = REC[(gap, "S1 기존처치·기존풀")], REC[(gap, "S4 확장처치·확장풀")]
    if min(len(r1), len(r4)) < 20: PC[f"G{gap}"] = {"note": "n<20"}; continue
    d1 = np.array([x["t"] - x["cs"].mean() for x in r1]); d4 = np.array([x["t"] - x["cs"].mean() for x in r4])
    bd = np.array([d1[rng.integers(0, len(d1), len(d1))].mean()
                   - d4[rng.integers(0, len(d4), len(d4))].mean() for _ in range(NB)])
    ci = qci(bd); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
    def p1(r, j):
        td = np.array([r[k]["t"] for k in j]); pc = np.concatenate([r[k]["cs"] for k in j])
        return (td < 0).mean() - (pc < 0).mean()
    bp = np.array([p1(r1, rng.integers(0, len(r1), len(r1))) - p1(r4, rng.integers(0, len(r4), len(r4)))
                   for _ in range(NB)])
    cip = qci(bp); sp = "✓" if (cip[0] > 0 or cip[1] < 0) else "✗"
    PC[f"G{gap}"] = {"DiD_diff": round(float(d1.mean() - d4.mean()), 4), "DiD_ci": ci, "DiD_sig": sg == "✓",
                     "P1_diff": round(float(p1(r1, np.arange(len(r1))) - p1(r4, np.arange(len(r4)))), 4),
                     "P1_ci": cip, "P1_sig": sp == "✓"}
    q = PC[f"G{gap}"]
    print(f"  G={gap}: DiD 차이 {q['DiD_diff']:+.4f} {ci} {sg} | P1 차이 {q['P1_diff']:+.4f} {cip} {sp}")

# ---------- Panel D : 구성 진단 ----------
print("\n[Panel D] 회수 96 vs 기존 283 구성")
def feats(FL):
    yr = [(m0 - 1) // 12 for _, m0, _ in FL]
    ep = []
    for _, m0, ti in FL:
        c = CACHE.get(m0)
        if c is not None and np.isfinite(c[0][ti]): ep.append(c[0][ti])
    return {"n": len(FL), "deal_year_median": float(np.median(yr)),
            "deal_year_p25_p75": [float(np.percentile(yr, 25)), float(np.percentile(yr, 75))],
            "share_after_2020": round(float(np.mean([y >= 2020 for y in yr])), 3),
            "size_median": round(float(np.median(ep)), 1) if ep else None}
PD = {"기존283": feats(FIRST_O), "회수96": feats(FIRST_R)}
for k, v in PD.items(): print(f"  {k}: {v}")
for gap in GAPS:
    a = PA.get(f"G{gap}|S5 회수전용·확장풀", {})
    if a.get("n", 0) >= 20:
        PD[f"G{gap}_rec_ncand"] = a["mean_ncand"]; PD[f"G{gap}_orig_ncand"] = PA[f"G{gap}|S2 기존처치·확장풀"]["mean_ncand"]
        print(f"  G={gap} 가용 대조후보 평균: 회수 {a['mean_ncand']} vs 기존 {PA[f'G{gap}|S2 기존처치·확장풀']['mean_ncand']}")

# ---------- 판정 ----------
est = any(PC.get(f"G{g}", {}).get("DiD_sig") or PC.get(f"G{g}", {}).get("P1_sig") for g in GAPS)
if est: status, concl = "GO", "붕괴가 통계적으로 확립됨 — 원고에 명시"
else: status, concl = "PARTIAL", "붕괴는 통계적으로 확립되지 않음 — '유의성 상실'과 '효과 변화'를 구별해 서술"
verdict = (f"S1 vs S4 차이: " + " | ".join(
    f"G{g} DiD {PC.get(f'G{g}',{}).get('DiD_diff')}{PC.get(f'G{g}',{}).get('DiD_ci')}"
    f"{'✓' if PC.get(f'G{g}',{}).get('DiD_sig') else '✗'}"
    f" P1 {PC.get(f'G{g}',{}).get('P1_diff')}{'✓' if PC.get(f'G{g}',{}).get('P1_sig') else '✗'}" for g in GAPS)
    + f" | {concl}")
emit("I-06", "not-yet-treated 붕괴 해부", status,
     {"panelA_specs": PA, "panelB_decomposition": PB, "panelC_collapse_test": PC,
      "panelD_composition": PD},
     "붕괴가 (A)대조풀 오염제거 때문인지 (B)회수기업 구성 때문인지 분해하고, 붕괴 자체가 유의한지 검정",
     verdict, kill_met=False, n=len(FIRST_A), extra={"conclusion": concl})
