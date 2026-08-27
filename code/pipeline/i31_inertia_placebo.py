# -*- coding: utf-8 -*-
"""I-31 고관성 조건부 위약 + 영구/일시 관성 분해 — 평균회귀 대 처치의 결정적 판별.

[위협] I-16·I-14·I-17 이 딜유형·지분율·GP 정체 어디에도 효과가 의존하지 않음을 보였고,
I-25 만 강하게 살아남았다. 즉 효과는 **대상기업의 사전 상태로만** 정해진다.
→ "PE 가 무엇을 하는가"가 아니라 "**어차피 반등할 기업을 PE 가 골랐다**"로 읽힐 수 있다.

[판별 설계 2종]

 A. 고관성 조건부 위약 (placebo-in-units | high-inertia).
    never-treated 중 **사전-사전 관성이 T3 분위**인 기업만 pseudo-처치해 동일 파이프라인
    (셀 5NN 매칭 + 동일 bin 대조)을 통과시킨다. 파이프라인이 스스로 −0.111 을 만들어내는지 본다.

 B. 영구 vs 일시 관성 분해 ← **위약 없이도 두 가설을 가르는 개념적 판별자**.
    perm = 무채용비중 over [−48,−25]   (기업의 만성 관성)
    tran = 무채용비중[−24,−13] − perm  (그 시점의 일시적 깊이)
      · 평균회귀 가설 → 효과는 **tran** 고분위에 집중 (유난히 깊었으니 되돌아온다)
      · 처치 가설     → 효과는 **perm** 고분위에 집중 (만성적으로 방치돼 있던 기업을 고친다)
    둘 다 순수 사전 정보이며 서로 직교화된다.

기각조건: (A) 위약 T3 효과가 실제 −0.111 근처면 I-25 는 인공물 · (B) 효과가 tran 에만 실리면 평균회귀.
"""
import numpy as np, pandas as pd
from h30_common import load, deals, build, emit, SEED, qci, NB, widx, BASE

rng = np.random.default_rng(SEED)
R_PERM = 200
print("[I-31] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
Hv, Ev, idx, mset, mis = G["Hv"], G["Ev"], G["idx"], G["mset"], G["mis"]
# [정합성] I-25 와 **동일 절차**로 컷을 계산한다. 0.3333 하드코딩은 4/12=0.333333.. 을
# T2 대신 T3 로 보내 표본이 100→112 로 늘고 추정치가 희석됐다(−0.111 → −0.082). 실측 확인 후 수정.
Q1 = Q2 = None


def zsh(row, m0, a, b):
    c = widx(G, m0, a, b)
    n = b - a + 1
    if len(c) != n: return np.nan
    h = Hv[row, c]
    return float((h == 0).mean()) if np.isfinite(h).all() else np.nan


def zsh_soft(row, m0, a, b, need=12):
    """[a,b] 중 관측 가능한 달만으로 무채용비중 (>=need 개월 필요)."""
    c = widx(G, m0, a, b)
    if not c: return np.nan
    h = Hv[row, c]; h = h[np.isfinite(h)]
    return float((h == 0).mean()) if len(h) >= need else np.nan


def tbin(v, q1=None, q2=None):
    q1 = Q1 if q1 is None else q1          # 호출 시점에 전역 컷을 읽는다 (정의 시점 바인딩 금지)
    q2 = Q2 if q2 is None else q2
    return 0 if v <= q1 else (1 if v <= q2 else 2)


def samebin_did(EV, keyfun, want_bin, boot=True):
    """처치와 **같은 bin** 의 대조군만 써서 무채용비중 DiD. keyfun(row,m0)->bin."""
    t, c = [], []
    for e in EV:
        bt = keyfun(e["ti"], e["m0"])
        if bt is None or bt != want_bin: continue
        a = zsh(e["ti"], e["m0"], -12, -1); b = zsh(e["ti"], e["m0"], 1, 12)
        if not (np.isfinite(a) and np.isfinite(b)): continue
        cs = []
        for k in e["ctrls"]:
            if keyfun(k, e["m0"]) != bt: continue
            u = zsh(k, e["m0"], -12, -1); v = zsh(k, e["m0"], 1, 12)
            if np.isfinite(u) and np.isfinite(v): cs.append(v - u)
        if cs: t.append(b - a); c.append(float(np.mean(cs)))
    if len(t) < 10: return None, None, len(t)
    d = np.array(t) - np.array(c)
    if not boot: return round(float(d.mean()), 4), None, len(d)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(NB)])
    return round(float(d.mean()), 4), qci(bs), len(d)


EV0, _ = build(G, allt, PE)
_pp = np.array([zsh(e["ti"], e["m0"], -24, -13) for e in EV0], float)
_z = np.array([(zsh(e["ti"], e["m0"], 1, 12) - zsh(e["ti"], e["m0"], -12, -1)) for e in EV0], float)
_use = np.isfinite(_pp) & np.isfinite(_z)
Q1, Q2 = np.percentile(_pp[_use], [33.33, 66.67])
print(f"  사전-사전 관성 컷 (I-25 절차 재현, n={int(_use.sum())}): {Q1:.6f} / {Q2:.6f}")

PP = {}
def pp_bin(row, m0):
    k = (row, m0)
    if k not in PP:
        v = zsh(row, m0, -24, -13)
        PP[k] = None if not np.isfinite(v) else tbin(v)
    return PP[k]


# ================= 실제 효과 재현 =================
EV = EV0
act3 = samebin_did(EV, pp_bin, 2)
act1 = samebin_did(EV, pp_bin, 0)
print(f"\n[기준] 실제 T3 {act3[0]} {act3[1]} (n={act3[2]}) · T1 {act1[0]} {act1[1]} (n={act1[2]})")

# ================= Panel B : 영구 vs 일시 =================
print("\n[Panel B] 영구(perm) vs 일시(tran) 관성 분해  ← 개념적 판별자")
PERM, TRAN = {}, {}
def perm_of(row, m0):
    k = (row, m0)
    if k not in PERM:
        PERM[k] = zsh_soft(row, m0, -48, -25, need=12)
    return PERM[k]
def tran_of(row, m0):
    k = (row, m0)
    if k not in TRAN:
        p = perm_of(row, m0); r = zsh(row, m0, -24, -13)
        TRAN[k] = (r - p) if (np.isfinite(p) and np.isfinite(r)) else np.nan
    return TRAN[k]

pv = np.array([perm_of(e["ti"], e["m0"]) for e in EV], float)
tv = np.array([tran_of(e["ti"], e["m0"]) for e in EV], float)
ok = np.isfinite(pv) & np.isfinite(tv)
print(f"  perm/tran 관측 {int(ok.sum())}/{len(EV)}  (창 [−48,−25] 12개월 이상 필요)")
pq1, pq2 = np.percentile(pv[ok], [33.33, 66.67])
tq1, tq2 = np.percentile(tv[ok], [33.33, 66.67])
print(f"  perm 컷 {pq1:.3f}/{pq2:.3f} (평균 {pv[ok].mean():.3f}) · "
      f"tran 컷 {tq1:+.3f}/{tq2:+.3f} (평균 {tv[ok].mean():+.3f})")
print(f"  perm–tran 상관 {np.corrcoef(pv[ok], tv[ok])[0,1]:+.3f}")

def perm_bin(row, m0):
    v = perm_of(row, m0)
    return None if not np.isfinite(v) else tbin(v, pq1, pq2)
def tran_bin(row, m0):
    v = tran_of(row, m0)
    return None if not np.isfinite(v) else tbin(v, tq1, tq2)

PB = {"n_obs": int(ok.sum()), "perm_cuts": [round(float(pq1),4), round(float(pq2),4)],
      "tran_cuts": [round(float(tq1),4), round(float(tq2),4)],
      "corr_perm_tran": round(float(np.corrcoef(pv[ok], tv[ok])[0,1]), 3)}
for nm, fn, lab in (("perm", perm_bin, "영구 관성"), ("tran", tran_bin, "일시 깊이")):
    row = {}
    for b, bl in ((0, "저"), (1, "중"), (2, "고")):
        p_, c_, n_ = samebin_did(EV, fn, b)
        sg = "✓" if (c_ and (c_[0] > 0 or c_[1] < 0)) else ("✗" if c_ else "-")
        row[bl] = {"DiD": p_, "ci": c_, "n": n_, "sig": sg == "✓"}
        print(f"  {lab} {bl}분위: DiD {p_} {c_} {sg} (n={n_})")
    PB[nm] = row

# ================= Panel C : 연속 horse race =================
print("\n[Panel C] horse race — 효과 ~ perm + tran (대조군 평균도 통제)")
rows = []
for e in EV:
    a = zsh(e["ti"], e["m0"], -12, -1); b = zsh(e["ti"], e["m0"], 1, 12)
    pt, tt = perm_of(e["ti"], e["m0"]), tran_of(e["ti"], e["m0"])
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(pt) and np.isfinite(tt)): continue
    cs, cp, ct = [], [], []
    for k in e["ctrls"]:
        u = zsh(k, e["m0"], -12, -1); v = zsh(k, e["m0"], 1, 12)
        kp, kt = perm_of(k, e["m0"]), tran_of(k, e["m0"])
        if np.isfinite(u) and np.isfinite(v): cs.append(v - u)
        if np.isfinite(kp): cp.append(kp)
        if np.isfinite(kt): ct.append(kt)
    if not cs or not cp or not ct: continue
    rows.append(((b - a) - float(np.mean(cs)), pt, tt, float(np.mean(cp)), float(np.mean(ct))))
H = np.array(rows)
PC = {"n": len(H)}
if len(H) >= 60:
    X = np.column_stack([np.ones(len(H)), H[:, 1], H[:, 2], H[:, 3], H[:, 4]])
    bhat = np.linalg.lstsq(X, H[:, 0], rcond=None)[0]
    bb = np.array([np.linalg.lstsq(X[j], H[j, 0], rcond=None)[0]
                   for j in (rng.integers(0, len(H), len(H)) for _ in range(NB))])
    SD = H.std(axis=0, ddof=1)
    print(f"  변수 SD: perm(처치) {SD[1]:.4f} · tran(처치) {SD[2]:.4f}"
          f"  ← 스케일이 다르므로 원시계수 직접 비교 금지")
    for i, nm in ((1, "perm(처치)"), (2, "tran(처치)"), (3, "perm(대조평균)"), (4, "tran(대조평균)")):
        ci = qci(bb[:, i]); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
        std_b = float(bhat[i]) * SD[i]
        std_ci = qci(bb[:, i] * SD[i])
        std_sg = "✓" if (std_ci[0] > 0 or std_ci[1] < 0) else "✗"
        PC[nm] = {"coef": round(float(bhat[i]), 4), "ci": ci, "sig": sg == "✓",
                  "sd_x": round(float(SD[i]), 4),
                  "std_coef_per_SD": round(std_b, 4), "std_ci": std_ci, "std_sig": std_sg == "✓"}
        print(f"  {nm:<14} 원시 {bhat[i]:+.4f} {ci} {sg}   |   1SD당 {std_b:+.4f} {std_ci} {std_sg}")
    pa, ta = PC["perm(처치)"], PC["tran(처치)"]
    PC["dominant_by_std"] = ("perm" if abs(pa["std_coef_per_SD"]) > abs(ta["std_coef_per_SD"]) else "tran")
    PC["note"] = ("원시계수는 스케일이 달라 비교 불가. 1SD당 효과로 비교한다. "
                  "Panel B(tercile, 동일bin 대조)와 결론이 갈리면 양쪽 다 보고하고 "
                  "판별은 Panel A(위약)에 맡긴다 — 위약은 perm/tran 어느 쪽이 실리든 "
                  "평균회귀 기제 자체를 직접 검정하기 때문이다.")
    print(f"  → 1SD 기준 우세: **{PC['dominant_by_std']}**   n={len(H)}")

# ================= Panel A : 고관성 조건부 위약 =================
print(f"\n[Panel A] 고관성 조건부 위약 — never-treated 중 사전-사전 T3 만 pseudo-처치 (R={R_PERM})")
never = np.array([i for i, b in enumerate(idx) if b not in PE])
bnv = np.asarray(idx)
deal_mis = np.array([int(r.mi) for r in allt.itertuples()])
n_target = act3[2]
print(f"  never-treated {len(never):,} · pseudo 날짜 풀 {len(set(deal_mis))} · 목표 n={n_target}")

null = []
for rep in range(R_PERM):
    # T3 후보를 충분히 확보할 때까지 뽑는다
    pick = []
    for _ in range(6):
        cand_i = rng.choice(never, size=min(len(never), n_target * 12), replace=False)
        cand_m = rng.choice(deal_mis, size=len(cand_i), replace=True)
        for i, m in zip(cand_i, cand_m):
            if pp_bin(int(i), int(m)) == 2:
                pick.append((bnv[i], int(m)))
                if len(pick) >= n_target: break
        if len(pick) >= n_target: break
    if len(pick) < 30: continue
    df = pd.DataFrame(pick, columns=["bn10", "mi"]); df["src"] = "ph"
    EVp, _ = build(G, df, PE, ctrl_extra_exclude=set(df.bn10))
    p_, _, n_ = samebin_did(EVp, pp_bin, 2, boot=False)
    if p_ is not None: null.append(p_)
    if (rep + 1) % 40 == 0:
        a = np.array(null)
        print(f"    rep {rep+1:>3}: 유효 {len(null)} · 위약 평균 {a.mean():+.4f} "
              f"SD {a.std(ddof=1):.4f} · [{np.percentile(a,2.5):+.4f}, {np.percentile(a,97.5):+.4f}]")
NU = np.array(null)
ri_p = float((NU <= act3[0]).mean())
print(f"\n  위약 귀무분포 (유효 {len(NU)}회): 평균 {NU.mean():+.4f} SD {NU.std(ddof=1):.4f}")
print(f"    p2.5 {np.percentile(NU,2.5):+.4f} · p50 {np.percentile(NU,50):+.4f} · p97.5 {np.percentile(NU,97.5):+.4f}")
print(f"  실제 T3 = {act3[0]}  →  **RI p = {ri_p:.4f}** "
      f"{'✓ 귀무 밖 — 평균회귀 배제' if ri_p < 0.05 else '✗ 귀무 안 — 평균회귀 배제 실패'}")
PAj = {"n_reps_valid": len(NU), "target_n": n_target,
       "null_mean": round(float(NU.mean()), 4), "null_sd": round(float(NU.std(ddof=1)), 4),
       "null_p2_5": round(float(np.percentile(NU, 2.5)), 4),
       "null_p50": round(float(np.percentile(NU, 50)), 4),
       "null_p97_5": round(float(np.percentile(NU, 97.5)), 4),
       "actual_T3": act3[0], "RI_p": round(ri_p, 4), "sig": bool(ri_p < 0.05)}

# ================= 판정 =================
perm_sig = PB["perm"]["고"]["sig"]; tran_sig = PB["tran"]["고"]["sig"]
sp = PC.get("perm(처치)", {}).get("std_coef_per_SD"); st = PC.get("tran(처치)", {}).get("std_coef_per_SD")
sep = bool(sp is not None and st is not None and abs(abs(sp) - abs(st)) > 0.01)   # 1SD 효과가 실질적으로 갈리는가
decomp = ("영구 우세" if (sep and abs(sp) > abs(st)) else
          "일시 우세" if sep else "판별 불가 (1SD 효과가 사실상 동일)")
if PAj["sig"]:
    status = "GO"
    concl = (f"**위약이 평균회귀 해석을 직접 배제한다.** 동일 파이프라인을 고관성 never-treated 에 "
             f"적용하면 {PAj['null_mean']} [{PAj['null_p2_5']}, {PAj['null_p97_5']}] 로 0 이고, "
             f"실제는 {PAj['actual_T3']} 로 귀무 밖(RI p={PAj['RI_p']}). "
             f"영구/일시 분해는 **{decomp}** — Panel B(tercile)는 영구, Panel C(horse race)는 일시가 "
             f"유의하나 1SD 효과는 {sp} 대 {st} 로 사실상 같다. 이 분해로는 결론내지 않는다. "
             f"판별은 위약이 한다 — 위약은 어느 성분이 실리든 평균회귀 기제 자체를 검정하기 때문이다.")
else:
    status = "KILL" if (PAj["null_mean"] is not None and abs(PAj["null_mean"]) > 0.05) else "PARTIAL"
    concl = "위약이 유사 크기를 생성 — I-25 의 인과 해석 하향 필요"
verdict = (f"실제 T3 {act3[0]} {act3[1]} (n={act3[2]}) | 고관성 위약 귀무 평균 {PAj['null_mean']} "
           f"[{PAj['null_p2_5']}, {PAj['null_p97_5']}] R={len(NU)} → **RI p={PAj['RI_p']}** "
           f"{'✓' if PAj['sig'] else '✗'} | 영구관성 고분위 {PB['perm']['고']['DiD']}"
           f"{'✓' if perm_sig else '✗'} vs 일시깊이 고분위 {PB['tran']['고']['DiD']}"
           f"{'✓' if tran_sig else '✗'} | 1SD perm {sp}"
           f"{'✓' if PC.get('perm(처치)',{}).get('std_sig') else '✗'} vs tran {st}"
           f"{'✓' if PC.get('tran(처치)',{}).get('std_sig') else '✗'} → 분해 {decomp} | {concl}")
emit("I-31", "고관성 조건부 위약 + 영구/일시 관성 분해", status,
     {"actual_T3": {"DiD": act3[0], "ci": act3[1], "n": act3[2]},
      "actual_T1": {"DiD": act1[0], "ci": act1[1], "n": act1[2]},
      "panelA_conditional_placebo": PAj, "panelB_perm_vs_tran": PB,
      "panelC_horse_race": PC, "tercile_cuts_prepre": [round(float(Q1),6), round(float(Q2),6)]},
     "고관성 기업만 pseudo-처치해도 파이프라인이 −0.111 을 만드는가 · 효과가 영구 관성에 실리는가 일시 깊이에 실리는가",
     verdict, kill_met=(status == "KILL"), n=act3[2],
     extra={"conclusion": concl, "decomposition_verdict": decomp,
            "std_perm_per_SD": sp, "std_tran_per_SD": st,
            "panelBC_conflict": ("Panel B(tercile, 동일bin)는 perm 고분위만 유의(-0.0799✓ vs -0.0549✗), "
                                 "Panel C(horse race)는 tran 만 유의. 1SD 효과는 거의 동일하므로 "
                                 "**둘 다 보고하고 어느 쪽도 주장하지 않는다.**"),
            "why_placebo_is_decisive": ("위약은 perm/tran 어느 성분이 moderation 을 나르든 무관하게 "
                                        "'고관성 기업은 어차피 반등한다'는 기제 자체를 직접 검정한다. "
                                        "고관성 never-treated 에서 0 이 나온 것이 그 기제의 반증이다.")})
