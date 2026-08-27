# -*- coding: utf-8 -*-
"""I-42 ★ 최종 판별 — Δlog 채용률 결과대상에 고관성 조건부 위약.

I-41 이 두 경고를 냈다. (B) 관성의 R²=0.639 가 규모·산업·성장·업력으로 설명되고 잔차 기울기는
무유의. (E) 사전 log총채용을 통제하면 관성 계수가 −0.381 로 부호가 뒤집힌다(과잉통제 의심).
공통 우려는 **로그 평균회귀** — 사전 채용률이 낮은 기업은 그것만으로 Δlog 가 크게 나온다.

I-31 의 조건부 위약은 **무채용비중**에만 걸었다. 이제 **Δlog 채용률**에 같은 위약을 건다.
  · 위약 T3 ≈ 0 이면 → 조절자는 진짜다 (로그 평균회귀로 설명되지 않음)
  · 위약 T3 가 실제(+0.3741)에 근접하면 → 조절자는 로그 평균회귀 인공물. 논문 재설계 필요

동시에 T1 위약도 내어 **위약의 기울기(T3−T1)** 를 실제 기울기(+0.3472)와 직접 비교한다.
"""
import numpy as np, pandas as pd
from h30_common import load, deals, build, emit, SEED, qci, NB, widx

rng = np.random.default_rng(SEED)
R = 200
print("[I-42] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV0, _ = build(G, allt, PE)
Hv, Ev, idx, mset = G["Hv"], G["Ev"], G["idx"], G["mset"]

def W(row, m0, a, b):
    c = widx(G, m0, a, b); n = b - a + 1
    if len(c) != n: return None
    h, e = Hv[row, c].astype(float), Ev[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.nanmean(e) < 5: return None
    return h, e
def lrate(row, m0, a, b):
    w = W(row, m0, a, b)
    if w is None: return np.nan
    N, E = w[0].sum(), np.mean(w[1])
    return float(np.log(N / E)) if (N > 0 and E > 0) else np.nan
PPc = {}
def pp(row, m0):
    k = (row, m0)
    if k not in PPc:
        w = W(row, m0, -24, -13)
        PPc[k] = float((w[0] == 0).mean()) if w else np.nan
    return PPc[k]

_p = np.array([pp(e["ti"], e["m0"]) for e in EV0], float)
_y = np.array([lrate(e["ti"], e["m0"], 1, 12) - lrate(e["ti"], e["m0"], -12, -1) for e in EV0], float)
u = np.isfinite(_p) & np.isfinite(_y)
Q1, Q2 = np.percentile(_p[u], [33.33, 66.67])
tb = lambda v: None if not np.isfinite(v) else (0 if v <= Q1 else (1 if v <= Q2 else 2))
print(f"  관성 컷 {Q1:.4f}/{Q2:.4f}")

def tercile_did(EV, want, same_bin=True):
    t, c = [], []
    for e in EV:
        b = tb(pp(e["ti"], e["m0"]))
        if b != want: continue
        y = lrate(e["ti"], e["m0"], 1, 12) - lrate(e["ti"], e["m0"], -12, -1)
        if not np.isfinite(y): continue
        cs = []
        for k in e["ctrls"]:
            if same_bin and tb(pp(k, e["m0"])) != b: continue
            v = lrate(k, e["m0"], 1, 12) - lrate(k, e["m0"], -12, -1)
            if np.isfinite(v): cs.append(v)
        if cs: t.append(y); c.append(float(np.mean(cs)))
    if len(t) < 10: return None, 0
    return np.array(t) - np.array(c), len(t)

for lab, sb in (("동일bin", True), ("전체대조", False)):
    a1, n1 = tercile_did(EV0, 0, sb); a3, n3 = tercile_did(EV0, 2, sb)
    if a1 is None or a3 is None: continue
    bs = np.array([a3[rng.integers(0, len(a3), len(a3))].mean()
                   - a1[rng.integers(0, len(a1), len(a1))].mean() for _ in range(NB)])
    print(f"  실제 [{lab}] T1 {a1.mean():+.4f}(n={n1}) · T3 {a3.mean():+.4f}(n={n3}) · "
          f"T3−T1 {a3.mean()-a1.mean():+.4f} {qci(bs)}")
    if sb: ACT1, ACT3, N1, N3 = a1.mean(), a3.mean(), n1, n3

print(f"\n[위약] 고관성·저관성 never-treated 를 동일 파이프라인에 (R={R})")
never = np.array([i for i, b in enumerate(idx) if b not in PE]); bnv = np.asarray(idx)
dm = np.array([int(r.mi) for r in allt.itertuples()])
nullT3, nullT1, nullDiff = [], [], []
for rep in range(R):
    picks = {0: [], 2: []}
    for _ in range(8):
        ci_ = rng.choice(never, size=6000, replace=False)
        cm = rng.choice(dm, size=len(ci_), replace=True)
        for i, m in zip(ci_, cm):
            b = tb(pp(int(i), int(m)))
            if b in (0, 2) and len(picks[b]) < (N1 if b == 0 else N3):
                picks[b].append((bnv[i], int(m)))
        if len(picks[0]) >= N1 and len(picks[2]) >= N3: break
    if len(picks[0]) < 30 or len(picks[2]) < 30: continue
    rows = [(b, m, "ph") for g in picks.values() for b, m in g]
    df = pd.DataFrame(rows, columns=["bn10", "mi", "src"])
    L, _ = build(G, df, PE, ctrl_extra_exclude=set(df.bn10))
    p1, _ = tercile_did(L, 0); p3, _ = tercile_did(L, 2)
    if p1 is None or p3 is None: continue
    nullT1.append(float(p1.mean())); nullT3.append(float(p3.mean()))
    nullDiff.append(float(p3.mean() - p1.mean()))
    if (rep + 1) % 50 == 0:
        print(f"    rep {rep+1}: 위약 T3 {np.mean(nullT3):+.4f} · T1 {np.mean(nullT1):+.4f} · "
              f"차이 {np.mean(nullDiff):+.4f}  (유효 {len(nullT3)})")
NT3, NT1, ND = np.array(nullT3), np.array(nullT1), np.array(nullDiff)
ACTD = ACT3 - ACT1
def ri(nu, act): return float((nu >= act).mean())
OUT = {}
print()
for lab, nu, act in (("T3 고관성", NT3, ACT3), ("T1 저관성", NT1, ACT1), ("T3−T1 기울기", ND, ACTD)):
    p = ri(nu, act)
    OUT[lab] = {"actual": round(float(act), 4), "null_mean": round(float(nu.mean()), 4),
                "null_ci": [round(float(np.percentile(nu, 2.5)), 4), round(float(np.percentile(nu, 97.5)), 4)],
                "RI_p": round(p, 4), "sig": bool(p < 0.05), "n_reps": len(nu)}
    print(f"  {lab:<12} 실제 {act:+.4f} · 위약 {nu.mean():+.4f} "
          f"[{np.percentile(nu,2.5):+.4f}, {np.percentile(nu,97.5):+.4f}] → RI p={p:.4f} "
          f"{'✓ 귀무 밖' if p < 0.05 else '✗ 귀무 안'}")

g = OUT["T3−T1 기울기"]; t3 = OUT["T3 고관성"]
if g["sig"] and t3["sig"]:
    status = "GO"; concl = ("**조절자는 로그 평균회귀 인공물이 아니다.** 고관성 never-treated 를 동일 "
                            "파이프라인에 넣어도 실제만큼의 Δlog 증가도, 기울기도 재현되지 않는다.")
elif g["sig"]:
    status = "GO"; concl = "**기울기가 위약 귀무 밖이다** — 조절자 자체는 진짜다 (수준은 일부 재현)"
else:
    status = "KILL"; concl = ("위약이 실제와 유사한 기울기를 만든다 — **조절자는 로그 평균회귀로 "
                              "설명된다.** 논문의 마지막 기둥이 무너진다.")
emit("I-42", "Δlog 채용률 고관성 조건부 위약 (최종 판별)", status,
     {"results": OUT, "tercile_cuts": [round(float(Q1), 4), round(float(Q2), 4)],
      "n_T1": N1, "n_T3": N3, "R": R},
     "Δlog 채용률에서도 조절자가 위약 귀무 밖이면 로그 평균회귀 인공물이 아니다",
     f"실제 T3 {t3['actual']} vs 위약 {t3['null_mean']} (RI p={t3['RI_p']}) · "
     f"기울기 실제 {g['actual']} vs 위약 {g['null_mean']} (RI p={g['RI_p']}) | {concl}",
     kill_met=(status == "KILL"), n=N1 + N3, extra={"conclusion": concl})
