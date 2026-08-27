# -*- coding: utf-8 -*-
"""I-33 핵심축 검증 — 수준(level)과 빈도(frequency) 중 무엇이 더 잘 식별되는가.

[왜 필요한가] v3 는 "순고용 프리미엄은 선택, 채용빈도가 결과" 로 서사를 짰다. 그 판정의 근거는
**재매칭 placebo(ARM-2)** 였는데, **본 연구의 방법론 기여(H18/§30-A)가 바로 그 arm 이 인공물임을
보였다.** 쌍고정 arm(ARM-1)에서는 실제일 +0.1130 vs 가짜일 +0.0446/+0.0034/−0.0629/+0.0447 로
사건 특정적이다. 원장도 "두 arm 은 상한·하한 괄호"라고 적었다(§1 line 53).
→ **"수준=선택, 빈도=처치" 구분은 확립되어 있지 않다.** 이 위에 논문을 쓸 수 없다.

[해법] 빈도 결과가 통과한 **동일한 세 검정**에 수준 결과를 넣는다. 더 잘 식별되는 쪽을 주 결과로
삼되, 그 이유를 '선택 여부'가 아니라 **'설계가 방어할 수 있는 범위'** 로 서술한다(규칙 10).

  T1 분기 이벤트스터디 q1..q12 — 점진 onset(개입) vs 즉시 점프(선택)
  T2 사전 관성 조절 (T3−T1)
  T3 고관성 조건부 위약 RI (I-31 과 동일 절차)

Panel E  외연/내연 분해 비중 재확인 (확장 표본 71% 검증)
"""
import gc
import numpy as np, pandas as pd
from h30_common import (load, deals, build, boot_did_ci, emit, SEED, qci, NB,
                        widx, rel_log, pi_parts, BASE)

rng = np.random.default_rng(SEED)
R_PERM = 200
print("[I-33] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
Hv, Ev, idx, mset = G["Hv"], G["Ev"], G["idx"], G["mset"]
EV, _ = build(G, allt, PE)

def zsh(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return np.nan
    x = Hv[row, c]
    return float((x == 0).mean()) if np.isfinite(x).all() else np.nan
def lev(ti, ctrls, m0, k=12):
    return rel_log(G, ti, ctrls, m0, k)

for e in EV:
    a = zsh(e["ti"], e["m0"], -12, -1); b = zsh(e["ti"], e["m0"], 1, 12)
    e["fq_t"] = b - a if (np.isfinite(a) and np.isfinite(b)) else np.nan
    cd = [zsh(k, e["m0"], 1, 12) - zsh(k, e["m0"], -12, -1) for k in e["ctrls"]]
    cd = [x for x in cd if np.isfinite(x)]
    e["fq_c"] = float(np.mean(cd)) if cd else np.nan
    e["fq"] = e["fq_t"] - e["fq_c"] if (np.isfinite(e["fq_t"]) and np.isfinite(e["fq_c"])) else np.nan
    e["lv"] = lev(e["ti"], e["ctrls"], e["m0"])
    e["pp"] = zsh(e["ti"], e["m0"], -24, -13)
_pp = np.array([e["pp"] for e in EV], float)
Q1, Q2 = np.percentile(_pp[np.isfinite(_pp)], [33.33, 66.67])
tb = lambda v: None if not np.isfinite(v) else (0 if v <= Q1 else (1 if v <= Q2 else 2))
for e in EV: e["pb"] = tb(e["pp"])

def m1(v, lab):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 20: return None
    bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(NB)])
    ci = qci(bs); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
    print(f"  {lab:<30} {v.mean():+.4f} {ci} {sg} (n={len(v)})")
    return {"est": round(float(v.mean()), 4), "ci": ci, "sig": sg == "✓", "n": len(v)}

print("\n[기준] 두 결과대상의 주 효과")
BASE_ = {"frequency(무채용비중)": m1([e["fq"] for e in EV], "빈도 DiD"),
         "level(rel12 log고용)": m1([e["lv"] for e in EV], "수준 rel12")}

# ---------- T1 분기 이벤트스터디 ----------
print("\n[T1] 분기 이벤트스터디 — 점진 onset 인가 즉시 점프인가")
KM = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
def es_level():
    rows = []
    for e in EV:
        pre = [mset.get(e["m0"] + k) for k in range(-6, 0)]
        if any(j is None for j in pre): continue
        vals = []
        ok = True
        for k in KM:
            j = mset.get(e["m0"] + k)
            if j is None: ok = False; break
            v = lev(e["ti"], e["ctrls"], e["m0"], k)
            if not np.isfinite(v): ok = False; break
            vals.append(v)
        if ok: rows.append(vals)
    return np.array(rows)
def es_freq():
    rows = []
    for e in EV:
        vals = []; ok = True
        pre_t = zsh(e["ti"], e["m0"], -12, -1)
        pre_c = [zsh(k, e["m0"], -12, -1) for k in e["ctrls"]]
        pre_c = [x for x in pre_c if np.isfinite(x)]
        if not np.isfinite(pre_t) or not pre_c: continue
        pc = float(np.mean(pre_c))
        for q in range(1, 13):
            a, b = (q - 1) * 3 + 1, q * 3
            t = zsh(e["ti"], e["m0"], a, b)
            cs = [zsh(k, e["m0"], a, b) for k in e["ctrls"]]
            cs = [x for x in cs if np.isfinite(x)]
            if not np.isfinite(t) or not cs: ok = False; break
            vals.append((t - pre_t) - (float(np.mean(cs)) - pc))
        if ok: rows.append(vals)
    return np.array(rows)
ESL, ESF = es_level(), es_freq()
T1 = {}
for lab, A, unit in (("level(rel12)", ESL, "log"), ("frequency", ESF, "pp")):
    if len(A) < 40: print(f"  {lab}: n={len(A)} (<40)"); continue
    b = A.mean(axis=0)
    bs = np.array([A[rng.integers(0, len(A), len(A))].mean(axis=0) for _ in range(NB)])
    sig = [(qci(bs[:, i])[0] > 0 or qci(bs[:, i])[1] < 0) for i in range(A.shape[1])]
    first = next((i for i, s in enumerate(sig) if s), None)
    ratio = float(b[0] / b[-1]) if b[-1] != 0 else np.nan
    T1[lab] = {"n": len(A), "path": [round(float(x), 4) for x in b],
               "sig": [bool(s) for s in sig],
               "first_sig_month": (KM[first] if lab == "level(rel12)" else (first + 1) * 3) if first is not None else None,
               "q1_over_q12": round(ratio, 3)}
    print(f"  {lab} (n={len(A)}): " + " ".join(
        f"{(KM[i] if lab=='level(rel12)' else (i+1)*3)}m:{b[i]:+.3f}{'✓' if sig[i] else ''}" for i in range(len(b))))
    print(f"    최초 유의 {T1[lab]['first_sig_month']}개월 · 1분기/12분기 비율 {ratio:.3f}"
          f"  ({'즉시 점프' if ratio > 0.6 else '점진 onset'})")

# ---------- T2 관성 조절 ----------
print("\n[T2] 사전 관성 조절")
T2 = {}
for lab, key in (("level(rel12)", "lv"), ("frequency", "fq")):
    d1 = np.array([e[key] for e in EV if e["pb"] == 0], float); d1 = d1[np.isfinite(d1)]
    d3 = np.array([e[key] for e in EV if e["pb"] == 2], float); d3 = d3[np.isfinite(d3)]
    r1 = m1(d1, f"{lab} T1저관성"); r3 = m1(d3, f"{lab} T3고관성")
    bs = np.array([d3[rng.integers(0, len(d3), len(d3))].mean()
                   - d1[rng.integers(0, len(d1), len(d1))].mean() for _ in range(NB)])
    ci = qci(bs); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
    T2[lab] = {"T1": r1, "T3": r3, "T3_T1": {"diff": round(float(d3.mean() - d1.mean()), 4),
                                             "ci": ci, "sig": sg == "✓"}}
    print(f"    {lab} T3−T1 {d3.mean()-d1.mean():+.4f} {ci} {sg}")

# ---------- T3 고관성 조건부 위약 ----------
print(f"\n[T3] 고관성 조건부 위약 (R={R_PERM}, I-31 동일 절차) — 두 결과대상 동시")
never = np.array([i for i, b in enumerate(idx) if b not in PE]); bnv = np.asarray(idx)
deal_mis = np.array([int(r.mi) for r in allt.itertuples()])
PPc = {}
def ppb(row, m0):
    k = (row, m0)
    if k not in PPc:
        v = zsh(row, m0, -24, -13); PPc[k] = None if not np.isfinite(v) else tb(v)
    return PPc[k]
actF = np.array([e["fq"] for e in EV if e["pb"] == 2], float); actF = actF[np.isfinite(actF)].mean()
actL = np.array([e["lv"] for e in EV if e["pb"] == 2], float); actL = actL[np.isfinite(actL)].mean()
nT = sum(1 for e in EV if e["pb"] == 2)
nullF, nullL = [], []
for rep in range(R_PERM):
    pick = []
    for _ in range(6):
        ci_ = rng.choice(never, size=min(len(never), nT * 12), replace=False)
        cm = rng.choice(deal_mis, size=len(ci_), replace=True)
        for i, m in zip(ci_, cm):
            if ppb(int(i), int(m)) == 2:
                pick.append((bnv[i], int(m)))
                if len(pick) >= nT: break
        if len(pick) >= nT: break
    if len(pick) < 30: continue
    df = pd.DataFrame(pick, columns=["bn10", "mi"]); df["src"] = "ph"
    L, _ = build(G, df, PE, ctrl_extra_exclude=set(df.bn10))
    fq, lv = [], []
    for e in L:
        if ppb(e["ti"], e["m0"]) != 2: continue
        a = zsh(e["ti"], e["m0"], -12, -1); b = zsh(e["ti"], e["m0"], 1, 12)
        cs = [zsh(k, e["m0"], 1, 12) - zsh(k, e["m0"], -12, -1) for k in e["ctrls"]]
        cs = [x for x in cs if np.isfinite(x)]
        if np.isfinite(a) and np.isfinite(b) and cs: fq.append((b - a) - float(np.mean(cs)))
        v = lev(e["ti"], e["ctrls"], e["m0"])
        if np.isfinite(v): lv.append(v)
    if len(fq) >= 20: nullF.append(float(np.mean(fq)))
    if len(lv) >= 20: nullL.append(float(np.mean(lv)))
    if (rep + 1) % 50 == 0:
        print(f"    rep {rep+1}: 빈도 귀무 {np.mean(nullF):+.4f} · 수준 귀무 {np.mean(nullL):+.4f}")
T3 = {}
for lab, nu, act, side in (("frequency", np.array(nullF), actF, "low"),
                           ("level(rel12)", np.array(nullL), actL, "high")):
    p = float((nu <= act).mean()) if side == "low" else float((nu >= act).mean())
    T3[lab] = {"actual_T3": round(float(act), 4), "null_mean": round(float(nu.mean()), 4),
               "null_ci": [round(float(np.percentile(nu, 2.5)), 4), round(float(np.percentile(nu, 97.5)), 4)],
               "RI_p": round(p, 4), "sig": bool(p < 0.05), "n_reps": len(nu)}
    print(f"  {lab:<14} 실제 {act:+.4f} · 귀무 {nu.mean():+.4f} "
          f"[{np.percentile(nu,2.5):+.4f}, {np.percentile(nu,97.5):+.4f}] → RI p={p:.4f} "
          f"{'✓ 귀무 밖' if p < 0.05 else '✗ 귀무 안'}")

# ---------- Panel E 분해 ----------
print("\n[Panel E] 외연/내연 분해 (확장 표본)")
num, den = [], []
for e in EV:
    a = pi_parts(G, e["ti"], e["m0"], -12, -1); b = pi_parts(G, e["ti"], e["m0"], 1, 12)
    if a is None or b is None: continue
    p0, i0 = a; p1, i1 = b
    tot = p1 * i1 - p0 * i0
    if abs(tot) < 1e-9: continue
    num.append(i0 * (p1 - p0)); den.append(tot)
num, den = np.array(num), np.array(den)
share = float(num.sum() / den.sum())
bs = np.array([num[j].sum() / den[j].sum() for j in (rng.integers(0, len(num), len(num)) for _ in range(NB))])
PE_ = {"extensive_share": round(share, 4), "ci": qci(bs), "n": len(num)}
print(f"  외연 성분 비중 {share:.4f} {qci(bs)} (n={len(num)})  [기존 기록 71%]")

# ---------- 판정 ----------
fq_pass = sum([T1.get("frequency", {}).get("q1_over_q12", 1) < 0.6,
               T2["frequency"]["T3_T1"]["sig"], T3["frequency"]["sig"]])
lv_pass = sum([T1.get("level(rel12)", {}).get("q1_over_q12", 1) < 0.6,
               T2["level(rel12)"]["T3_T1"]["sig"], T3["level(rel12)"]["sig"]])
if fq_pass > lv_pass: concl = f"**빈도가 더 잘 식별된다** ({fq_pass}/3 vs {lv_pass}/3) — 주 결과로 빈도 채택, 근거는 '선택 여부'가 아니라 '설계 방어 범위'"
elif lv_pass > fq_pass: concl = f"**수준이 더 잘 식별된다** ({lv_pass}/3 vs {fq_pass}/3) — 서사 재구성 필요"
else: concl = f"두 결과대상이 동등하게 식별된다 ({fq_pass}/3) — 빈도를 '새로운 분해'로, 수준을 '기존 문헌 대비'로 병기"
print(f"\n판정: {concl}")
emit("I-33", "핵심축 검증 — 수준 vs 빈도 식별력", "GO",
     {"baseline": BASE_, "T1_quarterly_ES": T1, "T2_inertia_moderator": T2,
      "T3_conditional_placebo": T3, "panelE_extensive_share": PE_,
      "tercile_cuts": [round(float(Q1), 4), round(float(Q2), 4)]},
     "빈도가 통과한 세 검정(점진 onset·관성 조절·고관성 위약)에 수준을 넣어 어느 쪽이 더 잘 식별되는지 판정",
     concl, kill_met=False, n=len(EV),
     extra={"why": "v3 의 '수준=선택' 판정은 재매칭 ARM-2 근거인데 H18 이 그 arm 을 인공물로 판정했다. "
                   "따라서 수준/빈도 구분을 재확립해야 논문을 쓸 수 있다.",
            "frequency_score": fq_pass, "level_score": lv_pass})
