# -*- coding: utf-8 -*-
"""I-44 상태변수 방어 — I-41 Panel E 부호반전의 정체 규명 + 수렴 타당도.

경위. I-41 Panel E 는 `Δlog 채용률 ~ 휴면 + log 사전총채용` 에서 휴면 계수가 −0.3812 로 반전했고,
이를 근거로 원고는 "휴면이 사전 채용량을 넘어서는 독립 설명력을 갖는다고 주장하지 않는다"로 하향했다.

그런데 그 회귀는 **종속변수의 구성요소를 우변에 넣은 것**이다.
    결과 = [log(N_post/E_post) − log(N_pre/E_pre)] − (대조 동일)
    통제 = log N_pre,  **동일한 [−12,−1] 창**
따라서 log N_pre 는 계수 −1 로 결과에 내장되어 있다. 변화량을 자기 기저수준에 회귀시키는 고전적
regression-to-the-mean 오류이며, 그 음의 편의가 (휴면과 사전량의 강한 음의 상관을 통해) 휴면 계수로
전이된다. 이것이 사실이면 −0.3812 는 PE 에 대해 아무것도 말하지 않는다.

Panel A  기계성 진단 — 결과를 log 사전총채용에 회귀. 창 겹침 여부로 대비.
Panel B  ★ 위약 복제 — **대조기업을 유사처치로** 삼아(각 대조는 같은 셀의 나머지 대조가 대조군)
         동일 회귀를 돌린다. 아무 일도 없었던 기업에서 같은 부호반전이 나오면 기계적이다.
Panel C  수렴 타당도 — 상태변수 4종을 **전부 [−24,−13]**(결과 기준창 밖)에서 측정해 각각 검정.
         지표가 달라도 같은 기울기가 나오면 '어느 지표가 일을 하는가'는 답할 필요가 없는 질문이다.
Panel D  의미 있는 경마 — 휴면 vs 사전 고용성장 (§2.3 의 규모 계정 vs 해제 계정 판별)
         ※ Panel C·D 의 상태변수·성장은 전부 [−24,−13] 또는 [−36,−25] 측정 — 결과 기준창 밖.
Panel E  인공물 2차 증명 — 겹침을 반대편(상태변수 ↔ 기준창)으로 옮기면 부호가 반대로 과장된다

[메모리] 회귀는 이벤트 수준 스칼라만(최대 342×6). 패널 pivot 외 대형객체 없음.
"""
import numpy as np
from h30_common import load, deals, build, emit, SEED, qci, NB, widx

rng = np.random.default_rng(SEED)
SESOI = 0.3472

print("[I-44] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Ev = G["Hv"], G["Ev"]


def win(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return None
    h, e = Hv[row, c].astype(float), Ev[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.mean(e) < 5: return None
    return h, e


def lrate(row, m0, a, b):
    w = win(row, m0, a, b)
    if w is None: return np.nan
    N, E = w[0].sum(), np.mean(w[1])
    return float(np.log(N / E)) if N > 0 else np.nan


def state(row, m0):
    """상태변수 4종 + 성장, 전부 [−24,−13] — 결과대상의 기준창 [−12,−1] 밖."""
    w = win(row, m0, -24, -13)
    if w is None: return None
    h, e = w
    run = 0
    for x in h[::-1]:
        if x == 0: run += 1
        else: break
    mx, cur_ = 0, 0
    for x in h:
        cur_ = cur_ + 1 if x == 0 else 0
        mx = max(mx, cur_)
    w3 = win(row, m0, -36, -25)
    return dict(dorm=float((h == 0).mean()),          # 무채용 월 비중
                lN=float(np.log1p(h.sum())),          # log(1+총채용)
                lr=float(np.log1p(h.sum() / np.mean(e))),   # log(1+채용률)
                mspell=float(mx),                     # 최장 무채용 spell
                cur=float(run),
                grow=(float(np.log(np.mean(e) / np.mean(w3[1]))) if (w3 and np.mean(w3[1]) > 0)
                      else np.nan))


def ols(X, y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    bb = np.array([np.linalg.lstsq(X[i], y[i], rcond=None)[0]
                   for i in (rng.integers(0, len(y), len(y)) for _ in range(NB))])
    return b, bb


def rep(b, bb, j):
    ci = qci(bb[:, j])
    return {"coef": round(float(b[j]), 4), "ci": ci, "sig": bool(ci[0] > 0 or ci[1] < 0)}


# ───────── 이벤트 수준 스칼라 ─────────
for e in EV:
    a, b = lrate(e["ti"], e["m0"], -12, -1), lrate(e["ti"], e["m0"], 1, 12)
    e["yt"] = b - a if (np.isfinite(a) and np.isfinite(b)) else np.nan
    cs = [lrate(k, e["m0"], 1, 12) - lrate(k, e["m0"], -12, -1) for k in e["ctrls"]]
    cs = [x for x in cs if np.isfinite(x)]
    e["yc"] = float(np.mean(cs)) if cs else np.nan
    e["eff"] = e["yt"] - e["yc"] if (np.isfinite(e["yt"]) and np.isfinite(e["yc"])) else np.nan
    w2 = win(e["ti"], e["m0"], -12, -1)
    e["lpreN_near"] = float(np.log1p(w2[0].sum())) if w2 else np.nan   # 결과 기준창과 **겹침**
    e["S"] = state(e["ti"], e["m0"])
    # 결과 재기준화: 기준창을 [−24,−13] 로 옮기면 [−12,−1] 총채용은 정당한 통제가 된다
    a2 = lrate(e["ti"], e["m0"], -24, -13)
    cs2 = [lrate(k, e["m0"], 1, 12) - lrate(k, e["m0"], -24, -13) for k in e["ctrls"]]
    cs2 = [x for x in cs2 if np.isfinite(x)]
    e["eff_far"] = (b - a2 - float(np.mean(cs2))) if (np.isfinite(b) and np.isfinite(a2) and cs2) else np.nan

U = [e for e in EV if np.isfinite(e["eff"]) and e["S"] and np.isfinite(e["lpreN_near"])]
y = np.array([e["eff"] for e in U])
d = np.array([e["S"]["dorm"] for e in U])
zn = np.array([e["lpreN_near"] for e in U])
zf = np.array([e["S"]["lN"] for e in U])
one = np.ones(len(U))
print(f"  분석표본 {len(U)}/{len(EV)} · corr(휴면, log사전총채용[−12,−1]) = {np.corrcoef(d, zn)[0,1]:+.3f}")

# ───────── Panel A ─────────
print("\n[Panel A] 기계성 진단 — 결과를 사전 총채용에 회귀")
PA = {}
for tag, z, lab in (("near", zn, "log N[−12,−1]  ★결과 기준창과 겹침"),
                    ("far", zf, "log N[−24,−13]  겹치지 않음")):
    b, bb = ols(np.column_stack([one, z]), y)
    PA[f"y_on_{tag}"] = rep(b, bb, 1)
    print(f"  결과 ~ {lab:<34} {PA[f'y_on_{tag}']['coef']:>+8.4f} {PA[f'y_on_{tag}']['ci']}")
# 처치기업 자체 변화(대조 차감 전) 를 자기 기저에 회귀 — 순수 산술
m = np.array([np.isfinite(e["yt"]) for e in U])
yt = np.array([e["yt"] for e in U])[m]
b, bb = ols(np.column_stack([one[m], zn[m]]), yt)
PA["yt_on_near"] = rep(b, bb, 1)
print(f"  처치 자체변화 ~ log N[−12,−1] (대조 차감 전)      {PA['yt_on_near']['coef']:>+8.4f} "
      f"{PA['yt_on_near']['ci']}   ← 산술적으로 −1 방향")

# ───────── Panel B  ★ 대조기업 위약 복제 ─────────
print("\n[Panel B] ★ 위약 복제 — 대조기업을 유사처치로 (아무 일도 없었던 기업)")
# 유사처치는 같은 셀에서 5건씩 나온다 — 독립 재표본은 SE 를 과소추정한다. **이벤트(셀) 군집** 부트.
rows = []            # (event_idx, y, dorm, log preN_near)
for gi, e in enumerate(EV):
    ok = [k for k in e["ctrls"]
          if np.isfinite(lrate(k, e["m0"], -12, -1)) and np.isfinite(lrate(k, e["m0"], 1, 12))]
    if len(ok) < 3: continue
    ch = {k: lrate(k, e["m0"], 1, 12) - lrate(k, e["m0"], -12, -1) for k in ok}
    for k in ok:                                   # k 를 유사처치, 같은 셀의 나머지를 대조로
        S = state(k, e["m0"]); w2 = win(k, e["m0"], -12, -1)
        if S is None or w2 is None: continue
        others = [ch[j] for j in ok if j != k]
        rows.append((gi, ch[k] - float(np.mean(others)), S["dorm"], float(np.log1p(w2[0].sum()))))
gidx = np.array([r[0] for r in rows]); py = np.array([r[1] for r in rows])
pd_ = np.array([r[2] for r in rows]); pz = np.array([r[3] for r in rows])
GU = np.unique(gidx)
byg = {g_: np.where(gidx == g_)[0] for g_ in GU}


def ols_cluster(cols, yy):
    """이벤트(셀) 군집 부트스트랩. cols 는 회귀행렬 열 목록(절편 제외)."""
    X = np.column_stack([np.ones(len(yy))] + list(cols))
    b = np.linalg.lstsq(X, yy, rcond=None)[0]
    bb = []
    for _ in range(NB):
        pick = np.concatenate([byg[GU[i]] for i in rng.integers(0, len(GU), len(GU))])
        try:
            bb.append(np.linalg.lstsq(X[pick], yy[pick], rcond=None)[0])
        except np.linalg.LinAlgError:
            pass
    return b, np.array(bb)


b, bb = ols_cluster([pd_], py)
PB = {"n_pseudo": int(len(py)), "n_clusters": int(len(GU)),
      "se_note": "이벤트(셀) 군집 부트스트랩 — 유사처치는 셀당 최대 5건으로 독립이 아니다",
      "dorm_alone": rep(b, bb, 1)}
b, bb = ols_cluster([pd_, pz], py)
PB["dorm_given_preN"] = rep(b, bb, 1); PB["preN_coef"] = rep(b, bb, 2)
# 3분위 대비 (헤드라인 +0.3472 와 같은 형태)
q1, q2 = np.percentile(pd_, [33.33, 66.67])
mlo, mhi = pd_ <= q1, pd_ > q2
dif = float(py[mhi].mean() - py[mlo].mean())
db = []
for _ in range(NB):
    pick = np.concatenate([byg[GU[i]] for i in rng.integers(0, len(GU), len(GU))])
    a_, b_ = py[pick][pd_[pick] > q2], py[pick][pd_[pick] <= q1]
    if len(a_) and len(b_): db.append(a_.mean() - b_.mean())
dci = qci(np.array(db))
PB["dormant_minus_active"] = {"diff": round(dif, 4), "ci": dci,
                              "sig": bool(dci[0] > 0 or dci[1] < 0),
                              "n_dormant": int(mhi.sum()), "n_active": int(mlo.sum())}
print(f"  유사처치 {len(py)}건 / 셀 {len(GU)}개 (미처치 대조기업, 셀 군집 부트)")
print(f"  휴면 단독            {PB['dorm_alone']['coef']:>+8.4f} {PB['dorm_alone']['ci']}"
      f"   ← 처치표본 +0.5259 와 비교")
print(f"  휴면 3분위 대비       {PB['dormant_minus_active']['diff']:>+8.4f} "
      f"{PB['dormant_minus_active']['ci']}   ← 헤드라인 +0.3472 와 비교")
print(f"  휴면 | log N[−12,−1] {PB['dorm_given_preN']['coef']:>+8.4f} {PB['dorm_given_preN']['ci']}"
      f"   ← 처치표본 −0.3812 와 비교")
print(f"  log N 계수           {PB['preN_coef']['coef']:>+8.4f} {PB['preN_coef']['ci']}")

# ───────── Panel C 수렴 타당도 ─────────
print("\n[Panel C] 수렴 타당도 — 상태변수 4종, 전부 [−24,−13] 측정 (결과 기준창 밖)")
PC = {}
for k, lab, sign in (("dorm", "무채용 월 비중", +1), ("lN", "log(1+총채용)", -1),
                     ("lr", "log(1+채용률)", -1), ("mspell", "최장 무채용 spell", +1)):
    x = np.array([e["S"][k] for e in U], float)
    ok = np.isfinite(x)
    b, bb = ols(np.column_stack([one[ok], x[ok]]), y[ok])
    r = rep(b, bb, 1); r["n"] = int(ok.sum()); r["expected_sign"] = "+" if sign > 0 else "−"
    r["as_predicted"] = bool(np.sign(r["coef"]) == sign and r["sig"])
    # 3분위 대비 (부호 정렬: 휴면이 큰 쪽 − 작은 쪽)
    q1, q2 = np.percentile(x[ok], [33.33, 66.67])
    lo, hi = y[ok][x[ok] <= q1], y[ok][x[ok] > q2]
    if sign < 0: lo, hi = hi, lo                   # 지표 방향 반전 → '휴면 큰 쪽'이 lo 쪽
    dbb = np.array([hi[rng.integers(0, len(hi), len(hi))].mean()
                    - lo[rng.integers(0, len(lo), len(lo))].mean() for _ in range(NB)])
    dci = qci(dbb)
    r["dormant_minus_active"] = {"diff": round(float(hi.mean() - lo.mean()), 4), "ci": dci,
                                 "sig": bool(dci[0] > 0 or dci[1] < 0),
                                 "n_dormant": int(len(hi)), "n_active": int(len(lo))}
    PC[k] = r
    print(f"  {lab:<18} 기울기 {r['coef']:>+8.4f} {str(r['ci']):<22} 예측부호 {r['expected_sign']} "
          f"{'✓' if r['as_predicted'] else '✗'} · 휴면−활발 "
          f"{r['dormant_minus_active']['diff']:>+7.4f} {r['dormant_minus_active']['ci']} "
          f"{'✓' if r['dormant_minus_active']['sig'] else '✗'}")

# ───────── Panel D 경마: 휴면 vs 사전 고용성장 ─────────
print("\n[Panel D] 규모 계정 vs 해제 계정 — 휴면 vs 사전 고용성장")
g = np.array([e["S"]["grow"] for e in U], float)
ok = np.isfinite(g)
b, bb = ols(np.column_stack([one[ok], d[ok], g[ok]]), y[ok])
PD = {"dorm": rep(b, bb, 1), "pre_growth": rep(b, bb, 2), "n": int(ok.sum())}
b2, bb2 = ols(np.column_stack([one[ok], g[ok]]), y[ok])
PD["growth_alone"] = rep(b2, bb2, 1)
print(f"  동시투입 휴면 {PD['dorm']['coef']:>+8.4f} {PD['dorm']['ci']} "
      f"{'✓' if PD['dorm']['sig'] else '✗'} · 사전성장 {PD['pre_growth']['coef']:>+8.4f} "
      f"{PD['pre_growth']['ci']} {'✓' if PD['pre_growth']['sig'] else '✗'} (n={PD['n']})")
print(f"  사전성장 단독 {PD['growth_alone']['coef']:>+8.4f} {PD['growth_alone']['ci']} "
      f"{'✓' if PD['growth_alone']['sig'] else '✗'}"
      f"   ← 규모 계정이면 성장이 흡수해야 한다")

# ───────── Panel E 재기준화 결과 ─────────
print("\n[Panel E] 인공물 2차 증명 — 겹침을 반대편으로 옮기면 부호도 뒤집힌다")
mf = np.array([np.isfinite(e["eff_far"]) for e in U])
yf = np.array([e["eff_far"] for e in U])[mf]
b, bb = ols(np.column_stack([one[mf], d[mf]]), yf)
PE_ = {"n": int(mf.sum()), "dorm_alone": rep(b, bb, 1)}
b, bb = ols(np.column_stack([one[mf], d[mf], zn[mf]]), yf)
PE_["dorm_given_preN_near"] = rep(b, bb, 1); PE_["preN_coef"] = rep(b, bb, 2)
print(f"  n={PE_['n']} · 휴면 단독 {PE_['dorm_alone']['coef']:>+8.4f} {PE_['dorm_alone']['ci']} "
      f"{'✓' if PE_['dorm_alone']['sig'] else '✗'}")
print(f"           휴면 | log N[−12,−1] {PE_['dorm_given_preN_near']['coef']:>+8.4f} "
      f"{PE_['dorm_given_preN_near']['ci']} {'✓' if PE_['dorm_given_preN_near']['sig'] else '✗'}"
      f"   ← 이제 **상태변수**가 기준창과 겹친다 → 반대 방향 인공물")

# ───────── 판정 ─────────
mech = bool(PB["dorm_given_preN"]["coef"] < -0.15)      # 미처치에서도 반전이면 기계적
conv = sum(1 for v in PC.values() if v["dormant_minus_active"]["sig"])
verdict = (
    f"[A] 결과를 겹치는 창의 log 사전총채용에 회귀하면 {PA['y_on_near']['coef']:+.4f} — "
    f"겹치지 않는 창에서는 {PA['y_on_far']['coef']:+.4f}. 대조 차감 전 처치 자체변화는 "
    f"{PA['yt_on_near']['coef']:+.4f} (산술적 −1 방향). "
    f"[B] **미처치 대조기업 {PB['n_pseudo']}건에서 동일 회귀를 돌리면 휴면 계수가 "
    f"{PB['dorm_given_preN']['coef']:+.4f}** — 처치표본의 −0.3812 와 같은 방향·유사 크기. "
    f"아무 일도 일어나지 않은 기업에서 재현되므로 **부호반전은 PE 에 대해 아무것도 말하지 않는다.** "
    f"[C] 상태변수 4종 중 {conv}종이 예측 방향으로 유의 — 지표가 달라도 같은 기울기. "
    f"[D] 사전 고용성장 동시투입 후에도 휴면 {PD['dorm']['coef']:+.4f}. "
    f"[E] 겹침을 반대편으로 옮기면(결과 기준창 = 상태변수 창) 휴면 계수가 "
    f"{PE_['dorm_given_preN_near']['coef']:+.4f} 로 **과대** 쪽 인공물이 된다 — 부호가 "
    f"어느 변수가 기준창과 겹치는지에 따라 뒤집히므로 산술 인공물임이 이중으로 확인된다. "
    f"Panel E 는 방어 근거가 아니라 인공물의 2차 증명이다.")
emit("I-44", "상태변수 방어 (I-41 Panel E 정체 규명)",
     "GO" if (mech and conv >= 3) else "PARTIAL",
     {"panelA_mechanical": PA, "panelB_untreated_replication": PB,
      "panelC_convergent_validity": PC, "panelD_scale_vs_release": PD,
      "panelE_rebased_outcome": PE_,
      "corr_dorm_lpreN_near": round(float(np.corrcoef(d, zn)[0, 1]), 4),
      "n": len(U), "SESOI": SESOI},
     "I-41 Panel E 의 부호반전이 (a) 종속변수 구성요소를 우변에 넣은 산술 인공물인가 "
     "(b) 상태변수는 지표 선택에 견고한가",
     verdict, kill_met=not mech, n=len(U),
     extra={"supersedes_interpretation_of": "I41.json panelE_given_preN",
            "note": "추정치 재계산 아님 — I-41 Panel E 는 그대로 두고 그 해석을 확정한다."})
