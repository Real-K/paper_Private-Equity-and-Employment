# -*- coding: utf-8 -*-
"""I-61 교정 설계(286건)에서 (a) 주 gradient 의 클러스터 부트스트랩 CI,
(b) 대조군 공유 구조 진단.

배경. I-60 은 위약 null 만 산출해 주 추정치 0.7101 의 **표본변동 구간이 없었다**
(원장 E01 이 null CI 를 추정치 CI 로 잘못 기재). §10 의 공유대조군 진단은
교정 이전 설계(301건, I-49)에서 나온 값이라 본문 주사양(286건)과 표본이 다르다.
둘 다 같은 조립 코드에서 다시 낸다.

Panel A  처치기업 클러스터 부트스트랩 CI (winsor 컷은 매 draw 내부 재산출)
Panel B  고정컷 CI (컷을 추정량 정의의 일부로 고정)
Panel C  대조군 재사용 분포 (distinct 대조기업 수, 재사용 횟수)
Panel D  대조기업 클러스터 부트스트랩 CI
Panel E  각 사건당 대조군 1개만 배정했을 때의 점추정
"""
import numpy as np, json
from collections import Counter
from h30_common import load, deals, build, emit, SEED, qci, widx
from h39_common import SIZE_B
rng = np.random.default_rng(SEED); NB = 2000
print("[I-65] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Sv, Ev, adpt, idx = G["Hv"], G["Sv"], G["Ev"], G["adpt_arr"], G["idx"]
mset, ind_arr = G["mset"], G["ind_arr"]; NOTPE = np.asarray(~idx.isin(set(PE)))
_c, _s = {}, {}
def cellarr(m0):
    if m0 in _c: return _c[m0]
    iw = [mset[m] for m in range(m0-6, m0) if m in mset]
    i18 = [mset[m] for m in range(m0-18, m0-12) if m in mset]
    if not iw or not i18: _c[m0] = None; return None
    with np.errstate(all="ignore"):
        Ep = np.nanmean(Ev[:, iw], axis=1); g = Ep/np.nanmean(Ev[:, i18], axis=1) - 1
    _c[m0] = (Ep, g, np.digitize(Ep, SIZE_B, right=False),
              np.where(np.isnan(g), -1, np.digitize(g, [-0.10, 0.10])),
              np.where(np.isnan(adpt), -1, np.digitize((m0-adpt)/12.0, [5, 15])))
    return _c[m0]
def Sall(m0):
    if m0 in _s: return _s[m0]
    c = widx(G, m0, -24, -13)
    if len(c) != 12: _s[m0] = (None, None); return _s[m0]
    h = Hv[:, c].astype(float); e = Ev[:, c].astype(float)
    ok = np.isfinite(h).all(1) & np.isfinite(e).all(1) & (np.nanmean(e, 1) >= 5)
    S = np.full(Hv.shape[0], np.nan); S[ok] = -np.log1p(h[ok].sum(1)/np.nanmean(e[ok], 1))
    fin = np.isfinite(S); b = np.full(Hv.shape[0], -9)
    if fin.sum() >= 50:
        q1, q2 = np.percentile(S[fin], [33.33, 66.67]); b = np.where(fin, np.digitize(S, [q1, q2]), -9)
    _s[m0] = (S, b); return _s[m0]
def match(focal, m0, k=5):
    c = cellarr(m0)
    if c is None: return None
    Ep, g, sb, gb, ageb = c
    if not (np.isfinite(Ep[focal]) and Ep[focal] >= 5): return None
    S, bins = Sall(m0)
    if S is None or not np.isfinite(S[focal]) or bins[focal] == -9: return None
    same = (NOTPE & (ind_arr == ind_arr[focal]) & (sb == sb[focal]) & (gb == gb[focal])
            & (ageb == ageb[focal]) & (Ep >= 5) & np.isfinite(Ep) & (bins == bins[focal]))
    cand = np.flatnonzero(same); cand = cand[cand != focal]
    if len(cand) == 0: return None
    gt = g[focal] if np.isfinite(g[focal]) else 0.0
    gc = np.where(np.isfinite(g[cand]), g[cand], 0.0)
    d = ((np.log(Ep[cand])-np.log(Ep[focal]))/0.9)**2 + ((np.clip(gc,-1,2)-np.clip(gt,-1,2))/0.35)**2
    return cand[np.argsort(d)[:k]]
def blk(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b-a+1): return None
    h = Hv[row, c].astype(float); e = Ev[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.mean(e) < 5: return None
    return float(h.sum()), float(np.mean(e))
def dlog(row, m0):
    po, pr = blk(row, m0, 1, 12), blk(row, m0, -12, -1)
    if po is None or pr is None or po[0] <= 0 or pr[0] <= 0: return None
    return np.log(po[0]/po[1]) - np.log(pr[0]/pr[1])

# ---- 조립: I-60 의 처치 units 에 대조군 id 와 개별 대조 결과를 보존한다 ----
print("[조립] 처치 units + 대조군 id 보존")
T = []
for gi, e in enumerate(EV):
    ct = match(e["ti"], e["m0"])
    if ct is None: continue
    st = blk(e["ti"], e["m0"], -24, -13); yt = dlog(e["ti"], e["m0"])
    if st is None or yt is None: continue
    cs = [(int(o), dlog(int(o), e["m0"])) for o in ct]
    cs = [(o, v) for o, v in cs if v is not None]
    if not cs: continue
    w36 = blk(e["ti"], e["m0"], -36, -25)
    T.append(dict(g=gi, ti=int(e["ti"]), m0=int(e["m0"]), yt=yt, ctrl=cs,
                  eff=yt - float(np.mean([v for _, v in cs])),
                  S=-float(np.log1p(st[0]/st[1])), lsize=np.log(st[1]),
                  grow=(np.log(st[1]/w36[1]) if (w36 and w36[1] > 0) else np.nan),
                  age=((e["m0"]-adpt[e["ti"]])/12.0 if np.isfinite(adpt[e["ti"]]) else np.nan),
                  ind=str(ind_arr[e["ti"]])[:1]))
print(f"  처치 units n={len(T)}")

def slope(rows, effs=None, cuts=None, wins=None):
    if len(rows) < 30: return None
    y = np.array(effs if effs is not None else [r["eff"] for r in rows], float)
    if wins is not None: cuts = tuple(np.percentile(y, wins))
    if cuts is not None: y = np.clip(y, cuts[0], cuts[1])
    x = np.array([r["S"] for r in rows])
    cols = [np.ones(len(y)), np.array([r["lsize"] for r in rows])]
    for k in ("grow", "age"):
        v = np.array([r[k] for r in rows], float); m = np.isfinite(v)
        cols.append(np.where(m, v, np.median(v[m]) if m.any() else 0.0))
    for s_ in sorted({r["ind"] for r in rows})[1:]:
        cols.append(np.array([1.0 if r["ind"] == s_ else 0.0 for r in rows]))
    C = np.column_stack(cols); r_ = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    yr, xr = r_(y), r_(x); d = float(np.sum(xr*xr))
    return float(np.sum(xr*yr)/d) if d > 0 else None

obs = slope(T, wins=(5, 95)); fixed_cuts = tuple(np.percentile([r["eff"] for r in T], (5, 95)))
print(f"  관측 gradient {obs:+.4f} (I-60 주사양 0.7101 과 일치해야 함)")

# ---- Panel A/B: 처치기업 클러스터 부트스트랩 ----
firms = sorted({r["ti"] for r in T}); byfirm = {f: [r for r in T if r["ti"] == f] for f in firms}
print(f"  처치기업 클러스터 {len(firms)}개 (사건 {len(T)}건)")
bA, bB = [], []
for _ in range(NB):
    d = []
    for f in rng.choice(firms, size=len(firms), replace=True): d += byfirm[int(f)]
    vA = slope(d, wins=(5, 95)); vB = slope(d, cuts=fixed_cuts)
    if vA is not None: bA.append(vA)
    if vB is not None: bB.append(vB)
bA, bB = np.array(bA), np.array(bB)
PA = {"observed": round(obs, 4), "n_events": len(T), "n_firms": len(firms),
      "ci95": qci(bA), "boot_sd": round(float(bA.std()), 4), "n_boot": len(bA),
      "excludes_zero": bool(qci(bA)[0] > 0)}
PB = {"observed": round(obs, 4), "ci95": qci(bB), "boot_sd": round(float(bB.std()), 4),
      "n_boot": len(bB), "note": "winsor cuts fixed at full-sample 5/95"}
print(f"  [A] 처치기업 클러스터 CI {PA['ci95']}  SD {PA['boot_sd']}")
print(f"  [B] 고정컷 CI            {PB['ci95']}  SD {PB['boot_sd']}")

# ---- Panel C: 대조군 재사용 ----
use = Counter()
for r in T:
    for o, _ in r["ctrl"]: use[o] += 1
cnt = Counter(use.values())
PC = {"n_events": len(T), "n_unique_controls": len(use),
      "controls_per_event_mean": round(float(np.mean([len(r["ctrl"]) for r in T])), 2),
      "share_used_once": round(sum(v for k, v in cnt.items() if k == 1)/len(use), 4),
      "max_reuse": max(use.values()),
      "reuse_hist": {str(k): cnt[k] for k in sorted(cnt)}}
print(f"  [C] distinct 대조 {PC['n_unique_controls']} · 1회사용 {PC['share_used_once']:.1%} · 최대재사용 {PC['max_reuse']}")

# ---- Panel D: 대조기업 클러스터 부트스트랩 ----
ctrl_ids = sorted(use)
bD = []
for _ in range(NB):
    draw = Counter(rng.choice(ctrl_ids, size=len(ctrl_ids), replace=True).tolist())
    rows, effs = [], []
    for r in T:
        num = sum(draw[o]*v for o, v in r["ctrl"] if draw[o]);  den = sum(draw[o] for o, _ in r["ctrl"] if draw[o])
        if den == 0: continue
        rows.append(r); effs.append(r["yt"] - num/den)
    v = slope(rows, effs=effs, wins=(5, 95))
    if v is not None: bD.append(v)
bD = np.array(bD)
PD = {"ci95": qci(bD), "boot_sd": round(float(bD.std()), 4), "n_boot": len(bD),
      "excludes_zero": bool(qci(bD)[0] > 0)}
print(f"  [D] 대조기업 클러스터 CI {PD['ci95']}  SD {PD['boot_sd']}")

# ---- Panel E: 사건당 대조군을 1개로 제한(중복 배제) ----
taken, effs1, rows1 = set(), [], []
for r in sorted(T, key=lambda z: z["m0"]):
    pick = next(((o, v) for o, v in r["ctrl"] if o not in taken), None)
    if pick is None: continue
    taken.add(pick[0]); rows1.append(r); effs1.append(r["yt"] - pick[1])
e1 = slope(rows1, effs=effs1, wins=(5, 95))
PE_ = {"observed": round(e1, 4) if e1 is not None else None, "n_events": len(rows1),
       "note": "each control firm assigned to at most one event, first-come by event month"}
print(f"  [E] 대조군 1:1 배정 {PE_['observed']} (n={PE_['n_events']})")

# ---- Panel F: 해석 가능한 크기 (S 의 IQR 이동) ----
Sarr = np.array([r["S"] for r in T]); q25, q75 = np.percentile(Sarr, [25, 75]); iqr = q75 - q25
eff_iqr = obs * iqr; ci_iqr = [round(float(np.percentile(bA, 2.5)*iqr), 4), round(float(np.percentile(bA, 97.5)*iqr), 4)]
PF = {"S_q25": round(float(q25), 4), "S_q75": round(float(q75), 4), "S_iqr": round(float(iqr), 4),
      "S_sd": round(float(Sarr.std(ddof=1)), 4), "effect_iqr": round(float(eff_iqr), 4),
      "ci95_iqr": ci_iqr, "effect_per_sd": round(float(obs*Sarr.std(ddof=1)), 4),
      "note": "gradient x IQR of the pre-deal state among the 286 treated events"}
print(f"  [F] S IQR {PF['S_iqr']:.4f} → IQR 효과 {PF['effect_iqr']:+.4f} {PF['ci95_iqr']}")

# ---- Panel G: 표본기간 ----
from collections import Counter as _C
_k2y = lambda k: (k//12 if k % 12 else k//12-1)
_f = lambda k: "%d-%02d" % ((k//12 if k % 12 else k//12-1), (k % 12 or 12))
_m0 = [r["m0"] for r in T]; _mA = [int(e["m0"]) for e in EV]
PG = {"panel_first": _f(int(G["mis"].min())), "panel_last": _f(int(G["mis"].max())),
      "gradient_first_deal_month": _f(min(_m0)), "gradient_last_deal_month": _f(max(_m0)),
      "gradient_by_year": dict(sorted(_C(_k2y(k) for k in _m0).items())),
      "baseline_first_deal_month": _f(min(_mA)), "baseline_last_deal_month": _f(max(_mA)),
      "n_baseline_events": len(_mA)}
print(f"  [G] 패널 {PG['panel_first']}~{PG['panel_last']} · 처치월 {PG['gradient_first_deal_month']}~{PG['gradient_last_deal_month']} (n={len(_m0)})")

wide = [min(PA["ci95"][0], PD["ci95"][0]), max(PA["ci95"][1], PD["ci95"][1])]
emit("I-65", "교정 설계 gradient 의 부트스트랩 CI 와 대조군 공유 진단",
     "GO" if PA["excludes_zero"] and PD["excludes_zero"] else "PARTIAL",
     {"panelA_firm_cluster": PA, "panelB_fixed_cuts": PB, "panelC_reuse": PC,
      "panelD_control_cluster": PD, "panelE_one_to_one": PE_, "panelF_magnitude": PF, "panelG_period": PG,
      "conservative_union_ci": [round(wide[0], 4), round(wide[1], 4)], "n_boot": NB,
      "design": "I-60 과 동일 (상태균형 매칭 + FWL), 위약 대신 클러스터 부트스트랩"},
     "주 gradient 의 표본변동 구간이 0 을 배제하는가, 대조군 공유가 구간을 좌우하는가",
     f"관측 {obs:+.4f}; 처치기업 클러스터 CI {PA['ci95']}, 대조기업 클러스터 CI {PD['ci95']}; "
     f"사건 {len(T)}건이 distinct 대조 {PC['n_unique_controls']}개를 쓰고 {PC['share_used_once']:.1%}는 1회만 쓰인다.",
     kill_met=False, n=len(T))
