# -*- coding: utf-8 -*-
"""I-49 (C-5b + C-4) 대조 재사용 감사 · 사전 재무제약.

C-5b (리뷰3 §9.4). 이벤트 수준 부트스트랩은 처치기업과 대조 5개를 함께 움직이지만, **같은 대조기업이
여러 이벤트에 재사용되면 이벤트는 독립이 아니다.** 재사용이 심하면 SE 가 과소추정되고 헤드라인 CI 가
과도하게 좁아진다. 우리 헤드라인의 유의성 자체가 걸린 문제다.

C-4 (리뷰3 §8-4). §2.2 에서 사후처치 현금 문장을 뺐으므로 "사전 재무제약에 기울기가 없다"는 주장이
현재 **근거 없이 서 있다.** 사전 지표(현금비율·레버리지·이자보상·수익성)로 직접 검정한다.

Panel A  재사용 분포 — 고유 대조기업 수 · 재사용 횟수 · 최대 · 상위 1%
Panel B  대조기업 군집 부트스트랩 (이벤트 군집과 비교)
Panel C  ★ 대조 1회 사용 제한(그리디 배정) 후 재추정
Panel D  이벤트 × 대조기업 이원 군집 (보수적 결합)
Panel E  사전 재무제약 — 현금/자산 · 부채/자산 · 이자보상 · ROA 3분위별 반응, 상태변수와의 경마
"""
import re
import numpy as np, pandas as pd
from h30_common import load, deals, build, emit, SEED, qci, NB, widx, BASE

rng = np.random.default_rng(SEED)
print("[I-49] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Ev, idx = G["Hv"], G["Ev"], G["idx"]


def lrate(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return np.nan
    h, e = Hv[row, c].astype(float), Ev[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.mean(e) < 5: return np.nan
    N, E = h.sum(), np.mean(e)
    return float(np.log(N / E)) if N > 0 else np.nan


# 이벤트 수준 스칼라 (상태 = −log(1+사전 채용률[−24,−13]))
for e in EV:
    a, b = lrate(e["ti"], e["m0"], -12, -1), lrate(e["ti"], e["m0"], 1, 12)
    e["t_ch"] = b - a if (np.isfinite(a) and np.isfinite(b)) else np.nan
    e["c_ch"] = {int(k): (lrate(k, e["m0"], 1, 12) - lrate(k, e["m0"], -12, -1)) for k in e["ctrls"]}
    c24 = widx(G, e["m0"], -24, -13); c36 = widx(G, e["m0"], -36, -25)
    e["x"] = e["lsize"] = e["grow"] = np.nan
    if len(c24) == 12:
        h24, e24 = Hv[e["ti"], c24].astype(float), Ev[e["ti"], c24].astype(float)
        if np.isfinite(h24).all() and np.isfinite(e24).all() and np.mean(e24) >= 5:
            e["x"] = -float(np.log1p(h24.sum() / np.mean(e24)))
            e["lsize"] = float(np.log(np.mean(e24)))
            if len(c36) == 12:
                e36 = Ev[e["ti"], c36].astype(float)
                if np.isfinite(e36).all() and np.mean(e36) > 0:
                    e["grow"] = float(np.log(np.mean(e24) / np.mean(e36)))
    e["age"] = ((e["m0"] - G["adpt_arr"][e["ti"]]) / 12.0
                if np.isfinite(G["adpt_arr"][e["ti"]]) else np.nan)
    e["ind1"] = str(G["ind_arr"][e["ti"]])[:1]

U = [e for e in EV if np.isfinite(e["t_ch"]) and np.isfinite(e["x"])
     and any(np.isfinite(v) for v in e["c_ch"].values())]
print(f"  분석표본 {len(U)}/{len(EV)}")


def eff_of(e, drop=None):
    cs = [v for k, v in e["c_ch"].items() if np.isfinite(v) and (drop is None or k not in drop)]
    return e["t_ch"] - float(np.mean(cs)) if cs else np.nan


def fit(sub, effs):
    m = np.isfinite(effs)
    y = np.asarray(effs)[m]; sub = [s for s, k in zip(sub, m) if k]
    x = np.array([s["x"] for s in sub]); sz = np.array([s["lsize"] for s in sub])
    num = [sz]
    for k in ("grow", "age"):                       # I-47 과 동일 공변량 집합
        v = np.array([s[k] for s in sub], float)
        num.append(np.where(np.isfinite(v), v, np.nanmedian(v[np.isfinite(v)])))
    inds = sorted({s["ind1"] for s in sub})[1:]
    C = np.column_stack([np.ones(len(y))] + num
                        + [np.array([1.0 if s["ind1"] == q else 0.0 for s in sub]) for q in inds])
    r_ = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    return r_(y), r_(x), sub


def slope_boot(yr, xr, groups=None, R=NB):
    b = float(np.polyfit(xr, yr, 1)[0])
    bb = []
    if groups is None:
        for _ in range(R):
            i = rng.integers(0, len(yr), len(yr))
            bb.append(np.polyfit(xr[i], yr[i], 1)[0])
    else:
        GUq = np.unique(groups); byg = {g: np.where(groups == g)[0] for g in GUq}
        for _ in range(R):
            p = np.concatenate([byg[GUq[i]] for i in rng.integers(0, len(GUq), len(GUq))])
            if len(p) < 20: continue
            try: bb.append(np.polyfit(xr[p], yr[p], 1)[0])
            except Exception: pass
    ci = qci(np.array(bb))
    return {"slope": round(b, 4), "ci": ci, "n": len(yr), "sig": bool(ci[0] > 0 or ci[1] < 0),
            "half_width": round(float((ci[1] - ci[0]) / 2), 4)}


# ════════ Panel A 재사용 분포 ════════
print("\n[Panel A] 대조 재사용 분포")
use = {}
for e in U:
    for k in e["ctrls"]: use[int(k)] = use.get(int(k), 0) + 1
cnt = np.array(sorted(use.values()))
PA = {"n_events": len(U), "n_control_slots": int(cnt.sum()), "n_unique_controls": int(len(cnt)),
      "mean_reuse": round(float(cnt.mean()), 3), "median_reuse": int(np.median(cnt)),
      "max_reuse": int(cnt.max()), "p99_reuse": int(np.percentile(cnt, 99)),
      "share_used_once": round(float((cnt == 1).mean()), 4),
      "share_slots_from_top1pct": round(float(cnt[cnt >= np.percentile(cnt, 99)].sum() / cnt.sum()), 4)}
for k, v in PA.items(): print(f"  {k:<28} {v}")

# ════════ Panel B 군집 비교 ════════
print("\n[Panel B] 군집 방식별 헤드라인 기울기")
effs = [eff_of(e) for e in U]
yr, xr, sub = fit(U, effs)
PB = {"cluster_event": slope_boot(yr, xr)}
# 대조기업 군집: 각 이벤트를 '가장 많이 재사용된 대조'로 대표시켜 군집
rep = np.array([max(((int(k), use[int(k)]) for k in s["ctrls"]), key=lambda t: t[1])[0] for s in sub])
PB["cluster_control_firm"] = slope_boot(yr, xr, groups=rep)
for k, v in PB.items():
    print(f"  {k:<24} {v['slope']:>+7.4f} {str(v['ci']):<22}{'✓' if v['sig'] else '✗'} "
          f"반폭 {v['half_width']:.4f}")

# ════════ Panel C ★ 대조 1회 사용 제한 ════════
print("\n[Panel C] ★ 대조기업을 이벤트당 배타 배정(그리디) 후 재추정")
taken = set(); keep, effs2 = [], []
for e in sorted(U, key=lambda z: len(z["c_ch"])):
    avail = [k for k in e["c_ch"] if k not in taken and np.isfinite(e["c_ch"][k])]
    if not avail: continue
    for k in avail: taken.add(k)
    keep.append(e); effs2.append(e["t_ch"] - float(np.mean([e["c_ch"][k] for k in avail])))
yr2, xr2, sub2 = fit(keep, effs2)
PC = {"n_events": len(keep), "n_controls_used": len(taken),
      "slope": slope_boot(yr2, xr2)}
print(f"  이벤트 {len(keep)} · 배타 대조 {len(taken)}개 · "
      f"기울기 {PC['slope']['slope']:>+7.4f} {PC['slope']['ci']} "
      f"{'✓' if PC['slope']['sig'] else '✗'}")

# ════════ Panel D 이원 군집 (보수적 결합) ════════
hw = max(PB["cluster_event"]["half_width"], PB["cluster_control_firm"]["half_width"])
b0 = PB["cluster_event"]["slope"]
PD = {"conservative_ci": [round(b0 - hw, 4), round(b0 + hw, 4)],
      "sig": bool((b0 - hw) > 0 or (b0 + hw) < 0),
      "rule": "이벤트 군집과 대조기업 군집 중 넓은 쪽 반폭을 채택(이원 군집 상한 근사)"}
print(f"\n[Panel D] 보수적 결합 CI {PD['conservative_ci']} {'✓' if PD['sig'] else '✗'}")

# ════════ Panel E 사전 재무제약 ════════
print("\n[Panel E] 사전 재무제약 지표별 반응")
PE_ = {}
try:
    cols = ["회계연도", "사업자등록번호", "자산총계(천원)", "현금및현금성자산(천원)",
            "부채총계(천원)", "영업이익(천원)", "이자비용(천원)"]
    fin = None
    for ch in pd.read_csv(f"{BASE}/PI/drops/재무데이터_2009_2025_통합.csv",
                          dtype=str, chunksize=300_000):
        have = [c for c in cols if c in ch.columns]
        if fin is None: print("  사용 컬럼:", have)
        t = ch[have].copy()
        t["bn10"] = t["사업자등록번호"].str.replace(r"\D", "", regex=True).str.zfill(10)
        fin = t if fin is None else pd.concat([fin, t], ignore_index=True)
    for c in have:
        if c not in ("회계연도", "사업자등록번호"): fin[c] = pd.to_numeric(fin[c], errors="coerce")
    fin["yr"] = pd.to_numeric(fin["회계연도"], errors="coerce")
    fin = fin.dropna(subset=["yr"]).set_index(["bn10", "yr"])
    A = "자산총계(천원)"
    ratios = {}
    if "현금및현금성자산(천원)" in have: ratios["cash_to_assets"] = ("현금및현금성자산(천원)", A, +1)
    if "부채총계(천원)" in have: ratios["leverage"] = ("부채총계(천원)", A, -1)
    if "영업이익(천원)" in have: ratios["roa"] = ("영업이익(천원)", A, +1)
    if "이자비용(천원)" in have and "영업이익(천원)" in have:
        ratios["interest_coverage"] = ("영업이익(천원)", "이자비용(천원)", +1)
    for nm, (num, den, sgn) in ratios.items():
        vals, subs = [], []
        for e in U:
            y0 = (e["m0"] - 1) // 12 - 1                       # 딜 전년도
            try: r0 = fin.loc[(e["bn"], y0)]
            except KeyError: continue
            if isinstance(r0, pd.DataFrame): r0 = r0.iloc[0]
            a_, b_ = r0.get(num), r0.get(den)
            if not (np.isfinite(a_) and np.isfinite(b_) and b_ != 0): continue
            vals.append(float(a_) / float(b_)); subs.append(e)
        if len(subs) < 60: PE_[nm] = {"n": len(subs), "note": "표본 부족"}; continue
        ee = [eff_of(s) for s in subs]
        yrr, xrr, s2 = fit(subs, ee)
        v = np.array([vals[i] for i, s in enumerate(subs) if np.isfinite(ee[i])], float)
        v = np.clip(v, *np.percentile(v, [1, 99]))
        # 제약이 심한 쪽(현금↓·ROA↓·이자보상↓ 또는 레버리지↑)이 큰 반응을 보이는가
        xf = -v * sgn
        Cn = np.column_stack([np.ones(len(v))])
        sl = slope_boot(yrr, xf - xf.mean())
        q1, q2 = np.percentile(xf, [33.33, 66.67])
        hi, lo = yrr[xf > q2], yrr[xf <= q1]
        dd = float(hi.mean() - lo.mean())
        tb = np.array([hi[rng.integers(0, len(hi), len(hi))].mean()
                       - lo[rng.integers(0, len(lo), len(lo))].mean() for _ in range(NB)])
        tci = qci(tb)
        # 상태변수와의 경마
        Xh = np.column_stack([np.ones(len(v)), xrr, xf - xf.mean()])
        bh = np.linalg.lstsq(Xh, yrr, rcond=None)[0]
        bbh = []
        for _ in range(NB):
            i = rng.integers(0, len(v), len(v))
            try: bbh.append(np.linalg.lstsq(Xh[i], yrr[i], rcond=None)[0])
            except Exception: pass
        bbh = np.array(bbh)
        PE_[nm] = {"n": len(v), "slope_constraint": sl,
                   "tercile": {"diff": round(dd, 4), "ci": tci,
                               "sig": bool(tci[0] > 0 or tci[1] < 0)},
                   "horserace_state": {"coef": round(float(bh[1]), 4), "ci": qci(bbh[:, 1])},
                   "horserace_constraint": {"coef": round(float(bh[2]), 4), "ci": qci(bbh[:, 2])}}
        print(f"  {nm:<18} n={len(v):>3} 제약 기울기 {sl['slope']:>+7.4f} {str(sl['ci']):<20}"
              f"{'✓' if sl['sig'] else '✗'} · 3분위 {dd:>+7.4f} {str(tci):<20}"
              f"{'✓' if PE_[nm]['tercile']['sig'] else '✗'}")
        print(f"{'':<20}경마: 상태 {PE_[nm]['horserace_state']['coef']:>+7.4f} "
              f"{PE_[nm]['horserace_state']['ci']} · 제약 "
              f"{PE_[nm]['horserace_constraint']['coef']:>+7.4f} "
              f"{PE_[nm]['horserace_constraint']['ci']}")
except Exception as ex:
    PE_["error"] = str(ex); print("  실패:", ex)

n_fin_sig = sum(1 for k, v in PE_.items() if isinstance(v, dict) and v.get("tercile", {}).get("sig"))
verdict = (
    f"[재사용] 이벤트 {PA['n_events']} · 대조 슬롯 {PA['n_control_slots']} · 고유 대조 "
    f"{PA['n_unique_controls']} (평균 재사용 {PA['mean_reuse']}, 최대 {PA['max_reuse']}, "
    f"1회만 사용 {PA['share_used_once']:.1%}). 이벤트 군집 CI {PB['cluster_event']['ci']} vs "
    f"대조기업 군집 {PB['cluster_control_firm']['ci']}; 보수적 결합 {PD['conservative_ci']} "
    f"{'유의 유지' if PD['sig'] else '유의 상실'}. 배타 배정(이벤트 {PC['n_events']}) "
    f"{PC['slope']['slope']:+.4f}{PC['slope']['ci']}{'✓' if PC['slope']['sig'] else '✗'}. "
    f"[사전 재무제약] 3분위 대비가 유의한 지표 {n_fin_sig}개 — "
    + ("사전 재무제약 gradient 부재 주장 지지." if n_fin_sig == 0 else "일부 지표에서 gradient 존재, 주장 수정 필요."))
emit("I-49", "대조 재사용 감사 + 사전 재무제약 (리뷰3 §9.4·§8-4)",
     "GO" if (PD["sig"] and PC["slope"]["sig"]) else "PARTIAL",
     {"panelA_reuse": PA, "panelB_clustering": PB, "panelC_exclusive_controls": PC,
      "panelD_conservative": PD, "panelE_pre_deal_financials": PE_,
      "state_variable": "-log(1+pre-deal hiring rate[-24,-13])", "n": len(U)},
     "대조 재사용이 헤드라인 유의성을 만들고 있는가 · 사전 재무제약이 반응을 예측하는가",
     verdict, kill_met=not PD["sig"], n=len(U))
