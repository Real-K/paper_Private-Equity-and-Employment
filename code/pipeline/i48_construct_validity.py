# -*- coding: utf-8 -*-
"""I-48 (C-3) 결과대상 construct validity + 표본 flow.

리뷰 3 MC3: NPS `신규` 가 경제적 의미의 채용인가, 아니면 PE 거래 전후의 **행정적 재등록**인가.
거래 시점은 법인구조·사업장 재편이 실제로 일어나는 시점이므로 이 공격은 치명적일 수 있다.

Panel A  **752 → 379 표본 flow** — 각 단계 탈락 수 (build() 로직 계측)
Panel B  **stock-flow 정합** E_t − E_{t−1} ≈ A_t − S_t. 오차 분포 · PE 직후 변화
Panel C  **대오차·고churn 월 배제** 후 헤드라인 재추정
Panel D  **사업장 구조 변화 배제** (n_sites 변동) 후 재추정
Panel E  포함/제외 기업 비교

헤드라인은 I-47 이 확정한 **사전 채용률** 상태변수 기준으로 재추정한다.
"""
import gc
import numpy as np, pandas as pd
from h30_common import load, deals, build, emit, SEED, qci, NB, widx, BASE

rng = np.random.default_rng(SEED)
SIZE_B = None
print("[I-48] 로딩...")
G = load(); orig, allt, PE, META = deals(G)
Hv, Sv, Ev, mset, idx = G["Hv"], G["Sv"], G["Ev"], G["mset"], G["idx"]
EV, _ = build(G, allt, PE)
print(f"  이벤트 {len(EV)}")

# ════════ Panel A 표본 flow ════════
print("\n[Panel A] 752 → 379 표본 flow")
from h39_common import SIZE_B as SB
excl = set(PE)
ctrl_ok = ~idx.isin(excl)
n_rows = len(allt)
step = {"canonical_treated_file": int(META.shape[0]) if hasattr(META, "shape") else None,
        "deal_rows_entering_build": int(n_rows)}
c_link = c_win = c_emp = c_cell = c_ok = 0
cache = {}
for r in allt.itertuples():
    m0 = int(r.mi)
    if r.bn10 not in idx: c_link += 1; continue
    ti = idx.get_loc(r.bn10)
    if m0 not in cache:
        iw = [mset[m] for m in range(m0 - 6, m0) if m in mset]
        i18 = [mset[m] for m in range(m0 - 18, m0 - 12) if m in mset]
        if not iw or not i18: cache[m0] = None
        else:
            with np.errstate(all="ignore"):
                Ep = np.nanmean(Ev[:, iw], axis=1); g = Ep / np.nanmean(Ev[:, i18], axis=1) - 1
            cache[m0] = (Ep, g, np.digitize(Ep, SB, right=False),
                         np.where(np.isnan(g), -1, np.digitize(g, [-0.10, 0.10])),
                         np.where(np.isnan(G["adpt_arr"]), -1,
                                  np.digitize((m0 - G["adpt_arr"]) / 12.0, [5, 15])))
    c = cache[m0]
    if c is None: c_win += 1; continue
    Ep, g, sb, gb, ageb = c
    if not (np.isfinite(Ep[ti]) and Ep[ti] >= 5): c_emp += 1; continue
    same = (ctrl_ok & (G["ind_arr"] == G["ind_arr"][ti]) & (sb == sb[ti]) & (gb == gb[ti])
            & (ageb == ageb[ti]) & (Ep >= 5) & np.isfinite(Ep))
    cand = np.flatnonzero(same); cand = cand[cand != ti]
    if len(cand) == 0: c_cell += 1; continue
    c_ok += 1
step |= {"drop_no_nps_link": c_link, "drop_insufficient_pre_window": c_win,
         "drop_employment_lt5": c_emp, "drop_no_control_cell": c_cell,
         "matched_events": c_ok}
tot = c_link + c_win + c_emp + c_cell + c_ok
for k, v in step.items(): print(f"  {k:<34} {v}")
print(f"  합계 확인: {tot} (= 진입 {n_rows}) · 최종 {c_ok}")
step["reconciles"] = bool(tot == n_rows and c_ok == len(EV))

# ════════ Panel B stock-flow 정합 ════════
print("\n[Panel B] stock-flow 정합  E_t − E_{t−1} ≈ A_t − S_t")
rows_ti = sorted({e["ti"] for e in EV} | {int(k) for e in EV for k in e["ctrls"]})
sub_E = Ev[rows_ti, :]; sub_A = Hv[rows_ti, :]; sub_S = Sv[rows_ti, :]
dE = np.diff(sub_E, axis=1)
net = (sub_A - sub_S)[:, 1:]
ok = np.isfinite(dE) & np.isfinite(net) & np.isfinite(sub_E[:, :-1]) & (sub_E[:, :-1] >= 5)
err = np.where(ok, dE - net, np.nan)
rel = np.where(ok & (sub_E[:, :-1] > 0), np.abs(err) / sub_E[:, :-1], np.nan)
PB = {"n_firm_months": int(ok.sum()),
      "mean_abs_error": round(float(np.nanmean(np.abs(err))), 4),
      "median_abs_error": round(float(np.nanmedian(np.abs(err))), 4),
      "exact_zero_share": round(float(np.nanmean(np.abs(err) < 1e-9)), 4),
      "abs_error_p90": round(float(np.nanpercentile(np.abs(err), 90)), 4),
      "abs_error_p99": round(float(np.nanpercentile(np.abs(err), 99)), 4),
      "rel_error_median": round(float(np.nanmedian(rel)), 5),
      "rel_error_p90": round(float(np.nanpercentile(rel, 90)), 5),
      "rel_error_p99": round(float(np.nanpercentile(rel, 99)), 5),
      "share_rel_gt_5pct": round(float(np.nanmean(rel > 0.05)), 4)}
for k, v in PB.items(): print(f"  {k:<24} {v}")

# PE 직후 오차 변화 — 처치기업만, 이벤트시간별
rmap = {t: i for i, t in enumerate(rows_ti)}
prof = {}
for lab, a, b in (("pre[-12,-1]", -12, -1), ("deal[0,0]", 0, 0), ("post[1,12]", 1, 12)):
    vals = []
    for e in EV:
        c = widx(G, e["m0"], a, b)
        c = [x for x in c if 1 <= x < sub_E.shape[1]]
        if not c: continue
        v = rel[rmap[e["ti"]], [x - 1 for x in c]]
        v = v[np.isfinite(v)]
        if len(v): vals.append(float(np.nanmean(v)))
    prof[lab] = {"n": len(vals), "mean_rel_error": round(float(np.mean(vals)), 5)}
    print(f"  처치기업 상대오차 {lab:<12} {prof[lab]['mean_rel_error']:.5f} (n={prof[lab]['n']})")
PB["treated_profile"] = prof

# ════════ 헤드라인 재추정 함수 (I-47 확정 상태변수 = 사전 채용률) ════════
def lrate(row, m0, a, b, mask=None):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return np.nan
    h, e = Hv[row, c].astype(float), Ev[row, c].astype(float)
    if mask is not None:
        keep = np.array([mask.get((row, m), True) for m in c])
        if keep.sum() < 8: return np.nan
        h, e = h[keep], e[keep]
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.mean(e) < 5: return np.nan
    N, E = h.sum(), np.mean(e)
    return float(np.log(N / E)) if N > 0 else np.nan


def headline(mask=None, keep_ev=None, tag=""):
    """상태변수 = −log(1+사전 채용률[−24,−13]).  FWL 조정 기울기 + 3분위 대비."""
    ys, xs, cv = [], [], []
    for e in EV:
        if keep_ev is not None and e["bn"] not in keep_ev: continue
        a, b = lrate(e["ti"], e["m0"], -12, -1, mask), lrate(e["ti"], e["m0"], 1, 12, mask)
        if not (np.isfinite(a) and np.isfinite(b)): continue
        cs = [lrate(k, e["m0"], 1, 12, mask) - lrate(k, e["m0"], -12, -1, mask) for k in e["ctrls"]]
        cs = [x for x in cs if np.isfinite(x)]
        if not cs: continue
        c24 = widx(G, e["m0"], -24, -13)
        if len(c24) != 12: continue
        h24, e24 = Hv[e["ti"], c24].astype(float), Ev[e["ti"], c24].astype(float)
        if not (np.isfinite(h24).all() and np.isfinite(e24).all()) or np.mean(e24) < 5: continue
        c36 = widx(G, e["m0"], -36, -25)
        gw = np.nan
        if len(c36) == 12:
            e36 = Ev[e["ti"], c36].astype(float)
            if np.isfinite(e36).all() and np.mean(e36) > 0:
                gw = float(np.log(np.mean(e24) / np.mean(e36)))
        ag = ((e["m0"] - G["adpt_arr"][e["ti"]]) / 12.0
              if np.isfinite(G["adpt_arr"][e["ti"]]) else np.nan)
        ys.append((b - a) - float(np.mean(cs)))
        xs.append(-float(np.log1p(h24.sum() / np.mean(e24))))
        cv.append([np.log(np.mean(e24)), str(G["ind_arr"][e["ti"]])[:1], gw, ag])
    if len(ys) < 40: return None
    y = np.array(ys); x = np.array(xs)
    # 공변량: log 규모 · 사전 고용성장 · 업력 · 산업 (I-47 과 동일 집합)
    num = [np.array([c[0] for c in cv])]
    for j in (2, 3):
        v = np.array([c[j] for c in cv], float)
        num.append(np.where(np.isfinite(v), v, np.nanmedian(v[np.isfinite(v)])))
    C = np.column_stack([np.ones(len(y))] + num
                        + [np.array([1.0 if c[1] == s else 0.0 for c in cv])
                           for s in sorted({c[1] for c in cv})[1:]])
    r_ = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    yr, xr = r_(y), r_(x)
    b_ = float(np.polyfit(xr, yr, 1)[0])
    bo = np.array([np.polyfit(xr[i], yr[i], 1)[0]
                   for i in (rng.integers(0, len(y), len(y)) for _ in range(NB))])
    ci = qci(bo)
    q1, q2 = np.percentile(xr, [33.33, 66.67])
    hi, lo = yr[xr > q2], yr[xr <= q1]
    dd = float(hi.mean() - lo.mean())
    tb = np.array([hi[rng.integers(0, len(hi), len(hi))].mean()
                   - lo[rng.integers(0, len(lo), len(lo))].mean() for _ in range(NB)])
    tci = qci(tb)
    out = {"n": len(y), "slope_adj": round(b_, 4), "ci": ci,
           "sig": bool(ci[0] > 0 or ci[1] < 0),
           "tercile_adj": round(dd, 4), "tercile_ci": tci,
           "tercile_sig": bool(tci[0] > 0 or tci[1] < 0)}
    print(f"  {tag:<30} n={out['n']:>3} 기울기 {out['slope_adj']:>+7.4f} {str(ci):<20}"
          f"{'✓' if out['sig'] else '✗'} · 3분위 {out['tercile_adj']:>+7.4f} {str(tci):<20}"
          f"{'✓' if out['tercile_sig'] else '✗'}")
    return out


print("\n[Panel C] 대오차·고churn 월 배제 후 헤드라인 재추정")
PC = {"baseline": headline(tag="기준 (배제 없음)")}
for thr, tag in ((0.10, "상대오차 >10% 월 배제"), (0.05, "상대오차 >5% 월 배제")):
    mask = {}
    for i, t in enumerate(rows_ti):
        bad = np.where(np.isfinite(rel[i]) & (rel[i] > thr))[0] + 1
        for m in bad: mask[(t, int(m))] = False
    PC[f"exclude_rel_gt_{int(thr*100)}pct"] = headline(mask=mask, tag=tag)
# 고churn: A+S 가 고용의 50% 초과인 월
mask = {}
churn = np.where(np.isfinite(sub_E[:, 1:]) & (sub_E[:, 1:] > 0),
                 (sub_A[:, 1:] + sub_S[:, 1:]) / sub_E[:, 1:], np.nan)
for i, t in enumerate(rows_ti):
    bad = np.where(np.isfinite(churn[i]) & (churn[i] > 0.50))[0] + 1
    for m in bad: mask[(t, int(m))] = False
PC["exclude_churn_gt_50pct"] = headline(mask=mask, tag="churn >50% 월 배제")

# ════════ Panel D 사업장 구조 변화 ════════
print("\n[Panel D] 사업장 구조 변화 (n_sites 변동) 기업 배제")
try:
    ns = pd.read_parquet(f"{BASE}/shared/data/processed/nps_monthly_matched_v2.parquet",
                         columns=["bn10", "data_ym", "n_sites"])
    ns["bn10"] = ns.bn10.astype(str).str.zfill(10)
    tb = {e["bn"] for e in EV}
    ns = ns[ns.bn10.isin(tb)]
    ns["ym"] = pd.to_numeric(ns.data_ym, errors="coerce")
    chg = ns.sort_values(["bn10", "ym"]).groupby("bn10")["n_sites"].nunique()
    changed = set(chg[chg > 1].index)
    print(f"  n_sites 가 변한 처치기업 {len(changed)} / {len(tb)}")
    PD = {"n_treated_with_site_change": int(len(changed)), "n_treated": int(len(tb)),
          "excluded": headline(keep_ev=tb - changed, tag="사업장수 변동 기업 배제")}
    del ns; gc.collect()
except Exception as ex:
    PD = {"error": str(ex)}; print("  실패:", ex)

# ════════ Panel E 포함/제외 비교 ════════
print("\n[Panel E] 매칭 진입 vs 미진입 비교")
inb = {e["bn"] for e in EV}
allbn = set(allt.bn10)
outb = allbn - inb
def prof_bn(bns):
    v = [Ev[idx.get_loc(b), :] for b in bns if b in idx]
    v = [np.nanmean(x[np.isfinite(x)]) for x in v if np.isfinite(x).any()]
    return {"n": len(bns), "n_linked": len(v),
            "median_employment": (round(float(np.median(v)), 1) if v else None)}
PE_ = {"included": prof_bn(inb), "excluded": prof_bn(outb)}
print(f"  진입 {PE_['included']}")
print(f"  미진입 {PE_['excluded']}")

base = PC["baseline"]
kept = [v for k, v in PC.items() if k != "baseline" and v]
stable = all(v["sig"] for v in kept) if kept else False
verdict = (
    f"[A] flow 정합 {'✓' if step['reconciles'] else '✗'}: 진입 {n_rows} = 미연결 {c_link} + "
    f"창부족 {c_win} + 고용<5 {c_emp} + 대조셀없음 {c_cell} + 매칭 {c_ok}. "
    f"[B] stock-flow 상대오차 중앙값 {PB['rel_error_median']:.5f} · p90 {PB['rel_error_p90']:.5f} · "
    f">5% 월 비중 {PB['share_rel_gt_5pct']:.1%}. 처치기업 사전 {prof['pre[-12,-1]']['mean_rel_error']:.5f} "
    f"→ 딜월 {prof['deal[0,0]']['mean_rel_error']:.5f} → 사후 {prof['post[1,12]']['mean_rel_error']:.5f}. "
    f"[C] 기준 기울기 {base['slope_adj']:+.4f}{base['ci']}; 배제 후 "
    + " · ".join(f"{v['slope_adj']:+.4f}{'✓' if v['sig'] else '✗'}" for v in kept) + ". "
    + ("배제에 견고." if stable else "일부 배제에서 미검출 — 한계로 보고."))
emit("I-48", "결과대상 construct validity + 표본 flow (리뷰3 MC3)",
     "GO" if (step["reconciles"] and stable) else "PARTIAL",
     {"panelA_sample_flow": step, "panelB_stock_flow": PB,
      "panelC_exclusions": PC, "panelD_site_change": PD, "panelE_in_vs_out": PE_},
     "NPS 신규가 행정 재등록이 아니라 경제적 채용인가 · 표본 flow 가 재현 가능한가",
     verdict, kill_met=not step["reconciles"], n=len(EV))
