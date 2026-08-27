# -*- coding: utf-8 -*-
"""I-17 GP 고정효과 / 스폰서 스타일 (Bertrand–Schoar 형).

왜 지금인가. I-03 이 메커니즘을 '비주의 제거'로, I-16·I-14 가 '통제권 이전 불요'로 확정했고,
I-05 는 가역성 판정이 불가로 닫혔다. 남은 유력 채널은 **스폰서의 관여**뿐인데 이사회·정보권
자료가 없다. GP 고정효과는 그 관여를 **간접적으로, 그러나 직접적인 반증가능성을 갖고** 친다.

논리: 효과가 대상기업의 추세나 산업 충격이 아니라 **소유주가 하는 일**이라면,
같은 GP 의 포트폴리오 전반에서 효과가 일관돼야 하고, **GP 정체가 보류된 딜의 효과를 예측**해야 한다.

Panel A  GP 부착 커버리지 · 딜수 분포 (검정 가능성 게이트)
Panel B  분산분해 — 효과 분산 중 GP 간 성분. 귀무(GP 라벨 순열)와 대조
Panel C  ★ leave-one-out 예측 — GP 의 다른 딜 평균이 보류 딜을 예측하는가 (순열 귀무 대조)
Panel D  GP 특성 — 딜 경험수 · 국내/해외

기각조건: LOO 예측계수가 순열 귀무분포 안이면 GP 성분 부재 → 스폰서 관여 가설 미지지.
"""
import re
import numpy as np, pandas as pd
from h30_common import (load, deals, build, attach, boot_did_ci, emit,
                        SEED, qci, NB, widx, BASE)

rng = np.random.default_rng(SEED)
NPERM = 2000
print("[I-17] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
EV, _ = build(G, allt, PE); EV = attach(G, EV)
Hv = G["Hv"]

def zsh(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != 12: return np.nan
    h = Hv[row, c]
    return float((h == 0).mean()) if np.isfinite(h).all() else np.nan
for e in EV:
    a = zsh(e["ti"], e["m0"], -12, -1); b = zsh(e["ti"], e["m0"], 1, 12)
    e["z_t"] = b - a if (np.isfinite(a) and np.isfinite(b)) else np.nan
    cd = [zsh(k, e["m0"], 1, 12) - zsh(k, e["m0"], -12, -1) for k in e["ctrls"]]
    cd = [x for x in cd if np.isfinite(x)]
    e["z_c"] = float(np.mean(cd)) if cd else np.nan
    e["eff"] = e["z_t"] - e["z_c"] if (np.isfinite(e["z_t"]) and np.isfinite(e["z_c"])) else np.nan

# ---------- GP 부착 ----------
pbf = pd.read_csv(f"{BASE}/shared/data/processed/pitchbook_deals_v1.csv", dtype=str)
pbf["bn10"] = pbf.bn.astype(str).str.zfill(10)
pbf["dd"] = pd.to_datetime(pbf["Deal Date"], errors="coerce")
BG = pbf[(pbf.is_bg == "True") & pbf.dd.notna()].sort_values("dd")
INV_BN = BG.drop_duplicates("bn10").set_index("bn10")["Investors"].to_dict()
# 회수 96사: pb_company 로 되붙임
A = pd.read_csv(f"{BASE}/P014_upgrade_package/matching/work/PB_RECOVERY_FINAL_ADOPTED.csv", dtype=str)
A["bn10"] = A.bn10.astype(str).str.zfill(10)
def cnorm(x): return re.sub(r"[^0-9a-z가-힣]", "", str(x).lower())
CMAP = BG.assign(k=BG["Companies"].map(cnorm)).drop_duplicates("k").set_index("k")["Investors"].to_dict()
rec_hit = 0
for r in A.itertuples():
    if r.bn10 not in INV_BN or not isinstance(INV_BN.get(r.bn10), str):
        v = CMAP.get(cnorm(r.pb_company))
        if isinstance(v, str) and v.strip(): INV_BN[r.bn10] = v; rec_hit += 1
print(f"  회수 96사 company명 되붙임 성공 {rec_hit}건")

def gplist(s):
    if not isinstance(s, str): return []
    out = []
    for t in re.split(r"[,;|]", s):
        t = re.sub(r"\s*\((?:[^)]*)\)\s*", " ", t).strip()
        t = re.sub(r"\b(co|ltd|inc|corp|llc|lp|l\.p\.|limited|company)\b\.?", "", t, flags=re.I).strip(" .,")
        if len(t) >= 3: out.append(t.lower())
    return out
for e in EV:
    g = gplist(INV_BN.get(e["bn"]))
    e["gps"] = g; e["gp"] = g[0] if g else None

U = [e for e in EV if e["gp"] and np.isfinite(e["eff"])]
cnt = pd.Series([e["gp"] for e in U]).value_counts()
multi = cnt[cnt >= 2]
print(f"\n[Panel A] GP 부착 {len(U)}/{len(EV)} · 고유 GP {len(cnt)} · "
      f"딜 2건+ GP {len(multi)}개 (해당 이벤트 {int(multi.sum())})")
print(f"  GP당 딜수: 중앙값 {cnt.median():.0f} p90 {cnt.quantile(.9):.0f} 최대 {cnt.max()}")
print("  상위 GP:", cnt.head(6).to_dict())
PA = {"n_with_gp": len(U), "n_events": len(EV), "n_unique_gp": int(len(cnt)),
      "n_gp_multi": int(len(multi)), "n_events_multi": int(multi.sum()),
      "gp_deals_median": float(cnt.median()), "gp_deals_max": int(cnt.max()),
      "top_gps": {k: int(v) for k, v in cnt.head(8).items()}}

M = [e for e in U if cnt[e["gp"]] >= 2]
if len(M) < 40:
    emit("I-17", "GP 고정효과 / 스폰서 스타일", "BLOCKED", {"panelA_coverage": PA},
         "GP 정체가 보류 딜의 효과를 예측하는가", f"딜 2건+ GP 의 이벤트 {len(M)} < 40 — 검정 불가",
         True, len(EV)); raise SystemExit

y = np.array([e["eff"] for e in M]); g = np.array([e["gp"] for e in M])
print(f"  → 검정 표본 {len(M)} (GP {len(set(g))}개), 효과 평균 {y.mean():+.4f} SD {y.std(ddof=1):.4f}")

# ---------- Panel B 분산분해 ----------
def between_share(yy, gg):
    df = pd.DataFrame({"y": yy, "g": gg})
    gm = df.groupby("g")["y"].agg(["mean", "size"])
    ssb = float((gm["size"] * (gm["mean"] - yy.mean()) ** 2).sum())
    return ssb / float(((yy - yy.mean()) ** 2).sum())
obs_b = between_share(y, g)
nb = np.array([between_share(y, rng.permutation(g)) for _ in range(NPERM)])
p_b = float((nb >= obs_b).mean())
print(f"\n[Panel B] GP 간 분산비중 {obs_b:.4f} · 순열귀무 p50 {np.percentile(nb,50):.4f} "
      f"p95 {np.percentile(nb,95):.4f} → 순열 p = {p_b:.4f} "
      f"{'✓' if p_b < 0.05 else '✗'}")
PB = {"between_share": round(obs_b, 4), "null_p50": round(float(np.percentile(nb, 50)), 4),
      "null_p95": round(float(np.percentile(nb, 95)), 4), "perm_p": round(p_b, 4),
      "sig": bool(p_b < 0.05), "n_perm": NPERM}

# ---------- Panel C ★ LOO 예측 ----------
def loo_beta(yy, gg):
    df = pd.DataFrame({"y": yy, "g": gg})
    s = df.groupby("g")["y"].transform("sum"); n = df.groupby("g")["y"].transform("size")
    loo = (s - df.y) / (n - 1)
    ok = np.isfinite(loo) & (n > 1)
    if ok.sum() < 20: return np.nan, 0
    x, yv = loo[ok].values, df.y[ok].values
    return float(np.polyfit(x, yv, 1)[0]), int(ok.sum())
obs_c, n_c = loo_beta(y, g)
nc = np.array([loo_beta(y, rng.permutation(g))[0] for _ in range(NPERM)])
nc = nc[np.isfinite(nc)]
p_c = float((nc >= obs_c).mean())
# [수정] 이벤트 단위 재표본은 LOO 구조를 깨 상방 편의를 만든다(같은 관측이 LOO 평균과 결과 양쪽에
# 들어감). 실측: 이벤트 부트 CI [0.2357, 0.7603] 가 점추정 0.0703 을 포함하지 않았다.
# → **GP 군집 부트스트랩**: GP 를 통째로 재표본하고 중복 GP 는 별개 라벨로 분리해 LOO 를 보존.
GPS = np.array(sorted(set(g)))
bo = []
for _ in range(NB):
    pick = rng.integers(0, len(GPS), len(GPS))
    yy, gg = [], []
    for r, k in enumerate(pick):
        m = g == GPS[k]
        yy.append(y[m]); gg.append(np.full(m.sum(), f"{GPS[k]}#{r}"))
    b_, n_ = loo_beta(np.concatenate(yy), np.concatenate(gg))
    if np.isfinite(b_): bo.append(b_)
bo = np.array(bo)
print(f"\n[Panel C] ★ LOO 예측계수 β = {obs_c:+.4f} (n={n_c})  GP군집 부트 CI {qci(bo)}")
print(f"  순열귀무: 평균 {nc.mean():+.4f} p50 {np.percentile(nc,50):+.4f} p95 {np.percentile(nc,95):+.4f}"
      f"  → 순열 p = {p_c:.4f} {'✓ GP 성분 존재' if p_c < 0.05 else '✗ 귀무 안'}")
PC = {"loo_beta": round(obs_c, 4), "n": n_c, "boot_ci_gpcluster": qci(bo),
      "boot_note": "GP 군집 부트스트랩. 이벤트 단위 재표본은 LOO 구조를 깨 무효(실측 CI가 점추정 미포함).",
      "null_mean": round(float(nc.mean()), 4), "null_p95": round(float(np.percentile(nc, 95)), 4),
      "perm_p": round(p_c, 4), "sig": bool(p_c < 0.05), "n_perm": len(nc)}

# ---------- Panel D GP 특성 ----------
print("\n[Panel D] GP 경험(포트폴리오 딜수) 3분위별 효과")
PD = {}
ex = np.array([cnt[x] for x in g], float)
q1, q2 = np.percentile(ex, [33.33, 66.67])
for lab, m in (("E1 저경험", ex <= q1), ("E2", (ex > q1) & (ex <= q2)), ("E3 고경험", ex > q2)):
    v = y[m]
    if len(v) < 15: PD[lab] = {"n": int(len(v)), "note": "n<15"}; print(f"  {lab:<8} n={len(v)} (<15)"); continue
    b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(NB)])
    ci = qci(b); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
    PD[lab] = {"n": int(len(v)), "mean": round(float(v.mean()), 4), "ci": ci, "sig": sg == "✓"}
    print(f"  {lab:<8} n={len(v):>3} 효과 {v.mean():+.4f} {ci} {sg}  (딜수 범위 {ex[m].min():.0f}-{ex[m].max():.0f})")
PD["cuts_deals"] = [float(q1), float(q2)]

# ---------- 판정 ----------
if PC["sig"] and PB["sig"]: status, concl = "GO", "GP 성분이 분산·예측 양쪽에서 확인 — 소유주 개입 가설 지지"
elif PC["sig"]: status, concl = "PARTIAL", "LOO 예측만 유의 — GP 성분 시사하나 분산분해는 미지지"
elif PB["sig"]: status, concl = "PARTIAL", "분산만 유의하고 예측 실패 — 과적합 가능, GP 성분 주장 불가"
else: status, concl = "KILL", "GP 성분 부재 — 효과는 소유주 정체로 설명되지 않는다"
verdict = (f"GP 부착 {len(U)}/{len(EV)} · 검정표본 {len(M)}(GP {len(set(g))}) | "
           f"GP간 분산비중 {PB['between_share']} 순열p {PB['perm_p']}{'✓' if PB['sig'] else '✗'} | "
           f"★LOO β {PC['loo_beta']} {PC['boot_ci_gpcluster']} 순열p {PC['perm_p']}"
           f"{'✓' if PC['sig'] else '✗'} (귀무 p95 {PC['null_p95']}) | {concl}")
emit("I-17", "GP 고정효과 / 스폰서 스타일", status,
     {"panelA_coverage": PA, "panelB_variance": PB, "panelC_loo_prediction": PC,
      "panelD_gp_experience": PD},
     "효과가 소유주가 하는 일이라면 같은 GP 포트폴리오에서 일관되고 GP 정체가 보류 딜을 예측해야 한다",
     verdict, kill_met=(status == "KILL"), n=len(M),
     extra={"conclusion": concl, "outcome": "무채용비중 DiD (외연마진)",
            "gp_def": "PitchBook Investors 첫 기재 투자자, 법인격 접미사 제거 후 소문자",
            "null": f"GP 라벨 순열 {NPERM}회"})
