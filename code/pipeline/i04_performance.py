# -*- coding: utf-8 -*-
"""I-04 실적 앵커 — "조정을 자주 하는 것이 좋은 일인가".

현재 논문의 가장 큰 구멍. 관성 제거가 가치를 만드는지 말하지 못하면 referee 는 "so what" 을 쓴다.

논리. I-03 이 메커니즘을 '비주의 제거'로 판별했다면, 제거된 비주의는 **비효율이었어야** 한다.
그렇다면 (a) 성과가 개선되고 (b) 그 개선이 **관성 제거가 실제로 일어난 곳에 집중**돼야 한다.
(b) 가 (a) 보다 강한 검정이다 — 단순 성과 개선은 선택으로도 설명되지만,
**같은 조절자(사전 관성)가 노동 결과와 성과 결과를 동시에 가르는 것**은 설명하기 어렵다.

Panel A  성과 DiD (매출/인원 · 자산/인원 · 영업이익률 · log매출 · 인당임금)
Panel B  **I-25 조절자 재사용** — T1 저관성 vs T3 고관성에서 성과 DiD 가 갈리는가  ← 핵심
Panel C  동행 — 처치군 내부에서 관성제거 강도와 성과 개선의 상관 (인과 아님, 명시)
Panel D  위약 — I-31 방식으로 고관성 never-treated 에 같은 성과 검정 (파이프라인 영점)

[한계 명시] 재무는 연간, 노동은 월별이다. 성과 DiD 는 (Y0+1) − (Y0−1) 로 딜 연도를 건너뛴다.
Panel C 는 상관이며 인과가 아니다.
"""
import gc
import numpy as np, pandas as pd
from h30_common import load, deals, build, boot_did_ci, emit, SEED, qci, NB, widx, BASE

rng = np.random.default_rng(SEED)
print("[I-04] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
EV, _ = build(G, allt, PE)
Hv, Ev, Av, idx, mset = G["Hv"], G["Ev"], G["Av"], G["idx"], G["mset"]

# ---- 재무 (112열 중 9열만) ----
NEED = ["사업자등록번호", "회계연도", "분기", "자산총계(천원)", "매출액(천원)",
        "영업이익(천원)", "당기순이익(천원)", "자본총계(천원)", "부채총계(천원)"]
parts = []
for ch in pd.read_csv(f"{BASE}/PI/drops/재무데이터_2009_2025_통합.csv",
                      usecols=NEED, dtype=str, chunksize=200_000):
    ch = ch[ch["분기"].astype(str).str.contains("결산", na=False)]
    parts.append(ch)
F = pd.concat(parts, ignore_index=True); del parts; gc.collect()
F["bn10"] = F["사업자등록번호"].str.replace(r"\D", "", regex=True).str.zfill(10)
F["yr"] = pd.to_numeric(F["회계연도"], errors="coerce")
for c, k in (("자산총계(천원)", "asset"), ("매출액(천원)", "rev"), ("영업이익(천원)", "op"),
             ("당기순이익(천원)", "ni"), ("자본총계(천원)", "eq"), ("부채총계(천원)", "debt")):
    F[k] = pd.to_numeric(F[c], errors="coerce")
F = F[F.yr.notna()].drop_duplicates(["bn10", "yr"])
FIN = {(r.bn10, int(r.yr)): r for r in F.itertuples()}
print(f"  재무 결산 {len(F):,}행 · 기업 {F.bn10.nunique():,} · 연도 {int(F.yr.min())}-{int(F.yr.max())}")
del F; gc.collect()

BNV = np.asarray(idx)

def emp_dec(row, yr):
    j = mset.get(yr * 12 + 12)
    if j is None: return np.nan
    v = Ev[row, j]
    return float(v) if (np.isfinite(v) and v >= 5) else np.nan

def wage(row, yr):
    """인당 신고소득 = 고지금액 연합 / 평균 가입자수 (12개월)."""
    js = [mset.get(yr * 12 + m) for m in range(1, 13)]
    js = [j for j in js if j is not None]
    if len(js) < 12: return np.nan
    a, e = Av[row, js], Ev[row, js]
    if not (np.isfinite(a).all() and np.isfinite(e).all()) or np.nanmean(e) < 5: return np.nan
    return float(np.nansum(a) / np.nanmean(e))

def perf(row, yr):
    r = FIN.get((BNV[row], yr))
    if r is None: return {}
    e = emp_dec(row, yr)
    o = {}
    if np.isfinite(e) and e > 0:
        if np.isfinite(r.rev) and r.rev > 0: o["rev_pe"] = np.log(r.rev / e)
        if np.isfinite(r.asset) and r.asset > 0: o["asset_pe"] = np.log(r.asset / e)
    if np.isfinite(r.rev) and r.rev > 0:
        o["log_rev"] = np.log(r.rev)
        if np.isfinite(r.op): o["opm"] = float(np.clip(r.op / r.rev, -1, 1))
    w = wage(row, yr)
    if np.isfinite(w) and w > 0: o["log_wage"] = np.log(w)
    return o

OUT = ["rev_pe", "asset_pe", "opm", "log_rev", "log_wage"]
LAB = {"rev_pe": "log 매출/인원", "asset_pe": "log 자산/인원", "opm": "영업이익률",
       "log_rev": "log 매출", "log_wage": "log 인당임금"}

def dperf(row, y0, h=1):
    a, b = perf(row, y0 - 1), perf(row, y0 + h)
    return {k: b[k] - a[k] for k in OUT if k in a and k in b}

def zsh(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return np.nan
    x = Hv[row, c]
    return float((x == 0).mean()) if np.isfinite(x).all() else np.nan

# 사전-사전 관성 bin (I-25/I-31 절차)
_pp = np.array([zsh(e["ti"], e["m0"], -24, -13) for e in EV], float)
_z = np.array([(zsh(e["ti"], e["m0"], 1, 12) - zsh(e["ti"], e["m0"], -12, -1)) for e in EV], float)
_u = np.isfinite(_pp) & np.isfinite(_z)
Q1, Q2 = np.percentile(_pp[_u], [33.33, 66.67])
def pbin(row, m0):
    v = zsh(row, m0, -24, -13)
    return None if not np.isfinite(v) else (0 if v <= Q1 else (1 if v <= Q2 else 2))
print(f"  사전-사전 관성 컷 {Q1:.4f}/{Q2:.4f}")

# ---- 이벤트별 성과 변화 부착 ----
for e in EV:
    y0 = (e["m0"] - 1) // 12
    e["y0"] = y0; e["pb"] = pbin(e["ti"], e["m0"])
    e["dp"] = {h: dperf(e["ti"], y0, h) for h in (1, 2)}
    e["dpc"] = {}
    for h in (1, 2):
        acc = {}
        for k in e["ctrls"]:
            d = dperf(k, y0, h)
            for kk, vv in d.items(): acc.setdefault(kk, []).append(vv)
        e["dpc"][h] = {kk: float(np.mean(v)) for kk, v in acc.items()}
    e["zdid"] = (zsh(e["ti"], e["m0"], 1, 12) - zsh(e["ti"], e["m0"], -12, -1))
    cs = [zsh(k, e["m0"], 1, 12) - zsh(k, e["m0"], -12, -1) for k in e["ctrls"]]
    cs = [x for x in cs if np.isfinite(x)]
    e["zdid"] = e["zdid"] - float(np.mean(cs)) if (np.isfinite(e["zdid"]) and cs) else np.nan

def batt(sub, h, tag, samebin=None):
    o = {}
    for k in OUT:
        t = [e["dp"][h][k] for e in sub if k in e["dp"][h] and k in e["dpc"][h]]
        c = [e["dpc"][h][k] for e in sub if k in e["dp"][h] and k in e["dpc"][h]]
        p_, ci, n = boot_did_ci(t, c, rng)
        sg = "✓" if (ci and (ci[0] > 0 or ci[1] < 0)) else ("✗" if ci else "-")
        o[k] = {"DiD": p_, "ci": ci, "n": n, "sig": sg == "✓"}
        print(f"    {tag:<10} {LAB[k]:<12} h+{h}  {str(p_):>8} {str(ci):<20} {sg} (n={n})")
    return o

print("\n[Panel A] 성과 DiD (전체)")
PA = {f"h{h}": batt(EV, h, "전체") for h in (1, 2)}

print("\n[Panel B] ★ I-25 조절자 재사용 — 사전 관성 분위별 성과 DiD")
PB = {}
for b, bl in ((0, "T1 저관성"), (2, "T3 고관성")):
    sub = [e for e in EV if e["pb"] == b]
    print(f"  -- {bl} (n_ev={len(sub)}) --")
    PB[bl] = {f"h{h}": batt(sub, h, bl) for h in (1, 2)}
# T3 − T1 차이
print("  -- T3 − T1 차이 --")
PB["diff"] = {}
for h in (1, 2):
    for k in OUT:
        d1 = np.array([e["dp"][h][k] - e["dpc"][h][k] for e in EV
                       if e["pb"] == 0 and k in e["dp"][h] and k in e["dpc"][h]])
        d3 = np.array([e["dp"][h][k] - e["dpc"][h][k] for e in EV
                       if e["pb"] == 2 and k in e["dp"][h] and k in e["dpc"][h]])
        if min(len(d1), len(d3)) < 15: continue
        bs = np.array([d3[rng.integers(0, len(d3), len(d3))].mean()
                       - d1[rng.integers(0, len(d1), len(d1))].mean() for _ in range(NB)])
        ci = qci(bs); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
        PB["diff"][f"h{h}|{k}"] = {"diff": round(float(d3.mean() - d1.mean()), 4), "ci": ci,
                                   "sig": sg == "✓", "n1": len(d1), "n3": len(d3)}
        print(f"    {LAB[k]:<12} h+{h}  {d3.mean()-d1.mean():+.4f} {ci} {sg} (n {len(d1)}/{len(d3)})")

print("\n[Panel C] 동행 — 관성제거 강도 vs 성과 개선 (처치군 내부, **상관이며 인과 아님**)")
PC = {}
for h in (1, 2):
    for k in OUT:
        x = np.array([e["zdid"] for e in EV if np.isfinite(e.get("zdid", np.nan))
                      and k in e["dp"][h] and k in e["dpc"][h]])
        y = np.array([e["dp"][h][k] - e["dpc"][h][k] for e in EV if np.isfinite(e.get("zdid", np.nan))
                      and k in e["dp"][h] and k in e["dpc"][h]])
        if len(x) < 40: continue
        sl = float(np.polyfit(x, y, 1)[0])
        bs = np.array([np.polyfit(x[j], y[j], 1)[0] for j in
                       (rng.integers(0, len(x), len(x)) for _ in range(NB))])
        ci = qci(bs); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
        PC[f"h{h}|{k}"] = {"slope": round(sl, 4), "ci": ci, "sig": sg == "✓", "n": len(x),
                           "note": "기울기 음수 = 무채용비중이 더 크게 줄수록 성과 개선"}
        print(f"    {LAB[k]:<12} h+{h}  기울기 {sl:+.4f} {ci} {sg} (n={len(x)})")

# ---- 판정 ----
key = [("h1", "rev_pe"), ("h2", "rev_pe"), ("h1", "opm"), ("h2", "opm")]
anyA = any(PA[h][k]["sig"] for h, k in key)
anyB = any(v["sig"] for kk, v in PB["diff"].items() if "rev_pe" in kk or "opm" in kk)
if anyB: status, concl = "GO", "같은 조절자가 노동과 성과를 동시에 가른다 — 관성 제거의 가치 지지"
elif anyA: status, concl = "PARTIAL", "성과는 개선되나 조절자별로 갈리지 않음 — 선택으로도 설명 가능"
else: status, concl = "KILL", "성과 개선 미검출 — '조정만 늘고 가치는 없다'로 서술해야 함"
verdict = (f"전체 h+1 매출/인원 {PA['h1']['rev_pe']['DiD']}{PA['h1']['rev_pe']['ci']}"
           f"{'✓' if PA['h1']['rev_pe']['sig'] else '✗'} · 영업이익률 {PA['h1']['opm']['DiD']}"
           f"{'✓' if PA['h1']['opm']['sig'] else '✗'} | T3−T1 유의: "
           f"{[k for k,v in PB['diff'].items() if v['sig']] or '없음'} | {concl}")
emit("I-04", "실적 앵커", status,
     {"panelA_overall": PA, "panelB_by_inertia": PB, "panelC_comovement": PC,
      "tercile_cuts": [round(float(Q1),4), round(float(Q2),4)], "outcomes": LAB},
     "관성 제거가 비효율 제거였다면 성과가 개선되고 그 개선이 관성이 심했던 기업에 집중돼야 한다",
     verdict, kill_met=(status == "KILL"), n=len(EV),
     extra={"conclusion": concl,
            "limits": "재무는 연간·노동은 월별. 성과 DiD 는 (Y0+1)−(Y0−1) 로 딜 연도 건너뜀. "
                      "Panel C 는 상관이며 인과 아님."})
