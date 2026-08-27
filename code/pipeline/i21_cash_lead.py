# -*- coding: utf-8 -*-
"""I-21 재무여유 매개 — 현금이 관성 변화에 선행하는가.

메커니즘 갭: `PE → X → 채용빈도` 에서 **X 를 측정한 적이 없다.** 이사회·경영관행 자료는 없고,
CEO·재무제약·slack 은 전부 null 이다. **자료가 있는 유일한 매개 후보가 현금이다.**
(§17: 현금 +0.1076✓ · 자산 +0.3191✓, 그러나 **재무제약 gradient 는 부재** → '제약 완화'가 아니라
'여유 자원'으로만 서술 가능하다.)

매개가 성립하려면 최소 두 조건:
  (A) **선행성** — 현금 증가가 관성 변화보다 먼저 와야 한다 (동시·역순이면 매개 아님)
  (B) **조건부 집중** — 현금이 늘어난 기업에 관성 효과가 몰려야 한다

Panel A  현금 DiD 시간경로 (Y0−1 → Y0, Y0+1, Y0+2) — 언제 들어오는가
Panel B  ★ 선행성 — Δ현금(Y0) → Δ관성(월 1~12) vs 역방향 Δ관성 → Δ현금(Y0+1)
Panel C  조건부 집중 — 현금 증가 여부/분위별 관성 효과
Panel D  반증 — 현금이 늘어도 관성이 안 변하는 집단이 있는가 (필요조건 검정)
"""
import gc
import numpy as np, pandas as pd
from h30_common import load, deals, build, boot_did_ci, emit, SEED, qci, NB, widx, BASE

rng = np.random.default_rng(SEED)
print("[I-21] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
EV, _ = build(G, allt, PE)
Hv, Ev, mset, idx = G["Hv"], G["Ev"], G["mset"], G["idx"]
BNV = np.asarray(idx); INP = set(BNV)

NEED = ["사업자등록번호", "회계연도", "분기", "현금및현금성자산(천원)", "자산총계(천원)",
        "*단기유동성자산(천원)", "매출액(천원)"]
parts = []
for ch in pd.read_csv(f"{BASE}/PI/drops/재무데이터_2009_2025_통합.csv",
                      usecols=NEED, dtype=str, chunksize=200_000):
    ch = ch[ch["분기"].astype(str).str.contains("결산", na=False)]
    ch["bn10"] = ch["사업자등록번호"].str.replace(r"\D", "", regex=True).str.zfill(10)
    parts.append(ch[ch.bn10.isin(INP)])
F = pd.concat(parts, ignore_index=True); del parts; gc.collect()
F["yr"] = pd.to_numeric(F["회계연도"], errors="coerce")
for c, k in (("현금및현금성자산(천원)", "cash"), ("자산총계(천원)", "asset"),
             ("*단기유동성자산(천원)", "liq"), ("매출액(천원)", "rev")):
    F[k] = pd.to_numeric(F[c], errors="coerce")
F = F[F.yr.notna()].drop_duplicates(["bn10", "yr"])
FIN = {(r.bn10, int(r.yr)): (r.cash, r.asset, r.liq) for r in F.itertuples()}
print(f"  재무 {len(F):,}행 / 기업 {F.bn10.nunique():,}"); del F; gc.collect()

def cashr(row, yr):
    """현금/자산 비율 (규모 중립)."""
    f = FIN.get((BNV[row], yr))
    if not f: return np.nan
    c, a, _ = f
    return float(np.clip(c / a, 0, 1)) if (np.isfinite(c) and np.isfinite(a) and a > 0) else np.nan

def zsh(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return np.nan
    x = Hv[row, c]
    return float((x == 0).mean()) if np.isfinite(x).all() else np.nan

for e in EV:
    y0 = (e["m0"] - 1) // 12; e["y0"] = y0
    base = cashr(e["ti"], y0 - 1)
    e["c"] = {h: (cashr(e["ti"], y0 + h) - base) if np.isfinite(base) else np.nan for h in (0, 1, 2)}
    e["cc"] = {}
    for h in (0, 1, 2):
        v = []
        for k in e["ctrls"]:
            b2 = cashr(k, y0 - 1); a2 = cashr(k, y0 + h)
            if np.isfinite(b2) and np.isfinite(a2): v.append(a2 - b2)
        e["cc"][h] = float(np.mean(v)) if v else np.nan
    a = zsh(e["ti"], e["m0"], -12, -1); b = zsh(e["ti"], e["m0"], 1, 12)
    e["z_t"] = b - a if (np.isfinite(a) and np.isfinite(b)) else np.nan
    cd = [zsh(k, e["m0"], 1, 12) - zsh(k, e["m0"], -12, -1) for k in e["ctrls"]]
    cd = [x for x in cd if np.isfinite(x)]
    e["z_c"] = float(np.mean(cd)) if cd else np.nan
    e["zeff"] = e["z_t"] - e["z_c"] if (np.isfinite(e["z_t"]) and np.isfinite(e["z_c"])) else np.nan
    e["ceff"] = {h: (e["c"][h] - e["cc"][h]) if (np.isfinite(e["c"][h]) and np.isfinite(e["cc"][h]))
                 else np.nan for h in (0, 1, 2)}
    # 관성 변화의 '앞·뒤' 반쪽 (선행성 판별용)
    e["z_h1"] = (zsh(e["ti"], e["m0"], 1, 6) - zsh(e["ti"], e["m0"], -6, -1)) \
        if np.isfinite(zsh(e["ti"], e["m0"], 1, 6)) and np.isfinite(zsh(e["ti"], e["m0"], -6, -1)) else np.nan
    e["z_h2"] = (zsh(e["ti"], e["m0"], 7, 12) - zsh(e["ti"], e["m0"], -6, -1)) \
        if np.isfinite(zsh(e["ti"], e["m0"], 7, 12)) and np.isfinite(zsh(e["ti"], e["m0"], -6, -1)) else np.nan

print("\n[Panel A] 현금/자산 DiD 시간경로")
PA = {}
for h in (0, 1, 2):
    p_, ci, n = boot_did_ci([e["c"][h] for e in EV], [e["cc"][h] for e in EV], rng)
    sg = "✓" if (ci and (ci[0] > 0 or ci[1] < 0)) else "✗"
    PA[f"Y0+{h}"] = {"DiD": p_, "ci": ci, "n": n, "sig": sg == "✓"}
    print(f"  Y0−1 → Y0+{h}  {str(p_):>8} {str(ci):<20} {sg} (n={n})")

print("\n[Panel B] ★ 선행성 — 현금이 먼저인가 관성이 먼저인가")
PB = {}
def slope(x, y, lab):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 40: print(f"  {lab:<34} n={int(m.sum())} (<40)"); return None
    xx, yy = x[m], y[m]
    s = float(np.polyfit(xx, yy, 1)[0])
    bs = np.array([np.polyfit(xx[j], yy[j], 1)[0] for j in
                   (rng.integers(0, len(xx), len(xx)) for _ in range(NB))])
    ci = qci(bs); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
    print(f"  {lab:<34} 기울기 {s:+.4f} {ci} {sg} (n={int(m.sum())})")
    return {"slope": round(s, 4), "ci": ci, "sig": sg == "✓", "n": int(m.sum())}
c0 = np.array([e["ceff"][0] for e in EV], float)
c1 = np.array([e["ceff"][1] for e in EV], float)
z = np.array([e["zeff"] for e in EV], float)
zh1 = np.array([e["z_h1"] for e in EV], float)
zh2 = np.array([e["z_h2"] for e in EV], float)
PB["cash_Y0 → 관성 전체"] = slope(c0, z, "현금(Y0) → 관성 변화 전체")
PB["cash_Y0 → 관성 후반"] = slope(c0, zh2, "현금(Y0) → 관성 후반(7~12월)")
PB["관성 전반 → cash_Y0+1"] = slope(zh1, c1, "관성 전반(1~6월) → 현금(Y0+1)  [역방향]")
print("  (매개면 앞의 두 개가 음(−)으로 유의하고 역방향은 무유의여야 한다)")

print("\n[Panel C] 조건부 집중 — 현금 증가 여부/분위별 관성 효과")
PC = {}
ok = np.isfinite(c0)
if ok.sum() >= 60:
    q1, q2 = np.percentile(c0[ok], [33.33, 66.67])
    print(f"  현금DiD 3분위 컷 {q1:+.4f} / {q2:+.4f}")
    for lab, m in (("C1 현금감소", c0 <= q1), ("C2 중간", (c0 > q1) & (c0 <= q2)),
                   ("C3 현금증가", c0 > q2)):
        sub = [EV[i] for i in np.flatnonzero(m & ok)]
        p_, ci, n = boot_did_ci([e["z_t"] for e in sub], [e["z_c"] for e in sub], rng)
        sg = "✓" if (ci and (ci[0] > 0 or ci[1] < 0)) else "✗"
        PC[lab] = {"DiD": p_, "ci": ci, "n": n, "sig": sg == "✓",
                   "cash_mean": round(float(np.mean(c0[m & ok])), 4)}
        print(f"  {lab:<10} n={n:>3} (현금 {PC[lab]['cash_mean']:+.3f}) 관성 DiD {p_} {ci} {sg}")
    d1 = z[(c0 <= q1) & np.isfinite(z)]; d3 = z[(c0 > q2) & np.isfinite(z)]
    if min(len(d1), len(d3)) >= 15:
        bs = np.array([d3[rng.integers(0, len(d3), len(d3))].mean()
                       - d1[rng.integers(0, len(d1), len(d1))].mean() for _ in range(NB)])
        ci = qci(bs); sg = "✓" if (ci[0] > 0 or ci[1] < 0) else "✗"
        S = 0.046; mg = [round(ci[0] + S, 4), round(S - ci[1], 4)]
        PC["C3−C1"] = {"diff": round(float(d3.mean() - d1.mean()), 4), "ci": ci, "sig": sg == "✓",
                       "equivalence": {"SESOI": S, "holds": bool(ci[0] > -S and ci[1] < S),
                                       "margin": mg, "knife": bool(min(mg) < 0.001)}}
        print(f"  C3−C1 {d3.mean()-d1.mean():+.4f} {ci} {sg}  "
              f"등가성(δ=0.046) {'성립' if PC['C3−C1']['equivalence']['holds'] else '미성립'} 여유 {mg}")

# ---- 판정 ----
lead = bool(PB.get("cash_Y0 → 관성 전체") and PB["cash_Y0 → 관성 전체"]["sig"]
            and PB["cash_Y0 → 관성 전체"]["slope"] < 0)
rev = bool(PB.get("관성 전반 → cash_Y0+1") and PB["관성 전반 → cash_Y0+1"]["sig"])
conc = bool(PC.get("C3−C1", {}).get("sig") and (PC["C3−C1"]["diff"] or 0) < 0)
if lead and conc and not rev: status, concl = "GO", "선행성 + 조건부 집중 모두 성립 — 현금이 매개 후보로 확립"
elif lead or conc: status, concl = "PARTIAL", f"부분 성립 (선행성 {lead} / 집중 {conc}) — 매개 주장 불가"
else: status, concl = "KILL", "선행성·조건부 집중 모두 미검출 — 현금은 매개변수가 아니다"
verdict = (f"현금 DiD Y0 {PA['Y0+0']['DiD']}{'✓' if PA['Y0+0']['sig'] else '✗'} · "
           f"Y0+1 {PA['Y0+1']['DiD']}{'✓' if PA['Y0+1']['sig'] else '✗'} | "
           f"선행성 기울기 {PB.get('cash_Y0 → 관성 전체',{}).get('slope')}"
           f"{'✓' if lead else '✗'} · 역방향 {'유의' if rev else '무유의'} | "
           f"C3−C1 {PC.get('C3−C1',{}).get('diff')}{'✓' if conc else '✗'} | {concl}")
emit("I-21", "재무여유 매개 (현금 선행성)", status,
     {"panelA_cash_path": PA, "panelB_lead_lag": PB, "panelC_conditional": PC},
     "매개가 성립하려면 현금 증가가 관성 변화에 선행하고, 현금이 늘어난 기업에 효과가 집중돼야 한다",
     verdict, kill_met=(status == "KILL"), n=len(EV),
     extra={"conclusion": concl,
            "framing": "재무제약 gradient 는 이미 부재(§17). 따라서 '제약 완화'가 아니라 "
                       "'여유 자원'으로만 서술 가능하며, 매개가 성립해도 그 한도 안에서만 쓴다."})
