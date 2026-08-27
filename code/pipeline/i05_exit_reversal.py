# -*- coding: utf-8 -*-
"""I-05 exit 가역성 본검정 — PE 이탈 후 관성이 되돌아오는가.

처치군: I-05a 정제 + 사용자 수동판정(work/'I05_EXIT_ADJUDICATION complete.csv')에서
        manual_verdict == EXIT (8) + auto CONFIRMED_STRONG (5) = **13건**.
제외: INTERNAL_TRANSFER 5 · UNKNOWN 4 · BELOW_THRESHOLD 1 · 사후패널부족 2.

[사전 기록된 검정력 한계 — §34-9] MDE(n=15)=0.181 · MDE(n=13)≈0.194.
진입효과가 무채용비중 −0.046(전체)/−0.111(고관성군)이므로 **본검정은 진입효과의 4배 이상인
반전만 검출할 수 있다.** 결과가 무유의여도 그것은 '반전 없음'의 증거가 아니다(규칙 11 §2).
이 문단은 결과를 보기 전에 작성됐다.

시점: manual_reason 에 YYYY.M 이 명시된 건은 그 월을, 없으면 first_zero_yr 중반(6월)을 exit 시점으로.
설계: exit 시점을 m0 로 삼아 h39_common.build 로 **재매칭**(never-PE 대조, 셀+5NN). 진입 설계와 동일.

Panel A  exit 이벤트 결과 배터리 (무채용비중·채용률·hazard)
Panel B  진입 vs 이탈 쌍대응 — 같은 13개 기업에서 두 효과의 합이 0인가
Panel C  시점 사양 민감도 (명시월 / 연중반 / 연말)
"""
import re
import numpy as np, pandas as pd
from h30_common import (load, deals, build, attach, summ, boot_did_ci, emit,
                        SEED, qci, NB, widx, dflow, BASE)

rng = np.random.default_rng(SEED)
W = f"{BASE}/P014_upgrade_package/harness30/work"
A = pd.read_csv(f"{W}/I05_EXIT_ADJUDICATION complete.csv", dtype=str)
A["mv"] = A.manual_verdict.fillna("").str.strip()
EX = A[(A.mv == "EXIT") | ((A.mv == "") & (A.auto_verdict == "CONFIRMED_STRONG"))].copy()
print(f"[I-05] 판정 집계: {A.mv.replace('','(자동)').value_counts().to_dict()}")
print(f"  → 처치군 {len(EX)}건 (수동 EXIT {int((EX.mv=='EXIT').sum())} + 자동확정 {int((EX.mv=='').sum())})")

def exit_mi(r, mode):
    fz = int(r.first_zero_yr)
    if mode == "stated":
        m = re.search(r"(20\d\d)\s*[.\-년/]\s*(\d{1,2})", str(r.manual_reason or ""))
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            if 1 <= mo <= 12 and int(r.last_pos_yr) <= y <= fz + 1: return y * 12 + mo
        return fz * 12 + 6
    return fz * 12 + (6 if mode == "mid" else 12)

n_stated = sum(1 for r in EX.itertuples()
               if re.search(r"(20\d\d)\s*[.\-년/]\s*(\d{1,2})", str(r.manual_reason or "")))
print(f"  월 명시 {n_stated}건 / {len(EX)} — 나머지는 first_zero_yr 6월로 근사")

print("\n로딩...")
G = load()
orig, allt, PE, META = deals(G)
Hv = G["Hv"]
EVd, _ = build(G, allt, PE); EVd = attach(G, EVd)      # 진입 설계 (비교용)
ENTRY = None   # attach_z 정의 이후 채운다

def zsh(row, m0, a, b):
    c = widx(G, m0, a, b)
    if len(c) != 12: return np.nan
    h = Hv[row, c]
    return float((h == 0).mean()) if np.isfinite(h).all() else np.nan

def attach_z(EV):
    for e in EV:
        a = zsh(e["ti"], e["m0"], -12, -1); b = zsh(e["ti"], e["m0"], 1, 12)
        e["z_t"] = b - a if (np.isfinite(a) and np.isfinite(b)) else np.nan
        cd = [zsh(k, e["m0"], 1, 12) - zsh(k, e["m0"], -12, -1) for k in e["ctrls"]]
        cd = [x for x in cd if np.isfinite(x)]
        e["z_c"] = float(np.mean(cd)) if cd else np.nan
    return EV

def raw_did(EV, tk, ck):
    """n<20 가드 우회 — 점추정치만 서술적으로. CI 는 참고용(n=13 에서 추론 무효)."""
    d = np.array([e[tk] - e[ck] for e in EV
                  if np.isfinite(e.get(tk, np.nan)) and np.isfinite(e.get(ck, np.nan))])
    if len(d) < 3: return None, None, len(d), None
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(NB)])
    return round(float(d.mean()), 4), qci(b), len(d), d


ENTRY = {e["bn"]: e for e in attach_z(EVd)}     # 진입 설계에도 무채용지표 부착
print(f"  진입 설계 {len(ENTRY)} · 그 중 exit 13건과 겹치는 bn "
      f"{len(set(ENTRY) & set(EX.bn10))}")

RES = {}
for mode, lab in (("stated", "명시월 우선(주)"), ("mid", "연중반 일괄"), ("end", "연말 일괄")):
    df = pd.DataFrame({"bn10": EX.bn10.values, "mi": [exit_mi(r, mode) for r in EX.itertuples()],
                       "src": "exit"})
    EVx, _ = build(G, df, PE); EVx = attach(G, attach_z(EVx))
    for e in EVx:
        e["h_c"] = float(e["cs"].mean()) if len(e.get("cs", [])) else np.nan
    hp, hc, hn, _ = raw_did(EVx, "t", "h_c")
    zp, zc, zn, zd = raw_did(EVx, "z_t", "z_c")
    sg = "✓" if (zc and (zc[0] > 0 or zc[1] < 0)) else "✗"
    RES[mode] = {"label": lab, "n_events_built": len(EVx),
                 "hire_DiD": hp, "hire_ci": hc, "hire_n": hn,
                 "zero_DiD": zp, "zero_ci": zc, "zero_n": zn, "zero_sig": sg == "✓",
                 "inference_valid": False}
    print(f"\n[{lab}] 설계 진입 {len(EVx)}/{len(EX)}")
    print(f"  채용률 DiD {hp} {hc} (n={hn})")
    print(f"  무채용비중 DiD {zp} {zc} {sg} (n={zn})   ← 반전이면 부호가 **양(+)**")
    if zd is not None and mode == "stated":
        print(f"  기업별 값: {np.round(zd,3).tolist()}")
    if mode == "stated": EVX_MAIN = EVx

# ---------- Panel B : 진입 vs 이탈 쌍대응 ----------
print("\n[Panel B] 같은 기업에서 진입효과 vs 이탈효과 (쌍대응)")
pairs = []
for e in EVX_MAIN:
    en = ENTRY.get(e["bn"])
    if en is None: continue
    if not (np.isfinite(e.get("z_t", np.nan)) and np.isfinite(e.get("z_c", np.nan))
            and np.isfinite(en.get("z_t", np.nan)) and np.isfinite(en.get("z_c", np.nan))):
        continue
    pairs.append((e["bn"], en["z_t"] - en["z_c"], e["z_t"] - e["z_c"]))
P = pd.DataFrame(pairs, columns=["bn", "entry", "exit"])
PB = {"n_pairs": len(P)}
if len(P) >= 3:
    P["sum"] = P.entry + P.exit
    for c in ("entry", "exit", "sum"):
        v = P[c].values
        b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(NB)])
        PB[c] = {"mean": round(float(v.mean()), 4), "ci": qci(b),
                 "sig": bool(qci(b)[0] > 0 or qci(b)[1] < 0)}
        print(f"  {c:<6} {v.mean():+.4f} {qci(b)} {'✓' if PB[c]['sig'] else '✗'}")
    print("  (완전 가역이면 entry + exit ≈ 0 · 비가역이면 exit ≈ 0 이고 sum ≈ entry)")
    print(P.round(4).to_string(index=False))
    PB["table"] = P.round(4).to_dict("records")

# ---------- 판별력 검정: sum 의 CI 가 두 대립가설을 가르는가 ----------
DISC = None
if PB.get("sum") and PB.get("entry"):
    lo, hi = PB["sum"]["ci"]; ent = PB["entry"]["mean"]
    h_full = bool(lo <= 0 <= hi)          # 완전 가역 (sum=0)
    h_none = bool(lo <= ent <= hi)        # 비가역 (sum=entry)
    DISC = {"sum_ci": [lo, hi], "entry_mean": ent,
            "contains_full_reversal_H0": h_full, "contains_no_reversal_H0": h_none,
            "discriminates": not (h_full and h_none)}
    print(f"\n[판별력] sum CI {[lo,hi]} 는 완전가역(0) {'포함' if h_full else '배제'} · "
          f"비가역({ent}) {'포함' if h_none else '배제'} → "
          f"{'두 가설을 가르지 못한다' if (h_full and h_none) else '판별 가능'}")

# ---------- 판정 ----------
main = RES["stated"]
rev = bool(main["zero_sig"] and main["zero_DiD"] and main["zero_DiD"] > 0)
MDE = round(2.8 * 0.25 / np.sqrt(max(main["zero_n"], 1)), 3)
if rev: status, concl = "GO", "이탈 후 관성이 유의하게 복귀 — 소유주 인과의 직접 증거"
elif main["zero_DiD"] is not None and abs(main["zero_DiD"]) < MDE:
    status, concl = "PARTIAL", (f"반전 미검출. 그러나 n={main['zero_n']} 의 MDE={MDE} 가 진입효과 "
                                f"0.046 의 {MDE/0.046:.1f}배라 **검정 자체가 판별력이 없다**. "
                                f"'가역성 없음'으로 해석 불가(규칙 11 §2).")
else: status, concl = "PARTIAL", "반전 미검출 (부호·크기 아래 표 참조)"
if DISC and not DISC["discriminates"]:
    status = "PARTIAL"
    concl = (f"**설계가 완전가역(sum=0)과 비가역(sum={DISC['entry_mean']})을 구별하지 못한다** — "
             f"sum CI {DISC['sum_ci']} 가 두 값을 모두 포함. n=13 의 구조적 한계이며 "
             f"§34-9에 결과 확인 전 기록된 검정력 계산과 일치한다. "
             f"참고: 이 13개 기업의 진입효과는 {PB['entry']['mean']} {PB['entry']['ci']} 로 "
             f"유의해 부분표본 자체는 정상이다. 시점 사양별 부호도 불안정"
             f"(명시월 {RES['stated']['zero_DiD']} / 연중반 {RES['mid']['zero_DiD']} / "
             f"연말 {RES['end']['zero_DiD']}).")
verdict = (f"처치군 {len(EX)}건(수동 EXIT 8 + 자동확정 5) · 설계 진입 {main['zero_n']} | "
           f"무채용비중 DiD {main['zero_DiD']}{main['zero_ci']} {'✓' if main['zero_sig'] else '✗'} "
           f"(반전이면 +) | 채용률 DiD {main.get('DiD')}{main.get('DiD_ci')} | "
           f"MDE={MDE} vs 진입효과 0.046 | {concl}")
emit("I-05", "exit 가역성 (PE 이탈 후 관성 복귀)", status,
     {"panelA_by_timing": RES, "panelB_entry_exit_pairs": PB,
      "treatment_group": {"n": len(EX), "manual_EXIT": int((EX.mv == "EXIT").sum()),
                          "auto_CONFIRMED_STRONG": int((EX.mv == "").sum()),
                          "n_month_stated": n_stated},
      "excluded": A[~A.index.isin(EX.index)].mv.replace("", "(자동/제외)").value_counts().to_dict(),
      "MDE_at_n": MDE, "discrimination": DISC},
     "PE 스폰서 이탈 후 무채용비중이 다시 올라가면(DiD > 0) 관성 제거가 소유주에 의한 것임을 기업내에서 확인",
     verdict, kill_met=False, n=len(EX),
     extra={"conclusion": concl,
            "power_note": "§34-9에 결과 확인 전 기록된 검정력 한계. 무유의는 '반전 없음'의 증거가 아니다.",
            "adjudication_source": "work/I05_EXIT_ADJUDICATION complete.csv (사용자 수동판정)"})
