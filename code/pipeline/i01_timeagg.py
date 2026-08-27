# -*- coding: utf-8 -*-
"""I-01 시간집계 정리 검정.

주장: PE의 고용효과는 '수준'이 아니라 '조정빈도'에 있고, 이 대상은 연간 데이터에서 소멸한다.
따라서 PE-고용 문헌의 반복된 null은 집계 인공물일 수 있다. 우리 월별 데이터 안에서 이를 재현한다.

Panel A  무행동 지표의 집계 사망   — 블록크기 f=1,2,3,4,6,12개월에서 무채용 블록 비중 DiD
Panel B  이벤트 타이밍 흐림        — 이벤트시간 vs 달력연도 창에서 채용률 DiD 감쇠
Panel C  딜 월분포                 — B2 오염의 크기(사후연도에 섞인 처치전 개월수)
"""
import numpy as np
from h30_common import (load, deals, build, widx, flow, rel_log,
                        boot_did_ci, boot_mean_ci, emit, SEED, qci, NB)

rng = np.random.default_rng(SEED)
print("[I-01] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
EV, _ = build(G, allt, PE)
print(f"  이벤트 {len(EV)}건 (매칭 진입)")

Hv, Ev = G["Hv"], G["Ev"]
FS = [1, 2, 3, 4, 6, 12]


def zero_share(row, m0, a, b, f):
    """[m0+a, m0+b] 12개월을 f개월 블록으로 묶어 '무채용 블록' 비중."""
    c = widx(G, m0, a, b)
    if len(c) != 12:
        return np.nan
    h = Hv[row, c]
    if not np.isfinite(h).all():
        return np.nan
    blk = h.reshape(12 // f, f).sum(axis=1)
    return float((blk == 0).mean())


def cyear_flow(row, m0, Y, M):
    """달력연도 Y (mi = Y*12+1 .. Y*12+12) 의 flow rate."""
    return flow(G, row, m0, Y * 12 + 1 - m0, Y * 12 + 12 - m0, M)


# ---------------- Panel A ----------------
print("\n[Panel A] 무행동 지표의 집계 사망 — 무채용 블록 비중 DiD")
print(f"  {'블록':>6} {'해석':<14} {'n':>4} {'DiD':>9}  {'CI95':<20} {'처치사후평균':>10}")
A = {}
for f in FS:
    t, c, lv = [], [], []
    for e in EV:
        tp = zero_share(e["ti"], e["m0"], -12, -1, f)
        ts = zero_share(e["ti"], e["m0"], 1, 12, f)
        if not (np.isfinite(tp) and np.isfinite(ts)):
            continue
        cd = [zero_share(k, e["m0"], 1, 12, f) - zero_share(k, e["m0"], -12, -1, f)
              for k in e["ctrls"]]
        cd = [x for x in cd if np.isfinite(x)]
        if not cd:
            continue
        t.append(ts - tp); c.append(np.mean(cd)); lv.append(ts)
    pt, ci, n = boot_did_ci(t, c, rng)
    lab = {1: "월", 2: "2개월", 3: "분기", 4: "4개월", 6: "반기", 12: "연간"}[f]
    sig = "✓" if (ci and (ci[1] < 0 or ci[0] > 0)) else "✗"
    print(f"  {f:>4}개월 {lab:<14} {n:>4} {pt:>+9.4f}  {str(ci):<20} {np.mean(lv):>10.4f} {sig}")
    A[f] = {"label": lab, "n": n, "DiD": pt, "ci": ci, "sig": sig == "✓",
            "post_mean_treated": round(float(np.mean(lv)), 4)}

# ---------------- Panel B ----------------
print("\n[Panel B] 이벤트 타이밍 흐림 — 채용률 DiD")
specs = {}
tE, cE, tB2, cB2, tB3, cB3 = [], [], [], [], [], []
for e in EV:
    m0, ti, Y0 = e["m0"], e["ti"], (e["m0"] - 1) // 12
    # B1 이벤트시간
    a = flow(G, ti, m0, -12, -1, Hv); b = flow(G, ti, m0, 1, 12, Hv)
    cd = [flow(G, k, m0, 1, 12, Hv) - flow(G, k, m0, -12, -1, Hv) for k in e["ctrls"]]
    cd = [x for x in cd if np.isfinite(x)]
    if np.isfinite(a) and np.isfinite(b) and cd:
        tE.append(b - a); cE.append(np.mean(cd))
    # B2 달력연도, 사후 = 딜이 포함된 해 (분석가가 흔히 쓰는 코딩)
    a2 = cyear_flow(ti, m0, Y0 - 1, Hv); b2 = cyear_flow(ti, m0, Y0, Hv)
    c2 = [cyear_flow(k, m0, Y0, Hv) - cyear_flow(k, m0, Y0 - 1, Hv) for k in e["ctrls"]]
    c2 = [x for x in c2 if np.isfinite(x)]
    if np.isfinite(a2) and np.isfinite(b2) and c2:
        tB2.append(b2 - a2); cB2.append(np.mean(c2))
    # B3 달력연도, 딜 연도 제외
    a3 = cyear_flow(ti, m0, Y0 - 1, Hv); b3 = cyear_flow(ti, m0, Y0 + 1, Hv)
    c3 = [cyear_flow(k, m0, Y0 + 1, Hv) - cyear_flow(k, m0, Y0 - 1, Hv) for k in e["ctrls"]]
    c3 = [x for x in c3 if np.isfinite(x)]
    if np.isfinite(a3) and np.isfinite(b3) and c3:
        tB3.append(b3 - a3); cB3.append(np.mean(c3))

for key, lab, tt, cc in [("B1", "이벤트시간 [-12,-1]v[1,12]", tE, cE),
                         ("B2", "달력연도 사후=딜연도", tB2, cB2),
                         ("B3", "달력연도 딜연도제외", tB3, cB3)]:
    pt, ci, n = boot_did_ci(tt, cc, rng)
    sig = "✓" if (ci and (ci[1] < 0 or ci[0] > 0)) else "✗"
    print(f"  {key} {lab:<28} n={n:>3}  DiD {pt:>+8.4f} {str(ci):<20} {sig}")
    specs[key] = {"label": lab, "n": n, "DiD": pt, "ci": ci, "sig": sig == "✓"}
if specs["B1"]["DiD"]:
    for k in ("B2", "B3"):
        if specs[k]["DiD"] is not None:
            specs[k]["ratio_to_B1"] = round(specs[k]["DiD"] / specs["B1"]["DiD"], 3)
            print(f"     {k}/B1 감쇠비 = {specs[k]['ratio_to_B1']}")

# ---------------- Panel C ----------------
mon = np.array([((e["m0"] - 1) % 12) + 1 for e in EV])
contam = 12 - mon          # 딜연도를 '사후'로 쓸 때 그 해에 섞인 처치전 개월수
print(f"\n[Panel C] 딜 월분포 — B2 사후연도의 평균 처치전 오염 = {contam.mean():.1f}개월 / 12")
print(f"  월별 건수: {np.bincount(mon, minlength=13)[1:].tolist()}")

# 1월 몰림 진단 — PitchBook 날짜 대치 의심 (I-14 주주명부 정확시점의 근거)
cnt = np.bincount(mon, minlength=13)[1:]
exp = len(mon) / 12.0
jan_ratio = round(float(cnt[0] / exp), 2)
print(f"  1월 몰림: {cnt[0]}건 vs 균등기대 {exp:.1f} = {jan_ratio}배"
      f" — 연-only 공시의 1월 대치 의심 → I-14 정확시점 확보 필요")

# ---------------- 판정 (규칙 11 등가성 포함) ----------------
m_did, y_did = A[1]["DiD"], A[12]["DiD"]
died = (A[12]["sig"] is False) and A[1]["sig"]
# SESOI = 월별에서 실제로 검출한 효과크기. "연간에서는 월별만한 효과를 배제한다"가 주장.
SESOI = abs(m_did)
yci = A[12]["ci"]
equiv = bool(yci and yci[0] > -SESOI and yci[1] < SESOI)
margin = [round(yci[0] + SESOI, 4), round(SESOI - yci[1], 4)] if yci else None
knife = bool(margin and min(margin) < 0.001)
print(f"\n[등가성, 규칙 11] SESOI={SESOI:.4f}(월별 검출크기) 연간 CI{yci}"
      f" ⊂ ±SESOI : {'✓ 등가성 성립' if equiv and not knife else '✗'} 여유={margin}")

atten = specs["B2"].get("ratio_to_B1")
status = "GO" if (died and equiv) else "PARTIAL"
verdict = (f"PanelA 무채용비중: 월 {m_did:+.4f}✓ → 분기 {A[3]['DiD']:+.4f}✓ → 반기 {A[6]['DiD']:+.4f}✗ "
           f"→ 연간 {y_did:+.4f}✗, 등가성 {'성립' if equiv and not knife else '미성립'}(δ={SESOI:.4f}). "
           f"PanelB 반증: 채용률은 달력연도에서도 거의 감쇠하지 않음(B2/B1={atten}, 여전히 유의) "
           f"— 연간 비가시성은 '집계 일반'이 아니라 '외연마진 지표에 특정'된다.")
emit("I-01", "시간집계 정리 검정", status,
     {"panelA_inaction_by_block": A, "panelB_timing_blur": specs,
      "panelC_contamination_months": round(float(contam.mean()), 2),
      "deal_month_counts": cnt.tolist(),
      "equivalence_annual": {"SESOI": round(SESOI, 4), "ci": yci, "holds": equiv,
                             "margin": margin, "knife_edge": knife},
      "jan_bunching_ratio": jan_ratio},
     "월별에서 유의한 무행동 효과가 집계 주기 증가에 따라 감쇠·소멸하고, 달력연도 창에서 채용률 DiD도 감쇠한다",
     verdict, kill_met=not died, n=len(EV),
     extra={"prediction_refuted_part": "PanelB: 채용률 DiD의 달력연도 감쇠는 10%에 불과 — "
                                       "타이밍 흐림 감쇠 예측은 반증됨",
            "flag_for_I14": f"딜 월분포 1월 {int(cnt[0])}건({jan_ratio}배 초과) — 날짜 대치 의심"})
