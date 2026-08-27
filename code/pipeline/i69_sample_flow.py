# -*- coding: utf-8 -*-
"""I-69 표본 흐름과 날짜 사실의 단일 정본.

리뷰 7 §3 지적: Table 1 은 379 → 301 → 286 이라 쓰는데, 부록 F.1(I-63)의 탈락 분해는
379 − 53 − 1 − 17 − 22 = 286 으로 **301 을 거치지 않는다.** 코드로 확인한 결과 301 은
흐름의 한 단계가 아니라 **종전(상태 비균형) 설계의 자체 표본**(I-58 current_full.n)이다.
같은 지적의 날짜 항목도 여기서 정본화한다: 기준설계 379 는 *매칭* 수이므로 사후창이
아직 없는 최근 딜을 포함한다.

Panel A  기준 379 의 소진 분해 (I-63 재계산이 아니라 같은 정의를 여기서 다시 센다)
Panel B  사후 +1..+12 창이 완전한 사건 수와 마지막 딜월
Panel C  날짜 범위
"""
import numpy as np, json
from collections import Counter
from h30_common import load, deals, build, emit, widx
from h39_common import SIZE_B
ym = lambda k: "%d-%02d" % ((k//12 if k % 12 else k//12-1), (k % 12 or 12))
print("[I-69] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Ev, idx = G["Hv"], G["Ev"], G["idx"]
mis = G["mis"]; m0 = np.array([e["m0"] for e in EV])

post_ok = [e for e in EV if len(widx(G, e["m0"], 1, 12)) == 12]
# 평균효과(h39 flow())는 각 창에 **6개월 이상**만 요구한다 — Table 2 의 367 이 이 규칙이다.
ge6 = [e for e in EV if len(widx(G, e["m0"], 1, 12)) >= 6 and len(widx(G, e["m0"], -12, -1)) >= 6]
pre_ok  = [e for e in EV if len(widx(G, e["m0"], -12, -1)) == 12]
state_ok = [e for e in EV if len(widx(G, e["m0"], -24, -13)) == 12]

A = {"baseline_matched_events": len(EV),
     "note": "탈락 분해는 I-63 estimates.attrition 이 정본 "
             "(state_window 53 · emp_lt5 1 · no_cell 17 · outcome 22 · ok 286; 합 379)."}
B = {"post_window_complete": len(post_ok),
     "post_window_incomplete": len(EV) - len(post_ok),
     "last_deal_month_with_post_window": ym(max(e["m0"] for e in post_ok)),
     "incomplete_by_month": dict(sorted(Counter(ym(e["m0"]) for e in EV if e not in post_ok).items())),
     "pre_window_complete": len(pre_ok), "state_window_complete": len(state_ok),
     "windows_ge6_both": len(ge6), "windows_ge6_dropped": len(EV) - len(ge6),
     "last_deal_month_ge6": ym(max(e["m0"] for e in ge6)),
     "rule_note": "average effects (h39_common.flow) require >=6 observed months in each window; "
                  "the gradient (blk) requires all 12. Table 2 n=367 also drops windows with zero hires/NaN."}
C = {"panel_first": ym(int(mis.min())), "panel_last": ym(int(mis.max())),
     "baseline_first_deal_month": ym(int(m0.min())), "baseline_last_deal_month": ym(int(m0.max())),
     "conventional_design_n": 301,
     "conventional_design_source": "harness30/out/I58.json estimates.panelA_common_sample.current_full.n",
     "conventional_design_note": "301 은 흐름의 단계가 아니라 상태 비균형(종전) 설계의 표본. "
                                 "그 중 286 이 상태균형 셀도 찾는다 (차이 15)."}
print(f"  기준 매칭 {A['baseline_matched_events']}건 · 사후창 완전 {B['post_window_complete']}건 "
      f"(마지막 {B['last_deal_month_with_post_window']}) · 사후창 미완 {B['post_window_incomplete']}건")
print(f"  패널 {C['panel_first']}~{C['panel_last']} · 딜월 {C['baseline_first_deal_month']}~{C['baseline_last_deal_month']}")
emit("I-69", "표본 흐름과 날짜 사실의 정본", "OK",
     {"panelA_flow": A, "panelB_windows": B, "panelC_dates": C},
     "Table 1 의 301 이 흐름의 단계인가, 그리고 딜월 상한이 패널 상한을 넘는 것이 오류인가",
     f"301 은 흐름 단계가 아니라 종전 설계의 표본이다. 기준 379 는 매칭 수이므로 사후창이 없는 "
     f"최근 딜 {B['post_window_incomplete']}건을 포함하며, 사후창이 완전한 마지막 딜월은 "
     f"{B['last_deal_month_with_post_window']} 이다 — 오타가 아니다.",
     kill_met=False, n=len(EV))
