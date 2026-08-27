#!/usr/bin/env bash
# H30 하네스 재현 — 완료 경로 순차 실행. 각 스크립트가 out/I##.json 을 덮어쓴다.
# 사용: bash run_harness30.sh [경로ID ...]   (인자 없으면 전부)
set -euo pipefail
cd "$(dirname "$0")"
ALL=(i01_timeagg i02_hazard i03_ss_band i04_performance i04b_performance_v2 i04c_valueadded
     i05a_exit_refine i05_exit_reversal i06_notyet_anatomy i11_honestdid i14_shareholder_dose
     i15_fund_pressure i16_dealtype i17_gp_style i25_pre_inertia i31_inertia_placebo)
TARGETS=("${@:-${ALL[@]}}")
for s in "${TARGETS[@]}"; do
  echo "=== $s ==="
  /usr/bin/time -f "  [RSS %M KB  wall %E]" python3 "${s}.py"
done
echo "완료. 산출: out/*.json · 원장: ../manuscript/RESULTS_LEDGER_v4.md §34"
