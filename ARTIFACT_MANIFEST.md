# Artifact manifest

Aggregate result files in `artifacts/`. Each is the JSON written by one pipeline script in `code/pipeline/` (the `code` and `sha256_16` fields inside each file identify that script and its hash at run time). Titles are the pipeline's own (Korean) labels. `Ledger rows` counts the claims in `CLAIMS_LEDGER_v4.csv` sourced from the file.

| Artifact | sha256₁₆ | Bytes | Pipeline script | Title | Status | Ledger rows |
|---|---|---:|---|---|---|---:|
| `I01.json` | `5780c45ca931b2b9` | 3,538 | `i01_timeagg.py` | 시간집계 정리 검정 | GO | 1 |
| `I02.json` | `582b04b1e3a78e04` | 6,518 | `i02_hazard.py` | hazard 전환 (이산시간 cloglog) | GO | 1 |
| `I03.json` | `db70361a6d437c6f` | 3,985 | `i03_ss_band.py` | (S,s) 무행동 밴드폭 추정 | KILL | 2 |
| `I04.json` | `7f87b84efb4972a2` | 11,468 | `i04_performance.py` | 실적 앵커 | PARTIAL | 0 |
| `I04b.json` | `94f17e7e266ef4b9` | 14,402 | `i04b_performance_v2.py` | 실적 앵커 v2 (분모오염·커버리지·지평 수정) | PARTIAL | 0 |
| `I04c.json` | `c18f7bbc062046d1` | 14,155 | `i04c_valueadded.py` | 실적 앵커 v3 (부가가치 + 등가성) | PARTIAL | 2 |
| `I05a.json` | `55fced24357fb7c3` | 1,500 | `i05a_exit_refine.py` | exit 후보 정제 (스폰서 단위 추적) | PARTIAL | 0 |
| `I06.json` | `27038f2628daacaa` | 6,699 | `i06_notyet_anatomy.py` | not-yet-treated 붕괴 해부 | PARTIAL | 0 |
| `I11.json` | `93e2d0fb8ebfb26c` | 3,999 | `i11_honestdid.py` | HonestDiD breakdown value 승격 | PARTIAL | 1 |
| `I14.json` | `6b8f8f3cba88c4ac` | 7,095 | `i14_shareholder_dose.py` | 주주명부 진입시점 검증 + 지분율 dose | GO | 2 |
| `I15.json` | `0d9f102eb6cafa73` | 3,862 | `i15_fund_pressure.py` | GP 펀드 소진압력 (선택 가설 반증 검정) | superseded — hazard triple interaction; Table 7 uses specification (4) of Appendix Table F.4.1 | 1 |
| `I16.json` | `521afc447557117e` | 6,782 | `i16_dealtype.py` | 딜유형 대비 — 통제권 vs 자본 | KILL | 1 |
| `I17.json` | `3b988ff9a3a8c37c` | 2,686 | `i17_gp_style.py` | GP 고정효과 / 스폰서 스타일 | KILL | 2 |
| `I19.json` | `cb16a708ea98533e` | 5,261 | `i19_succession.py` | 비-PE 지배구조 변화 대조 (최대주주 변경) | GO | 0 |
| `I19b.json` | `e217efd628db4388` | 4,433 | `i19b_own_subtypes.py` | OWN 하위유형 + positive control | PARTIAL | 0 |
| `I19c.json` | `48fd8d451913ed48` | 3,005 | `i19c_dose_gradient.py` | 지배권 이전 dose 기울기 + PE 위치 | PARTIAL | 0 |
| `I21.json` | `6b3c72ab246e5e67` | 3,006 | `i21_cash_lead.py` | 재무여유 매개 (현금 선행성) | KILL | 1 |
| `I22.json` | `dc45a7e680142d59` | 4,115 | `i22_wage_structure.py` | 임금구조 (서사 판별) | GO | 2 |
| `I25.json` | `0f8c5b0461ccfac8` | 6,732 | `i25_pre_inertia.py` | 사전 관성수준 조절 (평균회귀 차단) | GO | 1 |
| `I31.json` | `0a9d85b94070b291` | 6,178 | `i31_inertia_placebo.py` | 고관성 조건부 위약 + 영구/일시 관성 분해 | GO | 2 |
| `I32.json` | `90c4b64dbca095f2` | 6,628 | `i32_decay.py` | 효과 감쇠 (1회성 리뷰 vs 영구 체제변화) | PARTIAL | 3 |
| `I33.json` | `603470c6b778cad1` | 4,231 | `i33_linchpin.py` | 핵심축 검증 — 수준 vs 빈도 식별력 | GO | 2 |
| `I34.json` | `af60923e4af98d4f` | 2,656 | `i34_margin_decomp.py` | 외연/내연 마진 분해 (비율 대체) | PARTIAL | 0 |
| `I35.json` | `c9f1c40df4def371` | 4,747 | `i35_canonical.py` | 정본 수치 전수 재계산 | PARTIAL | 11 |
| `I36.json` | `9d23770592a7a7de` | 6,579 | `i36_regression_table.py` | hazard 회귀표 (중첩 사양 + 적합통계) | GO | 0 |
| `I37.json` | `951d1151ed939593` | 2,685 | `i37_balance.py` | 균형표 (Table 1 Panel D) | GO | 0 |
| `I38.json` | `424a138113200850` | 4,321 | `i38_excess_zeros.py` | 빈도의 기계성 검정 (리뷰 Major Comment 1) | KILL | 4 |
| `I39.json` | `23d61216468e66e1` | 4,206 | `i39_spell_benchmark.py` | spell·집중도 벤치마크 (I-38 완성) | KILL | 3 |
| `I40.json` | `cf1157fa0119ef38` | 4,085 | `i40_salvage.py` | 조절자 천장효과 검정 + 36개월 집중도 | GO | 9 |
| `I41.json` | `b06cc9df9d81cea7` | 3,307 | `i41_moderator_defense.py` | 조절자 방어 (리뷰 §5.4) | GO | 8 |
| `I41_RECLASS.json` | `810417eed5ad035d` | 3,704 | `` |  |  | 0 |
| `I42.json` | `931e5464a2641c0d` | 1,841 | `i42_placebo_lograte.py` | Δlog 채용률 고관성 조건부 위약 (최종 판별) | GO | 8 |
| `I43.json` | `05e25cfb4ea2f29c` | 4,900 | `i43_invariance_lograte.py` | 불변성 재계산 (Δlog 채용률) | PARTIAL | 6 |
| `I44.json` | `7e893dccc627d316` | 5,795 | `i44_state_variable.py` | 상태변수 방어 (I-41 Panel E 정체 규명) | GO | 11 |
| `I45.json` | `ca7e0e559cd38c48` | 6,598 | `i45_power_invariance.py` | §8 검정력 재구성 (예측력 대결 + 공변량 조정) | PARTIAL | 18 |
| `I46.json` | `0c3065edbfbadd7b` | 9,936 | `i46_state_vs_volume.py` | 휴면 vs 사전 물량 — 동일 비겹침 창 경마 (리뷰3 MC1) | PARTIAL | 1 |
| `I47.json` | `f772a3eba5697770` | 10,732 | `i47_state_final.py` | 상태변수 확정 (FWL 정합 조정 + 위약) | GO | 7 |
| `I48.json` | `30cab17b5d6b8f25` | 3,782 | `i48_construct_validity.py` | 결과대상 construct validity + 표본 flow (리뷰3 MC3) | PARTIAL | 18 |
| `I49.json` | `5f41571a977336b3` | 5,083 | `i49_reuse_and_finance.py` | 대조 재사용 감사 + 사전 재무제약 (리뷰3 §9.4·§8-4) | superseded — shared-control diagnostic on the 301-event design; replaced by I65 (286 events) | 6 |
| `I50.json` | `9ae2ca62aea4bc0c` | 5,738 | `i50_power.py` | 검정력 제고 — 풀링 ANCOVA · PPML · 강건추정 · 암묵채용 | PARTIAL | 0 |
| `I50_panelH.json` | `17ee8c96eb8ea26e` | 349 | `` |  |  | 0 |
| `I51.json` | `3f331e65e54ffe1a` | 2,391 | `i51_spec_adjudication.py` | 사양 판별 — 차분 강제 vs 자유 lag, 처치와 위약을 같은 사양으로 | PARTIAL | 3 |
| `I52.json` | `83b5474d8bb49853` | 4,655 | `i52_headline_final.py` | 헤드라인 확정 — 처치−위약 gradient 대비 | GO | 2 |
| `I53.json` | `1f8ed3bf680363e8` | 2,938 | `i53_randomization.py` | 무작위화 추론 — 위약 분포를 귀무로 직접 사용 | GO | 21 |
| `I54.json` | `5c94d61cfabeeca3` | 3,319 | `i54_limits.py` | 남은 한계 보완 — 2차 결과대상 · 층화 RI · RI 기준 검정력 | GO | 3 |
| `I55.json` | `338ac07afe0df07f` | 3,550 | `i55_reallocation.py` | 재배치 가설 — 채용·이직·총유량의 상태 gradient | superseded design (pre-state-balancing); kept for the record | 8 |
| `I55_employment_horizons.json` | `6fa4b2cab9e6b5d0` | 943 | `(inline, 원장 §41-3)` | 고용 gradient — rel_log 지평별 | superseded — inline computation; replaced by I67 | 3 |
| `I56.json` | `3fc5778db1b3136a` | 4,915 | `i56_efficiency.py` | 검정력 제고 — 설계 교정(상태균형·거울위약) + 효율 개선 | GO | 10 |
| `I57.json` | `ef478131c5c069ca` | 3,991 | `i57_reallocation2.py` | 두 번째 추정량(hazard) + 재배치 쌍대비 | GO | 19 |
| `I57_levels.json` | `5d1968ab781ff148` | 932 | `` | 수준 결과대상 회계 정합 | OK | 4 |
| `I58.json` | `8e2e54ce180fab92` | 2,509 | `i58_design_audit.py` | 설계 교정 감사 — 상태균형 매칭의 이득 검증 | GO | 19 |
| `I58_control_contamination.json` | `01d679757aa751c7` | 659 | `` | 반사실 오염 진단 | OK | 6 |
| `I59.json` | `530a4c2d2a1c8af5` | 1,126 | `` | 분모 오염 검정 (채용건수 vs 채용률) | OK | 8 |
| `I60.json` | `4613c9d41fff2fc1` | 2,275 | `i60_speccurve.py` | 교정 설계 사양곡선 | GO | 24 |
| `I61.json` | `fbd921816ffd5a02` | 3,720 | `i61_gradient_pretrend.py` | 상태별 사전추세 (리뷰4 §8 · 리뷰3 MC2-3) | GO | 9 |
| `I62.json` | `2cc8309e0f530966` | 3,274 | `i62_power3.py` | 검정력 라운드 3 — 결합검정·사전추세 기울기·매칭 변형 | GO | 5 |
| `I63.json` | `45c1d11886ffd02b` | 3,214 | `i63_sample_expansion.py` | 표본 확대 — 탈락 계측과 회수 레버 | GO | 7 |
| `I64.json` | `ed1df01b02ecb3f4` | 1,914 | `i64_pretrend_honest.py` | 사전추세 — 12개월 해상도 · 위약 상대 RI · gradient HonestDiD | GO | 8 |
| `I65.json` | `3da372c82a53ce5d` | 2,832 | `i65_bootci_reuse.py` | 교정 설계 gradient 의 부트스트랩 CI 와 대조군 공유 진단 | GO | 17 |
| `I66.json` | `e0338c3d7cd2892d` | 2,950 | `i66_pretrend_zeros.py` | 비중첩 사전추세와 0 채용 창을 살린 강건성 | PARTIAL | 14 |
| `I67.json` | `67e86ce7fe5489c8` | 1,963 | `i67_emp_horizons.py` | 교정 설계에서의 상대고용 경로 | GO | 9 |
| `I68.json` | `3b75315a55917528` | 3,316 | `i68_hiring_rate_es.py` | 분기 채용률 이벤트 경로 (Figure 1a 용) | GO | 8 |
| `I69.json` | `9eea16a0120a0bda` | 2,230 | `i69_sample_flow.py` | 표본 흐름과 날짜 사실의 정본 | OK | 4 |
| `h41_causal_gap.json` | `31b201a9dd5b8680` | 13,913 | `` |  |  | 0 |

## Excluded on purpose

- `I05.json` (exit reversibility) lists business-registration numbers of the exit sample and is **not** in this repository; the appendix text that cites it remains traceable through the ledger's `source_json` column, and the file is available to editors and referees on request.
- All firm-level derived files (business numbers, monthly headcounts, per-firm outcomes) are excluded — they derive from the licensed pension register.

`CLAIMS_LEDGER_v4.csv` is the claim-level ledger: one row per reported number, with value, interval, sample size, artifact, JSON path, generating script and its hash. `notebooks/03_traceability.ipynb` resolves every row.
