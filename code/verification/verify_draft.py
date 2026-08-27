# -*- coding: utf-8 -*-
"""원고 검증 — 본문 산문에 등장하는 모든 수치가 원장(또는 그 파생형)에 존재하는지 확인.

초판은 `C##` 인용 태그를 따라갔으나, 저널 포맷으로 옮기며 산문의 태그를 표 참조로 바꿨다.
이제는 **산문의 숫자 자체**를 원장과 대조한다 — 태그 없이도 조작을 잡을 수 있어야 한다.
표(`exhibits/tables.md`)는 원장에서 기계 생성되므로 별도 대조가 불필요하다.
"""
import csv, re, os, sys, json
D = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(D)
PKG = os.path.dirname(P); H = os.path.join(PKG, "harness30", "out")
L = list(csv.DictReader(open(f"{P}/CLAIMS_LEDGER_v4.csv", encoding="utf-8-sig")))

# ─────────────────────────────────────────────────────────────────────────────
# 폐기 처리. 전역 풀 방식의 약점은 **낡은 값이 풀 어딘가에 남아 있으면 통과한다**는 것이다.
# (실제로 원고의 hazard 1.2512·IQR 0.1593·공유대조군 1328·상대고용 -0.0564 가 이렇게 통과했다.)
# 두 겹으로 막는다: 산출물 단위 제외 + 값 단위 차단.
# ─────────────────────────────────────────────────────────────────────────────
RETIRED_ARTIFACTS = {                     # 원장에 살아있는 행이 하나도 없는 산출물
    "I15": "hazard triple-interaction; 표 7 의 사양 (4) 로 대체 (1.2881)",
    "I49": "대조군 공유 진단 (301건 설계); I-65 로 대체 (286건)",
    "I55_employment_horizons": "상대고용 경로 (inline 산출); I-67 로 대체",
}
# 값 단위 차단 — 폐기됐지만 다른 산출물에도 우연히 존재할 수 있는 수치
# 손으로 관리하는 목록은 다음에 또 놓친다. 아래 MANUAL 은 seed 일 뿐이고,
# 실제 차단 목록은 원장의 SUPERSEDED 행에서 **자동 도출**한다 (build_stale() 참조).
MANUAL_STALE = {
    1.2512: "hazard, 구설계 (I-15) → 1.2881",
    0.1593: "IQR 효과, 301건 설계 (I-47) → 0.2005 (I-65)",
    1328:   "distinct 대조군, 301건 설계 (I-49) → 1144 (I-65)",
    0.9473: "1회사용 비중, 301건 설계 (I-49) → 0.95 (I-65)",
    0.5624: "대조군 클러스터 gradient, 301건 설계 (I-49) → 0.7101 (I-65)",
    0.5627: "1:1 배정 gradient, 301건 설계 (I-49) → 0.7036 (I-65)",
    0.2128: "이직 gradient, 구설계 (I-55) → 0.1788 (I-57)",
    0.2534: "이직 gradient p, 구설계 (I-55) → 0.2349 (I-57)",
    0.2807: "총유량 gradient, 구설계 (I-55)",
    0.0834: "순증 gradient, 구설계 (I-55)",
    0.0564: "상대고용 +12m, 구설계 → 0.0316 (I-67)",
    0.0208: "상대고용 +24m, 구설계 → 0.1616 (I-67)",
    0.1199: "상대고용 +36m, 구설계 → 0.2092 (I-67)",
}

def build_stale(rows):
    """원장의 SUPERSEDED 행에서 차단 수치를 자동 도출한다.

    안전장치 두 겹: (a) 살아있는 행에도 등장하는 값은 제외 — 우연 일치를 오탐하지 않는다.
    (b) 소수 3자리 이상이거나 4자리 이상 정수인 '특징적인' 값만 쓴다 — 2, 12, 286 같은
    흔한 수를 차단하면 검증기가 쓸모없어진다."""
    live, live_raw = set(), []
    for r in rows:
        if r["path_status"] == "SUPERSEDED": continue
        for t in [r["value"], r["n"]] + re.findall(r"-?\d+(?:\.\d+)?", r["ci95"] or ""):
            try: x = abs(float(t))
            except (TypeError, ValueError): continue
            live.add(round(x, 6)); live_raw.append(x)
    # 살아있는 값이 **반올림하면** 폐기값과 같아지는 경우도 살아있는 것으로 본다.
    # (예: 새 사전추세 상한 0.0854 → 원고에 0.085 로 쓰이면 폐기된 0.085 와 충돌한다.)
    def collides_with_live(v, nd):
        return any(round(x, nd) == v for x in live_raw)
    def distinctive(t):
        try: x = abs(float(t))
        except (TypeError, ValueError): return None
        st = str(t).strip().lstrip("-")
        frac = len(st.split(".")[1]) if "." in st else 0
        if not (frac >= 3 or (frac == 0 and x >= 1000)): return None
        return round(x, 6), frac
    out = {}
    for r in rows:
        if r["path_status"] != "SUPERSEDED": continue
        why = f"원장 {r['claim_id']} 폐기 — {r['claim'][:70]}"
        for t in [r["value"]] + re.findall(r"-?\d+(?:\.\d+)?", r["ci95"] or ""):
            d = distinctive(t)
            if d is None: continue
            v, nd = d
            if v in live or collides_with_live(v, nd): continue
            out.setdefault(v, why)
    return out

# 허용 수치 풀: 원장의 값·CI 경계 + 그 백분율/pp 변환 + I-36 회귀표 수치
POOL = {}
from decimal import Decimal, ROUND_HALF_UP
def add(v, src):
    try: x = float(v)
    except (TypeError, ValueError): return
    if not (abs(x) < 1e12): return
    for f in (1, 100, -1, -100):
        y = x * f
        for nd in range(0, 7):
            POOL.setdefault(round(y, nd), src)                      # 파이썬 반올림
            try:  # half-up 반올림 (원고에서 흔히 쓰는 형태)
                POOL.setdefault(float(Decimal(repr(y)).quantize(Decimal(1).scaleb(-nd), ROUND_HALF_UP)), src)
            except Exception: pass
LIVE = [r for r in L if r["path_status"] != "SUPERSEDED"]
DEAD = [r for r in L if r["path_status"] == "SUPERSEDED"]
STALE_VALUES = dict(MANUAL_STALE); STALE_VALUES.update(build_stale(L))
print(f"폐기 차단 수치: 수동 {len(MANUAL_STALE)}종 + 원장 자동도출 "
      f"{len(STALE_VALUES)-len(MANUAL_STALE)}종 = {len(STALE_VALUES)}종")
for r in LIVE:
    add(r["value"], r["claim_id"])
    for m in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", r["ci95"] or ""): add(m, r["claim_id"] + ".ci")
    add(r["n"], r["claim_id"] + ".n")
# 산문은 저장된 산출 JSON 의 어떤 수치든 인용할 수 있다. 원장 소스 JSON 전체를 수확한다.
def harvest(o, src):
    if isinstance(o, dict):
        for k, v in o.items(): harvest(v, src)
    elif isinstance(o, (list, tuple)):
        for v in o: harvest(v, src)
    elif isinstance(o, (int, float)) and not isinstance(o, bool): add(o, src)
# 부록은 원장에 없는 경로도 인용한다. out 의 **모든** 산출 JSON 을 수확한다.
SRC = sorted(x[:-5] for x in os.listdir(H) if x.startswith("I") and x.endswith(".json"))
_skipped = [f for f in SRC if f in RETIRED_ARTIFACTS]
SRC = [f for f in SRC if f not in RETIRED_ARTIFACTS]
for f in _skipped: print(f"  (폐기 제외) {f}: {RETIRED_ARTIFACTS[f]}")
for f in SRC:
    try: harvest(json.load(open(f"{H}/{f}.json", encoding="utf-8")).get("estimates", {}), f)
    except FileNotFoundError: print(f"  (경고) {f}.json 없음")
# 동결 산출 중 원고가 인용하는 것 (재선택 통제실험 = 방법론 기여)
FROZEN = [os.path.join(PKG, "..", "shared", "outputs", "idea014_h41_2026-08-24",
                       "h41_causal_gap.json")]
for fp in FROZEN:
    fp = os.path.abspath(fp)
    if os.path.exists(fp):
        harvest(json.load(open(fp, encoding="utf-8")), os.path.basename(fp)); SRC.append(fp)
    else:
        print(f"  (경고) 동결산출 없음: {fp}")
print(f"수확한 산출 JSON: {len(SRC)}개 → 허용 수치 {len(POOL):,}종")

i36 = json.load(open(f"{H}/I36.json", encoding="utf-8"))["estimates"]
for k, sp in i36["specs"].items():
    for t, v in sp["terms"].items():
        for f in ("coef", "se", "HR"): add(v[f], f"I36{k}.{t}.{f}")
        for x in v["ci"]: add(x, f"I36{k}.{t}.ci")
    for f in ("pseudo_r2_mcfadden", "n_cells", "n_clusters", "n_firm_months"): add(sp[f], f"I36{k}.{f}")
add(i36["tercile_cut_inaction"], "I36.cut"); add(i36["pressure_cut_years"], "I36.press")
# 제도·표본 서술 상수 (Table 1·§2 에 근거, 원장 밖)
for v, why in [(752,"sample"),(379,"sample"),(1895,"sample"),(48853,"sample"),(79,"median emp"),
               (2021,"median year"),(5,"min emp / 5% rule"),(9,"contribution rate"),(0.26,"pre-deal zero share"),
               (0.055,"pre i"),(0.738,"pre p"),(0.083,"tercile cut"),(0.333,"tercile cut"),
               (112,"T1 n"),(92,"T3 n"),(0.0099,"pre gap"),(200,"placebo draws"),(999,"bootstrap"),
               (62,"confirmed n"),(209,"stake n"),(2000,"permutations"),(6.6,"wage bound"),
               (20,"wage bound"),(10,"equiv bound"),(15,"equiv bound"),(4.6,"equiv bound"),
               (0.046,"equiv bound"),(16,"hazard pct"),(60.8,"ext share"),(0.005,"RI resolution"),
               (18,"age"),(59,"age"),(0.15,"equiv"),(0.10,"equiv"),(3,"years"),(12,"months"),
               (24,"months"),(13,"months"),(1,"misc"),(2,"misc"),(4,"misc")]:
    add(v, why)

TARGETS = sys.argv[1:] or ["DRAFT_v4_0.md", "DRAFT_v4_IRFA.md", "DRAFT_v4_PBFJ.md",
                           "ONLINE_APPENDIX_v4_0.md"]
ALLBAD = []; ALLSTALE = []
for _t in TARGETS:
  txt = open(f"{P}/manuscript/{_t}", encoding="utf-8").read()
  prose = txt.split("## References")[0]
  prose = re.sub(r"^#{1,6} .*$", "", prose, flags=re.M)   # 제목행 제외 (절 번호)
  prose = re.sub(r"\|.*\|", "", prose)                    # 표 행 제외 (표는 기계 생성)
  prose = prose.replace("\u2212", "-").replace("\u2013", "-")   # 유니코드 마이너스/en-dash 정규화
  prose = re.sub(r"(?<=\d),(?=\d{3}\b)", "", prose)        # 천단위 쉼표 제거
  prose = re.sub(r"\((?:19|20)\d\d[a-z]?\)", "", prose)   # 인용 연도 제외
  prose = re.sub(r"\b(?:19|20)\d\d\b", "", prose)         # 연도 제외
  prose = re.sub(r"Section \d(?:\.\d)?|Table \d|Figure \d[ab]?|column \(?\d\)?|Panel [A-H]|"
                 r"NBER Working Paper \d+|US\$\d+", "", prose)
  nums = sorted({round(float(m), 6) for m in re.findall(r"-?\d+(?:\.\d+)?", prose)})
  bad = [n for n in nums if n not in POOL]
  stale = [(n, STALE_VALUES[n]) for n in nums if n in STALE_VALUES]
  print(f"  {_t:<26} 고유 수치 {len(nums):>4}개 → 미확인 {len(bad)}개 "
        f"{'✓' if not bad else '🔴 ' + str(bad[:6])}")
  for n, why in stale: print(f"      🔴 폐기 수치 {n} — {why}")
  ALLBAD += [(_t, n) for n in bad]; ALLSTALE += [(_t, n, why) for n, why in stale]
print(f"\n제외한 폐기 산출물 {len(_skipped)}개 · 폐기 원장행 {len(DEAD)}개 "
      f"(원장 {len(LIVE)}행 유효)")
print(f"총평: 검증 {len(TARGETS)}개 문서 · 미확인 수치 {len(ALLBAD)}건 · 폐기 수치 {len(ALLSTALE)}건")
sys.exit(1 if (ALLBAD or ALLSTALE) else 0)
