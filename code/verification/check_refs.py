# -*- coding: utf-8 -*-
"""원고의 절·표·그림·부록 상호참조가 실제로 존재하는 대상을 가리키는지 확인한다.

표 재번호(리뷰 5 §27)처럼 **수치를 바꾸지 않는 변경**은 verify_draft.py 를 그대로 통과한다.
그래서 참조 축은 따로 둔다. (설계 선례: P-016 `06_code/wp13_check_refs.py`)

사용: python3 build/check_refs.py [원고경로 ...]
종료코드 1 = 미해결 참조 존재.
"""
import os, re, sys
D = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(D)

def headings(path):
    return [ln.rstrip() for ln in open(path, encoding="utf-8") if ln.startswith("#")]

def collect():
    """존재하는 대상 집합을 만든다."""
    secs, subsecs, apps, tabs, atabs, figs, afigs = set(), set(), set(), set(), set(), set(), set()
    for h in headings(f"{P}/manuscript/DRAFT_v4_0.md"):
        m = re.match(r"##\s+(\d+)\.\s", h)
        if m: secs.add(m.group(1))
        m = re.match(r"###\s+(\d+\.\d+)\s", h)
        if m: subsecs.add(m.group(1))
        m = re.match(r"##\s+Appendix\s+([A-Z])\.", h)
        if m: apps.add(m.group(1))
    for h in headings(f"{P}/manuscript/ONLINE_APPENDIX_v4_0.md"):
        m = re.match(r"##\s+([A-Z])\.\s", h)
        if m: apps.add(m.group(1))
        m = re.match(r"###\s+([A-Z]\.\d+[a-z]?)\s", h)
        if m: apps.add(m.group(1))
    for h in headings(f"{P}/exhibits/tables.md"):
        m = re.match(r"###\s+Table\s+(\d+)\.", h)
        if m: tabs.add(m.group(1))
    if os.path.exists(f"{P}/exhibits/appendix_tables.md"):
        for h in headings(f"{P}/exhibits/appendix_tables.md"):
            m = re.match(r"###\s+Appendix Table\s+([A-Z]\.[\d.]+)\.", h)
            if m: atabs.add(m.group(1))
    for f in os.listdir(f"{P}/exhibits"):
        m = re.match(r"figure(A?\d+)_.*\.png$", f)
        if m: (afigs if m.group(1).startswith("A") else figs).add(m.group(1))
    return secs, subsecs, apps, tabs, atabs, figs, afigs

SECS, SUBSECS, APPS, TABS, ATABS, FIGS, AFIGS = collect()
print(f"대상: 절 {sorted(SECS)} · 소절 {len(SUBSECS)}개 · 부록 {sorted(APPS)}")
print(f"      표 {sorted(TABS, key=int)} · 부록표 {sorted(ATABS)} · 그림 {sorted(FIGS)} + {sorted(AFIGS)}")

PATS = [
    (re.compile(r"Appendix Table\s+([A-Z]\.\d+(?:\.\d+)*)"), lambda v: v in ATABS,            "부록표"),
    (re.compile(r"Appendix (?:Figure )?([A-Z]\d+)\b"),      lambda v: v in AFIGS,            "부록그림"),
    (re.compile(r"Appendix\s+([A-Z]\.\d+[a-z]?)\b"),        lambda v: v in APPS,             "부록절"),
    (re.compile(r"Appendix\s+([A-Z])\b(?![.\w])"),          lambda v: v in APPS,             "부록"),
    (re.compile(r"\bTable\s+(\d+)\b"),                      lambda v: v in TABS,             "표"),
    (re.compile(r"\bFigure\s+(\d+)\b"),                     lambda v: v in FIGS,             "그림"),
    (re.compile(r"\bSections?\s+(\d+\.\d+)\b"),             lambda v: v in SUBSECS,          "소절"),
    (re.compile(r"\bSections?\s+(\d+)\b(?!\.\d)"),          lambda v: v in SECS,             "절"),
]
# ─────────────────────────────────────────────────────────────────────────────
# 그림 캡션 검사 (2026-08-27 추가). 캡션은 `make_submission.py` 의 FIGS/FIGCAP 코드 안에만 있어
# 수치 검증기도 이 해석기도 보지 못했고, 그 결과 구판 캡션(Figure 1 중복·구 결과변수·"Table 4")이
# 네 라운드를 살아남았다. 여기서 (a) 캡션 번호 중복·누락, (b) 캡션 제목 ↔ make_exhibits 의 실제
# 그림 제목 ↔ FIGS 제목 일치, (c) 캡션 본문의 상호참조 해석, (d) 순서를 검사한다.
# ─────────────────────────────────────────────────────────────────────────────
import ast
SUB_PY = os.environ.get("P014_SUBMISSION_PY", f"{D}/make_submission.py")   # 음성 대조용 override
EXH_PY = f"{D}/make_exhibits.py"

def _assign(src, name):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == name for t in node.targets):
            return ast.literal_eval(node.value)
    return None
def _norm(t): return re.sub(r"\s+", " ", t).strip().rstrip(".")
def exhibit_titles():
    """make_exhibits.py 의 suptitle/set_title 에 박힌 실제 그림 제목 → {번호: 제목}."""
    out = {}
    for node in ast.walk(ast.parse(open(EXH_PY, encoding="utf-8").read())):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") in ("suptitle", "set_title") and node.args:
            a = node.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                m = re.match(r"Figure ([A-Z]?\d+)\. (.*)", a.value.strip(), re.S)
                if m: out[m.group(1)] = _norm(m.group(2))
    return out
def check_captions():
    src = open(SUB_PY, encoding="utf-8").read()
    figs, figcap = _assign(src, "FIGS"), _assign(src, "FIGCAP")
    bad = []
    if not figs or not figcap: return [("FIGS/FIGCAP", "make_submission.py 에서 읽지 못함", "")]
    caps = re.findall(r"\*\*Figure ([A-Z]?\d+)\. (.*?)\*\*", figcap, re.S)
    nums = [n for n, _ in caps]
    for n in {x for x in nums if nums.count(x) > 1}: bad.append(("캡션 중복", f"Figure {n}", ""))
    files = {re.match(r"figure(A?\d+)_", f).group(1) for f in os.listdir(f"{P}/exhibits") if re.match(r"figure(A?\d+)_.*\.png$", f)}
    for n in sorted(files - set(nums)): bad.append(("캡션 누락", f"Figure {n}", "exhibits 에 그림은 있으나 FIGCAP 에 캡션 없음"))
    for n in sorted(set(nums) - files): bad.append(("그림 없음", f"Figure {n}", "FIGCAP 에 캡션은 있으나 exhibits 에 그림 없음"))
    real = exhibit_titles()
    figs_title = {re.match(r"Figure ([A-Z]?\d+)\. (.*)", t, re.S).group(1): _norm(re.match(r"Figure ([A-Z]?\d+)\. (.*)", t, re.S).group(2)) for _, t in figs}
    for n, t in caps:
        if n in real and _norm(t) != real[n]: bad.append(("제목 불일치(캡션↔그림)", f"Figure {n}", f"캡션 '{_norm(t)[:60]}' vs 그림 '{real[n][:60]}'"))
        if n in figs_title and _norm(t) != figs_title[n]: bad.append(("제목 불일치(캡션↔FIGS)", f"Figure {n}", f"FIGS '{figs_title[n][:60]}'"))
    key = lambda n: (0, int(n)) if n.isdigit() else (1, int(n[1:]))
    if nums != sorted(nums, key=key): bad.append(("순서", " ".join(nums), "본문 그림 → 부록 그림 순이어야 함"))
    fig_order = [re.match(r"figure(A?\d+)_", f).group(1) for f, _ in figs]
    if fig_order != sorted(fig_order, key=key): bad.append(("FIGS 순서", " ".join(fig_order), ""))
    for i, ln in enumerate(figcap.split("\n"), 1):          # 캡션 본문의 상호참조
        if ln.startswith("#"): continue
        for pat, ok, lab in PATS:
            for m in pat.finditer(ln):
                if not ok(m.group(1)): bad.append((f"캡션 참조 {lab}", m.group(0), ln.strip()[:90]))
    return bad

targets = sys.argv[1:] or [f"{P}/manuscript/DRAFT_v4_0.md", f"{P}/manuscript/DRAFT_v4_IRFA.md",
                           f"{P}/manuscript/DRAFT_v4_PBFJ.md", f"{P}/manuscript/ONLINE_APPENDIX_v4_0.md",
                           f"{P}/exhibits/tables.md", f"{P}/exhibits/appendix_tables.md"]
bad = 0
for t in targets:
    if not os.path.exists(t): print(f"  (없음) {t}"); continue
    hits = []
    inref = False
    for i, ln in enumerate(open(t, encoding="utf-8"), 1):
        if re.match(r"#+\s*References", ln): inref = True
        if inref or ln.startswith("#"): continue   # 표제 자체는 참조가 아니다
        seen = set()
        for pat, ok, lab in PATS:
            for m in pat.finditer(ln):
                key = (m.start(), m.group(1))
                if key in seen: continue
                seen.add(key)
                if not ok(m.group(1)): hits.append((i, lab, m.group(0), ln.strip()[:110]))
    bad += len(hits)
    print(f"  {os.path.basename(t):<26} 미해결 {len(hits)} {'✓' if not hits else '🔴'}")
    for i, lab, ref, ctx in hits[:40]: print(f"      L{i:<4} {lab} '{ref}'  — {ctx}")
cap = check_captions()
print(f"  {'FIGCAP (make_submission.py)':<26} 문제 {len(cap)} {'✓' if not cap else '🔴'}")
for kind, what, ctx in cap[:40]: print(f"      {kind} '{what}'  — {ctx}")
print(f"\n총평: 미해결 참조 {bad}건 · 캡션 문제 {len(cap)}건")
sys.exit(1 if (bad or cap) else 0)
