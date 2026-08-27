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
print(f"\n총평: 미해결 참조 {bad}건")
sys.exit(1 if bad else 0)
