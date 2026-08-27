# -*- coding: utf-8 -*-
"""Build the three notebooks with stored outputs.

The table and figure cells are **cut verbatim from code/build/make_tables.py and make_exhibits.py** —
the same scripts that generated the exhibits in the paper — so the notebook cannot drift from the paper's
code. Each notebook ends with a consistency check against paper_exhibits/ that raises if anything differs.
Run from anywhere:  python code/build/make_nb_p014.py
"""
import os, re, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_notebooks import build
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(REPO); os.makedirs("notebooks", exist_ok=True)

def cut(path, marker):
    """Split a build script into (header, [block_source, ...]); drops the file-writing tail."""
    lines = open(path, encoding="utf-8").read().split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith(marker)]
    tail = next((i for i, l in enumerate(lines) if l.startswith("with open(")), len(lines))
    header = "\n".join(lines[:idx[0]])
    blocks = ["\n".join(lines[a:b]).rstrip() for a, b in zip(idx, idx[1:] + [tail])]
    return header, blocks

COMMON = ["Every number below is read from an aggregate result artifact in `../artifacts/` — the same files the paper's",
          "tables and figures were generated from. No licensed microdata is used or required (see `../DATA_ACCESS.md`).",
          "Outputs are stored in this notebook, so everything renders on GitHub without running anything.",
          "The code cells are cut verbatim from `../code/build/`; the last cell checks the result against `../paper_exhibits/`."]

# ───────────────────────── 01 tables ─────────────────────────
hdr, blocks = cut("code/build/make_tables.py", "# ─────────")
hdr = hdr[hdr.index("def st(p):"):]                     # keep helpers, drop the module docstring and path lines
SETUP_T = """import json, os, csv
ART = "../artifacts"
J = lambda f: json.load(open(os.path.join(ART, f + ".json"), encoding="utf-8"))
L = {r["claim_id"]: r for r in csv.DictReader(open(os.path.join(ART, "CLAIMS_LEDGER_v4.csv"), encoding="utf-8-sig"))}
print(f"claims ledger: {len(L)} rows · artifacts: {len([f for f in os.listdir(ART) if f.endswith('.json')])} JSON files")
""" + hdr
def tkey(b):
    m = re.search(r'^(T\[(\d)\]|AT\["F4\.1"\])\s*=', b, re.M)
    return ("T", int(m.group(2))) if m.group(2) else ("AT", "F4.1")
def ttitle(b):
    return re.search(r'### (Table \d\.[^\n]*|Appendix Table F\.4\.1\.[^\n]*)', b).group(1)
order = sorted(blocks, key=lambda b: (0, tkey(b)[1]) if tkey(b)[0] == "T" else (1, 0))
cells = [(["## Setup — helpers shared by all tables (verbatim from `make_tables.py`)"], SETUP_T)]
for b in order:
    kind, k = tkey(b); ref = f"T[{k}]" if kind == "T" else 'AT["F4.1"]'
    cells.append(([f"## {ttitle(b)}"], b + f"\n_md = {ref}\nprint({ref}.split(chr(10))[0])"))
CHECK_T = """def sections(path):
    body = open(path, encoding="utf-8").read().split("\\n", 3)[3]      # drop the generated-file header
    return [s.strip() for s in body.split("\\n\\n---\\n\\n")]
ref_main, ref_app = sections("../paper_exhibits/tables.md"), sections("../paper_exhibits/appendix_tables.md")
mine_main = [T[i].strip() for i in sorted(T)]; mine_app = [AT[k].strip() for k in sorted(AT)]
fails = 0
pairs = list(zip([f"Table {i}" for i in sorted(T)], mine_main, ref_main)) + list(zip([f"Appendix Table {k}" for k in sorted(AT)], mine_app, ref_app))
for lab, a, b in pairs:
    ok = a == b; fails += (not ok); print(f"{lab:<24} {'IDENTICAL' if ok else 'DIFFERS'}  ({len(a):,} chars)")
assert len(mine_main) == len(ref_main) == 7 and len(mine_app) == len(ref_app) == 1
assert fails == 0, f"{fails} table(s) differ from the paper"
print("\\nAll 8 tables are byte-identical to the paper's exhibits.")"""
cells.append((["## Consistency check — regenerated tables versus the paper's exhibits",
               "Byte-for-byte comparison with `../paper_exhibits/tables.md` and `appendix_tables.md`, the files attached",
               "to the manuscript. The cell raises if any table differs."], CHECK_T))
build("notebooks/01_main_tables.ipynb", "# Tables 1–7 and Appendix Table F.4.1", COMMON, cells)

# ───────────────────────── 02 figures ─────────────────────────
hdr, blocks = cut("code/build/make_exhibits.py", "# ════════════")
SETUP_F = """import json, os, csv, io
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ART, EX = "../artifacts", "../figures"; os.makedirs(EX, exist_ok=True)
J = lambda f: json.load(open(os.path.join(ART, f + ".json"), encoding="utf-8"))
L = {r["claim_id"]: r for r in csv.DictReader(open(os.path.join(ART, "CLAIMS_LEDGER_v4.csv"), encoding="utf-8-sig"))}
c = lambda cid, w="value": L[cid][w]
_figs = []
def SAVE(fig, n):
    \"\"\"Write PNG+PDF to ../figures (as make_exhibits.py does) and keep a preview for this notebook.\"\"\"
    for e in ("png", "pdf"): fig.savefig(f"{EX}/{n}.{e}", dpi=200, bbox_inches="tight")
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=110, bbox_inches="tight"); _figs.append(b.getvalue())
print("artifacts:", len([f for f in os.listdir(ART) if f.endswith(".json")]), "· ledger rows:", len(L))"""
TITLES = {"figure1_event_study": "Figure 1 — Hiring rises after the deal, and the response is concentrated in targets with low pre-deal hiring intensity",
          "figure2_turnover": "Figure 2 — The hiring response is not matched by net employment growth",
          "figure3_what_predicts": "Figure 3 — Pre-deal hiring state produces the clearest detectable heterogeneity",
          "figure4_volume_benchmark": "Figure 4 — Changes in monthly hiring patterns are largely accounted for by a volume-only benchmark",
          "figureA1_aggregation": "Appendix Figure A1 — Monthly no-hire measures attenuate under temporal aggregation",
          "figureA2_positioning": "Appendix Figure A2 — Where this paper sits"}
RANK = list(TITLES)
def fname(b): return re.search(r'SAVE\(fig, "(\w+)"\)', b).group(1)
order = sorted(blocks, key=lambda b: RANK.index(fname(b)))
cells = [(["## Setup (verbatim helpers from `make_exhibits.py`)"], SETUP_F)]
for b in order: cells.append(([f"## {TITLES[fname(b)]}"], b))
CHECK_F = """import hashlib
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
fails = 0
for n in %s:
    a, b = sha(f"../figures/{n}.png"), sha(f"../paper_exhibits/figures/{n}.png")
    ok = a == b; fails += (not ok); print(f"{n:<28} {'IDENTICAL' if ok else 'DIFFERS'}  sha256 {a[:16]}")
assert fails == 0, f"{fails} figure(s) differ from the paper"
print("\\nAll 6 figures are byte-identical to the paper's figure files.")""" % json.dumps(RANK)
cells.append((["## Consistency check — regenerated figures versus the paper's figure files",
               "SHA-256 of each PNG just written to `../figures/` against the file attached to the manuscript in `../paper_exhibits/figures/`."], CHECK_F))
build("notebooks/02_figures.ipynb", "# Figures 1–4 and Appendix Figures A1–A2", COMMON, cells)

# ───────────────────────── 03 traceability ─────────────────────────
C1 = """import json, os, csv
ART = "../artifacts"
rows = list(csv.DictReader(open(os.path.join(ART, "CLAIMS_LEDGER_v4.csv"), encoding="utf-8-sig")))
def resolve(o, path):
    for k in [k for k in path.split(".") if k]:
        o = o[int(k)] if isinstance(o, list) else o[k]
    return o
exact = derived = mismatch = missing = 0; bad = []
for r in rows:
    f = os.path.join(ART, os.path.basename(r["source_json"]))
    if not os.path.exists(f):
        missing += 1; bad.append((r["claim_id"], "artifact not in repository", os.path.basename(r["source_json"]))); continue
    try: o = resolve(json.load(open(f, encoding="utf-8")), r["json_path"])
    except Exception: mismatch += 1; bad.append((r["claim_id"], "path", r["json_path"])); continue
    if isinstance(o, (dict, list)): derived += 1; continue
    try: ok = abs(float(r["value"]) - float(o)) <= max(5e-5, abs(float(o)) * 1e-6)
    except Exception: ok = str(o) == r["value"]
    if ok: exact += 1
    else: mismatch += 1; bad.append((r["claim_id"], "value", f"{r['value']} vs {o}"))
print(f"ledger rows {len(rows)} · exact {exact} · derived {derived} · mismatch {mismatch} · artifact missing {missing}")
for b in bad: print("  ", b)
from collections import Counter
print("\\nrows by status:", dict(Counter(r["path_status"] for r in rows)))
print("SUPERSEDED rows point at retired designs; they are kept so that every number ever reported stays traceable.")
assert mismatch == 0"""
C2 = """import hashlib, re
man = open("../ARTIFACT_MANIFEST.md", encoding="utf-8").read()
listed = dict(re.findall(r"^\\| `([^`]+\\.json)` \\| `([0-9a-f]{16})` \\|", man, re.M))
bad = [f for f, h in listed.items() if hashlib.sha256(open(os.path.join(ART, f), "rb").read()).hexdigest()[:16] != h]
print(f"artifacts listed {len(listed)} · hash mismatches {len(bad)}"); assert not bad
extra = sorted(set(f for f in os.listdir(ART) if f.endswith(".json")) - set(listed)); print("unlisted artifacts:", extra); assert not extra"""
C3 = """import re
use = {}
for nb, lab in [("01_main_tables.ipynb", "tables"), ("02_figures.ipynb", "figures")]:
    src = "\\n".join("".join(c["source"]) for c in json.load(open(nb, encoding="utf-8"))["cells"] if c["cell_type"] == "code")
    for a in sorted(set(re.findall(r'J\\("([A-Za-z0-9_]+)"\\)', src))): use.setdefault(a, set()).add(lab)
led = {}
for r in rows:
    k = os.path.basename(r["source_json"]).replace(".json", ""); led[k] = led.get(k, 0) + 1
print(f"{'artifact':<28}{'read directly by':<20}{'ledger rows'}")
for a in sorted(set(use) | set(led)): print(f"{a:<28}{'/'.join(sorted(use.get(a, []))) or '—':<20}{led.get(a, 0)}")"""
cells = [(["## Every claim in the ledger resolves to its artifact",
           "`CLAIMS_LEDGER_v4.csv` lists each reported number with the artifact file and JSON path it comes from.",
           "This cell follows every path and compares the stored value with the artifact. Rows whose path points at a",
           "whole object (a sum or a range computed from several fields) are reported as *derived*. The one artifact",
           "deliberately excluded from the repository (`I05.json`, see `ARTIFACT_MANIFEST.md`) is reported as missing."], C1),
         (["## Artifact integrity", "Recomputes SHA-256 for every artifact and compares with `../ARTIFACT_MANIFEST.md`."], C2),
         (["## Which artifacts feed which exhibits"], C3)]
build("notebooks/03_traceability.ipynb", "# Traceability — ledger ↔ artifacts ↔ exhibits", COMMON[:2], cells)
print("notebooks built")
