# -*- coding: utf-8 -*-
"""H30 하네스 공용 — h39_common 재사용 + 표준 산출 계약.

처치표본 정본: shared/data/processed/p014_treated_sample_v2_expanded.csv (752행, 매칭진입 379).
PitchBook 재구성 금지 — h39_common.deals()가 정본 경로를 읽는다.
"""
import os, sys, json, time, hashlib, inspect
import numpy as np

BASE = os.environ.get("P014_BASE", "/path/to/project-root")   # project root holding shared/data (licensed; see DATA_ACCESS.md)
OUT  = f"{BASE}/P014_upgrade_package/harness30/out"
SEED = 42
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # h39_common.py sits alongside in this repository

from h39_common import (load, deals, build, attach, summ, tercile,
                        widx, flow, dflow, rel_log, pi_parts, qci, NB)   # noqa: F401

_T0 = time.time()


def sha16(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def emit(iid, title, status, estimates, prediction, verdict, kill_met, n=None, extra=None):
    """표준 산출 계약. out/I##.json 을 쓰고 요약을 stdout에 찍는다."""
    src = os.path.abspath(inspect.stack()[1].filename)
    rec = {
        "id": iid, "title": title, "status": status, "n": n,
        "estimates": estimates, "prediction": prediction,
        "verdict": verdict, "kill_met": kill_met,
        "code": os.path.relpath(src, BASE), "sha256_16": sha16(src),
        "seed": SEED, "runtime_s": round(time.time() - _T0, 1),
        "treated_sample": "p014_treated_sample_v2_expanded.csv (752/379)",
        "date": "2026-08-24",
    }
    if extra:
        rec.update(extra)
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/{iid.replace('-', '')}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"\n{'='*70}\n[{iid}] {status} — {verdict}\n  → {os.path.relpath(p, BASE)}\n{'='*70}")
    return rec


def boot_mean_ci(x, rng, nb=NB):
    """1표본 평균의 부트스트랩 CI."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 20:
        return None, None, len(x)
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(nb)])
    return round(float(x.mean()), 4), qci(b), len(x)


def boot_did_ci(t, c, rng, nb=NB):
    """처치 변화량 t 와 대조 평균 변화량 c 의 쌍 DiD 부트스트랩."""
    t = np.asarray(t, float); c = np.asarray(c, float)
    ok = np.isfinite(t) & np.isfinite(c)
    t, c = t[ok], c[ok]
    if len(t) < 20:
        return None, None, len(t)
    d = t - c
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(nb)])
    return round(float(d.mean()), 4), qci(b), len(d)
