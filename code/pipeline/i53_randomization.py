# -*- coding: utf-8 -*-
"""I-53 무작위화 추론 — 위약 분포를 귀무로 직접 사용.

I-52 는 처치−위약 차이를 **부트스트랩**으로 검정했다(+0.3626 [0.004, 0.763], MDE 0.542).
두 표본의 분산을 각각 추정해 더하므로 보수적이고 검정력을 잃는다.

더 강한 방법: 위약 pseudo-event 풀(1,373건 / 321셀)에서 **처치표본과 같은 크기**의 표본을
반복 추출해 gradient 귀무분포를 만들고, 관측된 처치 gradient 가 그 안 어디에 있는지 본다.
이것이 이 설계의 자연스러운 무작위화 검정이며, 귀무분포를 통째로 쓰므로 부트스트랩 차이보다 정밀하다.

Panel A  귀무분포 구축 (셀 단위 추출, 처치 표본크기에 맞춤) + RI p
Panel B  사양별 RI p (winsor 0/1/5/10, 암묵채용)
Panel C  귀무분포 요약과 관측치 위치
"""
import numpy as np
from h30_common import load, deals, build, emit, SEED, qci, widx

rng = np.random.default_rng(SEED)
NDRAW = 2000
print("[I-53] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Sv, Ev, adpt = G["Hv"], G["Sv"], G["Ev"], G["adpt_arr"]
Astar = np.zeros_like(Hv)
Astar[:, 1:] = np.maximum(0.0, np.diff(Ev, axis=1) + Sv[:, 1:])
Astar[:, 0] = Hv[:, 0]


def rate(row, m0, a, b, M=None):
    c = widx(G, m0, a, b)
    if len(c) != (b - a + 1): return None
    h = (M if M is not None else Hv)[row, c].astype(float)
    e = Ev[row, c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.mean(e) < 5: return None
    return float(h.sum()), float(np.mean(e))


def unit(focal, others, m0, M=None):
    po, pr = rate(focal, m0, 1, 12, M), rate(focal, m0, -12, -1, M)
    st = rate(focal, m0, -24, -13, M)
    if not (po and pr and st and po[0] > 0 and pr[0] > 0): return None
    cs = []
    for o in others:
        p2, r2 = rate(o, m0, 1, 12, M), rate(o, m0, -12, -1, M)
        if p2 and r2 and p2[0] > 0 and r2[0] > 0:
            cs.append(np.log(p2[0] / p2[1]) - np.log(r2[0] / r2[1]))
    if not cs: return None
    w36 = rate(focal, m0, -36, -25, M)
    return dict(eff=(np.log(po[0] / po[1]) - np.log(pr[0] / pr[1])) - float(np.mean(cs)),
                S=-np.log1p(st[0] / st[1]), lsize=np.log(st[1]),
                grow=(np.log(st[1] / w36[1]) if w36 and w36[1] > 0 else np.nan),
                age=((m0 - adpt[focal]) / 12.0 if np.isfinite(adpt[focal]) else np.nan),
                ind=str(G["ind_arr"][focal])[:1], g=None)


def assemble(M=None):
    T, P = [], []
    for gi, e in enumerate(EV):
        ctr = [int(k) for k in e["ctrls"]]
        u = unit(e["ti"], ctr, e["m0"], M)
        if u: u["g"] = gi; T.append(u)
        for k in ctr:
            v = unit(k, [j for j in ctr if j != k], e["m0"], M)
            if v: v["g"] = gi; P.append(v)
    return T, P


def slope(rows, cuts=None):
    if len(rows) < 30: return None
    y = np.array([r["eff"] for r in rows]); x = np.array([r["S"] for r in rows])
    if cuts is not None: y = np.clip(y, cuts[0], cuts[1])
    cols = [np.ones(len(y)), np.array([r["lsize"] for r in rows])]
    for k in ("grow", "age"):
        v = np.array([r[k] for r in rows], float); m = np.isfinite(v)
        cols.append(np.where(m, v, np.median(v[m]) if m.any() else 0.0))
    for s in sorted({r["ind"] for r in rows})[1:]:
        cols.append(np.array([1.0 if r["ind"] == s else 0.0 for r in rows]))
    C = np.column_stack(cols)
    r_ = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    yr, xr = r_(y), r_(x)
    d = float(np.sum(xr * xr))
    return float(np.sum(xr * yr) / d) if d > 0 else None


def ri(M=None, wins=None, tag=""):
    T, P = assemble(M)
    cuts = tuple(np.percentile([r["eff"] for r in T], wins)) if wins else None
    obs = slope(T, cuts)
    n_t = len(T)
    cells = sorted({r["g"] for r in P})
    byg = {c: [r for r in P if r["g"] == c] for c in cells}
    null = []
    for _ in range(NDRAW):
        draw, sel = [], rng.permutation(len(cells))
        for i in sel:                      # 셀 단위로 뽑아 처치 표본크기에 도달
            draw += byg[cells[i]]
            if len(draw) >= n_t: break
        s_ = slope(draw[:n_t], cuts)
        if s_ is not None: null.append(s_)
    null = np.array(null)
    p = float((null >= obs).mean())
    p_corr = (int((null >= obs).sum()) + 1) / (len(null) + 1)
    out = {"observed": round(obs, 4), "n_treated": n_t, "n_draws": len(null),
           "null_mean": round(float(null.mean()), 4),
           "null_sd": round(float(null.std()), 4),
           "null_p95": round(float(np.percentile(null, 95)), 4),
           "null_ci": qci(null), "RI_p": round(p, 4), "RI_p_corrected": round(p_corr, 4),
           "sig": bool(p_corr < 0.05),
           "z": round(float((obs - null.mean()) / null.std()), 2) if null.std() > 0 else None}
    print(f"  {tag:<24} 관측 {out['observed']:>+7.4f} · 귀무 평균 {out['null_mean']:>+7.4f} "
          f"SD {out['null_sd']:.4f} · p95 {out['null_p95']:>+7.4f} · "
          f"RI p = {out['RI_p_corrected']:.4f} {'✓' if out['sig'] else '✗'} (z={out['z']})")
    return out


print(f"\n[Panel A·B] 위약 귀무분포 {NDRAW}회 추출 (셀 단위, 처치 표본크기에 맞춤)")
R = {}
R["main_winsor_5_95"] = ri(wins=(5, 95), tag="★ 주 사양 winsor5/95")
R["raw"] = ri(tag="무조정")
R["winsor_1_99"] = ri(wins=(1, 99), tag="winsor 1/99")
R["winsor_10_90"] = ri(wins=(10, 90), tag="winsor 10/90")
R["implied_hires"] = ri(M=Astar, wins=(5, 95), tag="암묵 채용 A*")

m = R["main_winsor_5_95"]
print(f"\n[Panel C] 주 사양 귀무분포: 95% 범위 {m['null_ci']} · 관측 {m['observed']} "
      f"→ 귀무 상위 {m['RI_p_corrected']*100:.1f}%")
n_sig = sum(1 for v in R.values() if v["sig"])
verdict = (
    f"위약 pseudo-event 풀을 귀무분포로 직접 사용한 무작위화 검정. 주 사양(winsor 5/95): "
    f"관측 기울기 **{m['observed']:+.4f}** vs 귀무 평균 {m['null_mean']:+.4f} "
    f"(SD {m['null_sd']:.4f}, 95% 범위 {m['null_ci']}), **RI p = {m['RI_p_corrected']:.4f}** "
    f"{'✓' if m['sig'] else '✗'}, z = {m['z']}. 사양 {len(R)}종 중 {n_sig}종 유의. "
    f"부트스트랩 차이 검정(I-52, +0.3626 [0.004, 0.763])보다 정밀한 이유는 귀무분포를 "
    f"추정하지 않고 **직접 구성**하기 때문이다.")
emit("I-53", "무작위화 추론 — 위약 분포를 귀무로 직접 사용",
     "GO" if m["sig"] else "PARTIAL",
     {"specs": R, "n_draws": NDRAW,
      "design": "위약 pseudo-event 를 셀 단위로 추출해 처치 표본크기에 맞춘 귀무분포",
      "estimand": "state gradient in Δlog hiring rate, FWL-adjusted"},
     "위약 분포를 귀무로 직접 쓰면 상태 gradient 가 유의한가",
     verdict, kill_met=False, n=m["n_treated"])
