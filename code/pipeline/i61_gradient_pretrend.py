# -*- coding: utf-8 -*-
"""I-61 상태별 사전추세 — headline estimand 에 직접 대응하는 식별 진단 (리뷰4 §8 · 리뷰3 MC2-3).

지금까지의 이벤트스터디는 **평균효과**의 사전추세만 보였다. 그러나 헤드라인은 gradient 다.
가장 날카로운 질문은 "처치 전에 이미 저활동 표적의 처치−대조 궤적이 고활동 표적과 달랐는가"이다.

분기 q 마다
    g_q = [T−C]_저활동 − [T−C]_고활동
를 그리고, 사전 분기(q≤−1)가 0 근처에 머무는지 본다. 위약(거울설계 유사처치)에도 같은 절차를
적용해 기계적 기준선을 제거한다.

Panel A  분기별 gradient 경로 (사전 4분기 + 사후 4분기), 셀 군집 부트 CI
Panel B  사전 분기 결합검정 — 사전 gradient 평균이 0 과 구별되는가 (등가성 포함)
Panel C  위약에서의 동일 경로
"""
import numpy as np, json
from h30_common import load, deals, build, emit, SEED, qci, NB, widx
from h39_common import SIZE_B
rng = np.random.default_rng(SEED)
print("[I-61] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Ev, adpt, idx = G["Hv"], G["Ev"], G["adpt_arr"], G["idx"]
mset, ind_arr = G["mset"], G["ind_arr"]; NOTPE = np.asarray(~idx.isin(set(PE)))
_c, _s = {}, {}
def cellarr(m0):
    if m0 in _c: return _c[m0]
    iw=[mset[m] for m in range(m0-6,m0) if m in mset]; i18=[mset[m] for m in range(m0-18,m0-12) if m in mset]
    if not iw or not i18: _c[m0]=None; return None
    with np.errstate(all="ignore"):
        Ep=np.nanmean(Ev[:,iw],axis=1); g=Ep/np.nanmean(Ev[:,i18],axis=1)-1
    _c[m0]=(Ep,g,np.digitize(Ep,SIZE_B,right=False),
            np.where(np.isnan(g),-1,np.digitize(g,[-0.10,0.10])),
            np.where(np.isnan(adpt),-1,np.digitize((m0-adpt)/12.0,[5,15])))
    return _c[m0]
def Sall(m0):
    if m0 in _s: return _s[m0]
    c=widx(G,m0,-24,-13)
    if len(c)!=12: _s[m0]=(None,None); return _s[m0]
    h=Hv[:,c].astype(float); e=Ev[:,c].astype(float)
    ok=np.isfinite(h).all(1)&np.isfinite(e).all(1)&(np.nanmean(e,1)>=5)
    S=np.full(Hv.shape[0],np.nan); S[ok]=-np.log1p(h[ok].sum(1)/np.nanmean(e[ok],1))
    fin=np.isfinite(S); b=np.full(Hv.shape[0],-9)
    if fin.sum()>=50:
        q1,q2=np.percentile(S[fin],[33.33,66.67]); b=np.where(fin,np.digitize(S,[q1,q2]),-9)
    _s[m0]=(S,b); return _s[m0]
def match(focal,m0,k=5):
    c=cellarr(m0)
    if c is None: return None
    Ep,g,sb,gb,ageb=c
    if not (np.isfinite(Ep[focal]) and Ep[focal]>=5): return None
    S,bins=Sall(m0)
    if S is None or not np.isfinite(S[focal]) or bins[focal]==-9: return None
    same=(NOTPE&(ind_arr==ind_arr[focal])&(sb==sb[focal])&(gb==gb[focal])&(ageb==ageb[focal])
          &(Ep>=5)&np.isfinite(Ep)&(bins==bins[focal]))
    cand=np.flatnonzero(same); cand=cand[cand!=focal]
    if len(cand)==0: return None
    gt=g[focal] if np.isfinite(g[focal]) else 0.0
    gc=np.where(np.isfinite(g[cand]),g[cand],0.0)
    d=((np.log(Ep[cand])-np.log(Ep[focal]))/0.9)**2+((np.clip(gc,-1,2)-np.clip(gt,-1,2))/0.35)**2
    return cand[np.argsort(d)[:k]]

QS = [(-4,(-12,-10)),(-3,(-9,-7)),(-2,(-6,-4)),(-1,(-3,-1)),
      (1,(1,3)),(2,(4,6)),(3,(7,9)),(4,(10,12))]
BASE = (-24,-13)                    # 기준: 상태창. 사후·사전 분기를 모두 여기에 대비시킨다.

def qrate(row,m0,a,b):
    c=widx(G,m0,a,b)
    if len(c)!=(b-a+1): return None
    h,e=Hv[row,c].astype(float),Ev[row,c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.mean(e)<5: return None
    return (float(h.sum()),float(np.mean(e)))

def unit(focal,ctrls,m0,gi):
    bs=qrate(focal,m0,*BASE)
    if bs is None or bs[0]<=0: return None
    S,bins=Sall(m0)
    if S is None or not np.isfinite(S[focal]): return None
    lb=np.log(bs[0]/bs[1])*(3/12)          # 분기 환산 기준
    row={"g":gi,"S":float(S[focal]),"hi":int(bins[focal]==2),"lo":int(bins[focal]==0)}
    ok=False
    for q,(a,b) in QS:
        t=qrate(focal,m0,a,b)
        if t is None or t[0]<=0: row[f"q{q}"]=np.nan; continue
        yt=np.log(t[0]/t[1])-lb
        cs=[]
        for o in ctrls:
            cq=qrate(int(o),m0,a,b); cb=qrate(int(o),m0,*BASE)
            if cq and cb and cq[0]>0 and cb[0]>0:
                cs.append(np.log(cq[0]/cq[1])-np.log(cb[0]/cb[1])*(3/12))
        row[f"q{q}"]=(yt-float(np.mean(cs))) if cs else np.nan
        ok=True
    return row if ok else None

def assemble():
    T,P=[],[]
    for gi,e in enumerate(EV):
        ct=match(e["ti"],e["m0"])
        if ct is None: continue
        u=unit(e["ti"],[int(x) for x in ct],e["m0"],gi)
        if u: T.append(u)
        for k in ct:
            ck=match(int(k),e["m0"])
            if ck is None: continue
            v=unit(int(k),[int(x) for x in ck],e["m0"],gi)
            if v: P.append(v)
    return T,P

T,P=assemble()
print(f"  처치 {len(T)} · 위약 {len(P)}")

def grad_q(rows,q):
    hi=[r[f"q{q}"] for r in rows if r["hi"] and np.isfinite(r.get(f"q{q}",np.nan))]
    lo=[r[f"q{q}"] for r in rows if r["lo"] and np.isfinite(r.get(f"q{q}",np.nan))]
    if min(len(hi),len(lo))<15: return None,0,0
    return float(np.mean(hi)-np.mean(lo)),len(hi),len(lo)

def boot(rows,q,R=NB):
    cells=sorted({r["g"] for r in rows}); byg={c:[r for r in rows if r["g"]==c] for c in cells}
    out=[]
    for _ in range(R):
        d=[r for i in rng.integers(0,len(cells),len(cells)) for r in byg[cells[i]]]
        v,_a,_b=grad_q(d,q)
        if v is not None: out.append(v)
    return qci(np.array(out)) if out else None

print("\n[Panel A·C] 분기별 gradient  g_q = [T−C]_저활동 − [T−C]_고활동")
print(f"  {'분기':>4}  {'처치':>9} {'CI':<22} {'위약':>9} {'CI':<22}")
PA,PC={},{}
for q,_ in QS:
    gt,nh,nl=grad_q(T,q); gp,ph,pl=grad_q(P,q)
    if gt is None: continue
    ct=boot(T,q); cp=boot(P,q)
    sig="✓" if (ct and (ct[0]>0 or ct[1]<0)) else "✗"
    PA[f"q{q}"]={"grad":round(gt,4),"ci":ct,"n_hi":nh,"n_lo":nl,
                 "sig":bool(ct and (ct[0]>0 or ct[1]<0))}
    PC[f"q{q}"]={"grad":(round(gp,4) if gp is not None else None),"ci":cp}
    print(f"  {q:>4}  {gt:>+9.4f} {str(ct):<22} "
          f"{(f'{gp:+9.4f}' if gp is not None else '        -')} {str(cp):<22} {sig}")

print("\n[Panel B] 사전 분기 결합검정")
pre=[q for q,_ in QS if q<0 and f"q{q}" in PA]
vals=[PA[f"q{q}"]["grad"] for q in pre]
cells=sorted({r["g"] for r in T}); byg={c:[r for r in T if r["g"]==c] for c in cells}
bb=[]
for _ in range(NB):
    d=[r for i in rng.integers(0,len(cells),len(cells)) for r in byg[cells[i]]]
    vv=[grad_q(d,q)[0] for q in pre]
    if all(v is not None for v in vv): bb.append(float(np.mean(vv)))
ci=qci(np.array(bb)); mean_pre=float(np.mean(vals))
post=[q for q,_ in QS if q>0 and f"q{q}" in PA]
mean_post=float(np.mean([PA[f"q{q}"]["grad"] for q in post]))
SESOI=abs(mean_post)
equiv=bool(ci[0]>-SESOI and ci[1]<SESOI)
PB={"pre_quarters":pre,"pre_mean":round(mean_pre,4),"pre_ci":ci,
    "pre_sig":bool(ci[0]>0 or ci[1]<0),"post_mean":round(mean_post,4),
    "SESOI":round(SESOI,4),"equivalence_holds":equiv,
    "margin":[round(ci[0]+SESOI,4),round(SESOI-ci[1],4)],
    "max_abs_pre":round(max(abs(v) for v in vals),4)}
print(f"  사전 평균 {mean_pre:+.4f} {ci} {'✓유의' if PB['pre_sig'] else '✗미검출'} · "
      f"사후 평균 {mean_post:+.4f}")
print(f"  등가성(δ=사후 크기 {SESOI:.4f}): {'✓ 성립' if equiv else '✗ 미성립'} 여유 {PB['margin']}")
print(f"  사전 최대 |gradient| {PB['max_abs_pre']:.4f}")

verdict=(f"분기별 상태 gradient: 사전 4분기 평균 {mean_pre:+.4f} {ci} "
         f"({'유의' if PB['pre_sig'] else '미검출'}), 최대 절대값 {PB['max_abs_pre']:.4f}. "
         f"사후 4분기 평균 {mean_post:+.4f}. "
         f"등가성 δ={SESOI:.4f}: {'성립' if equiv else '미성립'}. "
         f"위약 경로도 함께 보고 — 기계적 기준선 제거.")
emit("I-61","상태별 사전추세 (리뷰4 §8 · 리뷰3 MC2-3)",
     "GO" if (not PB["pre_sig"]) else "PARTIAL",
     {"panelA_treated_path":PA,"panelC_placebo_path":PC,"panelB_pre_joint":PB,
      "quarters":[q for q,_ in QS],"base_window":"[-24,-13]",
      "design":"상태균형 매칭 + 거울 위약","n_treated":len(T),"n_placebo":len(P)},
     "처치 전에 이미 저활동 표적의 처치−대조 궤적이 고활동 표적과 달랐는가",
     verdict, kill_met=bool(PB["pre_sig"]), n=len(T))
