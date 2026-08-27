# -*- coding: utf-8 -*-
"""I-63 표본 확대 — 검정력의 마지막 큰 레버.

379 → 286 의 탈락 93건(25%)이 어디서 새는지 계측하고, **선택편의 없이** 회수 가능한 것을 회수한다.
탈락 경로: (a) 상태창 [−24,−13] 12개월 완전관측 (b) 고용≥5 (c) 상태 bin 안에 대조 없음
(d) 결과창 N>0.

레버 (실행 전 근거):
  A  상태창 부분관측 허용 (≥9/12개월) — 결측이 무작위면 편의 없음, 표본만 회수
  B  상태 bin 을 3분위→2분위 — 균형은 약해지지만 대조 풀이 넓어짐
  C  대조 셀 완화: 2자리 산업 → 1자리 (셀 없음 탈락 회수)
  D  기업당 여러 딜 허용 X (이미 고유) / **처치표본 752 중 미연결 168** 은 회수 불가
  E  결과창을 분기 단위 패널로 → 관측치 4배 (셀 군집이라 SE 는 √4 까지 안 줄지만 이득)

각 레버에서 (n, 기울기, 귀무, z, RI p) 를 보고하고, **회수된 표본의 특성**을 원 표본과 비교해
선택편의 여부를 판단한다.
"""
import numpy as np
from h30_common import load, deals, build, emit, SEED, qci, NB, widx
from h39_common import SIZE_B
rng=np.random.default_rng(SEED); NDRAW=2000
print("[I-63] 로딩...")
G=load(); orig,allt,PE,META=deals(G); EV,_=build(G,allt,PE)
Hv,Sv,Ev,adpt,idx=G["Hv"],G["Sv"],G["Ev"],G["adpt_arr"],G["idx"]
mset,ind_arr=G["mset"],G["ind_arr"]; NOTPE=np.asarray(~idx.isin(set(PE)))
IND1=np.array([str(x)[:1] for x in ind_arr])
_c={}
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
_s={}
def Sall(m0,need=12,nbins=3):
    key=(m0,need,nbins)
    if key in _s: return _s[key]
    c=widx(G,m0,-24,-13)
    if len(c)<need: _s[key]=(None,None); return _s[key]
    h=Hv[:,c].astype(float); e=Ev[:,c].astype(float)
    m=np.isfinite(h)&np.isfinite(e)
    ok=(m.sum(1)>=need)&(np.nanmean(np.where(m,e,np.nan),1)>=5)
    S=np.full(Hv.shape[0],np.nan)
    hs=np.nansum(np.where(m,h,np.nan),1); es=np.nanmean(np.where(m,e,np.nan),1)
    S[ok]=-np.log1p(hs[ok]/es[ok])
    fin=np.isfinite(S); b=np.full(Hv.shape[0],-9)
    if fin.sum()>=50:
        qs=np.percentile(S[fin],[100*i/nbins for i in range(1,nbins)])
        b=np.where(fin,np.digitize(S,qs),-9)
    _s[key]=(S,b); return _s[key]
def match(focal,m0,k=5,need=12,nbins=3,ind_level=2):
    c=cellarr(m0)
    if c is None: return None
    Ep,g,sb,gb,ageb=c
    if not (np.isfinite(Ep[focal]) and Ep[focal]>=5): return None
    S,bins=Sall(m0,need,nbins)
    if S is None or not np.isfinite(S[focal]) or bins[focal]==-9: return None
    indm=(ind_arr==ind_arr[focal]) if ind_level==2 else (IND1==IND1[focal])
    same=(NOTPE&indm&(sb==sb[focal])&(gb==gb[focal])&(ageb==ageb[focal])
          &(Ep>=5)&np.isfinite(Ep)&(bins==bins[focal]))
    cand=np.flatnonzero(same); cand=cand[cand!=focal]
    if len(cand)==0: return None
    gt=g[focal] if np.isfinite(g[focal]) else 0.0
    gc=np.where(np.isfinite(g[cand]),g[cand],0.0)
    d=((np.log(Ep[cand])-np.log(Ep[focal]))/0.9)**2+((np.clip(gc,-1,2)-np.clip(gt,-1,2))/0.35)**2
    return cand[np.argsort(d)[:k]]
def blk(row,m0,a,b):
    c=widx(G,m0,a,b)
    if len(c)!=(b-a+1): return None
    h,e=Hv[row,c].astype(float),Ev[row,c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.mean(e)<5: return None
    return float(h.sum()),float(np.mean(e))
def unit(focal,ctrls,m0,gi,need,nbins):
    S,_b=Sall(m0,need,nbins)
    po,pr=blk(focal,m0,1,12),blk(focal,m0,-12,-1)
    if po is None or pr is None or po[0]<=0 or pr[0]<=0: return None
    cs=[]
    for o in ctrls:
        p2,r2=blk(int(o),m0,1,12),blk(int(o),m0,-12,-1)
        if p2 and r2 and p2[0]>0 and r2[0]>0: cs.append(np.log(p2[0]/p2[1])-np.log(r2[0]/r2[1]))
    if not cs: return None
    w36=blk(focal,m0,-36,-25)
    # 공변량은 I-60 관례: log 규모·성장 모두 **상태창 [−24,−13]** 기준 (부분관측 시 관측월 평균)
    c24=widx(G,m0,-24,-13); e24=Ev[focal,c24].astype(float); e24=e24[np.isfinite(e24)]
    lsz=float(np.log(np.mean(e24))) if len(e24) else np.log(pr[1])
    return dict(g=gi,bn=focal,eff=(np.log(po[0]/po[1])-np.log(pr[0]/pr[1]))-float(np.mean(cs)),
                S=float(S[focal]),lsize=lsz,
                grow=(np.log(np.exp(lsz)/w36[1]) if (w36 and w36[1]>0) else np.nan),
                age=((m0-adpt[focal])/12.0 if np.isfinite(adpt[focal]) else np.nan),
                ind=str(ind_arr[focal])[:1])
def assemble(k=5,need=12,nbins=3,ind_level=2):
    T,P=[],[]
    for gi,e in enumerate(EV):
        ct=match(e["ti"],e["m0"],k,need,nbins,ind_level)
        if ct is None: continue
        u=unit(e["ti"],[int(x) for x in ct],e["m0"],gi,need,nbins)
        if u: T.append(u)
        for c_ in ct:
            ck=match(int(c_),e["m0"],k,need,nbins,ind_level)
            if ck is None: continue
            v=unit(int(c_),[int(x) for x in ck],e["m0"],gi,need,nbins)
            if v: P.append(v)
    return T,P
def design(rows):
    cols=[np.ones(len(rows)),np.array([r["lsize"] for r in rows])]
    for k in ("grow","age"):
        v=np.array([r[k] for r in rows],float); m=np.isfinite(v)
        cols.append(np.where(m,v,np.median(v[m]) if m.any() else 0.0))
    for s_ in sorted({r["ind"] for r in rows})[1:]:
        cols.append(np.array([1.0 if r["ind"]==s_ else 0.0 for r in rows]))
    return np.column_stack(cols)
def grad(rows,cuts=None):
    if len(rows)<30: return None
    y=np.array([r["eff"] for r in rows]); x=np.array([r["S"] for r in rows])
    if cuts is not None: y=np.clip(y,cuts[0],cuts[1])
    C=design(rows); r_=lambda v: v-C@np.linalg.lstsq(C,v,rcond=None)[0]
    yr,xr=r_(y),r_(x); d=float(np.sum(xr*xr))
    return float(np.sum(xr*yr)/d) if d>0 else None
def ri(T,P,tag):
    cuts=tuple(np.percentile([r["eff"] for r in T],[5,95]))
    obs=grad(T,cuts); n_t=len(T)
    cells=sorted({r["g"] for r in P}); byg={c:[r for r in P if r["g"]==c] for c in cells}
    null=[]
    for _ in range(NDRAW):
        d_=[]
        for i in rng.permutation(len(cells)):
            d_+=byg[cells[i]]
            if len(d_)>=n_t: break
        v=grad(d_[:n_t],cuts)
        if v is not None: null.append(v)
    null=np.array(null); p=(int((null>=obs).sum())+1)/(len(null)+1)
    o={"observed":round(obs,4),"n":n_t,"n_placebo":len(P),"null_mean":round(float(null.mean()),4),
       "null_sd":round(float(null.std()),4),"RI_p":round(float(p),4),
       "z":round(float((obs-null.mean())/null.std()),2),"sig":bool(p<0.05)}
    print(f"  {tag:<38} n={n_t:>3}/{len(P):>4} · obs {o['observed']:>+7.4f} · null {o['null_mean']:>+7.4f}"
          f"(SD {o['null_sd']:.4f}) · z={o['z']:>5.2f} · p {o['RI_p']:.4f} {'✓' if o['sig'] else '✗'}")
    return o

print("\n[탈락 계측] 379 → 주사양 286")
drop={"state_window":0,"emp_lt5":0,"no_cell":0,"outcome":0,"ok":0}
for e in EV:
    S,b=Sall(e["m0"]); 
    if S is None or not np.isfinite(S[e["ti"]]):
        c=widx(G,e["m0"],-24,-13)
        if len(c)<12: drop["state_window"]+=1
        else:
            h,ee=Hv[e["ti"],c].astype(float),Ev[e["ti"],c].astype(float)
            m=np.isfinite(h)&np.isfinite(ee)
            drop["state_window" if m.sum()<12 else "emp_lt5"]+=1
        continue
    ct=match(e["ti"],e["m0"])
    if ct is None: drop["no_cell"]+=1; continue
    if unit(e["ti"],[int(x) for x in ct],e["m0"],0,12,3) is None: drop["outcome"]+=1; continue
    drop["ok"]+=1
print("  ",drop)

R={"attrition":drop}
print("\n[레버 비교]")
CFG=[("기준: 12개월·3분위·2자리산업",dict(need=12,nbins=3,ind_level=2)),
     ("A 상태창 ≥9개월",dict(need=9,nbins=3,ind_level=2)),
     ("A' 상태창 ≥6개월",dict(need=6,nbins=3,ind_level=2)),
     ("B 상태 2분위",dict(need=12,nbins=2,ind_level=2)),
     ("C 산업 1자리",dict(need=12,nbins=3,ind_level=1)),
     ("A+B+C 결합",dict(need=9,nbins=2,ind_level=1))]
SETS={}
for tag,cfg in CFG:
    T,P=assemble(**cfg); SETS[tag]=(T,P)
    R[tag]=ri(T,P,tag)

print("\n[회수 표본의 특성] — 기준 대비 신규 진입 기업")
base_bn={r["bn"] for r in SETS[CFG[0][0]][0]}
for tag,_ in CFG[1:]:
    T,_p=SETS[tag]; new=[r for r in T if r["bn"] not in base_bn]; old=[r for r in T if r["bn"] in base_bn]
    if not new: print(f"  {tag:<28} 신규 0"); continue
    def m(rows,k): 
        v=np.array([r[k] for r in rows],float); return float(np.nanmean(v))
    R[tag]["recovered"]={"n_new":len(new),
        "S_new":round(m(new,"S"),4),"S_old":round(m(old,"S"),4),
        "lsize_new":round(m(new,"lsize"),3),"lsize_old":round(m(old,"lsize"),3),
        "eff_new":round(m(new,"eff"),4),"eff_old":round(m(old,"eff"),4)}
    r=R[tag]["recovered"]
    print(f"  {tag:<28} 신규 {r['n_new']:>3} · S {r['S_new']:+.3f} vs {r['S_old']:+.3f} · "
          f"log규모 {r['lsize_new']:.2f} vs {r['lsize_old']:.2f} · 효과 {r['eff_new']:+.3f} vs {r['eff_old']:+.3f}")

base=R[CFG[0][0]]
best=max((v for k,v in R.items() if isinstance(v,dict) and "z" in v),key=lambda v:v["z"])
verdict=(f"탈락 계측: {drop}. 기준 n={base['n']} z={base['z']}. "
         f"레버별 z: "+" · ".join(f"{t.split(':')[0].split(' ')[0]} {R[t]['z']}" for t,_ in CFG[1:])+
         f". 최대 z 는 '{[k for k,v in R.items() if v is best][0]}' ({best['z']}, n={best['n']}).")
emit("I-63","표본 확대 — 탈락 계측과 회수 레버",
     "GO" if best["z"]>base["z"]+0.3 else "PARTIAL", R|{"n_draws":NDRAW},
     "선택편의 없이 표본을 회수해 검정력을 올릴 수 있는가",verdict,kill_met=False,n=base["n"])
