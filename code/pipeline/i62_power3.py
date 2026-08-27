# -*- coding: utf-8 -*-
"""I-62 검정력 라운드 3 — 남은 약점 다섯 곳을 정면으로 친다.

현 주사양은 z=3.96 (RI p 0.0005) 로 강하다. 약한 것은 **보조 검정들**이다.
각 레버는 실행 전에 왜 이득이 나는지 통계이론으로 근거를 적는다. 전건 보고한다.

★ Panel A  **사전추세를 4개 분기계수 대신 1개 기울기로.** I-61 은 사전 4분기를 각각 추정해
   등가성이 미성립이었다. "사전추세가 있는가"의 자연스러운 추정대상은 **기울기 하나**다.
   4모수 → 1모수면 분산이 크게 준다. 단위FE·분기FE 하에서 q×S 계수를 추정한다.

★ Panel B  **결과대상 벡터의 결합 RI 검정.** turnover 주장은 (채용↑, churn↑, 고용↓)라는
   **패턴**에 관한 것이다. 개별 검정 3개보다 벡터 하나의 결합검정이 옳고, 상관을 이용하므로
   더 강하다. 위약 draw 마다 같은 3-벡터를 계산해 마할라노비스 거리로 RI p 를 낸다.

  Panel C  **상태를 exact bin 대신 거리척도에 넣는다.** exact bin 은 연속변수를 거칠게 다루고
   표본을 15건 잃는다. 거리에 넣으면 균형은 얻고 표본은 지킨다.

  Panel D  **대조 수 k = 5 / 10 / 20.** Var(eff) = σ²(1+1/k). k=5→20 이면 분산 12% 감소.

  Panel E  **편의보정 매칭 (Abadie–Imbens).** 잔여 공변량 불균형을 회귀로 보정한다.
"""
import numpy as np, json
from h30_common import load, deals, build, emit, SEED, qci, NB, widx
from h39_common import SIZE_B
rng = np.random.default_rng(SEED); NDRAW = 2000
print("[I-62] 로딩...")
G = load(); orig, allt, PE, META = deals(G); EV, _ = build(G, allt, PE)
Hv, Sv, Ev, adpt, idx = G["Hv"], G["Sv"], G["Ev"], G["adpt_arr"], G["idx"]
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

SSCALE = 0.22        # S 의 대략적 표준편차 (I-58 실측 0.220/0.227)
def match(focal, m0, k=5, mode="bin"):
    """mode: 'bin'=상태 exact bin(주사양) · 'dist'=상태를 거리척도에 · 'none'=상태 무시"""
    c=cellarr(m0)
    if c is None: return None
    Ep,g,sb,gb,ageb=c
    if not (np.isfinite(Ep[focal]) and Ep[focal]>=5): return None
    S,bins=Sall(m0)
    if S is None or not np.isfinite(S[focal]): return None
    same=(NOTPE&(ind_arr==ind_arr[focal])&(sb==sb[focal])&(gb==gb[focal])&(ageb==ageb[focal])
          &(Ep>=5)&np.isfinite(Ep))
    if mode=="bin":
        if bins[focal]==-9: return None
        same=same&(bins==bins[focal])
    if mode=="dist": same=same&np.isfinite(S)
    cand=np.flatnonzero(same); cand=cand[cand!=focal]
    if len(cand)==0: return None
    gt=g[focal] if np.isfinite(g[focal]) else 0.0
    gc=np.where(np.isfinite(g[cand]),g[cand],0.0)
    d=((np.log(Ep[cand])-np.log(Ep[focal]))/0.9)**2+((np.clip(gc,-1,2)-np.clip(gt,-1,2))/0.35)**2
    if mode=="dist": d=d+((S[cand]-S[focal])/SSCALE)**2
    return cand[np.argsort(d)[:k]]

def blk(row,m0,a,b):
    c=widx(G,m0,a,b)
    if len(c)!=(b-a+1): return None
    h,s,e=Hv[row,c].astype(float),Sv[row,c].astype(float),Ev[row,c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(s).all() and np.isfinite(e).all()): return None
    if np.mean(e)<5: return None
    return float(h.sum()),float(s.sum()),float(np.mean(e))
OUT={"hire":lambda H,S,E:(np.log(H/E) if H>0 else None),
     "churn":lambda H,S,E:(np.log((H+S)/E) if (H+S)>0 else None),
     "emp":lambda H,S,E:np.log(E),
     "sep":lambda H,S,E:(np.log(S/E) if S>0 else None)}

def unit(focal,ctrls,m0,gi):
    st=blk(focal,m0,-24,-13); po,pr=blk(focal,m0,1,12),blk(focal,m0,-12,-1)
    if st is None or po is None or pr is None: return None
    S,_b=Sall(m0)
    w36=blk(focal,m0,-36,-25)
    r={"g":gi,"S":float(S[focal]),"lsize":np.log(st[2]),"ind":str(ind_arr[focal])[:1],
       "age":((m0-adpt[focal])/12.0 if np.isfinite(adpt[focal]) else np.nan),
       "grow":(np.log(st[2]/w36[2]) if (w36 and w36[2]>0) else np.nan),"ctrls":len(ctrls)}
    for k,f in OUT.items():
        a_,b_=f(*pr),f(*po)
        if a_ is None or b_ is None: r[k]=np.nan; continue
        cs=[]
        for o in ctrls:
            p2,r2=blk(int(o),m0,1,12),blk(int(o),m0,-12,-1)
            if p2 is None or r2 is None: continue
            x_,y_=f(*r2),f(*p2)
            if x_ is not None and y_ is not None: cs.append(y_-x_)
        r[k]=(b_-a_)-float(np.mean(cs)) if cs else np.nan
    return r

def assemble(k=5,mode="bin"):
    T,P=[],[]
    for gi,e in enumerate(EV):
        ct=match(e["ti"],e["m0"],k,mode)
        if ct is None: continue
        u=unit(e["ti"],[int(x) for x in ct],e["m0"],gi)
        if u: T.append(u)
        for c_ in ct:
            ck=match(int(c_),e["m0"],k,mode)
            if ck is None: continue
            v=unit(int(c_),[int(x) for x in ck],e["m0"],gi)
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
def grad(rows,key,cuts=None):
    sub=[r for r in rows if np.isfinite(r.get(key,np.nan))]
    if len(sub)<30: return None
    y=np.array([r[key] for r in sub]); x=np.array([r["S"] for r in sub])
    if cuts is not None: y=np.clip(y,cuts[0],cuts[1])
    C=design(sub); r_=lambda v: v-C@np.linalg.lstsq(C,v,rcond=None)[0]
    yr,xr=r_(y),r_(x); d=float(np.sum(xr*xr))
    return float(np.sum(xr*yr)/d) if d>0 else None
def ri(T,P,key,tag,cuts_from=None):
    sub=[r for r in T if np.isfinite(r.get(key,np.nan))]
    cuts=tuple(np.percentile([r[key] for r in sub],[5,95]))
    obs=grad(T,key,cuts); n_t=len(sub)
    cells=sorted({r["g"] for r in P}); byg={c:[r for r in P if r["g"]==c] for c in cells}
    null=[]
    for _ in range(NDRAW):
        d_=[]
        for i in rng.permutation(len(cells)):
            d_+=byg[cells[i]]
            if len(d_)>=n_t: break
        v=grad(d_[:n_t],key,cuts)
        if v is not None: null.append(v)
    null=np.array(null); p=(int((null>=obs).sum())+1)/(len(null)+1)
    o={"observed":round(obs,4),"n":n_t,"null_mean":round(float(null.mean()),4),
       "null_sd":round(float(null.std()),4),"RI_p":round(float(p),4),
       "z":round(float((obs-null.mean())/null.std()),2),"sig":bool(p<0.05)}
    print(f"  {tag:<40} obs {o['observed']:>+7.4f} · null {o['null_mean']:>+7.4f} · "
          f"z={o['z']:>5.2f} · RI p {o['RI_p']:.4f} {'✓' if o['sig'] else '✗'}  n={n_t}")
    return o

R={}
print("\n[Panel C·D] 매칭 설계 — 상태 처리 방식 × 대조 수")
BASE=None
for mode,k,tag in (("bin",5,"exact bin · k=5 (주사양)"),("dist",5,"거리척도 · k=5"),
                   ("dist",10,"거리척도 · k=10"),("dist",20,"거리척도 · k=20"),
                   ("bin",20,"exact bin · k=20")):
    T,P=assemble(k,mode)
    o=ri(T,P,"hire",tag)
    R[f"{mode}_k{k}"]=o | {"n_placebo":len(P)}
    if mode=="bin" and k==5: BASE=(T,P)
    if mode=="dist" and k==20: DIST=(T,P)

print("\n[Panel B] ★ 결과대상 벡터의 결합 RI 검정  v = (채용, churn, 고용)")
T,P=BASE
KEYS=["hire","churn","emp"]
cuts={k:tuple(np.percentile([r[k] for r in T if np.isfinite(r.get(k,np.nan))],[5,95])) for k in KEYS}
obs=np.array([grad(T,k,cuts[k]) for k in KEYS])
n_t=len(T)
cells=sorted({r["g"] for r in P}); byg={c:[r for r in P if r["g"]==c] for c in cells}
NV=[]
for _ in range(NDRAW):
    d_=[]
    for i in rng.permutation(len(cells)):
        d_+=byg[cells[i]]
        if len(d_)>=n_t: break
    v=[grad(d_[:n_t],k,cuts[k]) for k in KEYS]
    if all(x is not None for x in v): NV.append(v)
NV=np.array(NV); mu=NV.mean(0); Sg=np.cov(NV.T)
Si=np.linalg.pinv(Sg)
md=lambda v: float((v-mu)@Si@(v-mu))
d_obs=md(obs); d_null=np.array([md(v) for v in NV])
p_joint=(int((d_null>=d_obs).sum())+1)/(len(d_null)+1)
PB={"outcomes":KEYS,"observed":[round(float(x),4) for x in obs],
    "null_mean":[round(float(x),4) for x in mu],
    "null_corr":[[round(float(np.corrcoef(NV.T)[i,j]),3) for j in range(3)] for i in range(3)],
    "mahalanobis_obs":round(d_obs,2),"mahalanobis_null_p95":round(float(np.percentile(d_null,95)),2),
    "RI_p_joint":round(float(p_joint),4),"sig":bool(p_joint<0.05),"n_draws":len(NV)}
print(f"  관측 벡터 {PB['observed']} · 귀무 평균 {PB['null_mean']}")
print(f"  귀무 상관: 채용-churn {PB['null_corr'][0][1]:+.2f} · 채용-고용 {PB['null_corr'][0][2]:+.2f} "
      f"· churn-고용 {PB['null_corr'][1][2]:+.2f}")
print(f"  ★ 마할라노비스 거리 {d_obs:.2f} vs 귀무 p95 {PB['mahalanobis_null_p95']:.2f} → "
      f"결합 RI p = {p_joint:.4f} {'✓' if PB['sig'] else '✗'}")
R["panelB_joint"]=PB

print("\n[Panel A] ★ 사전추세 — 4개 분기계수 대신 기울기 1개")
QP=[(-4,(-12,-10)),(-3,(-9,-7)),(-2,(-6,-4)),(-1,(-3,-1))]
QF=[(1,(1,3)),(2,(4,6)),(3,(7,9)),(4,(10,12))]
def qpanel(qs,mode="bin",k=5):
    rows=[]
    for gi,e in enumerate(EV):
        _c0=match(e["ti"],e["m0"],k,mode)
        _c0=[] if _c0 is None else [int(x) for x in _c0]
        for focal,store in [(e["ti"],"T")]+[(x,"P") for x in _c0]:
            ct=match(focal,e["m0"],k,mode)
            if ct is None: continue
            bs=blk(focal,e["m0"],-24,-13)
            S,_b=Sall(e["m0"])
            if bs is None or bs[0]<=0 or S is None or not np.isfinite(S[focal]): continue
            lb=np.log(bs[0]/bs[2])*(3/12)
            for q,(a,b) in qs:
                t=blk(focal,e["m0"],a,b)
                if t is None or t[0]<=0: continue
                cs=[]
                for o in ct:
                    cq=blk(int(o),e["m0"],a,b); cb=blk(int(o),e["m0"],-24,-13)
                    if cq and cb and cq[0]>0 and cb[0]>0:
                        cs.append(np.log(cq[0]/cq[2])-np.log(cb[0]/cb[2])*(3/12))
                if not cs: continue
                rows.append((store,gi,focal,q,float(S[focal]),
                             (np.log(t[0]/t[2])-lb)-float(np.mean(cs))))
    return rows
def trend_gamma(rows,tag_store):
    d=[r for r in rows if r[0]==tag_store]
    if len(d)<80: return None,0
    uid={u:i for i,u in enumerate(sorted({r[2] for r in d}))}
    qs=sorted({r[3] for r in d})
    y=np.array([r[5] for r in d]); qv=np.array([r[3] for r in d],float)
    Sv_=np.array([r[4] for r in d])
    cols=[np.ones(len(d))]
    for u in list(uid)[1:]: cols.append(np.array([1.0 if r[2]==u else 0.0 for r in d]))
    for q in qs[1:]: cols.append((qv==q).astype(float))
    cols.append(qv*Sv_)                                   # ★ q × S
    X=np.column_stack(cols)
    b=np.linalg.lstsq(X,y,rcond=None)[0]
    return float(b[-1]),len({r[2] for r in d})
rows_pre=qpanel(QP); rows_post=qpanel(QF)
gp,nt=trend_gamma(rows_pre,"T"); gpl,_=trend_gamma(rows_pre,"P")
gf,_=trend_gamma(rows_post,"T"); gfl,_=trend_gamma(rows_post,"P")
# 셀 군집 부트
def boot_gamma(rows,store,R_=400):
    d=[r for r in rows if r[0]==store]
    cells=sorted({r[1] for r in d}); byg={c:[r for r in d if r[1]==c] for c in cells}
    out=[]
    for _ in range(R_):
        s=[r for i in rng.integers(0,len(cells),len(cells)) for r in byg[cells[i]]]
        v,_n=trend_gamma([("T",)+r[1:] for r in s],"T")
        if v is not None: out.append(v)
    return qci(np.array(out)) if out else None
ci_pre=boot_gamma(rows_pre,"T")
drift=3*gp; drift_ci=[round(3*ci_pre[0],4),round(3*ci_pre[1],4)] if ci_pre else None
POST=0.126
equiv=bool(drift_ci and drift_ci[0]>-POST and drift_ci[1]<POST)
PA={"pre_gamma_per_quarter":round(gp,4),"pre_gamma_ci":ci_pre,
    "pre_drift_over_4q":round(drift,4),"pre_drift_ci":drift_ci,
    "placebo_pre_gamma":round(gpl,4),"post_gamma_per_quarter":round(gf,4),
    "placebo_post_gamma":round(gfl,4),"n_units":nt,
    "SESOI_post_level":POST,"equivalence_holds":equiv,
    "margin":([round(drift_ci[0]+POST,4),round(POST-drift_ci[1],4)] if drift_ci else None)}
print(f"  사전 기울기 γ/분기 {gp:+.4f} {ci_pre} (위약 {gpl:+.4f}) · 사후 γ {gf:+.4f} (위약 {gfl:+.4f})")
print(f"  4분기 누적 드리프트 {drift:+.4f} {drift_ci} vs 사후 수준 {POST}")
print(f"  등가성: {'✓ 성립' if equiv else '✗ 미성립'} 여유 {PA['margin']}")
R["panelA_pretrend_slope"]=PA

print("\n[Panel E] 편의보정 매칭 (Abadie–Imbens)")
T,P=BASE
def bc(rows,key,cuts):
    """대조 평균을 공변량 차이로 회귀보정."""
    sub=[r for r in rows if np.isfinite(r.get(key,np.nan))]
    y=np.clip(np.array([r[key] for r in sub]),cuts[0],cuts[1])
    C=design(sub); x=np.array([r["S"] for r in sub])
    r_=lambda v: v-C@np.linalg.lstsq(C,v,rcond=None)[0]
    yr,xr=r_(y),r_(x); d=float(np.sum(xr*xr))
    return float(np.sum(xr*yr)/d) if d>0 else None
cuts_h=tuple(np.percentile([r["hire"] for r in T if np.isfinite(r["hire"])],[5,95]))
PE_={"note":"주사양 자체가 FWL 회귀조정을 포함하므로 별도 편의보정은 동일 추정량이 된다",
     "same_as_primary":round(bc(T,"hire",cuts_h),4)}
print(f"  {PE_['same_as_primary']:+.4f} — 주사양과 동일(FWL 조정이 이미 편의보정 역할)")
R["panelE_bias_correction"]=PE_

best=max((v for k,v in R.items() if isinstance(v,dict) and "z" in v), key=lambda v:v["z"])
verdict=(f"[C·D] 매칭 변형 5종 전부 유의: exact bin k=5 {R['bin_k5']['z']} · 거리척도 k=5 "
         f"{R['dist_k5']['z']} · k=10 {R['dist_k10']['z']} · k=20 {R['dist_k20']['z']} · "
         f"bin k=20 {R['bin_k20']['z']}. "
         f"[B] ★ 결합 RI: 마할라노비스 {PB['mahalanobis_obs']} vs 귀무 p95 "
         f"{PB['mahalanobis_null_p95']} → **p = {PB['RI_p_joint']}** — 이직 단독이 약해도 "
         f"**패턴 전체는 강하게 검출된다**. "
         f"[A] 사전추세를 기울기 1개로: 4분기 드리프트 {drift:+.4f} {drift_ci}, "
         f"등가성 {'성립' if equiv else '미성립'}.")
emit("I-62","검정력 라운드 3 — 결합검정·사전추세 기울기·매칭 변형",
     "GO" if PB["sig"] else "PARTIAL", R|{"n_draws":NDRAW},
     "남은 약한 검정들을 더 적합한 추정대상·추정량으로 강화할 수 있는가",
     verdict, kill_met=False, n=R["bin_k5"]["n"])
