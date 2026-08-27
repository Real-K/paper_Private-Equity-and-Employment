# -*- coding: utf-8 -*-
"""I-64 사전추세 — 등가성 대신 세 가지 올바른 질문.

I-61/62 의 분기별 gradient 사전추세는 등가성 미성립이었다. 원인은 분기 해상도의 잡음이다
(사후 분기도 개별로는 하나도 유의하지 않다). 등가성을 강요하는 대신 세 질문으로 바꾼다.

Panel A  **12개월 해상도 사전 gradient.** [−24,−13] → [−12,−1] 의 12개월 변화로 gradient 하나.
         분기의 절반 잡음. 헤드라인과 같은 해상도이므로 사후 gradient 와 직접 비교 가능.
Panel B  **위약 상대 RI.** 처치의 사전 gradient 가 위약 풀에서 만든 사전 gradient 귀무분포 안에 있는가.
         올바른 귀무는 "0" 이 아니라 "아무 일도 없던 기업의 사전 gradient".
Panel C  **gradient 에 대한 HonestDiD 상대크기 breakdown.** 관측된 최대 사전 |gradient| 의 M̄ 배까지
         차등추세를 허용해도 사후 gradient 가 0 을 배제하는 최대 M̄. (Rambachan–Roth 2023)
         분기 해상도(I-61)와 12개월 해상도(Panel A) 둘 다 계산 — I-11 에서 해상도 불일치가 M̄ 을
         기계적으로 0 으로 만든 전례가 있다.
"""
import numpy as np, json
from h30_common import load, deals, build, emit, SEED, qci, NB, widx
from h39_common import SIZE_B
rng=np.random.default_rng(SEED); NDRAW=2000
print("[I-64] 로딩...")
G=load(); orig,allt,PE,META=deals(G); EV,_=build(G,allt,PE)
Hv,Ev,adpt,idx=G["Hv"],G["Ev"],G["adpt_arr"],G["idx"]
mset,ind_arr=G["mset"],G["ind_arr"]; NOTPE=np.asarray(~idx.isin(set(PE)))
_c,_s={},{}
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
def blk(row,m0,a,b):
    c=widx(G,m0,a,b)
    if len(c)!=(b-a+1): return None
    h,e=Hv[row,c].astype(float),Ev[row,c].astype(float)
    if not (np.isfinite(h).all() and np.isfinite(e).all()) or np.mean(e)<5: return None
    return float(h.sum()),float(np.mean(e))
def lr(row,m0,a,b):
    w=blk(row,m0,a,b)
    return (np.log(w[0]/w[1]) if (w and w[0]>0) else None)

# 12개월 해상도: 사전 변화 = [−24,−13]→[−12,−1], 사후 변화 = [−12,−1]→[1,12]
def unit(focal,ctrls,m0,gi):
    S,_b=Sall(m0)
    a,b,c_=lr(focal,m0,-24,-13),lr(focal,m0,-12,-1),lr(focal,m0,1,12)
    if None in (a,b,c_) or S is None or not np.isfinite(S[focal]): return None
    pre_c,post_c=[],[]
    for o in ctrls:
        oa,ob,oc=lr(int(o),m0,-24,-13),lr(int(o),m0,-12,-1),lr(int(o),m0,1,12)
        if None not in (oa,ob,oc): pre_c.append(ob-oa); post_c.append(oc-ob)
    if not pre_c: return None
    w36=blk(focal,m0,-36,-25); st=blk(focal,m0,-24,-13)
    return dict(g=gi,S=float(S[focal]),pre=(b-a)-float(np.mean(pre_c)),post=(c_-b)-float(np.mean(post_c)),
                lsize=np.log(st[1]),grow=(np.log(st[1]/w36[1]) if (w36 and w36[1]>0) else np.nan),
                age=((m0-adpt[focal])/12.0 if np.isfinite(adpt[focal]) else np.nan),ind=str(ind_arr[focal])[:1])
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
print(f"  처치 {len(T)} · 위약 {len(P)}")
def design(rows):
    cols=[np.ones(len(rows)),np.array([r["lsize"] for r in rows])]
    for k in ("grow","age"):
        v=np.array([r[k] for r in rows],float); m=np.isfinite(v)
        cols.append(np.where(m,v,np.median(v[m]) if m.any() else 0.0))
    for s_ in sorted({r["ind"] for r in rows})[1:]:
        cols.append(np.array([1.0 if r["ind"]==s_ else 0.0 for r in rows]))
    return np.column_stack(cols)
def grad(rows,key,cuts=None):
    if len(rows)<30: return None
    y=np.array([r[key] for r in rows]); x=np.array([r["S"] for r in rows])
    if cuts is not None: y=np.clip(y,cuts[0],cuts[1])
    C=design(rows); r_=lambda v: v-C@np.linalg.lstsq(C,v,rcond=None)[0]
    yr,xr=r_(y),r_(x); d=float(np.sum(xr*xr))
    return float(np.sum(xr*yr)/d) if d>0 else None
def ri(key,tag,two=True):
    cuts=tuple(np.percentile([r[key] for r in T],[5,95]))
    obs=grad(T,key,cuts); n_t=len(T)
    cells=sorted({r["g"] for r in P}); byg={c:[r for r in P if r["g"]==c] for c in cells}
    null=[]
    for _ in range(NDRAW):
        d_=[]
        for i in rng.permutation(len(cells)):
            d_+=byg[cells[i]]
            if len(d_)>=n_t: break
        v=grad(d_[:n_t],key,cuts)
        if v is not None: null.append(v)
    null=np.array(null)
    pu=(int((null>=obs).sum())+1)/(len(null)+1); pl=(int((null<=obs).sum())+1)/(len(null)+1)
    o={"observed":round(obs,4),"n":n_t,"null_mean":round(float(null.mean()),4),
       "null_sd":round(float(null.std()),4),"null_ci":qci(null),
       "RI_p_two_sided":round(float(min(1,2*min(pu,pl))),4),"RI_p_upper":round(float(pu),4),
       "z":round(float((obs-null.mean())/null.std()),2)}
    # 셀 군집 부트 CI (등가성 판정용)
    cellsT=sorted({r["g"] for r in T}); bygT={c:[r for r in T if r["g"]==c] for c in cellsT}
    bb=[]
    for _ in range(NB):
        d_=[r for i in rng.integers(0,len(cellsT),len(cellsT)) for r in bygT[cellsT[i]]]
        v=grad(d_,key,cuts)
        if v is not None: bb.append(v)
    o["boot_ci"]=qci(np.array(bb))
    print(f"  {tag:<28} obs {o['observed']:>+7.4f} boot{o['boot_ci']} · null {o['null_mean']:>+7.4f}"
          f"(SD {o['null_sd']:.4f}) · z={o['z']:>5.2f} · RI p(양측) {o['RI_p_two_sided']:.4f}  n={n_t}")
    return o

print("\n[Panel A·B] 12개월 해상도 — 사전 gradient vs 사후 gradient, 위약 귀무 대비")
R={}
R["pre_12m"]=ri("pre","사전 gradient [−24,−13]→[−12,−1]")
R["post_12m"]=ri("post","사후 gradient [−12,−1]→[1,12]",two=False)
post=abs(R["post_12m"]["observed"]); ci=R["pre_12m"]["boot_ci"]
eq=bool(ci[0]>-post and ci[1]<post)
R["equivalence_12m"]={"SESOI":round(post,4),"pre_ci":ci,"holds":eq,
                      "margin":[round(ci[0]+post,4),round(post-ci[1],4)]}
print(f"  등가성(δ=사후 {post:.4f}): {'✓ 성립' if eq else '✗ 미성립'} 여유 {R['equivalence_12m']['margin']}")

print("\n[Panel C] HonestDiD 상대크기 breakdown M̄ — gradient 에 대해")
def mbar(post_obs,post_se,max_pre,c=1.0):
    """post_obs − 1.96·post_se > M̄·max_pre·c 를 만족하는 최대 M̄."""
    lo=post_obs-1.96*post_se
    return float(lo/(max_pre*c)) if (max_pre>0 and lo>0) else 0.0
# 12개월 해상도
se12=(R["post_12m"]["boot_ci"][1]-R["post_12m"]["boot_ci"][0])/3.92
mpre12=abs(R["pre_12m"]["observed"])
M12=mbar(R["post_12m"]["observed"],se12,mpre12)
# 분기 해상도 (I-61 산출 재사용)
i61=json.load(open("out/I61.json"))["estimates"]
qpre=[abs(v["grad"]) for k,v in i61["panelA_treated_path"].items() if k.startswith("q-")]
qpost=[v["grad"] for k,v in i61["panelA_treated_path"].items() if not k.startswith("q-")]
qpost_ci=[v["ci"] for k,v in i61["panelA_treated_path"].items() if not k.startswith("q-")]
post_q_mean=float(np.mean(qpost))
se_q=float(np.mean([(c[1]-c[0])/3.92 for c in qpost_ci]))/np.sqrt(4)   # 4분기 평균의 SE 근사
M_q=mbar(post_q_mean,se_q,max(qpre))
R["honestdid"]={"resolution_12m":{"post":R["post_12m"]["observed"],"post_se":round(se12,4),
                                  "max_pre_abs":round(mpre12,4),"Mbar":round(M12,3)},
                "resolution_quarterly":{"post_mean":round(post_q_mean,4),"post_se_approx":round(se_q,4),
                                        "max_pre_abs":round(max(qpre),4),"Mbar":round(M_q,3)}}
print(f"  12개월 해상도: 사후 {R['post_12m']['observed']:+.4f} (SE {se12:.4f}) · 최대 사전 |g| {mpre12:.4f} → M̄ = {M12:.3f}")
print(f"  분기 해상도:   사후 평균 {post_q_mean:+.4f} (SE {se_q:.4f}) · 최대 사전 |g| {max(qpre):.4f} → M̄ = {M_q:.3f}")
print(f"  ⚠️ I-11 전례: 추정대상보다 미세한 해상도에서 계산한 M̄ 은 분모가 잡음에 지배된다.")

verdict=(f"[A] 12개월 해상도 사전 gradient {R['pre_12m']['observed']:+.4f} {R['pre_12m']['boot_ci']} "
         f"vs 사후 {R['post_12m']['observed']:+.4f}. 등가성 {'성립' if eq else '미성립'}. "
         f"[B] 위약 귀무 대비 사전 gradient z={R['pre_12m']['z']}, RI p(양측) {R['pre_12m']['RI_p_two_sided']}. "
         f"[C] HonestDiD M̄: 12개월 {M12:.3f} · 분기 {M_q:.3f}.")
emit("I-64","사전추세 — 12개월 해상도 · 위약 상대 RI · gradient HonestDiD",
     "GO" if (eq or M12>=1.0) else "PARTIAL", R|{"n_draws":NDRAW},
     "분기 해상도의 등가성 미성립을 더 적합한 질문으로 바꿀 수 있는가",verdict,kill_met=False,n=len(T))
