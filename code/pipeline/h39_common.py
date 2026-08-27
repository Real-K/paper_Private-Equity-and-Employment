# -*- coding: utf-8 -*-
"""H39/H40 공용 — 확장 표본(379) 및 기존 표본(283) 이벤트 구축 + 결과 함수.

H38에서 확립: 사용자 수동판정 103사 반영 → 통합 최초딜 752, 매칭 이벤트 379(기존 283 + 회수 96).
대조 제외집합도 752로 정정(회수기업이 대조 풀에 섞여 있던 오염 제거).
"""
import os,re,warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

BASE=os.environ.get("P014_BASE","/path/to/project-root")   # project root holding shared/data (licensed; see DATA_ACCESS.md)
NB=999
SIZE_B=[5,10,20,50,100,250,np.inf]

def load():
    p=pd.read_parquet(f"{BASE}/shared/data/processed/nps_monthly_matched_v2.parquet",
        columns=["bn10","data_ym","가입자수","신규","상실","고지금액","업종","시도","적용일자","n_sites"])
    p["mi"]=p["data_ym"].str[:4].astype(int)*12+p["data_ym"].str[5:7].astype(int)
    piv=lambda c: p.pivot_table(index="bn10",columns="mi",values=c,aggfunc="first")
    E=piv("가입자수"); H=piv("신규").reindex(E.index); S=piv("상실").reindex(E.index)
    A=piv("고지금액").reindex(E.index); NS=piv("n_sites").reindex(E.index)
    mis=E.columns.to_numpy(); mset={m:i for i,m in enumerate(mis)}
    G=dict(Ev=E.to_numpy(float),Hv=H.to_numpy(float),Sv=S.to_numpy(float),
           Av=A.to_numpy(float),NSv=NS.to_numpy(float),mis=mis,mset=mset,idx=pd.Index(E.index))
    firm=p.sort_values("mi").groupby("bn10").agg(ind=("업종","last"),sido=("시도","last"),adpt=("적용일자","last"))
    firm["ind2"]=firm["ind"].astype(str).str.zfill(6).str[:2]
    ay=pd.to_datetime(firm["adpt"],errors="coerce"); firm["adpt_mi"]=ay.dt.year*12+ay.dt.month
    firm=firm.reindex(G["idx"])
    G["ind_arr"]=firm["ind2"].to_numpy(); G["sido_arr"]=firm["sido"].astype(str).to_numpy()
    G["adpt_arr"]=firm["adpt_mi"].to_numpy(float)
    return G

def deals(G):
    pbf=pd.read_csv(f"{BASE}/shared/data/processed/pitchbook_deals_v1.csv",dtype=str)
    bg=pbf[(pbf.is_bg=="True")&pbf.bn.notna()&pbf["Deal Date"].notna()].copy()
    dd=pd.to_datetime(bg["Deal Date"],errors="coerce")
    bg=bg[dd.notna()].assign(mi=dd.dt.year*12+dd.dt.month)
    orig=bg.sort_values("mi").drop_duplicates("bn")
    meta_o=orig.set_index(orig.bn.astype(str).str.zfill(10))[["Deal Type","pct_acq"]]
    orig=orig[["bn","mi"]].rename(columns={"bn":"bn10"})
    orig["bn10"]=orig.bn10.astype(str).str.zfill(10); orig["src"]="orig"
    A=pd.read_csv(f"{BASE}/P014_upgrade_package/matching/work/PB_RECOVERY_FINAL_ADOPTED.csv",dtype=str)
    A["bn10"]=A.bn10.astype(str).str.zfill(10)
    rec=A[["bn10","deal_mi"]].copy(); rec["mi"]=rec.deal_mi.astype(float).astype(int); rec["src"]="rec"
    meta_r=A.set_index("bn10")[["pb_deal_type","pb_pct"]].rename(
        columns={"pb_deal_type":"Deal Type","pb_pct":"pct_acq"})
    META=pd.concat([meta_o,meta_r]); META=META[~META.index.duplicated()]
    allt=pd.concat([orig[["bn10","mi","src"]],rec[["bn10","mi","src"]]]).sort_values("mi").drop_duplicates("bn10")
    PE_OLD=set(pbf.loc[pbf.is_bg=="True","bn"].dropna().astype(str).str.zfill(10))
    PE_NEW=PE_OLD|set(rec.bn10)
    return orig[["bn10","mi","src"]],allt,PE_NEW,META

def build(G,ev_df,pe_exclude,ctrl_extra_exclude=None):
    idx=G["idx"]; Ev=G["Ev"]; mset=G["mset"]
    excl=set(pe_exclude)|set(ctrl_extra_exclude or [])
    ctrl_ok=~idx.isin(excl)
    cache={}; EV=[]
    for r in ev_df.itertuples():
        m0=int(r.mi)
        if r.bn10 not in idx: continue
        ti=idx.get_loc(r.bn10)
        if m0 not in cache:
            iw=[mset[m] for m in range(m0-6,m0) if m in mset]; i18=[mset[m] for m in range(m0-18,m0-12) if m in mset]
            if not iw or not i18: cache[m0]=None
            else:
                with np.errstate(all="ignore"):
                    Ep=np.nanmean(Ev[:,iw],axis=1); g=Ep/np.nanmean(Ev[:,i18],axis=1)-1
                cache[m0]=(Ep,g,np.digitize(Ep,SIZE_B,right=False),
                    np.where(np.isnan(g),-1,np.digitize(g,[-0.10,0.10])),
                    np.where(np.isnan(G["adpt_arr"]),-1,np.digitize((m0-G["adpt_arr"])/12.0,[5,15])))
        c=cache[m0]
        if c is None: continue
        Ep,g,sb,gb,ageb=c
        if not (np.isfinite(Ep[ti]) and Ep[ti]>=5): continue
        same=(ctrl_ok&(G["ind_arr"]==G["ind_arr"][ti])&(sb==sb[ti])&(gb==gb[ti])&(ageb==ageb[ti])
              &(Ep>=5)&np.isfinite(Ep))
        cand=np.flatnonzero(same); cand=cand[cand!=ti]
        if len(cand)==0: continue
        gt=g[ti] if np.isfinite(g[ti]) else 0.0
        gc=np.where(np.isfinite(g[cand]),g[cand],0.0)
        dist=((np.log(Ep[cand])-np.log(Ep[ti]))/0.9)**2+((np.clip(gc,-1,2)-np.clip(gt,-1,2))/0.35)**2
        EV.append({"bn":r.bn10,"ti":ti,"m0":m0,"ctrls":cand[np.argsort(dist)[:5]],
                   "Epre":float(Ep[ti]),"g":float(gt),"src":r.src,"year":(m0-1)//12})
    return EV,cache

def widx(G,m0,a,b): return [m for m in (G["mset"].get(x) for x in range(m0+a,m0+b+1)) if m is not None]
def flow(G,row,m0,a,b,M):
    c=widx(G,m0,a,b)
    if len(c)<6: return np.nan
    den=np.nanmean(G["Ev"][row,c])
    return np.nansum(M[row,c])/den if np.isfinite(den) and den>0 else np.nan
def dflow(G,row,m0,M):
    a=flow(G,row,m0,-12,-1,M); b=flow(G,row,m0,1,12,M)
    return b-a if (np.isfinite(a) and np.isfinite(b)) else np.nan
def rel_log(G,ti,ct,m0,k=12):
    cmi=m0+k; iw=widx(G,m0,-6,-1)
    if cmi not in G["mset"] or not iw: return np.nan
    Ev=G["Ev"]
    with np.errstate(all="ignore"):
        bt=np.nanmean(Ev[ti,iw]); bc=np.nanmean(Ev[np.ix_(ct,iw)],axis=1)
        te=Ev[ti,G["mset"][cmi]]; ce=Ev[ct,G["mset"][cmi]]
        if not (np.isfinite(te) and te>0 and np.isfinite(bt) and bt>=5): return np.nan
        lc=np.log(np.where((ce>0)&(bc>0),ce,np.nan))-np.log(np.where(bc>0,bc,np.nan))
        v=(np.log(te)-np.log(bt))-np.nanmean(lc)
    return float(v) if np.isfinite(v) else np.nan
def pi_parts(G,row,m0,a,b):
    c=widx(G,m0,a,b)
    if len(c)<6: return None
    h=G["Hv"][row,c]; e=G["Ev"][row,c]; ok=np.isfinite(h)
    if ok.sum()<6: return None
    h=h[ok]; base=np.nanmean(e[ok])
    if not (np.isfinite(base) and base>0): return None
    act=h[h>0]
    return float((h>0).mean()),(float(act.mean()/base) if len(act) else 0.0)
def qci(v): return [round(float(np.percentile(v,2.5)),4),round(float(np.percentile(v,97.5)),4)]
def attach(G,EV):
    for e in EV:
        e["t"]=dflow(G,e["ti"],e["m0"],G["Hv"])
        cs=np.array([dflow(G,c,e["m0"],G["Hv"]) for c in e["ctrls"]],float); e["cs"]=cs[np.isfinite(cs)]
        e["rel"]=rel_log(G,e["ti"],e["ctrls"],e["m0"])
    return EV
def summ(EV,rng,lab="",log=print):
    r=[e for e in EV if np.isfinite(e.get("t",np.nan)) and len(e.get("cs",[]))>0]; n=len(r)
    if n<20:
        if lab: log(f"      {lab:30s} n={n} (<20)")
        return {"n":n,"note":"n<20"}
    td=np.array([e["t"] for e in r]); cdm=np.array([e["cs"].mean() for e in r])
    rel=np.array([e["rel"] for e in r],float)
    bd=np.empty(NB);bp=np.empty(NB);br=np.empty(NB)
    for i in range(NB):
        j=rng.integers(0,n,n); bd[i]=(td[j]-cdm[j]).mean()
        pc=np.concatenate([r[k]["cs"] for k in j]); bp[i]=(td[j]<0).mean()-(pc<0).mean()
        rj=rel[j][np.isfinite(rel[j])]; br[i]=rj.mean() if len(rj) else np.nan
    pool=np.concatenate([e["cs"] for e in r]); rr=rel[np.isfinite(rel)]
    o={"n":n,"DiD":round(float((td-cdm).mean()),4),"DiD_ci":qci(bd),
       "P1":round(float((td<0).mean()-(pool<0).mean()),4),"P1_ci":qci(bp),
       "rel":round(float(rr.mean()),4) if len(rr)>=20 else None,
       "rel_ci":qci(br[np.isfinite(br)]) if len(rr)>=20 else None}
    if lab: log(f"      {lab:30s} n={o['n']:3d} | DiD {o['DiD']:+.4f}{o['DiD_ci']} | "
                f"P1 {o['P1']:+.4f}{o['P1_ci']} | rel {o['rel']}{o['rel_ci']}")
    return o
def tercile(EV,key,lab,names,rng,log=print):
    s=[e for e in EV if np.isfinite(e.get(key,np.nan)) and np.isfinite(e.get("t",np.nan)) and len(e.get("cs",[]))>0]
    if len(s)<60:
        log(f"    -- {lab}: n={len(s)} (<60)"); return {"n":len(s),"note":"n<60"}
    v=np.array([e[key] for e in s]); a1,a2=np.percentile(v,[33.33,66.67])
    out={"n":len(s),"cuts":[round(float(a1),4),round(float(a2),4)]}
    log(f"    -- {lab} 컷 {a1:.4f}/{a2:.4f} n={len(s)} --")
    ms=[v<=a1,(v>a1)&(v<=a2),v>a2]
    for nm,m in zip(names,ms): out[nm]=summ([s[i] for i in np.flatnonzero(m)],rng,f"{lab}:{nm}",log)
    s1=[s[i] for i in np.flatnonzero(ms[0])]; s3=[s[i] for i in np.flatnonzero(ms[2])]
    if len(s1)>=20 and len(s3)>=20:
        def p1(ss,j):
            td=np.array([ss[k]["t"] for k in j]); pc=np.concatenate([ss[k]["cs"] for k in j])
            return (td<0).mean()-(pc<0).mean()
        bs=np.array([p1(s1,rng.integers(0,len(s1),len(s1)))-p1(s3,rng.integers(0,len(s3),len(s3))) for _ in range(NB)])
        obs=p1(s1,np.arange(len(s1)))-p1(s3,np.arange(len(s3)))
        out["T1_minus_T3_P1"]={"pt":round(float(obs),4),"ci":qci(bs),"sig":bool(qci(bs)[0]>0 or qci(bs)[1]<0)}
        log(f"      T1−T3 P1 {obs:+.4f} {qci(bs)} {'✓' if (qci(bs)[0]>0 or qci(bs)[1]<0) else ''}")
    return out
