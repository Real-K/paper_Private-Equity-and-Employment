# -*- coding: utf-8 -*-
"""I-05a exit 후보 정제 (I-05 본검정 선행).

[v1 결함 3건 — v2에서 수정]
 (1) 같은 해 복수 기준일의 지분율을 합산해 180% 가 나왔다 → **연도별 마지막 기준일**만 사용.
 (2) '국민연금기금'(영문 National Pension **Fund**)이 PE 로 오탐 → 연기금·증권·은행 제외어 추가.
 (3) 진입지분 2~13% VC 조합을 딜 스폰서로 오인 → **스폰서 주주 1인을 특정해 그 주주만 추적**.

스폰서 특정 규칙: 딜연도 또는 그 직후 기준일에 처음 등장한 PE 성격 주주 중 **지분율 최대**,
그리고 그 지분율이 **>=10%** 인 경우만 스폰서로 인정한다. 그 주주가 이후 기준일에서 사라지면 exit 후보.

판정 근거: E1 소멸 직전 지분율 · E2 후속 PitchBook 딜(Secondary/IPO) · E3 신규 대형주주 ·
E4 사후 패널 가용월수(>=12) · E5 기업 생존.

산출: out/I05a.json · work/I05_EXIT_ADJUDICATION.csv · work/I05_WORK_PACKAGE.md
"""
import os, gc, re
from difflib import SequenceMatcher
import numpy as np, pandas as pd
from h30_common import load, deals, build, emit, BASE

WORK = f"{BASE}/P014_upgrade_package/harness30/work"
os.makedirs(WORK, exist_ok=True)
print("[I-05a v2] 로딩...")
G = load()
orig, allt, PE, META = deals(G)
EV, _ = build(G, allt, PE)
EVBN = {e["bn"]: e for e in EV}
Ev, mis = G["Ev"], G["mis"]
PANEL_END = int(mis.max())
print(f"  이벤트 {len(EV)} · 패널 종료 {PANEL_END//12}-{PANEL_END%12 or 12}")

T = pd.read_csv(f"{BASE}/shared/data/processed/p014_treated_sample_v2_expanded.csv", dtype=str)
TB = set(T.bn10.str.zfill(10))
cols = ["business_number", "company_name", "기준일", "주주명", "주주명_영문", "관계", "보통주_지분율"]
parts = []
for ch in pd.read_csv(f"{BASE}/PI/drops/외감_주주_시계열_2009plus.csv",
                      usecols=cols, dtype=str, chunksize=400_000):
    ch["bn10"] = ch.business_number.str.replace(r"\D", "", regex=True).str.zfill(10)
    parts.append(ch[ch.bn10.isin(TB)])
S = pd.concat(parts, ignore_index=True); del parts; gc.collect()
S["dt"] = pd.to_datetime(S["기준일"], format="%Y%m%d", errors="coerce")
S = S[S.dt.notna()].copy(); S["yr"] = S.dt.dt.year
S["pct"] = pd.to_numeric(S["보통주_지분율"], errors="coerce").fillna(0.0)

# (1) 연도별 마지막 기준일만
last_dt = S.groupby(["bn10", "yr"])["dt"].transform("max")
S = S[S.dt == last_dt].copy()

# (2) 제외어 강화
EXCL = (r"우리사주|자기주식|자사주|종업원지주|국민연금|공무원연금|사학연금|군인연금|"
        r"National Pension|Teachers|예금보험|산업은행|기업은행|신용보증|기술보증")
PAT = (r"투자|인베스트|캐피탈|사모|펀드|조합|파트너스|에쿼티|벤처|PEF|"
       r"Capital|Invest|Partner|Equity|Fund|Holdings")
nm = S["주주명"].fillna("") + " " + S["주주명_영문"].fillna("")
S["pe"] = nm.str.contains(PAT, case=False, regex=True) & ~nm.str.contains(EXCL, case=False, regex=True)
# [v3] 주주명 정규화 + 기업내 퍼지 클러스터링 — 표기 변형을 같은 주주로 묶는다.
# 실측 오탐 예시(회사명은 공개 저장소에서 제거): 영문 등기명 vs 한글 등기명, 법인격 표기 위치 차이((주) 접두/접미),
#           홀딩스/홀딩즈 표기 차이 등 — 이런 쌍이 원시 문자열 비교에서 불일치로 잡히므로 아래 정규화를 거친다.
def _norm(x):
    x = str(x).lower()
    x = re.sub(r"주식회사|유한회사|유한책임회사|합자회사|\(주\)|\(유\)|㈜|limited|ltd|inc|corp|company|co\b", "", x)
    x = x.replace("홀딩즈", "홀딩스").replace("홀딩", "홀딩")
    return re.sub(r"[^0-9a-z가-힣]", "", x)
S["nm_norm"] = S["주주명"].map(_norm)
CLU = {}
for bn, g in S.groupby("bn10"):
    reps = []
    for v in sorted(g.nm_norm.unique(), key=len, reverse=True):
        if not v: CLU[(bn, v)] = v; continue
        hit = next((r for r in reps if v == r or (len(v) >= 5 and len(r) >= 5 and
                    (v in r or r in v or SequenceMatcher(None, v, r).ratio() >= 0.85))), None)
        if hit is None: reps.append(v); hit = v
        CLU[(bn, v)] = hit
S["sh_key"] = [CLU[(b, v)] for b, v in zip(S.bn10, S.nm_norm)]
_raw, _clu = S.groupby("bn10")["nm_norm"].nunique().sum(), S.groupby("bn10")["sh_key"].nunique().sum()
print(f"  주주명 클러스터링: 정규화 고유 {_raw} → 클러스터 {_clu} ({_raw-_clu}건 병합)")
CNAME = S.groupby("bn10")["company_name"].last().to_dict()
print(f"  주주 {len(S):,}행(연말스냅샷) · 기업 {S.bn10.nunique()} · PE플래그 {int(S.pe.sum()):,}행")

# (3) 스폰서 주주 1인 특정 후 그 주주만 추적
MIN_SPONSOR = 10.0
rows = []
for bn, g in S.groupby("bn10"):
    if bn not in EVBN: continue
    e = EVBN[bn]; dy = (e["m0"] - 1) // 12
    yrs = sorted(g.yr.unique())
    pre = [y for y in yrs if y < dy]
    held_pre = set(g.loc[g.pe & (g.yr == max(pre)) & (g.pct > 0), "sh_key"]) if pre else set()
    cand = g[g.pe & (g.yr.isin([y for y in yrs if dy <= y <= dy + 2])) & (g.pct >= MIN_SPONSOR)]
    cand = cand[~cand.sh_key.isin(held_pre)]                       # 딜 전부터 있던 주주 제외
    if cand.empty: continue
    sp_row = cand.sort_values("pct", ascending=False).iloc[0]
    sp, spk = sp_row["주주명"], sp_row["sh_key"]
    tr = g[g.sh_key == spk].groupby("yr")["pct"].sum().reindex(yrs).fillna(0.0)
    posy = [y for y in yrs if tr[y] > 0]
    if not posy: continue
    ly = max(posy); aft = [y for y in yrs if y > ly]
    if not aft or any(tr[y] > 0 for y in aft): continue           # 소멸 미관측
    fz = min(aft)
    exit_mi = fz * 12 + 12
    alive = int(np.isfinite(Ev[e["ti"], [i for i, m in enumerate(mis) if m > exit_mi]]).sum())
    # SPC 합병 판별: 스폰서 소멸 후에도 다른 PE 주주가 남아있는가
    pe_after = g[(g.yr >= fz) & g.pe & (g.pct > 0) & (g.sh_key != spk)]
    pe_rem = round(float(pe_after.groupby("yr")["pct"].sum().max()), 2) if len(pe_after) else 0.0
    pe_rem_nm = "; ".join(sorted(set(pe_after.주주명))[:2])[:60]
    holdco = bool(pd.Series([sp]).str.contains(r"홀딩스|Holdings|홀딩|지주", case=False, regex=True).iloc[0])
    othr = g[(g.yr == fz) & (~g.pe)]["pct"]
    othb = g[(g.yr == ly) & (~g.pe)]["pct"]
    pre_top = float(othb.max()) if len(othb) else np.nan
    post_top = float(othr.max()) if len(othr) else np.nan
    newbig = bool(np.isfinite(post_top) and (not np.isfinite(pre_top) or post_top - pre_top > 15))
    rows.append(dict(bn10=bn, company_name=CNAME.get(bn, ""), deal_yr=dy,
                     deal_type=str(META["Deal Type"].get(bn, "")), sponsor=sp,
                     stake_at_entry=round(float(tr[min(posy)]), 2), last_pos_yr=int(ly),
                     first_zero_yr=int(fz), hold_yr=int(ly - dy),
                     stake_at_exit=round(float(tr[ly]), 2),
                     top_other_before=None if not np.isfinite(pre_top) else round(pre_top, 2),
                     top_other_after=None if not np.isfinite(post_top) else round(post_top, 2),
                     new_big_holder=newbig, pe_remains_after=pe_rem, pe_remains_name=pe_rem_nm,
                     sponsor_is_holdco=holdco, post_months_alive=alive))
X = pd.DataFrame(rows)
print(f"\n스폰서 특정(>={MIN_SPONSOR}%, 딜전 보유자 제외) 후 소멸 후보 **{len(X)}건** (v1: 44건)")
if len(X) == 0:
    emit("I-05a", "exit 후보 정제", "BLOCKED", {"n": 0}, "", "스폰서 특정 후 후보 0건", True, 0); raise SystemExit

# 후속 PitchBook 딜
pbf = pd.read_csv(f"{BASE}/shared/data/processed/pitchbook_deals_v1.csv", dtype=str)
pbf["bn10"] = pbf.bn.astype(str).str.zfill(10)
pbf["dd"] = pd.to_datetime(pbf["Deal Date"], errors="coerce"); pbf = pbf[pbf.dd.notna()]
pbf["dy"] = pbf.dd.dt.year
nxt, sec = [], []
for r in X.itertuples():
    s = pbf[(pbf.bn10 == r.bn10) & (pbf.dy > r.deal_yr) & (pbf.dy <= r.first_zero_yr + 1)]
    nxt.append("; ".join(f"{q.dy}:{q._8}" for q in s.itertuples())[:110] if len(s) else "")
    sec.append(bool(s["Deal Type 2"].fillna("").str.contains("Secondary", case=False).any()
                    or (s["Deal Type"] == "IPO").any()) if len(s) else False)
X["next_pb_deals"], X["pb_secondary_or_ipo"] = nxt, sec

def verdict(r):
    if r.post_months_alive < 12: return "EXCL_NO_POST_PANEL"
    # SPC 합병/재편 의심: 스폰서가 사라져도 다른 PE 주주가 유의미하게 남음
    if r.pe_remains_after >= 10:
        return "SUSPECT_SPC_MERGER" if r.sponsor_is_holdco else "AMBIGUOUS"
    if r.sponsor_is_holdco and r.hold_yr <= 1 and not r.pb_secondary_or_ipo:
        return "SUSPECT_SPC_MERGER"      # 홀딩스 SPC 가 1년 내 소멸 = 합병 전형
    if r.pb_secondary_or_ipo or r.new_big_holder:
        return "CONFIRMED_STRONG" if r.stake_at_exit >= 10 else "AMBIGUOUS"
    if r.stake_at_exit >= 20: return "CONFIRMED"
    if r.stake_at_exit < 10: return "LIKELY_ARTIFACT"
    return "AMBIGUOUS"
X["auto_verdict"] = [verdict(r) for r in X.itertuples()]
X["needs_manual"] = X.auto_verdict.isin(["AMBIGUOUS", "LIKELY_ARTIFACT", "SUSPECT_SPC_MERGER"])
X["manual_verdict"] = ""; X["manual_reason"] = ""
print(X.auto_verdict.value_counts().to_string())
use = X[X.auto_verdict.isin(["CONFIRMED_STRONG", "CONFIRMED"])]
print(f"\n자동확정 {len(use)} · 수동판정 {int(X.needs_manual.sum())} · 패널부족 제외 "
      f"{int((X.auto_verdict=='EXCL_NO_POST_PANEL').sum())}")
print(f"소멸 직전 지분율 중앙값 {X.stake_at_exit.median():.1f}% · 보유기간 중앙값 {X.hold_yr.median():.0f}년"
      f" · 사후 가용월 중앙값 {X.post_months_alive.median():.0f}")

C = ["bn10","company_name","deal_yr","deal_type","sponsor","sponsor_is_holdco","stake_at_entry",
     "last_pos_yr","first_zero_yr","hold_yr","stake_at_exit","pe_remains_after","pe_remains_name",
     "top_other_before","top_other_after","new_big_holder","next_pb_deals","pb_secondary_or_ipo",
     "post_months_alive","auto_verdict","needs_manual","manual_verdict","manual_reason"]
X[C].sort_values(["needs_manual","auto_verdict","stake_at_exit"], ascending=[False,True,False]) \
   .to_csv(f"{WORK}/I05_EXIT_ADJUDICATION.csv", index=False, encoding="utf-8-sig")
print(f"→ work/I05_EXIT_ADJUDICATION.csv ({len(X)}행)")

emit("I-05a", "exit 후보 정제 (스폰서 단위 추적)", "GO" if len(use) >= 30 else "PARTIAL",
     {"n_candidates": len(X), "v1_candidates": 44, "sponsor_min_stake": MIN_SPONSOR,
      "auto_verdict_counts": X.auto_verdict.value_counts().to_dict(),
      "n_usable_auto": int(len(use)), "n_needs_manual": int(X.needs_manual.sum()),
      "stake_at_exit_median": float(X.stake_at_exit.median()),
      "hold_yr_median": float(X.hold_yr.median()),
      "post_months_median": float(X.post_months_alive.median())},
     "스폰서 주주 1인을 특정해 그 주주의 소멸만 exit 로 세고, 지분율·후속딜·신규주주·사후패널로 정제",
     f"v1 44건 → v2 {len(X)}건 (오탐 제거). 자동확정 {len(use)} · 수동 {int(X.needs_manual.sum())} · "
     f"패널부족 {int((X.auto_verdict=='EXCL_NO_POST_PANEL').sum())}",
     kill_met=False, n=len(X),
     extra={"v1_defects_fixed": ["연도내 복수기준일 합산(180% 발생)", "국민연금기금 오탐",
                                 "진입 2~13% VC조합을 스폰서로 오인"],
            "work_package": "work/I05_EXIT_ADJUDICATION.csv",
            "spc_merger_guard": "스폰서(홀딩스 SPC) 소멸 후 다른 PE 주주가 >=10% 남거나, 홀딩스가 "
                                "1년 내 소멸+후속딜 없음 → SUSPECT_SPC_MERGER (한국 LBO 인수목적법인 "
                                "합병 전형). exit 으로 세면 안 된다."})
