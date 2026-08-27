# -*- coding: utf-8 -*-
"""Table 1~7 — 저널 게재 포맷. 회귀표(SE·FE·N·clusters·pseudo-R²)와 매칭추정치표(CI)를 분리한다.

[설계 판단] 주 추정치의 상당수는 회귀가 아니라 **매칭쌍 부트스트랩 차분**이다. 여기에 R² 를
붙이면 추정량을 왜곡한다. 따라서 hazard 모형만 정식 회귀표로, 나머지는 CI 기반 추정치표로 낸다.
"""
import json, os, csv
_HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.environ.get("P014_ARTIFACTS", os.path.join(_HERE, "..", "..", "artifacts"))      # aggregate result artifacts
OUTDIR = os.environ.get("P014_TABLES", os.path.join(_HERE, "..", "..", "figures"))         # where tables.md is written
J = lambda f: json.load(open(os.path.join(ART, f + ".json"), encoding="utf-8"))
L = {r["claim_id"]: r for r in csv.DictReader(open(os.path.join(ART, "CLAIMS_LEDGER_v4.csv"), encoding="utf-8-sig"))}

def st(p):
    return "***" if p < .01 else ("**" if p < .05 else ("*" if p < .10 else ""))
def E(cid, dec=4, pct=False):
    r = L[cid]; v = float(r["value"])
    s = f"{v*100:.1f}" if pct else f"{v:.{dec}f}"
    return s, r["ci95"], (r["n"] or "—")
def ROW(cid, lab, dec=4, pct=False, unit=""):
    v, ci, n = E(cid, dec, pct)
    return f"| {lab} | {v}{unit} | {ci} | {n} |"
EST_HD = "| | Estimate | 95% CI | Events |\n|:---|---:|:---:|---:|"

i36 = J("I36")["estimates"]; S = i36["specs"]
i37 = J("I37")["estimates"]
T = {}
AT = {}   # 부록 표 (리뷰 5 §27: hazard 를 본문에서 부록으로)

# ───────── Table 1 ─────────
BAL = "\n".join(
    f"| {r['var']} | {r['treated']:.{r['dec']}f} | {r['matched']:.{r['dec']}f} | "
    f"{r['nd_matched']:+.3f} | {r['pool']:.{r['dec']}f} | {r['nd_pool']:+.3f} |"
    for r in i37["rows"])
T[1] = f"""### Table 1. Sample construction, characteristics, and covariate balance

**Panel A. Sample**

| | |
|:---|---:|
| Private equity investments identified | 752 |
| &nbsp;&nbsp;no pension record | {L['C91']['value']} |
| &nbsp;&nbsp;incomplete pre-deal window | {L['C91a']['value']} |
| &nbsp;&nbsp;fewer than five insured employees | {L['C91b']['value']} |
| &nbsp;&nbsp;no eligible control cell | {L['C91c']['value']} |
| **Baseline matched design** | **379** |
| &nbsp;&nbsp;with ≥ 6 observed months in each window, target and a control (Table 2) | {L['C2']['n']} |
| &nbsp;&nbsp;*memo:* conventional heterogeneity design, no state in the cell | 301 |
| &nbsp;&nbsp;**with state-balanced controls (primary gradient)** | **286** |
| &nbsp;&nbsp;with all deal characteristics observed | 180 |
| &nbsp;&nbsp;with audited financial statements | 147 |
| &nbsp;&nbsp;with complete pre-deal covariates (Panel D) | {i37['n_events']} |
| Matched control firms (5 per target, never treated) | 1,895 |
| Firm-months in the hazard sample | {S['(2)']['n_firm_months']:,} |
| Two-digit industries represented | {i37['n_industries']} |

**Panel B. Targets at the deal**

| | |
|:---|---:|
| Median employees | 79 |
| Mean employees | {[r for r in i37['rows'] if r['var']=='Employees'][0]['treated']:.1f} |
| Median deal year | 2021 |
| Share of firm-months with no hiring, pre-deal | {[r for r in i37['rows'] if 'no-hire months, −12' in r['var']][0]['treated']:.3f} |
| Mean intensity within active months (hires / employees) | 0.055 |

**Panel C. The pre-deal hiring state**

| | |
|:---|---:|
| State variable | −log(1 + hires / employment), months −24 to −13 |
| Normalized difference, treated versus controls, previous design | {L['D06']['value']} |
| Normalized difference, state-balanced design | **{L['D06a']['value']}** |
| Correlation of treated and control state, previous / balanced | {L['D05d']['value']} / {L['D05e']['value']} |
| Auxiliary: share of no-hire months, tercile cut points | 0.083 / 0.333 |

**Panel D. Covariate balance**

| | Treated | Matched controls | ND | Unmatched pool | ND |
|:---|---:|---:|---:|---:|---:|
{BAL}

*Notes.* Matching cells are defined by two-digit industry, size bin, pre-deal growth bin, and firm
age bin, formed at the deal month; the five nearest neighbours within the cell serve as controls.
These baseline cells are used for the average effects (Table 2) and for the balance statistics in
Panel D; the primary gradient design (Tables 3 and 4) adds a tercile of the pre-deal hiring state to
the cell, and Panel C reports balance on the state under both designs.
The control pool excludes all 752 identified targets. ND is the normalized difference
(mean_t − mean_c) / sqrt((var_t + var_c)/2); values below 0.25 in absolute value are conventionally
taken to indicate adequate balance. The unmatched pool column draws {i37['pool_draws_per_event']}
eligible never-treated firms at each event month and reports the same statistics, so the two ND
columns show what the matching accomplishes. The largest absolute normalized difference after
matching is **{i37['max_abs_nd_matched']:.3f}**; before matching it reaches
{max(abs(r['nd_pool']) for r in i37['rows']):.3f}. Canonical sample file sha256₁₆
`65a7e0fc488df3bd`."""

# ───────── 부록 Table F.4.1 : hazard 회귀표 (리뷰 5 §27 — 본문에서 부록으로) ─────────
COLS = ["(1)", "(2)", "(3)", "(4)"]
TERMS = [("treated x post", "Treated × Post"),
         ("treated x post x high inaction", "Treated × Post × Low pre-deal hiring activity"),
         ("treated x post x pressure", "Treated × Post × Sponsor deployment pressure"),
         ("treated", "Treated"), ("post", "Post")]
lines = ["### Appendix Table F.4.1. Discrete-time hazard of hiring in a month",
         "", "| | (1) | (2) | (3) | (4) |", "|:---|:---:|:---:|:---:|:---:|"]
for key, lab in TERMS:
    a, b = [], []
    for cc in COLS:
        t = S[cc]["terms"].get(key)
        a.append(f"{t['coef']:+.4f}{st(t['p'])}" if t else "")
        b.append(f"({t['se']:.4f})" if t else "")
    lines.append(f"| {lab} | " + " | ".join(a) + " |")
    lines.append("| | " + " | ".join(b) + " |")
lines += ["| | | | | |",
          "| *Implied hazard ratio, Treated × Post* | " +
          " | ".join(f"{S[cc]['terms']['treated x post']['HR']:.3f}" for cc in COLS) + " |",
          "| | " + " | ".join("[" + ", ".join(f"{x:.3f}" for x in S[cc]['terms']['treated x post']['ci']) + "]"
                              for cc in COLS) + " |",
          "| | | | | |",
          "| Duration fixed effects | " + " | ".join("Yes" for _ in COLS) + " |",
          "| Event fixed effects | " + " | ".join("Yes" if S[cc]["fe_event"] else "No" for cc in COLS) + " |",
          "| Standard errors clustered by | " + " | ".join("Event" for _ in COLS) + " |",
          "| Clusters (events) | " + " | ".join(f"{S[cc]['n_clusters']:,}" for cc in COLS) + " |",
          "| Observations (cells) | " + " | ".join(f"{S[cc]['n_cells']:,}" for cc in COLS) + " |",
          "| Underlying firm-months | " + " | ".join(f"{S[cc]['n_firm_months']:,}" for cc in COLS) + " |",
          "| Pseudo-R² (McFadden) | " + " | ".join(f"{S[cc]['pseudo_r2_mcfadden']:.3f}" for cc in COLS) + " |",
          "| Log-likelihood | " + " | ".join(f"{S[cc]['loglik']:,.0f}" for cc in COLS) + " |",
          "",
          "*Notes.* Complementary log-log hazard of at least one hire in a firm-month, estimated on "
          "grouped cells, which is likelihood-equivalent to the individual-observation fit. Duration "
          "is the number of consecutive no-hire months preceding the observation, entered as bucket "
          "fixed effects. Low pre-deal hiring activity is the top tercile of the share of no-hire months "
          f"in months −24 to −13 (cut {i36['tercile_cut_inaction']}). Sponsor deployment pressure is the "
          f"top tercile of years since the sponsor's most recent fund close (cut {i36['pressure_cut_years']} "
          "years). Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.10."]
AT["F4.1"] = "\n".join(lines)

# ───────── Table 2 : 주 효과 (매칭 DiD) ─────────
T[2] = f"""### Table 2. Effect on hiring, separations, and the two margins

{EST_HD}
| **Panel A. Employment and rates** | | | |
{ROW('C52','Employment, 12 months after (Δ log)')}
{ROW('C2','12-month hiring rate')}
{ROW('C3','Share of months with no hiring')} †
{ROW('C4','Separation rate')}
| **Panel B. Margin decomposition (logs)** | | | |
{ROW('C5a','Extensive margin, Δ log p')}
{ROW('C5b','Intensive margin, Δ log i')}
| Extensive share of the log increase | {L['C5']['value']} | — | — |
| **Panel C. Margin decomposition (levels)** | | | |
{ROW('C5c','Intensive margin, Δ i')}

*Notes.* Each estimate is the mean difference between the treated firm's change and the mean change
of its five matched controls, comparing the twelve months after the deal with the twelve months
before. Confidence intervals are percentile intervals from 999 event-level bootstrap replications;
these are matched-pair differences rather than regression coefficients, so no R² is reported. The
hiring rate is 12·p·i, where p is the share of months with at least one hire and i the mean hires
per active month scaled by employment; pre-deal values are p = 0.738 and i = 0.055. The separation
interval lies inside ±0.046, the magnitude of the hiring effect, so a change in separations as
large as the change in hiring is excluded at that benchmark. The level form of the intensive
margin sits at the significance boundary and both forms are reported. † This measure, and the
decomposition in Panels B and C that rests on it, move as the increase in hiring volume
a volume-only benchmark implies; Table 6, Panel A separates the two. They are reported as descriptions of
how the increase is expressed in monthly data, not as evidence about the timing of hiring. The
estimates in this table are average effects within the matched population; the heterogeneity of the
response across targets is a separate estimand and is reported in Tables 3 and 4."""

# ───────── Table 3 : 주 상태 gradient + 위약 귀무 + 설계 교정 ─────────
T[3] = f"""### Table 3. The hiring response varies with the target's pre-deal hiring state

{EST_HD}
| **Panel A. Primary specification (state-balanced matching)** | | | |
| Gradient in the log hiring rate | **{L['E01']['value']}** | {L['E01h']['ci95']} | {L['E01']['n']} |
| Placebo null: mean (SD) [95% range] | {L['E01a']['value']} ({L['E01b']['value']}) | {L['E01']['ci95']} | 2,000 draws |
| Excess over the null | {L['E01e']['value']} | — | — |
| Standardized distance | **{L['E01d']['value']}** | — | — |
| Placebo *p* (upper tail) | **{L['E01c']['value']}** | — | — |
| Bootstrap SD (treated-firm clusters) | {L['E01i']['value']} | — | 2,000 draws |
| Bootstrap CI, winsorization cut-offs held fixed | {L['E01j']['value']} | {L['E01j']['ci95']} | — |
| Effect of an interquartile move in the state | {L['E01k']['value']} | {L['E01k']['ci95']} | IQR = {L['E01l']['value']} |
| Minimum detectable gradient (80% power) | {L['E01f']['value']} | — | — |
| **Panel B. What the design correction does** | | | |
| Previous design (no state in the matching cell) | {L['D02']['value']} | *z* = {L['D02c']['value']} | {L['D02']['n']} |
| Previous design, common sample | {L['D03']['value']} | *z* = {L['D03a']['value']} | {L['D03']['n']} |
| Corrected design, common sample | **{L['D03b']['value']}** | *z* = {L['D03c']['value']} | {L['D03']['n']} |
| Decomposition (unwinsorized): treated firms' own gradient, previous | {L['D04']['value']} | — | — |
| Decomposition (unwinsorized): treated firms' own gradient, corrected | {L['D04a']['value']} | — | — |
| Decomposition (unwinsorized): control-group gradient, previous | {L['D04b']['value']} | — | — |
| Decomposition (unwinsorized): control-group gradient, corrected | {L['D04c']['value']} | — | — |
| **Panel C. Not a denominator artefact** | | | |
| Log count of hires (numerator only) | {L['D07']['value']} | null {L['D07a']['value']}, *z* = {L['D07b']['value']} | {L['D07']['n']} |
| Log count of hires, holding Δ log employment fixed | **{L['D08']['value']}** | null {L['D08a']['value']}, *z* = {L['D08b']['value']} | {L['D08']['n']} |
| Log employment (denominator only) | {L['D09d']['value']} | null {L['D09e']['value']}, *z* = {L['D09f']['value']} | {L['D09d']['n']} |

*Notes.* The outcome is the change in the log hiring rate between the twelve months before and the
twelve months after the deal, differenced against matched controls. The state is the firm's hiring
intensity over months −24 to −13, entered as −log(1 + hires/employment) so that a positive gradient
means a larger response among less active firms; the window is separated from the outcome's base so
that no component of the outcome appears on the right-hand side. Matching cells are defined by
two-digit industry, size bin, pre-deal growth bin, age bin **and tercile of the pre-deal hiring
state**. Estimation partials out log pre-deal size, pre-deal employment growth, firm age and
one-digit industry from both the outcome and the state, and the primary specification winsorizes the
outcome at the 5th and 95th percentiles with cut points computed on the treated sample and applied
to the placebo sample at the same absolute values. The placebo null treats each matched
control firm as a pseudo-event, matches it to its own controls by the identical procedure, and draws
from that pool at the treated sample size 2,000 times; the null is centred at {L['E01a']['value']},
not at zero.

Panel B shows that the correction is not sample composition: adding the state to the matching cell
costs 15 of 301 events, and on the common sample the two designs still give {L['D03']['value']} and
{L['D03b']['value']}. The decomposition rows explain where the difference comes from — the treated
firms' own gradient is identical under both designs, and the entire change is in the control group,
whose gradient with respect to the treated firm's state falls from {L['D04b']['value']} to
{L['D04c']['value']}. Table 7 reports the corresponding placebo on the counterfactual.

The specification curve is reported in the text (Section 6.2) rather than repeated here. Panel C
addresses the fact that employment appears in the denominator of the hiring rate and is itself an
outcome. The decomposition rows in Panel B are estimated without winsorization and therefore sum to
the unwinsorized gradient of {L['E02']['value']}, not to the winsorized headline in Panel A. The
interval on the headline gradient is a bootstrap clustered on treated firms; the bracketed range on
the null row is the placebo distribution, not a confidence interval. The placebo *p* for the gradient
is upper-tail because the hypothesis is directional, as are the *p*-values of the alternative
hiring-gradient specifications in Section 6.2; Table 4's *p*-values are two-sided. For reference, the
corresponding two-sided placebo *p*-value for the headline gradient is {L['E01m']['value']}. Table 4
applies the same estimator to the other flow outcomes."""

# ───────── Table 4 : 유량 결과대상 + 쌍대비 (리뷰 5 §27 — Table 3 에서 분리) ─────────
T[4] = f"""### Table 4. The hiring response is not matched by net employment growth

{EST_HD}
| **Panel A. State gradient by outcome** | | | |
| Log hiring rate | {L['D09']['value']} | — | {L['D09']['n']} |
| Log churn, (hires + separations) / employment | {L['D09a']['value']} | *z* = {L['D09b']['value']} | {L['D09a']['n']} |
| Log separation rate | {L['D09h']['value']} | *z* = {L['D09i']['value']}, *p* = {L['D09j']['value']} | {L['D09h']['n']} |
| Log employment | {L['D09d']['value']} | *z* = {L['D09f']['value']}, *p* = {L['D09g']['value']} | {L['D09d']['n']} |
| Relative employment, +12 / +24 / +36 months | {L['C90n']['value']} / {L['C90an']['value']} / {L['C90bn']['value']} | *z* = {L['C90nz']['value']} / {L['C90anz']['value']} / {L['C90bnz']['value']} | {L['C90n']['n']} / {L['C90an']['n']} / {L['C90bn']['n']} |
| **Panel B. Paired contrasts (same events)** | | | |
| Hiring − employment | **{L['D10']['value']}** | null {L['D10a']['value']}, *z* = {L['D10b']['value']} | {L['D10']['n']} |
| Churn − employment | {L['D10c']['value']} | *z* = {L['D10d']['value']} | — |
| Hiring − separations | {L['D10e']['value']} | *p* = {L['D10f']['value']} | — |

*Notes.* The state, the matching cells, the estimator and the placebo null are identical to Table 3;
only the outcome changes. *p*-values in this table are two-sided because these cross-outcome gradients and paired
contrasts were not assigned directional alternatives; hiring-gradient specifications in Table 3 and
Section 6.2 use the upper tail. Panel B reports contrasts formed within event, which net out the
component common to the two outcomes and are correspondingly more precise than differencing two
separately estimated gradients. The separation gradient is positive but not detected, so the
gross-flow reading rests on the churn and paired-contrast rows and we do not claim the separation
channel. Relative employment measures log employment at the stated horizon against the mean of
months −6 to −1, differenced against controls; the twenty-four-month estimate is not individually
detected. The hiring row repeats the Table 3 estimate; its placebo band here is re-estimated alongside
the other outcomes' bands, and Table 3 reports its primary standardized distance."""

# ───────── Table 5 : 불변성 (공통표본 · 조정 · 결합 · 표본외) ─────────
T[5] = f"""### Table 5. Observed transaction characteristics add little detectable explanatory power

{EST_HD}
| **Panel A. Individual comparisons (available sample by characteristic)** | | | |
| Control transfer: buyout − growth, unadjusted | {L['C65']['value']} | {L['C65']['ci95']} | 301 |
| Control transfer: buyout − growth, covariate-adjusted | {L['C66']['value']} | {L['C66']['ci95']} | 301 |
| Acquired stake: slope per percentage point, adjusted | {L['C69']['value']} | {L['C69']['ci95']} | {L['C69']['n']} |
| Acquired stake: majority (≥ 50%) − minority, adjusted | {L['C67']['value']} | {L['C67']['ci95']} | {L['C69']['n']} |
| Sponsor experience: top − bottom tercile, adjusted | {L['C68']['value']} | {L['C68']['ci95']} | 301 |
| Sponsor identity: leave-one-out prediction | {L['C76']['value']} | — | {L['C76']['n']} |
| **Panel B. Joint explanatory power (n = {L['C70']['n']})** | | | |
| Deal characteristics together, R² | {L['C70']['value']} | permutation *p* = {L['C70a']['value']} | — |
| Pre-deal hiring state alone, R² | {L['C71']['value']} | permutation *p* = {L['C71a']['value']} | — |
| Incremental R²: deal characteristics on top of the state | {L['C72']['value']} | — | — |
| Incremental R²: the state on top of deal characteristics | {L['C72a']['value']} | — | — |
| **Panel C. Out-of-sample prediction (five-fold cross-fitting)** | | | |
| Deal characteristics, out-of-sample R² | {L['C73']['value']} | — | {L['C73']['n']} |
| Pre-deal hiring state, out-of-sample R² | {L['C74']['value']} | — | {L['C73']['n']} |
| Difference, state − deal characteristics | {L['C75']['value']} | {L['C75']['ci95']} | — |
| **Memo** | | | |
| Pre-deal hiring state, adjusted terciles | **{L['C84a']['value']}** | **{L['C84a']['ci95']}** | — |

*Notes.* The outcome throughout is the change in the log hiring rate, the same as Table 3. Panel A
reports each comparison on the available sample for that characteristic: the buyout and
sponsor-experience comparisons use 301 events, the stake comparisons 181, and the sponsor
leave-one-out test 189. Panel B provides the like-for-like joint comparison on the 180 events for
which all deal variables are observed, and Appendix E reports each comparison on its own maximal
sample. Covariate
adjustment residualises the event-level response on the pre-deal hiring state, size, growth, age, industry
and deal year — all measured before the transaction — which removes composition and narrows the
intervals by 2 to 10 percent. It matters here: buyout targets are less active before the deal than growth
targets, by {L['C64']['value']} on the share of no-hire months, so part of the unadjusted deal-type
difference is the state gradient rather than the deal type. None of the standard positive transaction contrasts is
detected; the sponsor leave-one-out coefficient is negative and is discussed below and in Appendix E. The
stake is the sum of private equity holders' common-share stakes in the shareholder register at the
entry year; the transaction-level percentage-acquired field is not used because its distribution is
massed at 100% and its upper tercile is empty. Panel B regresses the event-level response on a
buyout indicator, the stake, and log sponsor deal count jointly, and compares the fit with that of
the state alone; permutation p-values come from 2,000 relabellings, and the state used
throughout is the same index as in Table 3. Panel C repeats the comparison out of sample on the
same 180 events: a negative R² means the predictor does worse than assuming every deal has the
average response, and on this subsample **both** predictors are negative, with a difference that is
not statistically distinguishable. The in-sample permutation test separates them; the
out-of-sample exercise does not, and we report both. Sponsor identity is tested by predicting each event from the mean of the same
sponsor's other deals. That coefficient is negative — a sponsor's other deals predict the held-out
deal in the wrong direction. It is evidence against a persistent sponsor style, which is what the
row tests, but the sign is not something the data explain: 164 sponsors contribute 189 leave-one-out
observations, the median sponsor contributes one deal, and dropping the five most influential events
attenuates the coefficient to {L['C76b']['value']}. We report it and do not interpret it."""

# ───────── Table 6 : 무엇이 아닌가 ─────────
T[6] = f"""### Table 6. What the response is not

{EST_HD}
| **Panel A. Timing, benchmarked against the increase in volume** | | | |
{ROW('C41','Share of no-hire months: actual change')}
{ROW('C41a','Change implied by the volume increase alone')}
{ROW('C42','Excess, over and above the benchmark')}
{ROW('C43','Excess longest no-hire spell')}
{ROW('C44','Excess concentration of hiring across months')}
{ROW('C45','Excess no-hire share, 36-month window')}
{ROW('C45a','Excess longest spell, 36-month window')}
| **Panel B. Wages** | | | |
{ROW('C17','Assessed income per worker, Δ log')}
{ROW('C17a','Implied marginal wage of new hires, ratio to controls')}
| **Panel C. Value added, one year after** | | | |
{ROW('C18','Value added, Δ log')}
{ROW('C18a','Value added per worker, Δ log')}

*Notes.* Panel A holds each firm's realized total hires fixed and asks how they are distributed. The
benchmark is the number of no-hire months expected if those hires were allocated at random across
the same months, weighting months by employment; longest spell and concentration use the same
allocation, simulated. The actual changes are significant and the excesses are not: the equivalence
bounds are ±0.046 for the no-hire share, ±0.36 for the longest spell over twelve months, ±0.025 for
concentration, and ±1.0 for the longest spell over thirty-six months, and every excess interval
lies inside its bound. A firm that hires more will be active in more months even with unchanged
timing behaviour, so these measures do not reveal a separate timing effect comparable in magnitude
to the observed change once volume is held fixed.
Assessed income is the pension contribution base, which is top-coded, so only treated–control
differences are interpreted. The marginal-wage ratio is the change in monthly payroll divided by the
change in employment relative to the incumbent average; only its difference from controls is
interpretable because the level is inflated by incumbent wage growth. A wage discount of 20% is
excluded; discounts below about 6.6% are not. Value added is operating income plus payroll observed
in the pension records; the per-worker interval lies inside ±0.10."""

# ───────── Table 7 : 식별 ─────────
T[7] = f"""### Table 7. Identification

{EST_HD}
{ROW('C19','HonestDiD relative-magnitude breakdown value, M̄')}
| Sponsor deployment pressure: Treated × Post × Pressure (hazard ratio) | {S['(4)']['terms']['treated x post x pressure']['HR']:.4f} | [{S['(4)']['terms']['treated x post x pressure']['ci'][0]:.4f}, {S['(4)']['terms']['treated x post x pressure']['ci'][1]:.4f}] | 379 |
{ROW('C21','Effect in the independently date-confirmed subsample')}
| Employment level, least active tercile: treated | {L['C22']['value']} | — | — |
| Employment level: placebo null range, 200 draws | — | {L['C22']['ci95']} | 200 draws |
| Employment level: placebo *p* | < 0.005 | — | — |
{ROW('C23','Share of no-hire periods at annual frequency')}
| **Pre-trends in the gradient (twelve-month resolution)** | | | |
| Pre-deal gradient, state on months −36/−25 (no window overlap) | **{L['H06']['value']}** | excess {L['H06b']['ci95']} | {L['H06']['n']} |
| &nbsp;&nbsp;placebo null mean [95% range] | {L['H06a']['value']} | {L['H06a']['ci95']} | 2,000 draws |
| &nbsp;&nbsp;two-sided *p* | {L['H06c']['value']} | — | — |
| &nbsp;&nbsp;excess interval inside ±0.7101 | {'Yes' if L['H06d']['value'] == 'True' else 'No'} | — | — |
| Pre-deal gradient, state on months −24/−13 (shares a base window) | {L['H01']['value']} | {L['H01']['ci95']} | {L['H01']['n']} |
| Post-deal gradient, same events | {L['H02']['value']} | {L['H02']['ci95']} | {L['H01']['n']} |
| Pre-deal gradient among untreated pseudo-events (null mean) | {L['H03']['value']} | *z* = {L['H03a']['value']} | — |
| Relative-magnitude breakdown, gradient | {L['H04']['value']} | — | — |
| **A placebo on the counterfactual** | | | |
| Control change on treated state, previous design | {L['D05']['value']} | — | — |
| Control change on treated state, corrected design | **{L['D05a']['value']}** | — | — |
| Control change on its own state, previous design | {L['D05b']['value']} | — | — |
| Control change on its own state, corrected design | {L['D05c']['value']} | — | — |
| Correlation of treated and control state, previous | {L['D05d']['value']} | — | — |
| Correlation of treated and control state, corrected | {L['D05e']['value']} | — | — |
| **Inference: shared controls** | | | |
| Distinct control firms | {L['C98']['value']} | — | {L['C98']['n']} events |
| Share of control firms used once | {L['C98a']['value']} | — | — |
| Maximum events served by one control firm | {L['C98b']['value']} | — | — |
| Gradient, bootstrap clustered on treated firms | {L['E01h']['value']} | {L['E01h']['ci95']} | {L['E01h']['n']} |
| Gradient, bootstrap clustered on control firms | {L['C98c']['value']} | {L['C98c']['ci95']} | — |
| Gradient, each control assigned to one event | {L['C98d']['value']} | — | {L['C98d']['n']} |
| Conservative interval, wider of the two clusterings | {L['C98e']['value']} | {L['C98e']['ci95']} | — |

*Notes.* The pre-trend rows estimate the state gradient at the same twelve-month resolution as the
headline, over the year before the outcome's base window. The state's usual window, months −24 to
−13, is also the base window of the pre-deal outcome, so the two are mechanically related; the first
block moves the state to months −36 to −25, which removes the overlap, and the pre-trend claim rests
on that version. In the overlapping version the pre-deal interval lies inside ±0.725,
the size of the post-deal gradient, and the relative-magnitude breakdown is the multiple of the
largest observed pre-deal gradient that differential trends may reach before the post-deal gradient
ceases to be distinguishable from zero. The pre-deal gradient among untreated pseudo-events is
larger than among treated firms. The counterfactual placebo regresses the control group's own change on the treated firm's
pre-deal state while holding the control firms' own state fixed. A valid counterfactual gives zero;
matching only on industry, size, growth and age gives {L['D05']['value']}, and adding the state to
the matching cell gives {L['D05a']['value']}. The controls' response to their own state is
essentially unchanged, so what the correction removes is specifically the dependence of the
counterfactual on the treated firm's moderator. M̄ is the largest multiple of the greatest observed pre-period quarterly movement in
differential trends under which the estimate remains significant, computed at the resolution of the
estimand; we do not claim a value at or above one. Deployment pressure runs opposite to selection on
private information: the effect is larger where the sponsor had least scope to choose (column (4) of
Appendix Table F.4.1). Deal years were confirmed independently against the shareholder register for 62 events,
and the share of no-hire months falls there by more than in the full sample, so measurement error in
the event date does not appear to be the primary explanation; that estimate is on the extensive
measure and not on the primary outcome, the subsample is small, and the comparison is qualitative. For the employment level the bracketed range is
the placebo distribution, not a confidence interval. At annual frequency the no-hire interval lies
inside ±0.046. Control firms are shared across events only rarely. Clustering the bootstrap on
control firms gives a narrower interval than clustering on treated firms, so the treated side is the
binding source of sampling variation and the conservative row reports the wider of the two. The not-yet-treated design is underpowered in this sample and is reported in the
online appendix."""

with open(os.path.join(OUTDIR, "tables.md"), "w", encoding="utf-8") as f:
    f.write("# Tables\n\n*Generated by `build/make_tables.py` from the analysis outputs. "
            "Do not edit by hand.*\n\n" + "\n\n---\n\n".join(T[i] for i in sorted(T)) + "\n")
with open(os.path.join(OUTDIR, "appendix_tables.md"), "w", encoding="utf-8") as f:
    f.write("# Appendix tables\n\n*Generated by `build/make_tables.py`. Do not edit by hand.*\n\n"
            + "\n\n---\n\n".join(AT[k] for k in sorted(AT)) + "\n")
print(f"tables.md: {len(T)}개 표 · appendix_tables.md: {len(AT)}개 표")
print("  부록 Table F.4.1 hazard 회귀표: 4사양 · pseudo-R² " +
      " / ".join(f"{S[c]['pseudo_r2_mcfadden']:.3f}" for c in COLS) +
      f" · clusters {S['(2)']['n_clusters']}")
