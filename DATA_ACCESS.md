# Data access

None of the primary sources can be redistributed in this repository. This file states what each contains, what it is used for, and how to obtain it.

| Source | Used for | Redistributable here | How to obtain |
|---|---|---|---|
| **National Pension Service establishment register** (monthly, firm level, November 2015 – May 2026) | The outcomes: monthly insured headcount, new enrolments (hires) and losses (separations) by business-registration number; the pre-deal hiring state; all matching covariates | **No** — licensed microdata | Korean National Pension Service public-data portal, under its data-use terms |
| **PitchBook** deal records | Identification of private-equity investments in Korean companies, deal month, deal type, acquired stake, sponsor identity and fund vintage | **No** — licensed | PitchBook |
| **Business-registration linkage and name-matching recovery** | Linking deals to pension records; the manually adjudicated recovery step for deals without a registration number | **No** — derived from the above and identifying | Reproducible from the two sources above with `code/pipeline/`; the adjudication protocol is described in the paper |
| **Audited annual financial statements** | Assets, revenue, cash, debt, interest expense, operating income (Section 9 and the financial-constraint splits) | **No** — commercial database | The vendor |
| **Shareholder register** (annual reference dates) | Independent confirmation of deal years; the non-PE ownership-change comparison | **No** — commercial database | The vendor |

## What *is* here instead

`artifacts/` holds 64 aggregate result files (JSON). They contain estimates, confidence intervals, placebo-null distributions summarised as moments and quantiles, sample counts, balance diagnostics and event-path coefficients — the quantities reported in the paper — and no firm identifiers. Every table and figure in the paper is rebuilt from these files by the notebooks, and `notebooks/03_traceability.ipynb` resolves every row of the claims ledger against them.

Firm-level derived files (business numbers, monthly headcounts, per-firm outcomes, matched-pair lists) are **deliberately excluded**: they derive from the licensed register and would identify individual employers. One aggregate artifact, `I05.json` (exit reversibility), is also excluded because it lists the business-registration numbers of the exit sample; see `ARTIFACT_MANIFEST.md`.

## Reproducing from raw data

`code/pipeline/` is the complete analysis pipeline (`i01_…` to `i69_…`, with the shared loaders `h30_common.py` and `h39_common.py`). With the sources above in place and the environment variable `P014_BASE` pointed at a project root that holds them under `shared/data/processed/`, each script rebuilds its artifact in `artifacts/`. `run_harness30.sh` runs the sequence; `CODE_INDEX.md` and `HARNESS30.md` describe each script (in Korean, the pipeline's working language).
