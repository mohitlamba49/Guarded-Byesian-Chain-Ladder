# Loss Reserving Data Provenance and Empirical Pipeline

This repository contains the data preparation scripts and schema definitions required to reproduce the 1998–2007 accident-year sample and the 2007–2012 rolling evaluation frameworks.

## Data Provenance

### Source Files
To reproduce this paper's exact results, you **must use the 2011 historical data sets**. Do not substitute the newer December 2025 files. Download the following six source CSVs from the official [CAS Loss Reserving Data Page](https://www.casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p):

*   **Private Passenger Auto:** `ppauto_pos.csv`
*   **Commercial Auto:** `comauto_pos.csv`
*   **Workers Compensation:** `wkcomp_pos.csv`
*   **Medical Malpractice:** `medmal_pos.csv`
*   **Other Liability:** `othliab_pos.csv`
*   **Products Liability:** `prodliab_pos.csv`

*Note: Raw source files are not committed to this repository as their distribution is governed by the source provider.*

### Sample Construction
The sample filtering is handled sequentially by `scripts/prepare_schedule_p.py`:
1. **Initial Pool:** 772 insurer-line triangles.
2. **Completeness Filter:** Identifies 665 complete 10-by-10 triangles.
3. **Eligibility Filter:** Filters down to 64 paid-loss-eligible triangles.
4. **Positivity Filter:** Excludes 22 triangles with non-positive observed development-column totals under the 2007 cutoff.
5. **Final Matrix:** Consists of **42 insurer-line triangles** nested within **32 `GRCODE` insurer groups**. No data points are omitted due to anomaly flags alone.

---

## Input Schema & Processing

The empirical tracking scripts (`scripts/postprocess_empirical.py`) expect a structural output mapping containing one row per method and forecast iteration. 

### Field Definitions

| Column | Type | Meaning |
| :--- | :--- | :--- |
| `LOB` | String | Line-of-business code (e.g., B, C, D, F2, H1, R1) |
| `GRCODE` | Integer | NAIC insurer-group or single-entity identification code |
| `GRNAME` | String | Insurer-group or single-entity formal name |
| `ValuationYear` | Integer | Forecast evaluation cutoff year (2007–2012) |
| `Method` | String | Method identifier code |
| `PointEstimate` | Numeric | Finite-horizon reserve point estimate |
| `TrueRealizedReserve` | Numeric | Subsequently observed actual finite-horizon payment |
| `Observed` | Numeric | Cumulative paid loss recorded at the active valuation date |
| `Lower95` | Numeric | Lower 95% uncalibrated predictive endpoint *(Optional for point-only models)* |
| `Upper95` | Numeric | Upper 95% uncalibrated predictive endpoint *(Optional for point-only models)* |

### Required Method Identifiers
To execute the complete paper pipeline comparison, your model outputs must populate the `Method` column using these exact identifiers:
*   `ChainLadder`
*   `MackChainLadder`
*   `BayesianMack`
*   `BayesianODP`
*   `GuardedV2_Q050_Cap1pct`
*   `GuardedV2_Q050_Raw`
*   `GuardedV2_Q100_Raw`
*   `ODP-Calendar-AICc`


