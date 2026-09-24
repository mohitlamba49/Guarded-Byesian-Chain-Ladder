# Guarded Bayesian Chain-Ladder (GBCL)

Reference implementation and result archive for **“Guarded Bayesian Chain-Ladder Reserving under Uncertain Calendar-Year Change”** (Mohit Lamba and Harmanpreet Singh Kapoor).

This repository provides the core modules, data preparation tools, and empirical post-processing pipelines used in the paper. Large-scale simulation loops and backtest batch scripts are provided as verified mathematical pseudocode in `pseudocode/`.

---

## Repository Map

*   `src/` — Core executable GBCL reference implementation (Bayesian ODP reference, state space mapping, posterior reserve sampler, and constrained Bayes action).
*   `scripts/` — Pipeline scripts for Schedule P screening and empirical post-processing.
*   `examples/` — Minimal end-to-end single triangle execution script.
*   `pseudocode/` — Mathematical specifications for the batch simulation engine, two-change experiment, rolling backtests, and AICc/Mack benchmarks.
*   `data/` — Provenance documentation, exclusion mappings, and table data schemas.
*   `results/` — Final compact summary tables used in the manuscript.
*   `tests/` — Automated unit and verification checks.

---

## Baseline Specification

The locked empirical configuration used in the study is defined by:
*   **Prior Probability (`pi0`):** 0.70
*   **Multiplier Prior Concentration (`kappa`):** 25
*   **Terminal Continuation Scenario (`q`):** 0.50
*   **Reporting / Activation Threshold:** 0.75
*   **Reserve-Impact Tolerance (`L`):** 0.01

*Note: The target is the finite-horizon reserve through development age \(J-1\). No extrapolative tail factor is applied.*

---

## Installation

Python 3.10 or later is recommended. 

```bash
# Set up virtual environment
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install repository in editable mode
python -m pip install -e .
```

### Verify Installation
Run the automated test suite and the reference example:
```bash
python examples/run_single_triangle.py
python -m pytest -q
```
*Note: The example script uses truncated MCMC chains to verify the execution path quickly. Production estimation runs should use the full chain lengths specified in Appendix A.3.*

---

## Data Acquisition & Preparation

The study utilizes the historical CAS Schedule P data sets **updated September 1, 2011** (1998–2007 accident years). To automatically download and screen these files:

```bash
python scripts/prepare_schedule_p.py --download
```
Data cleaning filters row conditions, applies data quality flags, and isolates the final matrix of **42 insurer-line triangles** nested within **32 `GRCODE` insurer groups**.

---

## Empirical Post-Processing

After stacking model-specific forecasts into a unified, long-format CSV layout, execute the clustered bootstrap and interval calibration pipeline:

```bash
python scripts/postprocess_empirical.py --input path/to/method_results.csv --output results/postprocessed
```
The routine performs a leave-one-group-out validation, sampling by `GRCODE` clusters to handle internal within-group dependence structures.
