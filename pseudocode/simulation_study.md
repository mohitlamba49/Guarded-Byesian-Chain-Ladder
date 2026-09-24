# Simulation-study pseudocode

This file records the batch orchestration used for Sections 3 and B.1. It is
pseudocode, not a second implementation of `src/gbcl_core.py`.

## One-change experiment

```text
INPUT
    18 rows in results/simulation_design.csv
    replications = 200 per row
    candidate q = {0, .25, .50, .75, 1}
    candidate L = {0, .01, .025, .05, .10, infinity}
    primary settings fixed before final evaluation: q=.50, L=.01

FOR each scenario
    construct the complete 10 x 10 conditional mean triangle
    apply the stated temporary or persistent calendar multiplier
    FOR replication = 1,...,200
        generate the full incremental square from the stated cell distribution
        retain cells with calendar year <= 2007 as the observed upper triangle
        retain cells with calendar year > 2007 as the future outcome
        compute the conditional expected future reserve for point-error assessment

        fit Chain-Ladder, Mack, Bayesian Mack and Bayesian ODP
        fit GBCL once and reuse its stored draws for every q/L configuration
        fit the information-matched ODP-Calendar-AICc candidates

        save point reserve, realized future reserve, predictive interval,
             posterior state probabilities, leading state, impact weight,
             diagnostics and random seed
    END
END

Pool squared errors before taking square roots.  Pair every method comparison on
the same generated triangle.  Report false activation under no change, detection
of any changed state, correct structural type where identifiable, RMSE, coverage,
interval score and CRPS.
```

Each Bayesian fit uses two chains of 2,500 iterations, burn-in 700 and thinning 6,
giving 600 retained draws. The reporting label is activated at
`ProbabilityAnyChange >= 0.75`; prediction always averages over every state.

## Two-change robustness experiment

```text
INPUT
    six rows in results/two_change_design.csv
    replications = 200 per row

Repeat the one-change workflow, but generate two calendar effects while fitting
the unchanged one-change GBCL state space.  Compute pooled and scenario-specific
RMSE ratios.  Use a 5,000-replication scenario-stratified paired bootstrap for
the reported RMSE-ratio intervals.
```

## Sensitivity runs

```text
KAPPA SENSITIVITY
    refit all 18 designs at kappa = {10, 25, 50}
    retain pi0=.70, threshold=.75, q=.50 and L=.01

WEAK-CHANGE SENSITIVITY
    at calendar 2005, generate temporary and persistent multipliers
    {0.90, 0.95, 1.05, 1.10}; use 200 replications per design

L SENSITIVITY
    transport the same paired Bayesian-ODP and raw-GBCL draws with
    L = {0, .01, .025, .05, .10, infinity}; do not refit the likelihood
```

The seed must be recorded in every replication-level output. Failures are written
to a separate CSV and are never replaced silently.

