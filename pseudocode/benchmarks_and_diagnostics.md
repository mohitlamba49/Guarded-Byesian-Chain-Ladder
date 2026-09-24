# Benchmark and diagnostic pseudocode

## Information-matched ODP-Calendar-AICc

```text
Construct exactly the GBCL candidate support:
    no change
    temporary(c) for every eligible interior calendar
    persistent(c) for every eligible interior calendar
    unresolved(t) on the latest observed diagonal

FOR each candidate
    fit log(mu_ij) = alpha_i + beta_j + delta * z_ij by IPF
    stop when maximum relative coordinate change < 1e-10
    compute Poisson deviance D
    compute QAICc = D/phi0 + 2p + 2p(p+1)/(n-p-1)
END

Select the minimum-QAICc candidate.
Forecast temporary as ended, persistent as continuing, and unresolved with
future multiplier (1-q) + q*exp(delta), using the same q as GBCL.
```

## Bayesian Mack

Use the classical volume-weighted development factors and Mack variance estimates.
For development age `j`, use the Appendix A.2 prior
`sigma_j^2 ~ InvGamma(2, sigma_hat_j^2)` and
`F_j | sigma_j^2 ~ Normal(1, sigma_j^2/kappa0_j)`, where
`kappa0_j = 0.01 * mean(C_ij)` over contributing origins. Resample nonpositive
factor or cumulative-claim draws. Use the manuscript's final-development variance
extrapolation when the last variance cannot be estimated directly.

## GBCL diagnostics

```text
continuous diagnostic = maximum multi-chain R-hat over origin effects,
                         development effects and the change multiplier
discrete diagnostic   = max over chains and states of
                         abs(chain mean conditional state probability
                             - pooled mean conditional state probability)

flag if maximum R-hat > 1.05 or discrete diagnostic > .05
perform the targeted longer refit
retain any remaining discrete warning in the audit table
```

The discrete diagnostic describes sensitivity in allocating posterior mass among
nearby structural states. It is not a continuous-parameter convergence diagnostic
and must not be used to remove a portfolio from the empirical sample.
