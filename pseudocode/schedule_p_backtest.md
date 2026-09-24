# Schedule P backtest pseudocode

This file maps the empirical design in Section 4 to the repository functions.

```text
DOWNLOAD the six legacy (2011) CAS Schedule P CSV files
RUN scripts/prepare_schedule_p.py

KEEP complete 10 x 10 paid-loss insurer-line triangles satisfying the paid-loss
eligibility flag
SCREEN at the 2007 valuation for positive observed development-column totals
KEEP the same final set of 42 insurer-line triangles (32 GRCODE groups) throughout

FOR valuation year t in {2007, 2008, 2009, 2010, 2011, 2012}
    FOR each of the 42 insurer-line triangles
        observed cells = cells with AccidentYear + DevelopmentLag <= t
        future cells   = remaining cells through development lag 9
        realized target = sum of the recorded future incremental payments

        fit Chain-Ladder
        fit Mack Chain-Ladder
        fit Bayesian Mack
        fit Bayesian ODP
        fit raw GBCL and transport draws to the primary L=.01 action
        fit ODP-Calendar-AICc on exactly the same admissible state support

        save one long row per method with the schema in data/README.md
        save GBCL state probabilities, leading state and convergence diagnostics
    END
END

RUN scripts/postprocess_empirical.py on the long method-result file
```

The finite horizon ends at development lag 9; no tail factor is added. Monetary
amounts remain in the Schedule P reported unit (USD thousands).

Empirical uncertainty is clustered by `GRCODE`. A bootstrap draw retains every
line and, for the rolling experiment, all six valuation years belonging to that
insurer group. Interval calibration leaves out the entire `GRCODE`; its score is
the maximum standardized error over that group's lines and valuation years. The
calibrated lower bound is intersected with `[0, infinity)`.

Initial empirical GBCL fits use four chains of 5,000 iterations, burn-in 1,000 and
thinning 5. Flagged fits are rerun with the targeted schedules stated in Appendix
A.3. A remaining discrete-state warning is retained and reported, not deleted.

