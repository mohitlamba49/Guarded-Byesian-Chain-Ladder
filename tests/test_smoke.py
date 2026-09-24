import os
import numpy as np
import pandas as pd
from gbcl_core import fit_gbcl


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def test_short_end_to_end_fit():
    triangle = pd.read_csv(os.path.join(ROOT, "examples", "example_triangle.csv"), index_col="AccidentYear")
    fit = fit_gbcl(triangle, n_chains=2, n_iter=120, burn_in=40, thin=4, seed=123)
    summary = fit["summary"]
    assert np.isfinite(summary["GuardedReserve"])
    assert summary["BayesianODPReserve"] > 0
    assert abs(summary["ReportedChangePercent"]) <= 1.0000001
    assert np.isclose(fit["state_summary"]["PosteriorProbability"].sum(), 1.0)
