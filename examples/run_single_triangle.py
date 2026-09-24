"""Small end-to-end GBCL example."""

import os
import sys
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from gbcl_core import fit_gbcl


triangle = pd.read_csv(os.path.join(HERE, "example_triangle.csv"), index_col="AccidentYear")

fit = fit_gbcl(triangle, q=0.50, impact_limit=0.01, n_chains=2, n_iter=1000, burn_in=300, thin=5, seed=20260904)

print("\nPOINT-RESERVE SUMMARY")
print(fit["summary"].round(4).to_string())

print("\nCALENDAR-STATE SUMMARY")
print(fit["state_summary"].round(4).to_string(index=False))

print("\nDIAGNOSTICS")
print(fit["diagnostics"].round(4).to_string())
