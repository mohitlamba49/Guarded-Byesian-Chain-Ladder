import numpy as np
import pandas as pd
from postprocess_empirical import leave_one_group_out_calibration


def test_calibration_intersects_nonnegative_support():
    rows = []
    for group in range(1, 6):
        rows.append({"LOB": "x", "GRCODE": group, "GRNAME": str(group), "ValuationYear": 2007, "Method": "BayesianODP", "Lower95": -20.0, "Upper95": 20.0, "TrueRealizedReserve": float(group)})
    results = pd.DataFrame(rows)
    intervals, summary = leave_one_group_out_calibration(results, ["BayesianODP"], "test", alpha=0.20)
    assert (intervals["CalibratedLower95"] >= 0).all()
    assert np.isfinite(summary["MeanCalibratedIntervalWidth95"]).all()
