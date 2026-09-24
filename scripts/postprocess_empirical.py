"""Insurer-group bootstrap and leave-one-GRCODE-out interval calibration.

Expected input: one long CSV with one row per LOB, GRCODE, valuation year and
method.  See data/README.md for the required column names.  The script keeps all
lines and all valuation years belonging to a sampled insurer group together.
"""

import argparse
import os
import numpy as np
import pandas as pd


def prepare_results(path):
    results = pd.read_csv(path)
    required = ["LOB", "GRCODE", "GRNAME", "ValuationYear", "Method", "PointEstimate", "TrueRealizedReserve", "Observed"]
    missing = [column for column in required if column not in results.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    results["GRCODE"] = results["GRCODE"].astype(int)
    results["Error"] = results["PointEstimate"] - results["TrueRealizedReserve"]
    results["AbsoluteError"] = results["Error"].abs()
    results["ErrorPercentObserved"] = 100.0 * results["Error"] / results["Observed"]
    return results


def dependence_audit(results, label):
    base = results[results["Method"] == "ChainLadder"][["LOB", "GRCODE"]].drop_duplicates()
    counts = base.groupby("GRCODE")["LOB"].nunique().sort_values(ascending=False)
    audit = pd.DataFrame({"Sample": [label], "InsurerLineTriangles": [len(base)], "DistinctGRCODE": [counts.size], "MultiLOBGroups": [int((counts > 1).sum())], "MaximumLOBsInOneGroup": [int(counts.max())]})
    detail = base.groupby("GRCODE")["LOB"].agg(lambda values: ", ".join(sorted(values))).reset_index(name="LOBs")
    detail["NumberLOBs"] = detail["LOBs"].str.count(",") + 1
    detail = detail[detail["NumberLOBs"] > 1].sort_values(["NumberLOBs", "GRCODE"], ascending=[False, True])
    return audit, detail


def clustered_bootstrap(results, comparisons, label, replications=10000, seed=20260921):
    random = np.random.default_rng(seed)
    groups = np.sort(results["GRCODE"].unique())
    sampled = random.integers(0, len(groups), size=(replications, len(groups)))
    counts = np.zeros((replications, len(groups)), dtype=int)
    for replication in range(replications):
        counts[replication] = np.bincount(sampled[replication], minlength=len(groups))
    group_number = {group: number for number, group in enumerate(groups)}
    rows = []
    keys = ["LOB", "GRCODE", "ValuationYear"]
    measures = ["Error", "AbsoluteError", "ErrorPercentObserved"]
    for first_method, second_method in comparisons:
        first = results[results["Method"] == first_method][keys + measures]
        second = results[results["Method"] == second_method][keys + measures]
        paired = first.merge(second, on=keys, suffixes=("First", "Second"), validate="one_to_one")
        if paired["GRCODE"].nunique() != len(groups):
            raise ValueError(f"Incomplete GRCODE matching for {first_method} versus {second_method}.")
        statistics = []
        for group, block in paired.groupby("GRCODE", sort=False):
            statistics.append({"GroupNumber": group_number[group], "N": len(block), "SSEFirst": np.sum(block["ErrorFirst"] ** 2), "SSESecond": np.sum(block["ErrorSecond"] ** 2), "NSSEFirst": np.sum(block["ErrorPercentObservedFirst"] ** 2), "NSSESecond": np.sum(block["ErrorPercentObservedSecond"] ** 2), "AbsoluteFirst": block["AbsoluteErrorFirst"].sum(), "AbsoluteSecond": block["AbsoluteErrorSecond"].sum()})
        statistics = pd.DataFrame(statistics).set_index("GroupNumber").reindex(range(len(groups)))
        sample_size = counts @ statistics["N"].to_numpy()
        rmse_ratio = np.sqrt((counts @ statistics["SSEFirst"].to_numpy()) / sample_size) / np.sqrt((counts @ statistics["SSESecond"].to_numpy()) / sample_size)
        nrmse_ratio = np.sqrt((counts @ statistics["NSSEFirst"].to_numpy()) / sample_size) / np.sqrt((counts @ statistics["NSSESecond"].to_numpy()) / sample_size)
        mae_difference = ((counts @ statistics["AbsoluteFirst"].to_numpy()) - (counts @ statistics["AbsoluteSecond"].to_numpy())) / sample_size
        observed_rmse_ratio = np.sqrt(np.mean(paired["ErrorFirst"] ** 2)) / np.sqrt(np.mean(paired["ErrorSecond"] ** 2))
        observed_nrmse_ratio = np.sqrt(np.mean(paired["ErrorPercentObservedFirst"] ** 2)) / np.sqrt(np.mean(paired["ErrorPercentObservedSecond"] ** 2))
        observed_mae_difference = paired["AbsoluteErrorFirst"].mean() - paired["AbsoluteErrorSecond"].mean()
        rows.append({"Sample": label, "FirstMethod": first_method, "SecondMethod": second_method, "InsurerGroups": len(groups), "InsurerLineTriangles": paired[["LOB", "GRCODE"]].drop_duplicates().shape[0], "Forecasts": len(paired), "BootstrapUnit": "GRCODE with all LOBs and valuation years", "BootstrapReplications": replications, "ObservedRMSERatio": observed_rmse_ratio, "RMSERatioLower2.5": np.percentile(rmse_ratio, 2.5), "RMSERatioUpper97.5": np.percentile(rmse_ratio, 97.5), "ProbabilityFirstLowerRMSE": np.mean(rmse_ratio < 1.0), "ObservedNormalizedRMSERatio": observed_nrmse_ratio, "NormalizedRMSERatioLower2.5": np.percentile(nrmse_ratio, 2.5), "NormalizedRMSERatioUpper97.5": np.percentile(nrmse_ratio, 97.5), "ProbabilityFirstLowerNormalizedRMSE": np.mean(nrmse_ratio < 1.0), "ObservedMAEDifference": observed_mae_difference, "MAEDifferenceLower2.5": np.percentile(mae_difference, 2.5), "MAEDifferenceUpper97.5": np.percentile(mae_difference, 97.5), "ProbabilityFirstLowerMAE": np.mean(mae_difference < 0.0)})
    return pd.DataFrame(rows)


def interval_score(lower, upper, truth, alpha=0.05):
    score = upper - lower
    if truth < lower:
        score += 2.0 * (lower - truth) / alpha
    if truth > upper:
        score += 2.0 * (truth - upper) / alpha
    return score


def leave_one_group_out_calibration(results, methods, label, alpha=0.05):
    rows = []
    for method in methods:
        selected = results[(results["Method"] == method) & results["Lower95"].notna() & results["Upper95"].notna()].copy()
        if selected.empty:
            continue
        selected["Center"] = (selected["Lower95"] + selected["Upper95"]) / 2.0
        selected["HalfWidth"] = np.maximum((selected["Upper95"] - selected["Lower95"]) / 2.0, 1e-12)
        selected["Score"] = np.abs(selected["TrueRealizedReserve"] - selected["Center"]) / selected["HalfWidth"]
        group_scores = selected.groupby("GRCODE")["Score"].max()
        for held_out in group_scores.index:
            calibration_scores = group_scores.drop(index=held_out).to_numpy()
            rank = int(np.ceil((len(calibration_scores) + 1) * (1.0 - alpha)))
            rank = min(max(rank, 1), len(calibration_scores))
            factor = float(np.sort(calibration_scores)[rank - 1])
            held_rows = selected[selected["GRCODE"] == held_out]
            lower_all = np.maximum(0.0, held_rows["Center"] - factor * held_rows["HalfWidth"])
            upper_all = held_rows["Center"] + factor * held_rows["HalfWidth"]
            truth_all = held_rows["TrueRealizedReserve"]
            simultaneous = int(((lower_all <= truth_all) & (truth_all <= upper_all)).all())
            for row_number, row in held_rows.iterrows():
                lower = max(0.0, float(row["Center"] - factor * row["HalfWidth"]))
                upper = float(row["Center"] + factor * row["HalfWidth"])
                truth = float(row["TrueRealizedReserve"])
                rows.append({"Sample": label, "LOB": row["LOB"], "GRCODE": int(row["GRCODE"]), "GRNAME": row["GRNAME"], "ValuationYear": int(row["ValuationYear"]), "Method": method, "CalibrationScope": "Leave one GRCODE out; group-maximum score", "CalibrationGroups": len(calibration_scores), "CalibrationFactor": factor, "CalibratedLower95": lower, "CalibratedUpper95": upper, "TrueRealizedReserve": truth, "CalibratedCoverage95": int(lower <= truth <= upper), "GRCODESimultaneousCoverage95": simultaneous, "CalibratedIntervalWidth95": upper - lower, "CalibratedIntervalScore95": interval_score(lower, upper, truth, alpha)})
    intervals = pd.DataFrame(rows)
    summaries = []
    for method, group in intervals.groupby("Method", sort=False):
        group_coverage = group[["GRCODE", "GRCODESimultaneousCoverage95"]].drop_duplicates()
        factors = group[["GRCODE", "CalibrationFactor"]].drop_duplicates()
        summaries.append({"Sample": label, "Method": method, "InsurerGroups": group["GRCODE"].nunique(), "Forecasts": len(group), "MeanCalibrationFactor": factors["CalibrationFactor"].mean(), "MedianCalibrationFactor": factors["CalibrationFactor"].median(), "MarginalCoverage95": group["CalibratedCoverage95"].mean(), "GRCODESimultaneousCoverage95": group_coverage["GRCODESimultaneousCoverage95"].mean(), "MeanCalibratedIntervalWidth95": group["CalibratedIntervalWidth95"].mean(), "MeanCalibratedIntervalScore95": group["CalibratedIntervalScore95"].mean()})
    return intervals, pd.DataFrame(summaries)


def available_comparisons(results):
    methods = set(results["Method"])
    requested = [("GuardedV2_Q050_Cap1pct", "ChainLadder"), ("GuardedV2_Q050_Cap1pct", "BayesianODP"), ("BayesianODP", "ChainLadder"), ("ODP-Calendar-AICc", "ChainLadder"), ("GuardedV2_Q050_Cap1pct", "ODP-Calendar-AICc")]
    return [pair for pair in requested if pair[0] in methods and pair[1] in methods]


def run_postprocessing(input_file, output_directory, replications=10000, seed=20260921):
    results = prepare_results(input_file)
    os.makedirs(output_directory, exist_ok=True)
    label = f"{results['ValuationYear'].min()}-{results['ValuationYear'].max()}" if results["ValuationYear"].nunique() > 1 else str(results["ValuationYear"].iloc[0])
    audit, detail = dependence_audit(results, label)
    bootstrap = clustered_bootstrap(results, available_comparisons(results), label, replications=replications, seed=seed)
    interval_methods = [method for method in ["MackChainLadder", "BayesianMack", "BayesianODP", "GuardedV2_Q050_Cap1pct", "GuardedV2_Q050_Raw", "GuardedV2_Q100_Raw"] if method in set(results["Method"])]
    intervals, calibration = leave_one_group_out_calibration(results, interval_methods, label)
    audit.to_csv(os.path.join(output_directory, "dependence_audit.csv"), index=False)
    detail.to_csv(os.path.join(output_directory, "multi_lob_groups.csv"), index=False)
    bootstrap.to_csv(os.path.join(output_directory, "clustered_bootstrap.csv"), index=False)
    intervals.to_csv(os.path.join(output_directory, "calibrated_intervals.csv"), index=False)
    calibration.to_csv(os.path.join(output_directory, "calibration_summary.csv"), index=False)
    return audit, detail, bootstrap, intervals, calibration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Long method-results CSV.")
    parser.add_argument("--output", default="results/postprocessed", help="Output directory.")
    parser.add_argument("--replications", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260921)
    arguments = parser.parse_args()
    run_postprocessing(arguments.input, arguments.output, arguments.replications, arguments.seed)
