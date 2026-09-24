"""Download and flag the six legacy CAS Schedule P files used in the paper.

The script preserves every source row.  Eligibility variables are flags; no zero,
negative or non-monotone loss value is silently deleted.
"""

import argparse
import os
import urllib.request
import numpy as np
import pandas as pd


FILES = {
    "ppauto": "https://www.casact.org/sites/default/files/2021-04/ppauto_pos.csv",
    "comauto": "https://www.casact.org/sites/default/files/2021-04/comauto_pos.csv",
    "wkcomp": "https://www.casact.org/sites/default/files/2021-04/wkcomp_pos.csv",
    "medmal": "https://www.casact.org/sites/default/files/2021-04/medmal_pos.csv",
    "othliab": "https://www.casact.org/sites/default/files/2021-04/othliab_pos.csv",
    "prodliab": "https://www.casact.org/sites/default/files/2021-04/prodliab_pos.csv",
}


def download_files(directory):
    os.makedirs(directory, exist_ok=True)
    for lob, url in FILES.items():
        path = os.path.join(directory, f"{lob}_pos.csv")
        if not os.path.exists(path):
            print(f"Downloading {lob}: {url}")
            urllib.request.urlretrieve(url, path)


def normalize_columns(data):
    data = data.copy()
    data.columns = [str(column).strip().rstrip("_") for column in data.columns]
    aliases = {"IncurLosses": "IncurLoss", "CumulativePaidLoss": "CumPaidLoss", "GroupCode": "GRCODE", "GroupName": "GRNAME"}
    data = data.rename(columns={old: new for old, new in aliases.items() if old in data.columns and new not in data.columns})
    required = ["GRCODE", "GRNAME", "AccidentYear", "DevelopmentLag", "CumPaidLoss", "IncurLoss"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("The CAS file is missing: " + ", ".join(missing))
    for column in ["GRCODE", "AccidentYear", "DevelopmentLag", "CumPaidLoss", "IncurLoss"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data["DevelopmentLag"].min() == 1 and data["DevelopmentLag"].max() == 10:
        data["DevelopmentLag"] = data["DevelopmentLag"] - 1
    data["DevelopmentYear"] = data["AccidentYear"] + data["DevelopmentLag"]
    return data


def company_flags(block):
    unique_cells = block[["AccidentYear", "DevelopmentLag"]].drop_duplicates().shape[0]
    complete = int(unique_cells == 100 and block["AccidentYear"].nunique() == 10 and block["DevelopmentLag"].nunique() == 10)
    paid_nonpositive = int((block["CumPaidLoss"] <= 0).any())
    incurred_nonpositive = int((block["IncurLoss"] <= 0).any())
    paid_nonmonotone = 0
    incurred_nonmonotone = 0
    for accident_year, row in block.sort_values("DevelopmentLag").groupby("AccidentYear"):
        paid_nonmonotone = max(paid_nonmonotone, int((row["CumPaidLoss"].diff().dropna() < 0).any()))
        incurred_nonmonotone = max(incurred_nonmonotone, int((row["IncurLoss"].diff().dropna() < 0).any()))
    return pd.Series({"Complete10x10": complete, "PaidHasNonpositive": paid_nonpositive, "PaidHasDecrease": paid_nonmonotone, "IncurredHasNonpositive": incurred_nonpositive, "IncurredHasDecrease": incurred_nonmonotone, "EligiblePaidCL": int(complete == 1 and paid_nonpositive == 0 and paid_nonmonotone == 0), "EligibleIncurredCL": int(complete == 1 and incurred_nonpositive == 0 and incurred_nonmonotone == 0)})


def clean_schedule_p(directory):
    all_lines = []
    summaries = []
    for lob in FILES:
        path = os.path.join(directory, f"{lob}_pos.csv")
        data = normalize_columns(pd.read_csv(path))
        before = len(data)
        data = data.drop_duplicates().copy()
        print(f"{lob}: removed {before - len(data)} exact duplicate rows")
        data["LOB"] = lob
        flags = data.groupby(["GRCODE", "GRNAME"], dropna=False).apply(company_flags, include_groups=False).reset_index()
        data = data.merge(flags, on=["GRCODE", "GRNAME"], validate="many_to_one")
        all_lines.append(data)
        summaries.append({"LOB": lob, "Companies": flags[["GRCODE", "GRNAME"]].drop_duplicates().shape[0], "Complete10x10": int(flags["Complete10x10"].sum()), "EligiblePaidCL": int(flags["EligiblePaidCL"].sum()), "EligibleIncurredCL": int(flags["EligibleIncurredCL"].sum())})
    return pd.concat(all_lines, ignore_index=True), pd.DataFrame(summaries)


def positive_support_screen(cleaned, valuation_year=2007):
    eligible = cleaned[cleaned["EligiblePaidCL"] == 1].copy()
    rows = []
    for (lob, grcode, grname), block in eligible.groupby(["LOB", "GRCODE", "GRNAME"]):
        triangle = block.pivot(index="AccidentYear", columns="DevelopmentLag", values="CumPaidLoss").sort_index().sort_index(axis=1)
        incremental = triangle.diff(axis=1)
        incremental.iloc[:, 0] = triangle.iloc[:, 0]
        calendars = triangle.index.to_numpy()[:, None] + triangle.columns.to_numpy()[None, :]
        observed = calendars <= valuation_year
        values = incremental.to_numpy(dtype=float)
        observed_values = np.where(observed, values, np.nan)
        column_totals = np.nansum(observed_values, axis=0)
        used_columns = np.any(observed & np.isfinite(values), axis=0)
        positive_support = int(np.all(column_totals[used_columns] > 0))
        rows.append({"LOB": lob, "GRCODE": int(grcode), "GRNAME": grname, "PositiveObservedDevelopmentTotals2007": positive_support})
    return pd.DataFrame(rows)


def run(directory, output_file):
    cleaned, summary = clean_schedule_p(directory)
    support = positive_support_screen(cleaned)
    cleaned = cleaned.merge(support, on=["LOB", "GRCODE", "GRNAME"], validate="many_to_one")
    cleaned["EmpiricalSample"] = ((cleaned["EligiblePaidCL"] == 1) & (cleaned["PositiveObservedDevelopmentTotals2007"] == 1)).astype(int)
    cleaned.to_csv(output_file, index=False)
    summary.to_csv(os.path.splitext(output_file)[0] + "_summary.csv", index=False)
    support.to_csv(os.path.splitext(output_file)[0] + "_support_screen.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved {len(cleaned):,} rows to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="data/processed/CAS_clean_all_lines.csv")
    parser.add_argument("--download", action="store_true")
    arguments = parser.parse_args()
    if arguments.download:
        download_files(arguments.data_dir)
    os.makedirs(os.path.dirname(arguments.output), exist_ok=True)
    run(arguments.data_dir, arguments.output)
