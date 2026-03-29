from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def first_existing_path(candidates, label):
    for path in candidates:
        if path.exists():
            return path
    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find {label}. Checked:\n{checked}")


predicted_path = first_existing_path(
    [
        BASE_DIR / "predicted_standings.csv",
        PROJECT_ROOT / "predicted_standings.csv",
        PROJECT_ROOT / "simulated_season_table.csv",
        BASE_DIR / "simulated_season_table.csv",
    ],
    "predicted standings CSV",
)

actual_path = first_existing_path(
    [
        BASE_DIR / "actual_standings_all.csv",
        PROJECT_ROOT / "actual_standings_all.csv",
    ],
    "actual standings CSV",
)

predicted = pd.read_csv(predicted_path)
actual = pd.read_csv(actual_path)

# Sim output uses P instead of MP. Keep this rename here in case you use MP later.
if "P" in predicted.columns and "MP" not in predicted.columns:
    predicted = predicted.rename(columns={"P": "MP"})

# Derive predicted position if it is missing.
if "Position" not in predicted.columns:
    sort_cols = [c for c in ["Pts", "GD", "GF"] if c in predicted.columns]
    if "Pts" not in sort_cols:
        raise ValueError("Predicted table must include Pts to derive Position.")
    predicted = predicted.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
    predicted["Position"] = predicted.index + 1

# If predicted data has no season, compare it to the latest season in actual standings.
if "Season" not in predicted.columns:
    latest_season = actual["Season"].astype(str).max()
    predicted["Season"] = latest_season

required_pred_cols = {"Team", "Season", "Pts", "Position"}
required_actual_cols = {"Team", "Season", "Pts", "Position"}

missing_pred = required_pred_cols.difference(predicted.columns)
missing_actual = required_actual_cols.difference(actual.columns)
if missing_pred:
    raise ValueError(f"Predicted table is missing required columns: {sorted(missing_pred)}")
if missing_actual:
    raise ValueError(f"Actual table is missing required columns: {sorted(missing_actual)}")

merged = predicted[["Season", "Team", "Pts", "Position"]].merge(
    actual[["Season", "Team", "Pts", "Position"]],
    on=["Team", "Season"],
    suffixes=("_Predicted", "_Actual"),
)

merged["Position_Diff"] = merged["Position_Actual"] - merged["Position_Predicted"]
merged["Pts_Diff"] = merged["Pts_Actual"] - merged["Pts_Predicted"]

output_path = BASE_DIR / "predicted_vs_actual.csv"
merged.to_csv(output_path, index=False)
print(f"Merged comparison saved to {output_path}")
print(merged.head())