import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

def process_season(filepath, season_label):
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Assign matchday by rank of unique dates
    unique_dates = sorted(df["Date"].unique())
    date_to_matchday = {date: i+1 for i, date in enumerate(unique_dates)}
    df["Matchday"] = df["Date"].map(date_to_matchday)
    df["Season"] = season_label

    form_rows = []
    for _, row in df.iterrows():
        md = row["Matchday"]
        if row["FTR"] == "H":
            form_rows.append({"Season": season_label, "Team": row["HomeTeam"], "Matchday": md, "Points": 3})
            form_rows.append({"Season": season_label, "Team": row["AwayTeam"], "Matchday": md, "Points": 0})
        elif row["FTR"] == "A":
            form_rows.append({"Season": season_label, "Team": row["HomeTeam"], "Matchday": md, "Points": 0})
            form_rows.append({"Season": season_label, "Team": row["AwayTeam"], "Matchday": md, "Points": 3})
        else:
            form_rows.append({"Season": season_label, "Team": row["HomeTeam"], "Matchday": md, "Points": 1})
            form_rows.append({"Season": season_label, "Team": row["AwayTeam"], "Matchday": md, "Points": 1})

    form = pd.DataFrame(form_rows).sort_values(["Team", "Matchday"])
    form["CumulativePoints"] = form.groupby("Team")["Points"].cumsum()
    return form

seasons = [
    ("Bundesliga2223.csv", "2022/23"),
    ("Bundesliga2324.csv", "2023/24"),
    ("Bundesliga2425.csv", "2024/25"),
]

form_all = []
for filepath, label in seasons:
    csv_path = BASE_DIR / filepath
    if not csv_path.exists():
        print(f"Skipping {label}: file not found at {csv_path}")
        continue
    form_all.append(process_season(csv_path, label))

if not form_all:
    raise FileNotFoundError("No season CSV files were found in the data directory.")

combined_form = pd.concat(form_all, ignore_index=True)
combined_form.to_csv(BASE_DIR / "form_all.csv", index=False)
print(combined_form[combined_form["Team"] == "Bayern Munich"].head(10))