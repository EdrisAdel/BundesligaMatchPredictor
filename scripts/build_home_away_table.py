import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

dfs = []
seasons = [
    ("Bundesliga2223.csv", "2022/23"),
    ("Bundesliga2324.csv", "2023/24"),
    ("Bundesliga2425.csv", "2024/25"),
]

for filepath, label in seasons:
    df = pd.read_csv(BASE_DIR / filepath)
    df["Season"] = label
    dfs.append(df)

df = pd.concat(dfs)

rows = []
for season in df["Season"].unique():
    sdf = df[df["Season"] == season]
    teams = pd.unique(sdf[["HomeTeam", "AwayTeam"]].values.ravel())
    
    for team in teams:
        home = sdf[sdf["HomeTeam"] == team]
        away = sdf[sdf["AwayTeam"] == team]

        rows.append({
            "Season": season,
            "Team": team,
            "Context": "Home",
            "W": (home["FTR"] == "H").sum(),
            "D": (home["FTR"] == "D").sum(),
            "L": (home["FTR"] == "A").sum(),
            "GF": home["FTHG"].sum(),
            "GA": home["FTAG"].sum(),
            "Pts": (home["FTR"] == "H").sum() * 3 + (home["FTR"] == "D").sum()
        })

        rows.append({
            "Season": season,
            "Team": team,
            "Context": "Away",
            "W": (away["FTR"] == "A").sum(),
            "D": (away["FTR"] == "D").sum(),
            "L": (away["FTR"] == "H").sum(),
            "GF": away["FTAG"].sum(),
            "GA": away["FTHG"].sum(),
            "Pts": (away["FTR"] == "A").sum() * 3 + (away["FTR"] == "D").sum()
        })

home_away = pd.DataFrame(rows)
home_away.to_csv(BASE_DIR / "home_away.csv", index=False)
print(home_away[home_away["Team"] == "Bayern Munich"].head(8))