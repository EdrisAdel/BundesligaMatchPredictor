import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

form = pd.read_csv(BASE_DIR / "form_all.csv")

# Reset cumulative points per team per season
form = form.sort_values(["Season", "Team", "Matchday"])
form["CumulativePoints"] = form.groupby(["Season", "Team"])["Points"].cumsum()

# Make sure Matchday is an integer
form["Matchday"] = form["Matchday"].astype(int)

form.to_csv(BASE_DIR / "form_all.csv", index=False)
print(form[form["Team"] == "Bayern Munich"].head(10))