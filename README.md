Simple tool to train a match-outcome model on historical Bundesliga CSVs and simulate a 34-match season for 18 teams.
Requirements
- Python 3.8+
- Packages: pandas, numpy, scikit-learn, joblib
  Install: python -m pip install pandas numpy scikit-learn joblib

Notes
- The simulator uses team-level attack/defence estimates and Poisson goal sampling; results are stochastic unless a seed is set.
- To change input files, update CSV_PATHS in DataScraper.py.
- Make sure CV files have correct columns in data.

License
- Use freely for experimentation. No warranty.
