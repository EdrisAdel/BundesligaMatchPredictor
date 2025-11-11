# Edris Adel
# Bundesliga Match Predictor and Simulator using data from 2022-2025 season data

import os
import pandas as pd
import numpy as np
from datetime import datetime
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import joblib
import copy
from typing import List, Tuple, Dict
import glob

# change these paths as needed
CSV_PATHS = [
    r"c:\Users\itzni\OneDrive\Documents\DataScraper\Bundesliga2223.csv",
    r"c:\Users\itzni\OneDrive\Documents\DataScraper\Bundesliga2324.csv",
    r"c:\Users\itzni\OneDrive\Documents\DataScraper\Bundesliga2425.csv"
]

# where to save/load trained model and scaler
MODEL_PATH = r"c:\Users\itzni\OneDrive\Documents\DataScraper\rf_match_model.joblib"
SCALER_PATH = r"c:\Users\itzni\OneDrive\Documents\DataScraper\scaler.joblib"

# columns to keep (pure football numbers) -- moved above loader so load_csv can reference it
CORE_COLS = [
    "Date","Time","HomeTeam","AwayTeam","FTHG","FTAG","FTR",
    "HTHG","HTAG","HTR",
    "HS","AS","HST","AST","HC","AC","HF","AF",
    "HY","AY","HR","AR"
]

def _expand_csv_input(paths):
    """Return a list of file paths from a single path/string, list, or glob."""
    if isinstance(paths, (list, tuple)):
        files = []
        for p in paths:
            if os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, "*.csv"))))
            elif ("*" in p) or ("?" in p):
                files.extend(sorted(glob.glob(p)))
            else:
                files.append(p)
        return files
    else:
        p = paths
        if os.path.isdir(p):
            return sorted(glob.glob(os.path.join(p, "*.csv")))
        if ("*" in p) or ("?" in p):
            return sorted(glob.glob(p))
        return [p]

def load_csv(paths=CSV_PATHS, dayfirst=True):
    """Load one or more CSVs (file list, directory or glob) and concatenate.
       Keeps only CORE_COLS and parses Date/Time into Datetime, sorts by it."""
    files = _expand_csv_input(paths)
    if not files:
        raise FileNotFoundError(f"No CSV files found for input: {paths}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, dayfirst=dayfirst, low_memory=False)
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}; skipping.")
            continue

        # keep only relevant columns (add any additional columns your CSVs may have)
        for c in CORE_COLS:
            if c not in df.columns:
                df[c] = 0

        # parse Date and Time
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=dayfirst, errors="coerce")
        if "Time" in df.columns:
            # when Time is missing or malformed, coerce to NaT
            df["Datetime"] = pd.to_datetime(df["Date"].dt.strftime("%Y-%m-%d") + " " + df["Time"].astype(str),
                                            errors="coerce")
        else:
            df["Datetime"] = df["Date"]

        keep = CORE_COLS + ["Datetime"]
        df = df[keep].copy()
        dfs.append(df)

    if not dfs:
        raise ValueError("No readable CSVs were loaded.")

    # concatenate and sort by datetime
    full = pd.concat(dfs, ignore_index=True, sort=False)
    full = full.sort_values("Datetime").reset_index(drop=True)

    # ensure required columns exist and set types
    for c in CORE_COLS:
        if c not in full.columns:
            full[c] = 0

    # drop incomplete rows
    full = full.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])

    full["FTR"] = full["FTR"].astype(str)

    # ensure numeric columns are numeric and fill missing with 0
    num_cols = ["FTHG","FTAG","HS","AS","HST","AST","HC","AC","HF","AF","HY","AY","HR","AR","HTHG","HTAG"]
    for c in num_cols:
        if c in full.columns:
            full[c] = pd.to_numeric(full[c], errors="coerce").fillna(0).astype(int)
        else:
            full[c] = 0

    return full

def init_team_stats(teams):
    keys = [
        "played","wins","draws","losses","gf","ga","pts",
        "home_played","home_wins","home_draws","home_losses","home_gf","home_ga","home_hs","home_hst","home_hc",
        "away_played","away_wins","away_draws","away_losses","away_gf","away_ga","away_hs","away_hst","away_hc",
        "shots_for","shots_against","shots_on_target_for","shots_on_target_against","corners_for","corners_against",
        "fouls_for","fouls_against","yellow","red"
    ]
    base = dict.fromkeys(keys, 0)
    return {t: base.copy() for t in teams}

def update_stats(stats, home, away, hgoal, agoal, hs, ast, hst, ast_on, hc, ac, hf, af, hy, ay, hr, ar):
    stats[home]["played"] += 1
    stats[away]["played"] += 1
    stats[home]["home_played"] += 1
    stats[away]["away_played"] += 1

    stats[home]["gf"] += hgoal
    stats[home]["ga"] += agoal
    stats[away]["gf"] += agoal
    stats[away]["ga"] += hgoal

    stats[home]["home_gf"] += hgoal
    stats[home]["home_ga"] += agoal
    stats[away]["away_gf"] += agoal
    stats[away]["away_ga"] += hgoal

    stats[home]["shots_for"] += hs
    stats[away]["shots_for"] += ast
    stats[home]["shots_against"] += ast
    stats[away]["shots_against"] += hs

    stats[home]["shots_on_target_for"] += hst
    stats[away]["shots_on_target_for"] += ast_on
    stats[home]["shots_on_target_against"] += ast_on
    stats[away]["shots_on_target_against"] += hst

    stats[home]["corners_for"] += hc
    stats[away]["corners_for"] += ac
    stats[home]["corners_against"] += ac
    stats[away]["corners_against"] += hc

    stats[home]["fouls_for"] += hf
    stats[away]["fouls_for"] += af
    stats[home]["fouls_against"] += af
    stats[away]["fouls_against"] += hf

    stats[home]["yellow"] += hy
    stats[away]["yellow"] += ay
    stats[home]["red"] += hr
    stats[away]["red"] += ar

    if hgoal > agoal:
        stats[home]["wins"] += 1
        stats[away]["losses"] += 1
        stats[home]["home_wins"] += 1
        stats[away]["away_losses"] += 1
        stats[home]["pts"] += 3
    elif hgoal < agoal:
        stats[away]["wins"] += 1
        stats[home]["losses"] += 1
        stats[away]["away_wins"] += 1
        stats[home]["home_losses"] += 1
        stats[away]["pts"] += 3
    else:
        stats[home]["draws"] += 1
        stats[away]["draws"] += 1
        stats[home]["home_draws"] += 1
        stats[away]["away_draws"] += 1
        stats[home]["pts"] += 1
        stats[away]["pts"] += 1

def team_vector(s):
    played = s["played"] if s["played"] > 0 else 1
    home_played = s["home_played"] if s["home_played"] > 0 else 1
    away_played = s["away_played"] if s["away_played"] > 0 else 1
    return [
        s["played"], s["wins"], s["draws"], s["losses"], s["gf"], s["ga"], s["pts"],
        s["home_played"], s["home_wins"], s["home_draws"], s["home_losses"], s["home_gf"], s["home_ga"],
        s["away_played"], s["away_wins"], s["away_draws"], s["away_losses"], s["away_gf"], s["away_ga"],
        s["shots_for"]/played, s["shots_against"]/played,
        s["shots_on_target_for"]/played, s["shots_on_target_against"]/played,
        s["corners_for"]/played, s["corners_against"]/played,
        s["fouls_for"]/played, s["fouls_against"]/played,
        s["yellow"]/played, s["red"]/played,
        (s["gf"]-s["ga"])/played
    ]

def build_features(df):
    teams = sorted(pd.unique(df[["HomeTeam","AwayTeam"]].values.ravel()))
    stats = init_team_stats(teams)
    X_rows = []
    y = []
    for _, r in df.iterrows():
        h = r["HomeTeam"]; a = r["AwayTeam"]
        hgoal = int(r.get("FTHG",0)); agoal = int(r.get("FTAG",0))
        hs = int(r.get("HS",0)); ast = int(r.get("AS",0))
        hst = int(r.get("HST",0)); ast_on = int(r.get("AST",0))
        hc = int(r.get("HC",0)); ac = int(r.get("AC",0))
        hf = int(r.get("HF",0)); af = int(r.get("AF",0))
        hy = int(r.get("HY",0)); ay = int(r.get("AY",0))
        hr = int(r.get("HR",0)); ar = int(r.get("AR",0))

        hvec = team_vector(stats.get(h, {}))
        avec = team_vector(stats.get(a, {}))

        feat = hvec + avec + list(np.array(hvec)-np.array(avec))
        X_rows.append(feat)
        y.append(r["FTR"])

        update_stats(stats, h, a, hgoal, agoal, hs, ast, hst, ast_on, hc, ac, hf, af, hy, ay, hr, ar)

    base_names = [
        "played","wins","draws","losses","gf","ga","pts",
        "home_played","home_wins","home_draws","home_losses","home_gf","home_ga",
        "away_played","away_wins","away_draws","away_losses","away_gf","away_ga",
        "shots_for_per_game","shots_against_per_game",
        "sot_for_per_game","sot_against_per_game",
        "corners_for_per_game","corners_again_game",
        "fouls_for_per_game","fouls_against_per_game",
        "yellow_per_game","red_per_game",
        "gd_per_game"
    ]
    feature_names = []
    for p in ["home_","away_"]:
        for n in base_names:
            feature_names.append(p + n)
    for n in base_names:
        feature_names.append("diff_" + n)

    X = pd.DataFrame(X_rows, columns=feature_names)
    y = pd.Series(y, name="FTR")
    return X, y, stats

def train_and_save(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=90,stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base_clf = RandomForestClassifier(n_estimators=1000, random_state=90, class_weight=None)
    clf = CalibratedClassifierCV(base_clf, cv=3)
    clf.fit(X_train_s, y_train)

    preds = clf.predict(X_test_s)
    print("Accuracy:", accuracy_score(y_test,preds))
    print("Report:\n", classification_report(y_test,preds))
    print("Confusion:\n", confusion_matrix(y_test,preds))

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return clf, scaler

def predict_new_season(fixtures, final_stats):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Train model first.")
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    rows = []
    idx = []
    for h,a in fixtures:
        hs = final_stats.get(h, None)
        as_ = final_stats.get(a, None)
        if hs is None:
            hs = list(final_stats.values())[0].copy()
            for k in hs: hs[k]=0
        if as_ is None:
            as_ = list(final_stats.values())[0].copy()
            for k in as_: as_[k]=0
        hvec = team_vector(hs); avec = team_vector(as_)
        feat = hvec + avec + list(np.array(hvec)-np.array(avec))
        rows.append(feat); idx.append(f"{h} vs {a}")

    # build feature names same as training
    base_names = [
        "played","wins","draws","losses","gf","ga","pts",
        "home_played","home_wins","home_draws","home_losses","home_gf","home_ga",
        "away_played","away_wins","away_draws","away_losses","away_gf","away_ga",
        "shots_for_per_game","shots_against_per_game",
        "sot_for_per_game","sot_against_per_game",
        "corners_for_per_game","corners_again_game",
        "fouls_for_per_game","fouls_against_per_game",
        "yellow_per_game","red_per_game",
        "gd_per_game"
    ]
    feature_names = []
    for p in ["home_","away_"]:
        for n in base_names:
            feature_names.append(p + n)
    for n in base_names:
        feature_names.append("diff_" + n)

    Xp = pd.DataFrame(rows, columns=feature_names, index=idx)
    Xp_s = scaler.transform(Xp)
    preds = clf.predict(Xp_s)
    proba = clf.predict_proba(Xp_s) if hasattr(clf, "predict_proba") else None
    out = pd.DataFrame({"prediction": preds}, index=Xp.index)
    if proba is not None:
        classes = list(clf.classes_)
        out["prob_H"] = proba[:, classes.index("H")] if "H" in classes else 0
        out["prob_D"] = proba[:, classes.index("D")] if "D" in classes else 0
        out["prob_A"] = proba[:, classes.index("A")] if "A" in classes else 0
    return out

def round_robin_schedule(teams: List[str]) -> List[List[Tuple[str,str]]]:
    """Return rounds of a single round-robin (each team plays each other once).
       If odd number of teams, a bye is inserted and ignored in output fixtures."""
    teams = list(teams)
    if len(teams) % 2:
        teams.append(None) 
    n = len(teams)
    half = n // 2
    rounds = []
    for i in range(n-1):
        pairs = []
        for j in range(half):
            t1 = teams[j]
            t2 = teams[n-1-j]
            if t1 is not None and t2 is not None:
                pairs.append((t1, t2))
        rounds.append(pairs)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return rounds

def double_round_robin(teams: List[str]) -> List[Tuple[str,str]]:
    """Create full home-and-away fixture list as flat list of (home, away)."""
    single = round_robin_schedule(teams)
    fixtures = []
    # first half: as generated
    for rnd in single:
        fixtures.extend(rnd)
    # second half: reverse home/away for return legs
    for rnd in single:
        fixtures.extend([(away, home) for (home, away) in rnd])
    return fixtures

def _make_feature_names() -> List[str]:
    base_names = [
        "played","wins","draws","losses","gf","ga","pts",
        "home_played","home_wins","home_draws","home_losses","home_gf","home_ga",
        "away_played","away_wins","away_draws","away_losses","away_gf","away_ga",
        "shots_for_per_game","shots_against_per_game",
        "sot_for_per_game","sot_against_per_game",
        "corners_for_per_game","corners_again_game",
        "fouls_for_per_game","fouls_against_per_game",
        "yellow_per_game","red_per_game",
        "gd_per_game"
    ]
    names = []
    for p in ["home_","away_"]:
        for n in base_names:
            names.append(p + n)
    for n in base_names:
        names.append("diff_" + n)
    return names

def simulate_season(fixtures: List[Tuple[str,str]],
                    final_stats: Dict[str, Dict],
                    model,
                    scaler,
                    league_avg_goals: float = 1.35,
                    seed: int | None = None) -> pd.DataFrame:
    """Simulate a season. Use league_avg_goals to scale Poisson lambdas so draws are realistic."""
    if seed is not None:
        np.random.seed(seed)

    stats = copy.deepcopy(final_stats)
    teams = sorted(stats.keys())
    table = {t: {"P":0,"W":0,"D":0,"L":0,"GF":0,"GA":0,"GD":0,"Pts":0} for t in teams}
    feat_names = _make_feature_names()

    # zero template for unknown teams
    template_vals = next(iter(stats.values()))
    zero_template = {k: 0 for k in template_vals.keys()}

    for home, away in fixtures:
        if home not in stats:
            stats[home] = copy.deepcopy(zero_template)
            table.setdefault(home, {"P":0,"W":0,"D":0,"L":0,"GF":0,"GA":0,"GD":0,"Pts":0})
        if away not in stats:
            stats[away] = copy.deepcopy(zero_template)
            table.setdefault(away, {"P":0,"W":0,"D":0,"L":0,"GF":0,"GA":0,"GD":0,"Pts":0})

        # model input
        hvec = team_vector(stats[home])
        avec = team_vector(stats[away])
        feat = hvec + avec + list(np.array(hvec) - np.array(avec))
        Xp = pd.DataFrame([feat], columns=feat_names)
        Xp_s = scaler.transform(Xp)

        proba = model.predict_proba(Xp_s) if hasattr(model, "predict_proba") else None
        classes = list(model.classes_)

        if proba is not None:
            p_home = float(proba[0][classes.index("H")]) if "H" in classes else 0.0
            p_draw = float(proba[0][classes.index("D")]) if "D" in classes else 0.0
            p_away = float(proba[0][classes.index("A")]) if "A" in classes else 0.0
        else:
            pred = model.predict(Xp_s)[0]
            p_home = 1.0 if pred == "H" else 0.0
            p_draw = 1.0 if pred == "D" else 0.0
            p_away = 1.0 if pred == "A" else 0.0

        # compute attack/defense rates
        def attack_rate(s):
            played = s["played"] if s["played"]>0 else 1
            return (s.get("gf",0) / played)

        def defense_rate(s):
            played = s["played"] if s["played"]>0 else 1
            return (s.get("ga",0) / played)

        ha = attack_rate(stats[home])
        hd = defense_rate(stats[home])
        aa = attack_rate(stats[away])
        ad = defense_rate(stats[away])

        # scale lambdas by league_avg_goals so typical team means ~ realistic values
        home_adv = 1.04  # home advantage factor
        eps = 1e-7
        lam_h = max(0.05, home_adv * (ha + 0.1) / (ad + 0.1) * league_avg_goals)
        lam_a = max(0.05, (aa + 0.1) / (hd + 0.1) * league_avg_goals)

        # small nudge using model probabilities
        lam_h *= 1.0 + (p_home - p_away) * 0.1
        lam_a *= 1.0 + (p_away - p_home) * 0.1
        lam_h = max(0.02, lam_h); lam_a = max(0.02, lam_a)

        # sample goals from Poisson (do NOT force match to proba outcome)
        hgoal = int(np.random.poisson(lam_h))
        agoal = int(np.random.poisson(lam_a))

        update_stats(stats, home, away, hgoal, agoal,
                     hs=max(hgoal,1), ast=max(agoal,1), hst=hgoal, ast_on=agoal,
                     hc=0, ac=0, hf=0, af=0, hy=0, ay=0, hr=0, ar=0)

        # update season table
        table[home]["P"] += 1; table[away]["P"] += 1
        table[home]["GF"] += hgoal; table[home]["GA"] += agoal
        table[away]["GF"] += agoal; table[away]["GA"] += hgoal
        if hgoal > agoal:
            table[home]["W"] += 1; table[away]["L"] += 1; table[home]["Pts"] += 3
        elif hgoal < agoal:
            table[away]["W"] += 1; table[home]["L"] += 1; table[away]["Pts"] += 3
        else:
            table[home]["D"] += 1; table[away]["D"] += 1; table[home]["Pts"] += 1; table[away]["Pts"] += 1

    # finalize and sort
    rows = []
    for t, vals in table.items():
        vals["GD"] = vals["GF"] - vals["GA"]
        row = {"Team": t, **vals}
        rows.append(row)
    df_table = pd.DataFrame(rows)
    df_table = df_table.sort_values(["Pts","GD","GF"], ascending=[False, False, False]).reset_index(drop=True)
    return df_table

def get_latest_season_teams(paths):
    """Return the unique teams from the last CSV in the expanded input list."""
    files = _expand_csv_input(paths)
    if not files:
        return []
    last = files[-1]
    try:
        df = pd.read_csv(last, dayfirst=True, low_memory=False)
    except Exception:
        return []
    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    teams = [t for t in teams if pd.notna(t)]
    return sorted(teams)

if __name__ == "__main__":
    # load all historical matches (training data)
    df = load_csv(CSV_PATHS)

    total_goals = df["FTHG"].sum() + df["FTAG"].sum()
    matches = len(df)
    league_avg_goals = (total_goals / (2 * matches)) if matches > 0 else 1.35

    X, y, final_stats = build_features(df)
    clf, scaler = train_and_save(X, y)

    # determine teams to simulate from the most recent season file (ensure 18 teams)
    latest_teams = get_latest_season_teams(CSV_PATHS)
    if len(latest_teams) < 18:
        print(f"Warning: only {len(latest_teams)} teams found in latest CSV — simulation will use available teams.")
    if len(latest_teams) > 18:
        print(f"Note: {len(latest_teams)} teams found in latest CSV; trimming to first 18.")
        latest_teams = latest_teams[:18]

    teams = latest_teams

    # build a filtered baseline stats dict containing only the teams we'll simulate
    if final_stats:
        template = next(iter(final_stats.values()))
    else:
        template = {}
    zero_template = {k: 0 for k in template.keys()} if template else {}
    filtered_stats = {t: copy.deepcopy(final_stats.get(t, zero_template)) for t in teams}

    # create fixtures and simulate
    fixtures = double_round_robin(teams)
    sim_table = simulate_season(fixtures, filtered_stats, clf, scaler, league_avg_goals=league_avg_goals, seed=42)
    # change as needed
    out_path = r"c:\Users\itzni\OneDrive\Documents\DataScraper\simulated_season_table.csv"
    sim_table.to_csv(out_path, index=False)
    print("Simulated season table saved to", out_path)
    print(sim_table.head(20))