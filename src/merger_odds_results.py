import pandas as pd
from pathlib import Path
from fuzzywuzzy import fuzz  # install python-Levenshtein for speed

def merge_odds_and_results(odds_file, results_file, output_file):
    odds = pd.read_csv(odds_file)
    results = pd.read_csv(results_file)
    
    # Normalize team names (simple lowercase and strip, but often fuzzy matching is needed)
    odds['home_key'] = odds['home_team'].str.strip().str.lower()
    odds['away_key'] = odds['away_team'].str.strip().str.lower()
    results['home_key'] = results['home_team'].str.strip().str.lower()
    results['away_key'] = results['away_team'].str.strip().str.lower()
    
    merged = pd.merge(odds, results, on=['home_key', 'away_key'], how='inner')
    # Add result label (1, X, 2)
    def map_result(row):
        if row['home_goals'] > row['away_goals']:
            return 1
        elif row['home_goals'] == row['away_goals']:
            return 0   # or 'X'
        else:
            return 2
    merged['result'] = merged.apply(map_result, axis=1)
    merged.to_csv(output_file, index=False)
    print(f"Merged dataset saved to {output_file}")
    return merged