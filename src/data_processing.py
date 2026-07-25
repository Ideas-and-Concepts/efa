import pandas as pd
from pathlib import Path
from datetime import datetime

def save_odds(df, site_name):
    date_str = datetime.now().strftime('%Y-%m-%d')
    path = Path(f'data/raw/{site_name}_{date_str}.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)