# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import plotly.express as px
from datetime import datetime, timedelta
import glob
import os
import sys

# Add src to path so we can import scrapers
sys.path.append(str(Path(__file__).parent.parent))

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Uganda Betting Analysis",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Uganda Betting Analysis & Predictions")
st.caption("Educational project – Not for real gambling")

with st.expander("⚠️ Disclaimer", expanded=False):
    st.warning(
        """
        This application is for **educational and research purposes only**.  
        Scraping betting websites may violate their Terms of Service.  
        Predictions are experimental and carry no guarantee of accuracy.  
        **Do not use this for actual gambling.**
        """
    )

# ----------------------------------------------------------------------
# Paths & model loading
# ----------------------------------------------------------------------
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")

model = None
model_path = MODELS_DIR / "rf_model.pkl"
if model_path.exists():
    try:
        model = joblib.load(model_path)
        st.sidebar.success("Prediction model loaded")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def get_available_sites():
    if not RAW_DIR.exists():
        return []
    files = glob.glob(str(RAW_DIR / "*.csv"))
    sites = set()
    for f in files:
        name = Path(f).stem
        if "_" in name:
            site = name.rsplit("_", 1)[0]
            sites.add(site)
    return sorted(sites)

def load_odds_for_site_and_date(site, date_str):
    file_path = RAW_DIR / f"{site}_{date_str}.csv"
    if not file_path.exists():
        return None
    df = pd.read_csv(file_path)
    required_cols = {"home_team", "away_team", "odd_1", "odd_X", "odd_2"}
    if not required_cols.issubset(df.columns):
        st.warning(f"File missing required columns: {required_cols - set(df.columns)}")
        return None
    return df

def get_available_dates(site):
    files = glob.glob(str(RAW_DIR / f"{site}_*.csv"))
    dates = []
    for f in files:
        name = Path(f).stem
        date_part = name.replace(f"{site}_", "")
        if len(date_part) == 10:
            dates.append(date_part)
    return sorted(dates, reverse=True)

# ----------------------------------------------------------------------
# Feature engineering for model (must match training)
# ----------------------------------------------------------------------
def prepare_features(row):
    features = {}
    features['prob_1'] = 1 / row['odd_1']
    features['prob_X'] = 1 / row['odd_X']
    features['prob_2'] = 1 / row['odd_2']
    margin = features['prob_1'] + features['prob_X'] + features['prob_2']
    features['margin'] = margin
    features['norm_prob_1'] = features['prob_1'] / margin
    features['norm_prob_X'] = features['prob_X'] / margin
    features['norm_prob_2'] = features['prob_2'] / margin
    return features

# ----------------------------------------------------------------------
# Fuzzy match for arbitrage
# ----------------------------------------------------------------------
from fuzzywuzzy import fuzz

def match_teams_series(series1, series2, threshold=80):
    """Return list of matched indices from series2 for each item in series1."""
    matches = []
    for name1 in series1:
        best_score = 0
        best_idx = None
        for idx2, name2 in series2.items():
            score = fuzz.token_set_ratio(name1.lower(), name2.lower())
            if score > best_score:
                best_score = score
                best_idx = idx2
        if best_score >= threshold:
            matches.append(best_idx)
        else:
            matches.append(None)
    return matches

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("Data Selection")
available_sites = get_available_sites()

if not available_sites:
    st.error("No data files found. Run scraper first or use the 'Run Scraper' button.")
    # Even if no data, we still show the scraper button
    st.stop()

site = st.sidebar.selectbox("Bookmaker", available_sites)
dates = get_available_dates(site)
if not dates:
    st.error(f"No data files for {site}.")
    st.stop()
date = st.sidebar.selectbox("Date", dates, index=0)

df = load_odds_for_site_and_date(site, date)
if df is None:
    st.error(f"Could not load data for {site} on {date}.")
    st.stop()

st.sidebar.info(f"Loaded {len(df)} matches from {site} ({date})")

# ----------------------------------------------------------------------
# One‑click scraper button
# ----------------------------------------------------------------------
if st.sidebar.button("🔄 Run Fortebet Scraper Now"):
    try:
        from src.scraper.site_scrapers import FortebetScraper
        scraper = FortebetScraper(headless=True)
        df_scraped = scraper.scrape_football_odds()
        scraper.save_data(df_scraped)
        scraper.close()
        st.sidebar.success(f"Scraped {len(df_scraped)} matches. Refresh to see new data.")
        st.experimental_rerun()
    except Exception as e:
        st.sidebar.error(f"Scraper failed: {e}")

# ----------------------------------------------------------------------
# Tabs for different analyses
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📋 Odds & Predictions", "💹 Value & Arbitrage", "📈 Trends", "📊 Model Performance"])

# ===================== TAB 1: Odds & Predictions =====================
with tab1:
    st.subheader(f"Odds from {site} – {date}")
    display_df = df.copy()
    display_df[['odd_1', 'odd_X', 'odd_2']] = display_df[['odd_1', 'odd_X', 'odd_2']].round(2)

    def highlight_favorite(row):
        fav_idx = np.argmin([row['odd_1'], row['odd_X'], row['odd_2']])
        styles = [''] * 3
        styles[fav_idx] = 'background-color: #d4edda; font-weight: bold'
        return styles

    styled_df = display_df.style.apply(
        lambda row: highlight_favorite(row),
        subset=['odd_1', 'odd_X', 'odd_2'],
        axis=1
    )
    st.dataframe(styled_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x='odd_1', nbins=20, title='Home Win Odds (1)')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x='odd_2', nbins=20, title='Away Win Odds (2)')
        st.plotly_chart(fig, use_container_width=True)

    # Model predictions (if model)
    if model is not None:
        st.subheader("🤖 Model Predictions")
        features_list = []
        for _, row in df.iterrows():
            features_list.append(prepare_features(row))
        X_pred = pd.DataFrame(features_list)
        expected_features = ['prob_1', 'prob_X', 'prob_2', 'margin',
                             'norm_prob_1', 'norm_prob_X', 'norm_prob_2']
        try:
            X_pred = X_pred[expected_features]
        except KeyError as e:
            st.error(f"Feature mismatch: {e}")
            st.stop()

        predictions = model.predict(X_pred)
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_pred)
            confidence = np.max(probas, axis=1) * 100
        else:
            probas = None
            confidence = None

        outcome_map = {1: "Home Win", 0: "Draw", 2: "Away Win"}
        pred_df = df[['home_team', 'away_team']].copy()
        pred_df['Predicted'] = [outcome_map.get(p, str(p)) for p in predictions]
        if confidence is not None:
            pred_df['Confidence (%)'] = confidence.round(1)
        st.dataframe(pred_df, use_container_width=True)

# ===================== TAB 2: Value & Arbitrage =====================
with tab2:
    st.subheader("💹 Arbitrage Opportunities")
    if len(available_sites) < 2:
        st.info("Need at least two bookmakers for arbitrage detection.")
    else:
        # Load data for selected date across all sites
        arb_date = st.date_input("Arbitrage date", datetime.strptime(date, "%Y-%m-%d"))
        arb_date_str = arb_date.strftime("%Y-%m-%d")
        dfs = {}
        for s in available_sites:
            tmp = load_odds_for_site_and_date(s, arb_date_str)
            if tmp is not None:
                dfs[s] = tmp

        if len(dfs) < 2:
            st.warning("Not enough data on this date.")
        else:
            # Use the first bookmaker as the reference for matching
            ref_site = list(dfs.keys())[0]
            ref_df = dfs[ref_site].copy()
            ref_df['match'] = ref_df['home_team'] + ' vs ' + ref_df['away_team']

            # For each other bookmaker, find matching matches and collect best odds
            all_odds = []
            for i, row in ref_df.iterrows():
                best_1 = row['odd_1']
                best_X = row['odd_X']
                best_2 = row['odd_2']
                # Compare with other bookmakers
                for s, sdf in dfs.items():
                    if s == ref_site:
                        continue
                    # Fuzzy match
                    home_s = sdf['home_team']
                    away_s = sdf['away_team']
                    # Simple exact match on concatenated names (lowercase)
                    match_idx = sdf[(sdf['home_team'].str.lower() == row['home_team'].lower()) &
                                    (sdf['away_team'].str.lower() == row['away_team'].lower())].index
                    if len(match_idx) == 1:
                        idx = match_idx[0]
                        best_1 = max(best_1, sdf.at[idx, 'odd_1'])
                        best_X = max(best_X, sdf.at[idx, 'odd_X'])
                        best_2 = max(best_2, sdf.at[idx, 'odd_2'])
                # Calculate arbitrage
                inv_sum = 1/best_1 + 1/best_X + 1/best_2
                if inv_sum < 1:
                    profit = (1 - inv_sum) * 100
                    stakes = [1/best_1, 1/best_X, 1/best_2] / inv_sum  # normalized
                    all_odds.append({
                        'match': row['match'],
                        'Best 1': best_1,
                        'Best X': best_X,
                        'Best 2': best_2,
                        'Arbitrage %': round(profit, 2),
                        'Stake 1 %': round(stakes[0]*100, 1),
                        'Stake X %': round(stakes[1]*100, 1),
                        'Stake 2 %': round(stakes[2]*100, 1),
                    })
            if all_odds:
                arb_df = pd.DataFrame(all_odds)
                st.success(f"Found {len(arb_df)} arbitrage opportunities!")
                st.dataframe(arb_df, use_container_width=True)
            else:
                st.info("No arbitrage opportunities found.")

    st.subheader("✨ Value Bets (Model vs Market)")
    if model is None:
        st.info("Load a model to detect value bets.")
    else:
        # Use the current page's df (already loaded in tab1)
        features_list = []
        for _, row in df.iterrows():
            features_list.append(prepare_features(row))
        X_pred = pd.DataFrame(features_list)[expected_features]
        probas = model.predict_proba(X_pred)  # shape (n, 3) for 0,1,2
        classes = model.classes_  # e.g., [0,1,2]

        # Calculate market implied probabilities (adjusted for margin)
        margin = 1/df['odd_1'] + 1/df['odd_X'] + 1/df['odd_2']
        market_prob_1 = (1/df['odd_1']) / margin
        market_prob_X = (1/df['odd_X']) / margin
        market_prob_2 = (1/df['odd_2']) / margin

        val_df = df[['home_team', 'away_team']].copy()
        for outcome, label in zip([1, 0, 2], ['Home', 'Draw', 'Away']):
            idx = list(classes).index(outcome)
            model_prob = probas[:, idx]
            if outcome == 1:
                market_prob = market_prob_1
            elif outcome == 0:
                market_prob = market_prob_X
            else:
                market_prob = market_prob_2
            val_df[f'Model_{label}%'] = (model_prob * 100).round(1)
            val_df[f'Market_{label}%'] = (market_prob * 100).round(1)
            val_df[f'Value_{label}'] = (model_prob - market_prob) > 0.02  # 2% threshold

        def highlight_value(row):
            styles = [''] * len(row)
            for i, col in enumerate(row.index):
                if col.startswith('Value_') and row[col]:
                    styles[i] = 'background-color: #d4edda'
            return styles

        styled_val = val_df.style.apply(highlight_value, axis=1)
        st.dataframe(styled_val, use_container_width=True)

# ===================== TAB 3: Trends =====================
with tab3:
    st.subheader("📈 Odds Movement Over Time")
    if len(dates) < 2:
        st.info("Need at least two dates of data to show trends.")
    else:
        date2 = st.selectbox("Compare with", dates[1:], key="trend_date")
        df1 = load_odds_for_site_and_date(site, date)
        df2 = load_odds_for_site_and_date(site, date2)
        if df1 is not None and df2 is not None:
            # Merge on team names (exact match)
            merged = pd.merge(
                df1[['home_team', 'away_team', 'odd_1', 'odd_X', 'odd_2']],
                df2[['home_team', 'away_team', 'odd_1', 'odd_X', 'odd_2']],
                on=['home_team', 'away_team'],
                suffixes=('_new', '_old')
            )
            if not merged.empty:
                for outcome in ['1', 'X', '2']:
                    merged[f'change_{outcome}'] = merged[f'odd_{outcome}_new'] - merged[f'odd_{outcome}_old']
                # Show matches with significant moves
                st.dataframe(merged, use_container_width=True)
                # Plot a specific match if selected
                match_choice = st.selectbox("Pick a match", merged['home_team'] + " vs " + merged['away_team'])
                if match_choice:
                    idx = merged[merged['home_team'] + " vs " + merged['away_team'] == match_choice].index[0]
                    row = merged.iloc[idx]
                    fig = px.bar(
                        x=['Home', 'Draw', 'Away'],
                        y=[row['odd_1_old'], row['odd_X_old'], row['odd_2_old']],
                        title=f"Odds on {date2}",
                    )
                    fig.add_bar(x=['Home', 'Draw', 'Away'], y=[row['odd_1_new'], row['odd_X_new'], row['odd_2_new']],
                                name=date)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No common matches between these dates.")

# ===================== TAB 4: Model Performance =====================
with tab4:
    st.subheader("📊 Model Accuracy Over Time")
    # This requires historical merged data with actual results
    merged_path = PROCESSED_DIR / "merged_data.csv"
    if not merged_path.exists():
        st.info("Merged dataset (odds + results) not found. Please run the merge script first.")
    elif model is None:
        st.info("Model not loaded.")
    else:
        hist = pd.read_csv(merged_path)
        # Ensure we have result column
        if 'result' not in hist.columns:
            hist['result'] = np.where(hist['home_goals'] > hist['away_goals'], 1,
                                      np.where(hist['home_goals'] == hist['away_goals'], 0, 2))
        # Prepare features for historical data
        hist_features = build_features(hist)
        hist_preds = model.predict(hist_features)
        hist['predicted'] = hist_preds
        hist['correct'] = hist['predicted'] == hist['result']
        # Group by match_date (if available) or by index order
        if 'match_date' in hist.columns:
            perf = hist.groupby('match_date')['correct'].mean().reset_index()
            perf.columns = ['Date', 'Accuracy']
            fig = px.line(perf, x='Date', y='Accuracy', title='Model Daily Accuracy')
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Overall accuracy
            acc = hist['correct'].mean()
            st.metric("Overall Accuracy", f"{acc:.2%}")