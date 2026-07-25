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

# ----------------------------------------------------------------------
# Page config & disclaimer
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

# Try to load the trained model (if available)
model = None
model_path = MODELS_DIR / "rf_model.pkl"
if model_path.exists():
    try:
        model = joblib.load(model_path)
        st.sidebar.success("Prediction model loaded")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")

# ----------------------------------------------------------------------
# Helper functions to load available data
# ----------------------------------------------------------------------
def get_available_sites():
    """Detect which bookmakers have raw data files."""
    if not RAW_DIR.exists():
        return []
    files = glob.glob(str(RAW_DIR / "*.csv"))
    sites = set()
    for f in files:
        name = Path(f).stem
        # Expected pattern: site_YYYY-MM-DD.csv
        if "_" in name:
            site = name.rsplit("_", 1)[0]
            sites.add(site)
    return sorted(sites)

def load_odds_for_site_and_date(site, date_str):
    """Load the CSV file for a given site and date."""
    file_path = RAW_DIR / f"{site}_{date_str}.csv"
    if not file_path.exists():
        return None
    df = pd.read_csv(file_path)
    # Ensure required columns exist
    required_cols = {"home_team", "away_team", "odd_1", "odd_X", "odd_2"}
    if not required_cols.issubset(df.columns):
        st.warning(f"File missing required columns: {required_cols - set(df.columns)}")
        return None
    return df

def get_available_dates(site):
    """Return sorted list of dates for which we have data for a given site."""
    files = glob.glob(str(RAW_DIR / f"{site}_*.csv"))
    dates = []
    for f in files:
        name = Path(f).stem
        date_part = name.replace(f"{site}_", "")
        if len(date_part) == 10:  # YYYY-MM-DD
            dates.append(date_part)
    return sorted(dates, reverse=True)

# ----------------------------------------------------------------------
# Feature engineering for model prediction (if needed)
# ----------------------------------------------------------------------
def prepare_features(row):
    """
    Convert a row of odds into features the model expects.
    Adapt to match your actual model training features.
    Here we use simple odds-derived features as a placeholder.
    """
    features = {}
    # Implied probabilities (raw, without margin adjustment)
    features['prob_1'] = 1 / row['odd_1']
    features['prob_X'] = 1 / row['odd_X']
    features['prob_2'] = 1 / row['odd_2']
    # Odds margin (overround)
    margin = features['prob_1'] + features['prob_X'] + features['prob_2']
    features['margin'] = margin
    # Normalized probabilities
    features['norm_prob_1'] = features['prob_1'] / margin
    features['norm_prob_X'] = features['prob_X'] / margin
    features['norm_prob_2'] = features['prob_2'] / margin
    # Additional features could be added if you have them (team form, H2H, etc.)
    return features

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("Data Selection")

available_sites = get_available_sites()
if not available_sites:
    st.error(
        "No data files found. Please run the scraper first to populate `data/raw/`."
    )
    st.stop()

site = st.sidebar.selectbox("Bookmaker", available_sites)
dates = get_available_dates(site)
if not dates:
    st.error(f"No data files for {site}.")
    st.stop()

date = st.sidebar.selectbox("Date", dates, index=0)

# Load data
df = load_odds_for_site_and_date(site, date)
if df is None:
    st.error(f"Could not load data for {site} on {date}.")
    st.stop()

st.sidebar.info(f"Loaded {len(df)} matches from {site} ({date})")

# ----------------------------------------------------------------------
# Show raw odds data
# ----------------------------------------------------------------------
st.subheader(f"📋 Odds from {site} – {date}")

# Format odds to 2 decimal places
display_df = df.copy()
display_df[['odd_1', 'odd_X', 'odd_2']] = display_df[['odd_1', 'odd_X', 'odd_2']].round(2)

# Highlight the lowest odd (favorite)
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

# ----------------------------------------------------------------------
# Exploratory visuals
# ----------------------------------------------------------------------
st.subheader("📊 Odds Distribution")

col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(df, x='odd_1', nbins=20, title='Home Win Odds (1)')
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.histogram(df, x='odd_2', nbins=20, title='Away Win Odds (2)')
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# Model predictions (if model exists)
# ----------------------------------------------------------------------
if model is not None:
    st.subheader("🤖 Model Predictions")

    # Prepare features from odds
    features_list = []
    for _, row in df.iterrows():
        features_list.append(prepare_features(row))
    X_pred = pd.DataFrame(features_list)

    # Ensure correct feature order as during training
    # Replace with your actual feature names
    expected_features = ['prob_1', 'prob_X', 'prob_2', 'margin',
                         'norm_prob_1', 'norm_prob_X', 'norm_prob_2']
    # If model was trained with different features, adjust accordingly.
    # This example uses only the above; real training would include team form etc.
    try:
        X_pred = X_pred[expected_features]
    except KeyError as e:
        st.error(f"Feature mismatch. Model expects: {expected_features}. Got: {list(X_pred.columns)}")
        st.stop()

    # Make prediction
    try:
        predictions = model.predict(X_pred)
        # If classifier, get class probabilities
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_pred)
            classes = model.classes_
        else:
            probas = None
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Map numeric predictions to labels (1=Home, X=Draw, 2=Away)
    outcome_map = {1: "Home Win", 0: "Draw", 2: "Away Win"}  # adjust based on your encoding
    # If your model uses strings, adjust accordingly
    if isinstance(predictions[0], str):
        outcome_map = None  # already string

    # Add predictions to display dataframe
    pred_df = df[['home_team', 'away_team']].copy()
    if outcome_map:
        pred_df['Predicted'] = [outcome_map.get(p, str(p)) for p in predictions]
    else:
        pred_df['Predicted'] = predictions

    if probas is not None:
        confidence = np.max(probas, axis=1) * 100
        pred_df['Confidence (%)'] = confidence.round(1)

    st.dataframe(pred_df, use_container_width=True)

    # Confidence chart
    if probas is not None:
        fig = px.histogram(confidence, nbins=20, title='Prediction Confidence Distribution',
                           labels={'value': 'Confidence (%)'})
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No prediction model found. Train and save a model to `models/rf_model.pkl` to see predictions.")

# ----------------------------------------------------------------------
# Comparison across bookmakers (if multiple files exist)
# ----------------------------------------------------------------------
st.subheader("🔁 Cross‑Bookmaker Comparison (Same Date)")

if len(available_sites) > 1:
    comparison_date = st.date_input("Select date for comparison", datetime.strptime(date, "%Y-%m-%d"))
    date_str = comparison_date.strftime("%Y-%m-%d")
    all_dfs = {}
    for s in available_sites:
        df_tmp = load_odds_for_site_and_date(s, date_str)
        if df_tmp is not None:
            all_dfs[s] = df_tmp

    if len(all_dfs) > 1:
        # Merge on team names (requires exact matching – simplistic)
        merged = None
        for s, sdf in all_dfs.items():
            sdf = sdf.copy()
            sdf['match'] = sdf['home_team'] + ' vs ' + sdf['away_team']
            sdf = sdf[['match', 'odd_1', 'odd_X', 'odd_2']].add_prefix(f'{s}_')
            sdf.rename(columns={f'{s}_match': 'match'}, inplace=True)
            if merged is None:
                merged = sdf
            else:
                merged = pd.merge(merged, sdf, on='match', how='outer')
        st.dataframe(merged, use_container_width=True)
    else:
        st.write("Not enough data for cross‑bookmaker comparison on this date.")
else:
    st.info("Only one bookmaker data available – add more scrapers.")