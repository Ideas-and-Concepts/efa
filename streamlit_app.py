# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import glob
import sys
import traceback

# Add project root to sys.path for imports
sys.path.append(str(Path(__file__).parent.parent))

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Uganda Betting Analysis",
    page_icon="⚽",
    layout="wide",
)

# ----------------------------------------------------------------------
# Compatibility wrapper for st.rerun (newer Streamlit versions)
def rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# ----------------------------------------------------------------------
# Cached data loader
@st.cache_data(ttl=300)  # cache for 5 minutes
def get_available_sites():
    RAW_DIR = Path("data/raw")
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

@st.cache_data(ttl=300)
def load_odds_for_site_and_date(site: str, date_str: str):
    file_path = Path("data/raw") / f"{site}_{date_str}.csv"
    if not file_path.exists():
        return None
    df = pd.read_csv(file_path)
    required_cols = {"home_team", "away_team", "odd_1", "odd_X", "odd_2"}
    if not required_cols.issubset(df.columns):
        return None
    return df

@st.cache_data(ttl=300)
def get_available_dates(site: str):
    files = glob.glob(str(Path("data/raw") / f"{site}_*.csv"))
    dates = []
    for f in files:
        name = Path(f).stem
        date_part = name.replace(f"{site}_", "")
        if len(date_part) == 10:
            dates.append(date_part)
    return sorted(dates, reverse=True)

# ----------------------------------------------------------------------
# Model loading
@st.cache_resource
def load_model():
    model_path = Path("models/rf_model.pkl")
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception:
            return None
    return None

# ----------------------------------------------------------------------
# Feature engineering (must match training)
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

def prepare_features_df(df):
    """Apply prepare_features to entire DataFrame and return ordered feature matrix."""
    features_list = df.apply(lambda row: pd.Series(prepare_features(row)), axis=1)
    expected = ['prob_1', 'prob_X', 'prob_2', 'margin', 'norm_prob_1', 'norm_prob_X', 'norm_prob_2']
    return features_list[expected]

# ----------------------------------------------------------------------
# Fuzzy matching for team names
from fuzzywuzzy import fuzz

def fuzzy_match_teams(df_a, df_b, threshold=80):
    """Find matching rows between two DataFrames based on home+away team names."""
    matches = []
    for i, row_a in df_a.iterrows():
        best_score = -1
        best_idx = None
        for j, row_b in df_b.iterrows():
            score_home = fuzz.token_set_ratio(row_a['home_team'].lower(), row_b['home_team'].lower())
            score_away = fuzz.token_set_ratio(row_a['away_team'].lower(), row_b['away_team'].lower())
            avg = (score_home + score_away) / 2
            if avg > best_score:
                best_score = avg
                best_idx = j
        if best_score >= threshold:
            matches.append((i, best_idx, best_score))
    return matches

# ----------------------------------------------------------------------
# Start of UI
# ----------------------------------------------------------------------
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

model = load_model()

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.header("Data Selection")

available_sites = get_available_sites()
if not available_sites:
    st.error("No data files found. Please run a scraper first (use the button below).")
    # scraper button still visible
    if st.sidebar.button("🔄 Run Fortebet Scraper Now"):
        try:
            from src.scraper.site_scrapers import FortebetScraper
            scraper = FortebetScraper(headless=True)
            df_scraped = scraper.scrape_football_odds()
            scraper.save_data(df_scraped)
            scraper.close()
            st.sidebar.success(f"Scraped {len(df_scraped)} matches. Refreshing...")
            rerun()
        except ImportError:
            st.sidebar.error("Scraper module not found. Check src/scraper/site_scrapers.py")
        except Exception as e:
            st.sidebar.error(f"Scraper error: {e}")
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

# Data freshness badge
latest_date = datetime.strptime(dates[0], "%Y-%m-%d")
days_old = (datetime.now() - latest_date).days
if days_old == 0:
    freshness = "🟢 Today"
elif days_old == 1:
    freshness = "🟡 Yesterday"
else:
    freshness = f"🔴 {days_old} days ago"
st.sidebar.info(f"Latest data: {freshness}")

# Quick stats
st.sidebar.metric("Matches loaded", len(df))
st.sidebar.metric("Bookmakers available", len(available_sites))

if st.sidebar.button("🔄 Run Fortebet Scraper Now"):
    try:
        from src.scraper.site_scrapers import FortebetScraper
        scraper = FortebetScraper(headless=True)
        df_new = scraper.scrape_football_odds()
        scraper.save_data(df_new)
        scraper.close()
        st.sidebar.success(f"Scraped {len(df_new)} matches. Refreshing...")
        # Clear caches so new data appears
        get_available_sites.clear()
        load_odds_for_site_and_date.clear()
        get_available_dates.clear()
        rerun()
    except ImportError:
        st.sidebar.error("Scraper module not found (src/scraper/site_scrapers.py)")
    except Exception as e:
        st.sidebar.error(f"Scraper error: {e}")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📋 Odds & Predictions", "💹 Value & Arbitrage", "📈 Trends", "📊 Model Performance"])

# ===================== TAB 1: Odds & Predictions =====================
with tab1:
    st.subheader(f"Odds from {site} – {date}")

    # Formatting
    display_df = df.copy()
    for col in ['odd_1', 'odd_X', 'odd_2']:
        display_df[col] = display_df[col].round(2)

    # Highlight favorite
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

    # Model predictions (if model exists)
    if model:
        st.subheader("🤖 Model Predictions")
        try:
            X_pred = prepare_features_df(df)
            predictions = model.predict(X_pred)
            probas = model.predict_proba(X_pred) if hasattr(model, "predict_proba") else None

            # Map predictions to labels
            outcome_map = {1: "Home Win", 0: "Draw", 2: "Away Win"}
            pred_df = df[['home_team', 'away_team']].copy()
            pred_df['Predicted'] = [outcome_map.get(int(p), str(p)) for p in predictions]

            if probas is not None:
                confidence = np.max(probas, axis=1) * 100
                pred_df['Confidence (%)'] = confidence.round(1)

            st.dataframe(pred_df, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ===================== TAB 2: Value & Arbitrage =====================
with tab2:
    st.subheader("💹 Arbitrage Opportunities")

    if len(available_sites) < 2:
        st.info("Need at least two bookmakers for arbitrage detection.")
    else:
        arb_date = st.date_input("Arbitrage date", datetime.strptime(date, "%Y-%m-%d"))
        arb_date_str = arb_date.strftime("%Y-%m-%d")
        dfs = {}
        for s in available_sites:
            tmp = load_odds_for_site_and_date(s, arb_date_str)
            if tmp is not None:
                dfs[s] = tmp

        if len(dfs) < 2:
            st.warning("Not enough data on this date from multiple bookmakers.")
        else:
            # Arbitrage calculation using best odds across all bookmakers
            # For each match in the first bookmaker, find matches in others (fuzzy) and combine odds
            ref_site = list(dfs.keys())[0]
            ref_df = dfs[ref_site]
            all_arbs = []

            for i, row in ref_df.iterrows():
                home = row['home_team']
                away = row['away_team']
                best_1 = row['odd_1']
                best_X = row['odd_X']
                best_2 = row['odd_2']

                # Collect odds from other bookmakers for the same match
                for s, sdf in dfs.items():
                    if s == ref_site:
                        continue
                    # Use fuzzy matching to find the same match
                    matches = fuzzy_match_teams(ref_df.iloc[[i]], sdf, threshold=80)
                    if matches:
                        matched_idx = matches[0][1]  # (i_ref, j_other, score)
                        best_1 = max(best_1, sdf.iloc[matched_idx]['odd_1'])
                        best_X = max(best_X, sdf.iloc[matched_idx]['odd_X'])
                        best_2 = max(best_2, sdf.iloc[matched_idx]['odd_2'])

                # Check for arbitrage
                inv_sum = 1 / best_1 + 1 / best_X + 1 / best_2
                if inv_sum < 1:
                    profit = (1 - inv_sum) * 100
                    # Stake proportions
                    stakes = [1 / best_1, 1 / best_X, 1 / best_2] / inv_sum
                    all_arbs.append({
                        'Match': f"{home} vs {away}",
                        'Best 1': round(best_1, 2),
                        'Best X': round(best_X, 2),
                        'Best 2': round(best_2, 2),
                        'Arbitrage %': round(profit, 2),
                        'Stake 1 %': round(stakes[0] * 100, 1),
                        'Stake X %': round(stakes[1] * 100, 1),
                        'Stake 2 %': round(stakes[2] * 100, 1),
                    })

            if all_arbs:
                arb_df = pd.DataFrame(all_arbs)
                st.success(f"Found {len(arb_df)} arbitrage opportunities!")
                st.dataframe(arb_df, use_container_width=True)
            else:
                st.info("No arbitrage opportunities found on this date.")

    st.subheader("✨ Value Bets (Model vs Market)")
    if model is None:
        st.info("Load a model to detect value bets.")
    else:
        try:
            X_pred = prepare_features_df(df)
            probas = model.predict_proba(X_pred)
            classes = model.classes_  # e.g., [0, 1, 2]

            # Market implied probabilities (margin-adjusted)
            margin = 1 / df['odd_1'] + 1 / df['odd_X'] + 1 / df['odd_2']
            market_prob = {
                1: (1 / df['odd_1']) / margin,
                0: (1 / df['odd_X']) / margin,
                2: (1 / df['odd_2']) / margin,
            }

            val_df = df[['home_team', 'away_team']].copy()
            threshold = 0.02  # 2% edge

            for outcome, label in zip([1, 0, 2], ['Home', 'Draw', 'Away']):
                idx = list(classes).index(outcome)
                model_prob = probas[:, idx]
                market_p = market_prob[outcome]
                val_df[f'Model_{label}%'] = (model_prob * 100).round(1)
                val_df[f'Market_{label}%'] = (market_p * 100).round(1)
                val_df[f'Value_{label}'] = (model_prob - market_p) > threshold

            # Highlight rows where any value bet exists
            def highlight_value(row):
                styles = [''] * len(row)
                for i, col in enumerate(row.index):
                    if col.startswith('Value_') and row[col]:
                        styles[i] = 'background-color: #d4edda'
                return styles

            styled_val = val_df.style.apply(highlight_value, axis=1)
            st.dataframe(styled_val, use_container_width=True)
        except Exception as e:
            st.error(f"Error calculating value bets: {e}")

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
            # Merge on exact team names
            merged = pd.merge(
                df1[['home_team', 'away_team', 'odd_1', 'odd_X', 'odd_2']],
                df2[['home_team', 'away_team', 'odd_1', 'odd_X', 'odd_2']],
                on=['home_team', 'away_team'],
                suffixes=('_new', '_old')
            )

            if not merged.empty:
                # Compute absolute changes
                for outcome in ['1', 'X', '2']:
                    merged[f'Δ_{outcome}'] = merged[f'odd_{outcome}_new'] - merged[f'odd_{outcome}_old']

                st.dataframe(merged, use_container_width=True)

                # Interactive line chart for a selected match
                matches_list = merged['home_team'] + " vs " + merged['away_team']
                selected_match = st.selectbox("Pick a match to see odds trend", matches_list)
                if selected_match:
                    match_row = merged[matches_list == selected_match].iloc[0]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['Home', 'Draw', 'Away'],
                        y=[match_row['odd_1_old'], match_row['odd_X_old'], match_row['odd_2_old']],
                        name=f'Odds on {date2}',
                        marker_color='lightslategray'
                    ))
                    fig.add_trace(go.Bar(
                        x=['Home', 'Draw', 'Away'],
                        y=[match_row['odd_1_new'], match_row['odd_X_new'], match_row['odd_2_new']],
                        name=f'Odds on {date}',
                        marker_color='indianred'
                    ))
                    fig.update_layout(barmode='group', title=f"Odds Change: {selected_match}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No common matches between these two dates.")
        else:
            st.warning("Could not load data for one of the dates.")

# ===================== TAB 4: Model Performance =====================
with tab4:
    st.subheader("📊 Model Performance Over Time")

    merged_path = Path("data/processed/merged_data.csv")
    if not merged_path.exists():
        st.info("Merged dataset (odds + results) not found. Run the merge script to create it.")
    elif model is None:
        st.info("Model not loaded.")
    else:
        try:
            hist = pd.read_csv(merged_path)
            # Ensure we have result column
            if 'result' not in hist.columns:
                if 'home_goals' in hist.columns and 'away_goals' in hist.columns:
                    hist['result'] = np.where(hist['home_goals'] > hist['away_goals'], 1,
                                              np.where(hist['home_goals'] == hist['away_goals'], 0, 2))
                else:
                    st.error("Merged data must contain either 'result' or 'home_goals'/'away_goals'.")
                    st.stop()

            # Prepare features and predict
            X_hist = prepare_features_df(hist)
            hist['predicted'] = model.predict(X_hist)
            hist['correct'] = hist['predicted'] == hist['result']

            # Daily accuracy
            if 'match_date' in hist.columns:
                hist['match_date'] = pd.to_datetime(hist['match_date'])
                daily_acc = hist.groupby('match_date')['correct'].mean().reset_index()
                daily_acc.columns = ['Date', 'Accuracy']
                # 7-day rolling average
                daily_acc['Rolling_7D'] = daily_acc['Accuracy'].rolling(7, min_periods=1).mean()

                fig = px.line(daily_acc, x='Date', y=['Accuracy', 'Rolling_7D'],
                              title='Model Daily Accuracy (with 7‑day rolling)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                overall = hist['correct'].mean()
                st.metric("Overall Accuracy", f"{overall:.2%}")
                # Show by index (chronological order) if date missing
                hist['index'] = range(len(hist))
                rolling = hist['correct'].rolling(50, min_periods=1).mean()
                fig = px.line(x=hist['index'], y=rolling, title='Accuracy (50‑match rolling window)')
                st.plotly_chart(fig, use_container_width=True)

            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = pd.crosstab(hist['result'], hist['predicted'], rownames=['Actual'], colnames=['Predicted'])
            st.dataframe(cm)

        except Exception as e:
            st.error(f"Error computing performance: {e}")
            st.text(traceback.format_exc())

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown("---")
st.caption("Built for educational purposes. Data sourced from public betting websites.")
