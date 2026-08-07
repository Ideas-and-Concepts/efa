# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import glob
import time
import os
import sys

# ----------------------------------------------------------------------
# Page config (must be first Streamlit command)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Uganda Betting Analysis",
    page_icon="⚽",
    layout="wide",
)

# Detect cloud environment
IS_CLOUD = bool(os.getenv('STREAMLIT_SERVER_ADDRESS', '')) or bool(os.getenv('RENDER', ''))

# Lazy‑import helpers (used only when needed)
def lazy_import(module_name):
    """Import a module only when called – avoids heavy startup."""
    import importlib
    return importlib.import_module(module_name)

# Compatibility wrapper for newer Streamlit versions
def rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# ----------------------------------------------------------------------
# Cached data inventory (combines site list + dates)
# ----------------------------------------------------------------------
@st.cache_data(ttl=300)
def get_data_inventory():
    RAW_DIR = Path("data/raw")
    if not RAW_DIR.exists():
        return [], {}
    files = glob.glob(str(RAW_DIR / "*.csv"))
    sites = set()
    site_dates = {}
    for f in files:
        name = Path(f).stem
        if "_" in name:
            site = name.rsplit("_", 1)[0]
            date_part = name.replace(f"{site}_", "")
            if len(date_part) == 10:
                sites.add(site)
                site_dates.setdefault(site, []).append(date_part)
    for s in site_dates:
        site_dates[s] = sorted(site_dates[s], reverse=True)
    return sorted(sites), site_dates

@st.cache_data(ttl=300)
def load_odds(site: str, date_str: str):
    """Load only necessary columns from the CSV."""
    file_path = Path("data/raw") / f"{site}_{date_str}.csv"
    if not file_path.exists():
        return None
    usecols = ['home_team', 'away_team', 'odd_1', 'odd_X', 'odd_2']
    try:
        df = pd.read_csv(file_path, usecols=usecols)
        return df
    except Exception:
        return None

# ----------------------------------------------------------------------
# Model loading (lazy)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    joblib = lazy_import('joblib')
    model_path = Path("models/rf_model.pkl")
    if model_path.exists():
        return joblib.load(model_path)
    return None

# ----------------------------------------------------------------------
# Feature engineering (matches training exactly)
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

def prepare_features_df(df):
    """Apply prepare_features to whole DataFrame, return ordered feature matrix."""
    features_list = df.apply(lambda row: pd.Series(prepare_features(row)), axis=1)
    expected = ['prob_1', 'prob_X', 'prob_2', 'margin', 'norm_prob_1', 'norm_prob_X', 'norm_prob_2']
    return features_list[expected]

# ----------------------------------------------------------------------
# Fuzzy matching (lazy import)
# ----------------------------------------------------------------------
def fuzzy_match_teams(df_a, df_b, threshold=80):
    """Find matching rows based on home+away team names."""
    fuzz = lazy_import('fuzzywuzzy.fuzz')
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
        "This application is for **educational and research purposes only**.\n"
        "Scraping betting websites may violate their Terms of Service.\n"
        "Predictions are experimental and carry no guarantee of accuracy.\n"
        "**Do not use this for actual gambling.**"
    )

# Load model lazily
model = load_model()

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.header("Data Selection")
sites, dates_dict = get_data_inventory()

if not sites:
    st.error("No data files found. Run the scraper locally or push data to the repo.")
    # Scraper button only if not in cloud
    if not IS_CLOUD:
        if st.sidebar.button("🔄 Run Fortebet Scraper Now"):
            try:
                from src.scraper.site_scrapers import FortebetScraper
                scraper = FortebetScraper(headless=True)
                df_new = scraper.scrape_football_odds()
                scraper.save_data(df_new)
                scraper.close()
                st.sidebar.success(f"Scraped {len(df_new)} matches. Refreshing...")
                get_data_inventory.clear()
                load_odds.clear()
                rerun()
            except ImportError:
                st.sidebar.error("Scraper module not found (src/scraper/site_scrapers.py).")
            except Exception as e:
                st.sidebar.error(f"Scraper error: {e}")
    else:
        st.sidebar.info("ℹ️ Scraper disabled in cloud – data is updated via GitHub Actions.")
    st.stop()

site = st.sidebar.selectbox("Bookmaker", sites)
dates = dates_dict.get(site, [])

if not dates:
    st.error(f"No data files for {site}.")
    st.stop()

date = st.sidebar.selectbox("Date", dates, index=0)

# Load data with a spinner
with st.spinner("Loading odds data..."):
    df = load_odds(site, date)

if df is None:
    st.error(f"Could not load data for {site} on {date}.")
    st.stop()

# Data freshness
latest_date = datetime.strptime(dates[0], "%Y-%m-%d")
days_old = (datetime.now() - latest_date).days
freshness = "🟢 Today" if days_old == 0 else ("🟡 Yesterday" if days_old == 1 else f"🔴 {days_old} days ago")
st.sidebar.info(f"Latest data: {freshness}")
st.sidebar.metric("Matches loaded", len(df))
st.sidebar.metric("Bookmakers available", len(sites))

# Scraper button (cloud‑safe)
if not IS_CLOUD:
    if st.sidebar.button("🔄 Run Fortebet Scraper Now"):
        try:
            from src.scraper.site_scrapers import FortebetScraper
            scraper = FortebetScraper(headless=True)
            df_new = scraper.scrape_football_odds()
            scraper.save_data(df_new)
            scraper.close()
            st.sidebar.success(f"Scraped {len(df_new)} matches. Refreshing...")
            get_data_inventory.clear()
            load_odds.clear()
            rerun()
        except ImportError:
            st.sidebar.error("Scraper module not found (src/scraper/site_scrapers.py)")
        except Exception as e:
            st.sidebar.error(f"Scraper error: {e}")
else:
    st.sidebar.info("ℹ️ Scraper disabled in cloud – data pushed via GitHub Actions.")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📋 Odds & Predictions", "💹 Value & Arbitrage", "📈 Trends", "📊 Model Performance"])

# ===================== TAB 1: Odds & Predictions =====================
with tab1:
    st.subheader(f"Odds from {site} – {date}")

    display_df = df.copy()
    for col in ['odd_1', 'odd_X', 'odd_2']:
        display_df[col] = display_df[col].round(2)

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
        px = lazy_import('plotly.express')
        fig = px.histogram(df, x='odd_1', nbins=20, title='Home Win Odds (1)')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        px = lazy_import('plotly.express')
        fig = px.histogram(df, x='odd_2', nbins=20, title='Away Win Odds (2)')
        st.plotly_chart(fig, use_container_width=True)

    if model:
        st.subheader("🤖 Model Predictions")
        try:
            X_pred = prepare_features_df(df)
            predictions = model.predict(X_pred)
            probas = model.predict_proba(X_pred) if hasattr(model, "predict_proba") else None

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

    if len(sites) < 2:
        st.info("Need at least two bookmakers for arbitrage detection.")
    else:
        arb_date = st.date_input("Arbitrage date", datetime.strptime(date, "%Y-%m-%d"))
        arb_date_str = arb_date.strftime("%Y-%m-%d")
        dfs = {}
        for s in sites:
            tmp = load_odds(s, arb_date_str)
            if tmp is not None:
                dfs[s] = tmp

        if len(dfs) < 2:
            st.warning("Not enough data on this date from multiple bookmakers.")
        else:
            ref_site = list(dfs.keys())[0]
            ref_df = dfs[ref_site]
            all_arbs = []

            for i, row in ref_df.iterrows():
                home = row['home_team']
                away = row['away_team']
                best_1 = row['odd_1']
                best_X = row['odd_X']
                best_2 = row['odd_2']

                for s, sdf in dfs.items():
                    if s == ref_site:
                        continue
                    matches = fuzzy_match_teams(ref_df.iloc[[i]], sdf, threshold=80)
                    if matches:
                        matched_idx = matches[0][1]
                        best_1 = max(best_1, sdf.iloc[matched_idx]['odd_1'])
                        best_X = max(best_X, sdf.iloc[matched_idx]['odd_X'])
                        best_2 = max(best_2, sdf.iloc[matched_idx]['odd_2'])

                inv_sum = 1 / best_1 + 1 / best_X + 1 / best_2
                if inv_sum < 1:
                    profit = (1 - inv_sum) * 100
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
            classes = model.classes_

            margin = 1 / df['odd_1'] + 1 / df['odd_X'] + 1 / df['odd_2']
            market_prob = {
                1: (1 / df['odd_1']) / margin,
                0: (1 / df['odd_X']) / margin,
                2: (1 / df['odd_2']) / margin,
            }

            val_df = df[['home_team', 'away_team']].copy()
            threshold = 0.02

            for outcome, label in zip([1, 0, 2], ['Home', 'Draw', 'Away']):
                idx = list(classes).index(outcome)
                model_prob = probas[:, idx]
                market_p = market_prob[outcome]
                val_df[f'Model_{label}%'] = (model_prob * 100).round(1)
                val_df[f'Market_{label}%'] = (market_p * 100).round(1)
                val_df[f'Value_{label}'] = (model_prob - market_p) > threshold

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
        df1 = load_odds(site, date)
        df2 = load_odds(site, date2)

        if df1 is not None and df2 is not None:
            merged = pd.merge(
                df1[['home_team', 'away_team', 'odd_1', 'odd_X', 'odd_2']],
                df2[['home_team', 'away_team', 'odd_1', 'odd_X', 'odd_2']],
                on=['home_team', 'away_team'],
                suffixes=('_new', '_old')
            )

            if not merged.empty:
                for outcome in ['1', 'X', '2']:
                    merged[f'Δ_{outcome}'] = merged[f'odd_{outcome}_new'] - merged[f'odd_{outcome}_old']

                st.dataframe(merged, use_container_width=True)

                matches_list = merged['home_team'] + " vs " + merged['away_team']
                selected_match = st.selectbox("Pick a match to see odds trend", matches_list)
                if selected_match:
                    go = lazy_import('plotly.graph_objects')
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
        st.info("Merged dataset (odds + results) not found. Create it with the merge script.")
    elif model is None:
        st.info("Model not loaded.")
    else:
        try:
            hist = pd.read_csv(merged_path)
            if 'result' not in hist.columns:
                if 'home_goals' in hist.columns and 'away_goals' in hist.columns:
                    hist['result'] = np.where(hist['home_goals'] > hist['away_goals'], 1,
                                              np.where(hist['home_goals'] == hist['away_goals'], 0, 2))
                else:
                    st.error("Merged data must contain either 'result' or 'home_goals'/'away_goals'.")
                    st.stop()

            X_hist = prepare_features_df(hist)
            hist['predicted'] = model.predict(X_hist)
            hist['correct'] = hist['predicted'] == hist['result']

            if 'match_date' in hist.columns:
                hist['match_date'] = pd.to_datetime(hist['match_date'])
                daily_acc = hist.groupby('match_date')['correct'].mean().reset_index()
                daily_acc.columns = ['Date', 'Accuracy']
                daily_acc['Rolling_7D'] = daily_acc['Accuracy'].rolling(7, min_periods=1).mean()

                px = lazy_import('plotly.express')
                fig = px.line(daily_acc, x='Date', y=['Accuracy', 'Rolling_7D'],
                              title='Model Daily Accuracy (with 7‑day rolling)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                overall = hist['correct'].mean()
                st.metric("Overall Accuracy", f"{overall:.2%}")
                hist['index'] = range(len(hist))
                rolling = hist['correct'].rolling(50, min_periods=1).mean()
                px = lazy_import('plotly.express')
                fig = px.line(x=hist['index'], y=rolling, title='Accuracy (50‑match rolling window)')
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Confusion Matrix")
            cm = pd.crosstab(hist['result'], hist['predicted'], rownames=['Actual'], colnames=['Predicted'])
            st.dataframe(cm)

        except Exception as e:
            st.error(f"Error computing performance: {e}")
            import traceback
            st.text(traceback.format_exc())

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
elapsed = time.time() - globals().get('_start_time', 0)
st.markdown("---")
st.caption(f"Built for educational purposes. ⚡ Loaded in {elapsed:.1f}s")