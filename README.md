# eFa
Sports betting AI

# Uganda Betting Analysis & Predictions

This project collects sports odds from Ugandan betting websites, performs exploratory analysis, and builds machine learning models to predict match outcomes.

⚠️ **Disclaimer**: This is for educational purposes only. Scraping may violate site terms. Do not use for actual gambling.

## Features
- Web scraping of odds from multiple Ugandan bookmakers
- Historical data storage and processing
- Match outcome prediction using Random Forest / XGBoost
- Streamlit dashboard for visualization and predictions

## Setup
1. Clone the repo
2. Install requirements: `pip install -r requirements.txt`
3. Run scrapers: `python -m src.scraper.site_scrapers`
4. Train model: `python src/model.py`
5. Launch dashboard: `streamlit run app/dashboard.py`

## Data Sources (adapt as needed)
- Betway Uganda
- SportPesa Uganda
- 1xBet Uganda
- ForteBet
-BetPawa

## License
MIT