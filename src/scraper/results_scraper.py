# src/scraper/results_scraper.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
import re

class FlashScoreResultsScraper:
    """
    Scrapes match results from FlashScore.
    Works for any league by providing the correct URL.
    Example URLs (adjust as needed):
    - Uganda Premier League: https://www.flashscore.com/football/uganda/premier-league/results/
    - English Premier League: https://www.flashscore.com/football/england/premier-league/results/
    """

    def __init__(self, league_url, delay=2):
        self.league_url = league_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "X-Requested-With": "XMLHttpRequest",  # often needed for FlashScore
        })

    def _get_page(self, url):
        time.sleep(self.delay)
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.text

    def _parse_date(self, date_text):
        """Convert FlashScore date format like '25.07.2026' to YYYY-MM-DD."""
        try:
            return datetime.strptime(date_text.strip(), "%d.%m.%Y").strftime("%Y-%m-%d")
        except:
            return date_text.strip()

    def scrape_season(self, season_url_suffix=""):
        """
        Scrape all match results from the league's results page.
        The page usually loads all matches for a season if you scroll enough.
        For simplicity we fetch the base URL; FlashScore loads data via AJAX.
        We'll use a static fallback by mimicking their internal API.
        """
        # FlashScore's results page uses dynamic loading; 
        # we can fetch data via their hidden API endpoint:
        # https://www.flashscore.com/x/feed/f_1_{event_id}_{part}_4_en_1
        # but we need the tournament ID first.
        # Instead, we'll use the simpler method: load the standard HTML and parse.
        # Note: FlashScore heavily relies on JavaScript, so requests might not render.
        # For a robust solution, we'd use Selenium.
        # Here's a Selenium-based version:
        pass

# ---- Selenium version (recommended for FlashScore) ----
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class FlashScoreResultsSelenium:
    def __init__(self, league_url, headless=True, delay=2):
        self.league_url = league_url
        self.delay = delay
        options = Options()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=options)

    def _load_full_page(self, url):
        """Load a FlashScore results page, scroll to reveal all matches."""
        self.driver.get(url)
        time.sleep(self.delay * 2)
        # Try to click "Show more matches" if present
        try:
            show_more = self.driver.find_element(By.CSS_SELECTOR, "a.event__more--static")
            self.driver.execute_script("arguments[0].click();", show_more)
            time.sleep(self.delay)
        except:
            pass
        # Scroll to bottom a few times
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        for _ in range(5):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(self.delay)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        return self.driver.page_source

    def scrape_results(self):
        html = self._load_full_page(self.league_url)
        soup = BeautifulSoup(html, "html.parser")

        matches = []
        # FlashScore match container classes (as of 2025)
        for event in soup.select(".event__match"):
            try:
                # Date/time
                date_el = event.select_one(".event__time")
                date_text = date_el.get_text(strip=True) if date_el else ""
                date_clean = self._parse_date(date_text)

                # Teams
                home_el = event.select_one(".event__participant--home")
                away_el = event.select_one(".event__participant--away")
                home = home_el.get_text(strip=True) if home_el else None
                away = away_el.get_text(strip=True) if away_el else None

                # Score
                score_home = event.select_one(".event__score--home")
                score_away = event.select_one(".event__score--away")
                if score_home and score_away:
                    home_goals = int(score_home.get_text(strip=True))
                    away_goals = int(score_away.get_text(strip=True))
                else:
                    continue  # not finished or no score

                matches.append({
                    "home_team": home,
                    "away_team": away,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "match_date": date_clean,
                })
            except Exception as e:
                print(f"Skipping event: {e}")
                continue

        df = pd.DataFrame(matches)
        print(f"Scraped {len(df)} completed matches.")
        return df

    def save_results(self, df, league_name):
        if df.empty:
            print("No results to save.")
            return
        out_dir = Path("data/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / f"{league_name}_results.csv"
        # Append if file exists to accumulate history
        if filepath.exists():
            existing = pd.read_csv(filepath)
            df = pd.concat([existing, df]).drop_duplicates(
                subset=["home_team", "away_team", "match_date"]
            )
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")

    def _parse_date(self, text):
        # FlashScore date format can be "DD.MM." or "Today", "Yesterday"
        if "today" in text.lower():
            return datetime.now().strftime("%Y-%m-%d")
        if "yesterday" in text.lower():
            return (datetime.now() - timedelta(1)).strftime("%Y-%m-%d")
        try:
            return datetime.strptime(text, "%d.%m.").replace(year=datetime.now().year).strftime("%Y-%m-%d")
        except:
            return text

    def close(self):
        self.driver.quit()

# ---- Example usage ----
if __name__ == "__main__":
    # Uganda Premier League results
    url = "https://www.flashscore.com/football/uganda/premier-league/results/"
    scraper = FlashScoreResultsSelenium(url)
    try:
        df = scraper.scrape_results()
        scraper.save_results(df, "uganda_premier_league")
    finally:
        scraper.close()