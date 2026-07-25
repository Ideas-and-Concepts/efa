from .base_scraper import BaseScraper
import pandas as pd

class BetwayUGScraper(BaseScraper):
    def __init__(self):
        super().__init__('https://www.betway.co.ug')

    def scrape_odds(self):
        soup = self._get_soup(f'{self.base_url}/sport/soccer')
        matches = []
        for row in soup.select('.matchRow'):  # adjust selector
            home = row.select_one('.homeTeam').text.strip()
            away = row.select_one('.awayTeam').text.strip()
            odd_1 = row.select_one('.outcome-1 .odds').text.strip()
            odd_X = row.select_one('.outcome-X .odds').text.strip()
            odd_2 = row.select_one('.outcome-2 .odds').text.strip()
            matches.append({
                'home_team': home,
                'away_team': away,
                'odd_1': float(odd_1),
                'odd_X': float(odd_X),
                'odd_2': float(odd_2),
                'timestamp': pd.Timestamp.now()
            })
        return pd.DataFrame(matches)