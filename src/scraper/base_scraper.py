import time
import random
import requests
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod

class BaseScraper(ABC):
    def __init__(self, base_url, delay=2):
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def _get_soup(self, url):
        time.sleep(self.delay + random.uniform(0, 1))
        resp = self.session.get(url)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, 'html.parser')

    @abstractmethod
    def scrape_odds(self, match_url):
        pass

    @abstractmethod
    def scrape_fixtures(self, date):
        pass