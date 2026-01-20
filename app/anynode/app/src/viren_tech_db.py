import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
import hvac
import os
import feedparser
from datetime import datetime

class VirenTechDatabase:
    def __init__(self):
        self.vault_client = hvac.Client(url=os.getenv('VAULT_URL'), token=os.getenv('VAULT_TOKEN'))
        self.mongo_client = MongoClient(os.getenv('MONGO_URI'))
        self.db = self.mongo_client['viren_tech_db']
        self.tech_collection = self.db['technologies']

    def update_tech_data(self):
        """Update tech database with new software and tech from web and APIs."""
        # Scrape tech news (e.g., TechCrunch)
        response = requests.get('https://techcrunch.com/category/software/')
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article', class_='post')
        for article in articles:
            title = article.find('h2').text.strip()
            link = article.find('a')['href']
            self.tech_collection.update_one(
                {'title': title},
                {'$set': {'title': title, 'link': link, 'source': 'techcrunch', 'updated_at': datetime.now()}},
                upsert=True
            )

        # Fetch from GitHub trending
        github_trending = requests.get('https://api.github.com/search/repositories?q=stars:>1&sort=stars&order=desc')
        for repo in github_trending.json()['items'][:10]:
            self.tech_collection.update_one(
                {'name': repo['name']},
                {'$set': {'name': repo['name'], 'url': repo['html_url'], 'source': 'github', 'updated_at': datetime.now()}},
                upsert=True
            )

        # Fetch from RSS feeds (e.g., Ars Technica)
        feed = feedparser.parse('https://arstechnica.com/feed/')
        for entry in feed.entries[:10]:
            self.tech_collection.update_one(
                {'title': entry.title},
                {'$set': {'title': entry.title, 'link': entry.link, 'source': 'arstechnica', 'updated_at': datetime.now()}},
                upsert=True
            )

    def query_tech(self, query):
        """Query tech database for troubleshooting or deployment."""
        return list(self.tech_collection.find({'$text': {'$search': query}}))