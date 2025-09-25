"""
Multi-Source Data Collection Module
Web scraping and data collection from multiple sources for retail pricing analysis
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import random
import re
from urllib.parse import urljoin, urlparse
import logging
from typing import List, Dict, Optional
import feedparser
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebScraper:
    """
    Enhanced web scraper for collecting data from multiple sources
    """
    
    def __init__(self, delay_range=(1, 3), max_retries=3):
        """
        Initialize the web scraper with retry logic and delay settings
        
        Args:
            delay_range (tuple): Range for random delays between requests
            max_retries (int): Maximum number of retry attempts
        """
        self.delay_range = delay_range
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def _make_request(self, url: str) -> Optional[requests.Response]:
        """
        Make HTTP request with retry logic and error handling
        
        Args:
            url (str): URL to request
            
        Returns:
            requests.Response or None: Response object if successful
        """
        for attempt in range(self.max_retries):
            try:
                # Random delay between requests
                time.sleep(random.uniform(*self.delay_range))
                
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                logger.info(f"Successfully retrieved: {url}")
                return response
                
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to retrieve {url} after {self.max_retries} attempts")
                    
        return None
    
    def scrape_books_toscrape(self, max_pages: int = 5) -> List[Dict]:
        """
        Scrape book data from books.toscrape.com
        
        Args:
            max_pages (int): Maximum number of pages to scrape
            
        Returns:
            List[Dict]: List of book data dictionaries
        """
        books_data = []
        base_url = "http://books.toscrape.com"
        
        for page in range(1, max_pages + 1):
            if page == 1:
                url = f"{base_url}/index.html"
            else:
                url = f"{base_url}/catalogue/page-{page}.html"
                
            response = self._make_request(url)
            if not response:
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            books = soup.find_all('article', class_='product_pod')
            
            if not books:
                logger.info(f"No books found on page {page}, stopping pagination")
                break
                
            for book in books:
                try:
                    # Extract book information
                    title = book.find('h3').find('a')['title']
                    price_text = book.find('p', class_='price_color').text
                    price = float(re.sub(r'[£$,]', '', price_text))
                    
                    # Extract rating
                    rating_class = book.find('p', class_='star-rating')['class'][1]
                    rating_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
                    rating = rating_map.get(rating_class, 0)
                    
                    # Check availability
                    availability = book.find('p', class_='instock availability')
                    in_stock = 1 if availability and 'in stock' in availability.text.lower() else 0
                    
                    # Get book URL for additional details
                    book_url = urljoin(base_url, book.find('h3').find('a')['href'])
                    
                    # Extract category (requires additional request)
                    category = self._get_book_category(book_url)
                    
                    book_data = {
                        'title': title,
                        'price': price,
                        'rating': rating,
                        'in_stock': in_stock,
                        'category': category,
                        'url': book_url,
                        'source': 'books_toscrape',
                        'scraped_at': datetime.now().isoformat()
                    }
                    
                    books_data.append(book_data)
                    
                except Exception as e:
                    logger.warning(f"Error parsing book data: {str(e)}")
                    continue
                    
            logger.info(f"Scraped {len(books)} books from page {page}")
            
        logger.info(f"Total books scraped: {len(books_data)}")
        return books_data
    
    def _get_book_category(self, book_url: str) -> str:
        """
        Get book category from individual book page
        
        Args:
            book_url (str): URL of the book page
            
        Returns:
            str: Book category
        """
        try:
            response = self._make_request(book_url)
            if not response:
                return "Unknown"
                
            soup = BeautifulSoup(response.content, 'html.parser')
            breadcrumb = soup.find('ul', class_='breadcrumb')
            if breadcrumb:
                category_links = breadcrumb.find_all('a')
                if len(category_links) >= 3:  # Home > Books > Category
                    return category_links[2].text.strip()
                    
        except Exception as e:
            logger.warning(f"Error getting category for {book_url}: {str(e)}")
            
        return "Unknown"
    
    def scrape_rss_feeds(self, rss_urls: List[str]) -> List[Dict]:
        """
        Scrape data from RSS feeds
        
        Args:
            rss_urls (List[str]): List of RSS feed URLs
            
        Returns:
            List[Dict]: List of RSS item data
        """
        rss_data = []
        
        for rss_url in rss_urls:
            try:
                logger.info(f"Parsing RSS feed: {rss_url}")
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries:
                    item_data = {
                        'title': getattr(entry, 'title', ''),
                        'description': getattr(entry, 'description', ''),
                        'link': getattr(entry, 'link', ''),
                        'published': getattr(entry, 'published', ''),
                        'source': rss_url,
                        'scraped_at': datetime.now().isoformat()
                    }
                    rss_data.append(item_data)
                    
                logger.info(f"Scraped {len(feed.entries)} items from RSS feed")
                
            except Exception as e:
                logger.error(f"Error parsing RSS feed {rss_url}: {str(e)}")
                
        return rss_data
    
    def save_data(self, data: List[Dict], filename: str, format: str = 'csv'):
        """
        Save scraped data to file
        
        Args:
            data (List[Dict]): Data to save
            filename (str): Output filename
            format (str): Output format ('csv' or 'json')
        """
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            filepath = os.path.join('data', filename)
            
            if format.lower() == 'csv':
                df = pd.DataFrame(data)
                df.to_csv(filepath, index=False)
            elif format.lower() == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            logger.info(f"Data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")

class DataCollector:
    """
    Main data collection orchestrator
    """
    
    def __init__(self):
        self.scraper = WebScraper()
        
    def collect_all_data(self):
        """
        Collect data from all sources
        """
        logger.info("Starting comprehensive data collection...")
        
        # 1. Scrape books data
        logger.info("Collecting book data from books.toscrape.com...")
        books_data = self.scraper.scrape_books_toscrape(max_pages=10)
        self.scraper.save_data(books_data, 'books_data.csv', 'csv')
        self.scraper.save_data(books_data, 'books_data.json', 'json')
        
        # 2. Sample RSS feeds (news/tech feeds for additional data)
        rss_feeds = [
            'https://feeds.feedburner.com/oreilly/radar/atom',
            'https://www.wired.com/feed/rss',
        ]
        
        logger.info("Collecting RSS feed data...")
        rss_data = self.scraper.scrape_rss_feeds(rss_feeds)
        if rss_data:
            self.scraper.save_data(rss_data, 'rss_data.csv', 'csv')
            self.scraper.save_data(rss_data, 'rss_data.json', 'json')
        
        # 3. Generate summary report
        self._generate_collection_report(books_data, rss_data)
        
        logger.info("Data collection completed!")
        return books_data, rss_data
    
    def _generate_collection_report(self, books_data: List[Dict], rss_data: List[Dict]):
        """
        Generate a summary report of collected data
        """
        report = {
            'collection_timestamp': datetime.now().isoformat(),
            'books_collected': len(books_data),
            'rss_items_collected': len(rss_data),
            'books_price_range': {
                'min': min([book['price'] for book in books_data]) if books_data else 0,
                'max': max([book['price'] for book in books_data]) if books_data else 0
            },
            'unique_categories': len(set([book['category'] for book in books_data])) if books_data else 0,
            'sources': list(set([book['source'] for book in books_data] + [item['source'] for item in rss_data]))
        }
        
        self.scraper.save_data([report], 'collection_report.json', 'json')
        logger.info(f"Collection report: {report}")

if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    books_data, rss_data = collector.collect_all_data()
    
    print(f"Collected {len(books_data)} books and {len(rss_data)} RSS items")
