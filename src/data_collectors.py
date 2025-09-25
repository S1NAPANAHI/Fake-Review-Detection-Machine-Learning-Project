"""
Comprehensive Data Collection System for Multi-Platform Content Authenticity

This module provides data collectors for various platforms including:
- E-commerce platforms (Amazon, etc.)
- Social media platforms (Twitter, Reddit)
- App stores (Google Play, Apple App Store)
- Review platforms (Yelp, TripAdvisor)
- Public datasets for training

Features:
- Ethical scraping with rate limiting
- Multiple data sources
- Automatic labeling heuristics
- Data quality validation
- Export in multiple formats
"""

import asyncio
import aiohttp
import time
import random
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import json
import csv
from datetime import datetime, timedelta
import logging
from urllib.parse import urljoin, urlparse
import hashlib
import re
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import our utilities
from . import utils

logger = utils.setup_logging(__name__)

@dataclass
class ContentItem:
    """Standard format for collected content across all platforms."""
    text: str
    platform: str
    content_type: str  # review, comment, post, etc.
    user_id: Optional[str] = None
    rating: Optional[float] = None
    timestamp: Optional[datetime] = None
    product_id: Optional[str] = None
    business_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Labels (for training data)
    is_fake: Optional[bool] = None
    confidence: Optional[float] = None
    label_source: Optional[str] = None  # manual, heuristic, verified
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Generate unique ID
        content_hash = hashlib.md5(f"{self.text}_{self.user_id}_{self.timestamp}".encode()).hexdigest()
        self.metadata['content_id'] = f"{self.platform}_{content_hash[:12]}"
        
        # Add collection timestamp
        self.metadata['collected_at'] = datetime.now().isoformat()

class BaseCollector:
    """Base class for all data collectors."""
    
    def __init__(self, rate_limit: float = 1.0, max_retries: int = 3):
        self.rate_limit = rate_limit  # seconds between requests
        self.max_retries = max_retries
        self.session = self._create_session()
        self.ua = UserAgent()
        self.collected_items = []
        self.last_request_time = 0
        
    def _create_session(self) -> requests.Session:
        """Create a session with retry strategy."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _rate_limit_delay(self):
        """Implement rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            sleep_time = self.rate_limit - elapsed + random.uniform(0.1, 0.5)
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _get_headers(self) -> Dict[str, str]:
        """Generate random headers for requests."""
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    def collect(self, **kwargs) -> List[ContentItem]:
        """Main collection method to be implemented by subclasses."""
        raise NotImplementedError
    
    def save_data(self, filename: str, format: str = 'json'):
        """Save collected data to file."""
        if not self.collected_items:
            logger.warning("No data to save")
            return
        
        filepath = utils.resolve_path(f"data/raw/{filename}")
        utils.ensure_dir(filepath.parent)
        
        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump([asdict(item) for item in self.collected_items], f, 
                         indent=2, default=str, ensure_ascii=False)
        
        elif format == 'csv':
            df = pd.DataFrame([asdict(item) for item in self.collected_items])
            df.to_csv(filepath, index=False)
        
        logger.info(f"Saved {len(self.collected_items)} items to {filepath}")

class PublicDatasetCollector(BaseCollector):
    """Collector for public fake review datasets."""
    
    def __init__(self):
        super().__init__(rate_limit=0.5)
        self.datasets = {
            'amazon_reviews': {
                'url': 'https://snap.stanford.edu/data/web-Amazon.html',
                'description': 'Amazon product reviews dataset'
            },
            'yelp_dataset': {
                'url': 'https://www.yelp.com/dataset',
                'description': 'Yelp Open Dataset'
            },
            'opspam': {
                'url': 'https://myleott.com/op_spam/',
                'description': 'Opinion spam dataset (hotel reviews)'
            },
            'fake_review_dataset': {
                'url': 'https://github.com/roneya/FakeReviewDataset',
                'description': 'Labeled fake review dataset'
            }
        }
    
    def download_opspam_dataset(self) -> List[ContentItem]:
        """Download and process the OpSpam dataset (gold standard for fake reviews)."""
        logger.info("Downloading OpSpam dataset...")
        
        # URLs for the OpSpam dataset
        urls = {
            'truthful_positive': 'https://myleott.com/op_spam/positive_polarity/truthful_from_TripAdvisor/fold1/truthful_positive.txt',
            'deceptive_positive': 'https://myleott.com/op_spam/positive_polarity/deceptive_from_MTurk/fold1/deceptive_positive.txt',
            'truthful_negative': 'https://myleott.com/op_spam/negative_polarity/truthful_from_Web/fold1/truthful_negative.txt',
            'deceptive_negative': 'https://myleott.com/op_spam/negative_polarity/deceptive_from_MTurk/fold1/deceptive_negative.txt'
        }
        
        items = []
        
        for label, url in urls.items():
            try:
                self._rate_limit_delay()
                response = self.session.get(url, headers=self._get_headers(), timeout=30)
                response.raise_for_status()
                
                # Parse reviews (each line is a review)
                reviews = response.text.strip().split('\n')
                
                is_fake = 'deceptive' in label
                sentiment = 'positive' if 'positive' in label else 'negative'
                
                for i, review_text in enumerate(reviews):
                    if review_text.strip():
                        item = ContentItem(
                            text=review_text.strip(),
                            platform='hotel_reviews',
                            content_type='review',
                            user_id=f"{label}_user_{i}",
                            rating=5.0 if sentiment == 'positive' else 1.0,
                            is_fake=is_fake,
                            confidence=1.0,  # This is manually labeled data
                            label_source='manual_expert',
                            metadata={
                                'dataset': 'opspam',
                                'sentiment': sentiment,
                                'fold': 1
                            }
                        )
                        items.append(item)
                
                logger.info(f"Downloaded {len(reviews)} {label} reviews")
                
            except Exception as e:
                logger.error(f"Failed to download {label}: {e}")
        
        self.collected_items.extend(items)
        return items
    
    def generate_synthetic_reviews(self, count: int = 1000) -> List[ContentItem]:
        """Generate synthetic reviews for testing."""
        logger.info(f"Generating {count} synthetic reviews...")
        
        # Templates for fake reviews (generic, repetitive patterns)
        fake_templates = [
            "This product is amazing! Best purchase ever!",
            "Great quality, fast shipping. Highly recommend!",
            "Perfect item, exactly as described. 5 stars!",
            "Excellent service and product. Will buy again!",
            "Outstanding quality and value. Love it!",
            "Fantastic product, exceeded my expectations!",
            "Top quality item, great seller. Recommended!",
            "Amazing deal, perfect condition. Very happy!"
        ]
        
        # Templates for authentic reviews (more specific, varied)
        authentic_templates = [
            "I bought this for my kitchen renovation and it fits perfectly with my decor. The installation was straightforward, though I needed to buy extra screws.",
            "Used this during my recent camping trip to Yellowstone. Held up well in the rain, but the zipper got stuck once. Overall satisfied for the price.",
            "My daughter loves this for her college dorm. The size is perfect for her small space, and the color matches her bedding nicely.",
            "Works as advertised, but took longer to arrive than expected. Customer service was helpful when I called about the delay.",
            "Good alternative to the more expensive brand. Not quite the same quality, but decent for occasional use.",
            "The instructions could be clearer, but once I figured it out, it works great. My wife is happy with the results.",
            "Bought this as a gift for my brother's birthday. He uses it every weekend now for his woodworking projects.",
            "Sturdy construction and good value. I've had it for six months now with no issues. Would consider buying another."
        ]
        
        items = []
        products = [f"product_{i}" for i in range(1, 101)]  # 100 different products
        
        for i in range(count):
            is_fake = random.choice([True, False])
            
            if is_fake:
                text = random.choice(fake_templates)
                # Add some variation
                text = text.replace("amazing", random.choice(["incredible", "fantastic", "outstanding"]))
                text = text.replace("great", random.choice(["excellent", "wonderful", "perfect"]))
                rating = random.choice([4.0, 5.0])  # Fake reviews tend to be very positive
            else:
                text = random.choice(authentic_templates)
                rating = random.uniform(2.0, 5.0)  # More varied ratings
            
            item = ContentItem(
                text=text,
                platform='ecommerce',
                content_type='review',
                user_id=f"synthetic_user_{i}",
                product_id=random.choice(products),
                rating=rating,
                timestamp=datetime.now() - timedelta(days=random.randint(1, 365)),
                is_fake=is_fake,
                confidence=0.8,  # Lower confidence for synthetic data
                label_source='synthetic',
                metadata={
                    'dataset': 'synthetic',
                    'template_based': True
                }
            )
            items.append(item)
        
        self.collected_items.extend(items)
        return items

class AmazonReviewCollector(BaseCollector):
    """Collector for Amazon product reviews (for research purposes only)."""
    
    def __init__(self):
        super().__init__(rate_limit=2.0)  # Be respectful with rate limiting
    
    def collect_product_reviews(self, product_ids: List[str], max_reviews_per_product: int = 50) -> List[ContentItem]:
        """Collect reviews for specific products."""
        logger.info(f"Collecting reviews for {len(product_ids)} products...")
        
        items = []
        
        for product_id in product_ids:
            try:
                product_reviews = self._get_product_reviews(product_id, max_reviews_per_product)
                items.extend(product_reviews)
                logger.info(f"Collected {len(product_reviews)} reviews for product {product_id}")
                
            except Exception as e:
                logger.error(f"Failed to collect reviews for product {product_id}: {e}")
        
        self.collected_items.extend(items)
        return items
    
    def _get_product_reviews(self, product_id: str, max_reviews: int) -> List[ContentItem]:
        """Get reviews for a specific product."""
        # Note: This would need to be implemented with proper Amazon API or 
        # ethical scraping practices. For now, return mock data.
        
        logger.warning("Amazon scraping not implemented - returning mock data")
        
        items = []
        for i in range(min(max_reviews, 10)):  # Mock 10 reviews
            text = f"Mock review {i+1} for product {product_id}. This is a placeholder review."
            
            item = ContentItem(
                text=text,
                platform='amazon',
                content_type='review',
                user_id=f"amazon_user_{i}",
                product_id=product_id,
                rating=random.uniform(1.0, 5.0),
                timestamp=datetime.now() - timedelta(days=random.randint(1, 100)),
                metadata={
                    'source': 'mock_amazon_collector',
                    'verified_purchase': random.choice([True, False])
                }
            )
            items.append(item)
        
        return items

class SocialMediaCollector(BaseCollector):
    """Collector for social media content (Twitter, Reddit, etc.)."""
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        super().__init__(rate_limit=1.0)
        self.api_keys = api_keys or {}
    
    def collect_reddit_comments(self, subreddits: List[str], max_comments: int = 100) -> List[ContentItem]:
        """Collect comments from Reddit subreddits."""
        logger.info(f"Collecting comments from {len(subreddits)} subreddits...")
        
        items = []
        
        for subreddit in subreddits:
            try:
                # Use Reddit API or PRAW library in real implementation
                # For now, generate mock data
                subreddit_comments = self._get_reddit_comments_mock(subreddit, max_comments // len(subreddits))
                items.extend(subreddit_comments)
                
            except Exception as e:
                logger.error(f"Failed to collect from subreddit {subreddit}: {e}")
        
        self.collected_items.extend(items)
        return items
    
    def _get_reddit_comments_mock(self, subreddit: str, max_comments: int) -> List[ContentItem]:
        """Mock Reddit comment collection."""
        items = []
        
        comment_templates = [
            "I agree with this completely!",
            "This is exactly what I was thinking.",
            "Great point, thanks for sharing.",
            "I had a similar experience last week.",
            "Can you provide more details about this?",
            "This doesn't seem right to me.",
            "Has anyone else tried this approach?",
            "I disagree, here's why..."
        ]
        
        for i in range(max_comments):
            text = random.choice(comment_templates)
            
            # Simple heuristic: very short, generic comments might be fake
            is_potentially_fake = len(text.split()) < 5 and "this" in text.lower()
            
            item = ContentItem(
                text=text,
                platform='reddit',
                content_type='comment',
                user_id=f"reddit_user_{i}",
                timestamp=datetime.now() - timedelta(hours=random.randint(1, 168)),
                is_fake=is_potentially_fake,
                confidence=0.6,  # Low confidence for heuristic labeling
                label_source='heuristic',
                metadata={
                    'subreddit': subreddit,
                    'source': 'mock_reddit_collector'
                }
            )
            items.append(item)
        
        return items

class AppStoreCollector(BaseCollector):
    """Collector for app store reviews."""
    
    def __init__(self):
        super().__init__(rate_limit=1.5)
    
    def collect_app_reviews(self, app_ids: List[str], stores: List[str] = ['google', 'apple']) -> List[ContentItem]:
        """Collect reviews for mobile apps."""
        logger.info(f"Collecting reviews for {len(app_ids)} apps from {stores}...")
        
        items = []
        
        for app_id in app_ids:
            for store in stores:
                try:
                    app_reviews = self._get_app_reviews_mock(app_id, store)
                    items.extend(app_reviews)
                    
                except Exception as e:
                    logger.error(f"Failed to collect {store} reviews for app {app_id}: {e}")
        
        self.collected_items.extend(items)
        return items
    
    def _get_app_reviews_mock(self, app_id: str, store: str, max_reviews: int = 20) -> List[ContentItem]:
        """Mock app review collection."""
        items = []
        
        review_templates = {
            'positive': [
                "Love this app! Use it every day.",
                "Great features and easy to use.",
                "Perfect for what I needed.",
                "Highly recommend this app!"
            ],
            'negative': [
                "App crashes constantly.",
                "Too many ads, unusable.",
                "Doesn't work as advertised.",
                "Waste of time and money."
            ],
            'fake_positive': [
                "Amazing app! 5 stars!",
                "Best app ever! Perfect!",
                "Incredible! Love it!",
                "Outstanding! Recommend!"
            ]
        }
        
        for i in range(max_reviews):
            # Simulate different types of reviews
            review_type = random.choices(
                ['positive', 'negative', 'fake_positive'],
                weights=[0.5, 0.3, 0.2]
            )[0]
            
            text = random.choice(review_templates[review_type])
            rating = {
                'positive': random.uniform(3.5, 5.0),
                'negative': random.uniform(1.0, 2.5),
                'fake_positive': 5.0
            }[review_type]
            
            is_fake = review_type == 'fake_positive'
            
            item = ContentItem(
                text=text,
                platform=f'{store}_play_store' if store == 'google' else 'ios_app_store',
                content_type='review',
                user_id=f"{store}_user_{i}",
                rating=rating,
                timestamp=datetime.now() - timedelta(days=random.randint(1, 90)),
                is_fake=is_fake,
                confidence=0.7,
                label_source='heuristic',
                metadata={
                    'app_id': app_id,
                    'store': store,
                    'source': 'mock_app_store_collector'
                }
            )
            items.append(item)
        
        return items

class DataCollectionManager:
    """Manager class to coordinate data collection from multiple sources."""
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.collectors = {
            'public_datasets': PublicDatasetCollector(),
            'amazon': AmazonReviewCollector(),
            'social_media': SocialMediaCollector(),
            'app_stores': AppStoreCollector()
        }
        self.all_collected_items = []
    
    def collect_all(self, config: Dict[str, Any]) -> List[ContentItem]:
        """Collect data from all configured sources."""
        logger.info("Starting comprehensive data collection...")
        
        # Collect from public datasets
        if config.get('public_datasets', {}).get('enabled', True):
            logger.info("Collecting from public datasets...")
            
            # OpSpam dataset (gold standard)
            if config.get('public_datasets', {}).get('opspam', True):
                self.collectors['public_datasets'].download_opspam_dataset()
            
            # Synthetic data for testing
            synthetic_count = config.get('public_datasets', {}).get('synthetic_count', 1000)
            if synthetic_count > 0:
                self.collectors['public_datasets'].generate_synthetic_reviews(synthetic_count)
        
        # Collect from social media
        if config.get('social_media', {}).get('enabled', False):
            subreddits = config.get('social_media', {}).get('subreddits', ['reviews', 'products'])
            max_comments = config.get('social_media', {}).get('max_comments', 100)
            self.collectors['social_media'].collect_reddit_comments(subreddits, max_comments)
        
        # Collect from app stores
        if config.get('app_stores', {}).get('enabled', False):
            app_ids = config.get('app_stores', {}).get('app_ids', ['com.example.app'])
            self.collectors['app_stores'].collect_app_reviews(app_ids)
        
        # Collect from e-commerce (mock)
        if config.get('ecommerce', {}).get('enabled', False):
            product_ids = config.get('ecommerce', {}).get('product_ids', ['B001', 'B002'])
            self.collectors['amazon'].collect_product_reviews(product_ids)
        
        # Combine all collected items
        for collector in self.collectors.values():
            self.all_collected_items.extend(collector.collected_items)
        
        logger.info(f"Total collected items: {len(self.all_collected_items)}")
        return self.all_collected_items
    
    def save_combined_dataset(self, filename: str = "combined_dataset.json"):
        """Save all collected data to a single file."""
        if not self.all_collected_items:
            logger.warning("No data to save")
            return
        
        filepath = self.output_dir / filename
        utils.ensure_dir(filepath.parent)
        
        # Convert to dictionaries and save
        data = [asdict(item) for item in self.all_collected_items]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"Saved combined dataset with {len(data)} items to {filepath}")
        
        # Also save statistics
        self._save_collection_stats(filepath.with_suffix('.stats.json'))
    
    def _save_collection_stats(self, filepath: Path):
        """Save collection statistics."""
        stats = {
            'total_items': len(self.all_collected_items),
            'collection_date': datetime.now().isoformat(),
            'platform_breakdown': {},
            'content_type_breakdown': {},
            'label_distribution': {},
            'data_quality': {}
        }
        
        # Platform breakdown
        for item in self.all_collected_items:
            platform = item.platform
            stats['platform_breakdown'][platform] = stats['platform_breakdown'].get(platform, 0) + 1
            
            content_type = item.content_type
            stats['content_type_breakdown'][content_type] = stats['content_type_breakdown'].get(content_type, 0) + 1
            
            if item.is_fake is not None:
                label = 'fake' if item.is_fake else 'authentic'
                stats['label_distribution'][label] = stats['label_distribution'].get(label, 0) + 1
        
        # Data quality metrics
        total_with_labels = len([item for item in self.all_collected_items if item.is_fake is not None])
        stats['data_quality']['labeled_percentage'] = (total_with_labels / len(self.all_collected_items)) * 100
        
        avg_text_length = sum(len(item.text) for item in self.all_collected_items) / len(self.all_collected_items)
        stats['data_quality']['avg_text_length'] = avg_text_length
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"Saved collection statistics to {filepath}")

def main():
    """Main function to run data collection."""
    # Configuration for data collection
    config = {
        'public_datasets': {
            'enabled': True,
            'opspam': True,
            'synthetic_count': 2000
        },
        'social_media': {
            'enabled': False,  # Disable for now
            'subreddits': ['reviews', 'BuyItForLife', 'amazon'],
            'max_comments': 200
        },
        'app_stores': {
            'enabled': False,  # Disable for now
            'app_ids': ['com.example.app1', 'com.example.app2']
        },
        'ecommerce': {
            'enabled': False,  # Disable for now
            'product_ids': ['B001', 'B002', 'B003']
        }
    }
    
    # Initialize manager and collect data
    manager = DataCollectionManager()
    collected_items = manager.collect_all(config)
    
    # Save the dataset
    manager.save_combined_dataset(f"training_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    print(f"\n✅ Data collection completed!")
    print(f"📊 Total items collected: {len(collected_items)}")
    
    # Print summary statistics
    platforms = {}
    labels = {'fake': 0, 'authentic': 0, 'unlabeled': 0}
    
    for item in collected_items:
        platforms[item.platform] = platforms.get(item.platform, 0) + 1
        
        if item.is_fake is None:
            labels['unlabeled'] += 1
        elif item.is_fake:
            labels['fake'] += 1
        else:
            labels['authentic'] += 1
    
    print(f"\n📱 Platform breakdown:")
    for platform, count in platforms.items():
        print(f"  - {platform}: {count} items")
    
    print(f"\n🏷️ Label distribution:")
    for label, count in labels.items():
        percentage = (count / len(collected_items)) * 100
        print(f"  - {label}: {count} items ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
