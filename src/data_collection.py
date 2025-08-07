"""
Data Collection Module for Fake Review Detection System

This module implements the DataCollector class that handles data collection from multiple sources:
- Yelp Open Dataset (chunked read)
- Amazon Kaggle dataset download
- Yelp Fusion API with rate limiting

The collected data is stored as parquet files in organized directory structure.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Generator
import gzip
import shutil
from datetime import datetime, timedelta

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
from tqdm import tqdm
import yaml


class DataCollectionError(Exception):
    """Custom exception for data collection errors"""
    pass


class DataCollector:
    """
    Comprehensive data collector for fake review detection system.
    
    Supports multiple data sources:
    - Yelp Open Dataset (JSON files)
    - Amazon product reviews (Kaggle datasets)
    - Yelp Fusion API (real-time data)
    
    Features:
    - Chunked processing for large files
    - Rate limiting for API calls
    - Data validation and cleaning
    - Parquet storage with compression
    - Comprehensive error handling and logging
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize DataCollector with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Setup paths
        self.raw_data_path = Path(self.config['paths']['raw_data'])
        self.yelp_raw_path = self.raw_data_path / 'yelp'
        self.amazon_raw_path = self.raw_data_path / 'amazon'
        
        # Create directories
        self._create_directories()
        
        # Setup HTTP session with retry strategy
        self.session = self._setup_http_session()
        
        # API configurations
        self.yelp_api_key = os.getenv('YELP_API_KEY')
        self.kaggle_username = os.getenv('KAGGLE_USERNAME')
        self.kaggle_key = os.getenv('KAGGLE_KEY')
        
        # Rate limiting settings
        self.api_rate_limit = 5000  # requests per day for Yelp Fusion API
        self.api_requests_made = 0
        self.api_last_reset = datetime.now()
        
        self.logger.info("DataCollector initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            # Fallback configuration if file not found
            return {
                'paths': {
                    'raw_data': 'data/raw',
                    'logs': 'logs'
                },
                'logging': {
                    'level': 'INFO'
                },
                'data': {
                    'quality': {
                        'min_review_length': 10,
                        'max_review_length': 5000,
                        'min_rating': 1,
                        'max_rating': 5,
                        'required_fields': ['review_text', 'rating', 'user_id', 'product_id']
                    }
                }
            }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Create logs directory
            logs_dir = Path(self.config['paths'].get('logs', 'logs'))
            logs_dir.mkdir(exist_ok=True)
            
            # File handler
            file_handler = logging.FileHandler(logs_dir / 'data_collection.log')
            file_handler.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def _create_directories(self):
        """Create necessary directories for data storage"""
        directories = [
            self.raw_data_path,
            self.yelp_raw_path,
            self.amazon_raw_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")
    
    def _setup_http_session(self) -> requests.Session:
        """Setup HTTP session with retry strategy"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _check_api_rate_limit(self):
        """Check and enforce API rate limiting"""
        now = datetime.now()
        
        # Reset counter if a day has passed
        if now - self.api_last_reset > timedelta(days=1):
            self.api_requests_made = 0
            self.api_last_reset = now
        
        # Check if we've exceeded the daily limit
        if self.api_requests_made >= self.api_rate_limit:
            time_to_wait = (self.api_last_reset + timedelta(days=1) - now).total_seconds()
            if time_to_wait > 0:
                self.logger.warning(f"API rate limit reached. Waiting {time_to_wait:.0f} seconds")
                time.sleep(time_to_wait)
                self.api_requests_made = 0
                self.api_last_reset = datetime.now()
    
    def collect_yelp_open_dataset(
        self,
        dataset_path: str,
        chunk_size: int = 10000,
        file_types: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Collect and process Yelp Open Dataset with chunked reading.
        
        Args:
            dataset_path: Path to Yelp dataset files
            chunk_size: Number of records to process at once
            file_types: List of file types to process (business, review, user, checkin, tip)
        
        Returns:
            Dictionary with file paths of processed data
        """
        if file_types is None:
            file_types = ['review', 'business', 'user']
        
        self.logger.info(f"Starting Yelp Open Dataset collection from {dataset_path}")
        
        processed_files = {}
        
        for file_type in file_types:
            try:
                json_file = Path(dataset_path) / f"yelp_academic_dataset_{file_type}.json"
                
                if not json_file.exists():
                    self.logger.warning(f"File not found: {json_file}")
                    continue
                
                self.logger.info(f"Processing {file_type} data")
                
                # Process file in chunks
                processed_data = self._process_yelp_file_chunked(
                    json_file, file_type, chunk_size
                )
                
                # Save as parquet
                output_file = self.yelp_raw_path / f"yelp_{file_type}.parquet"
                processed_data.to_parquet(
                    output_file,
                    compression='snappy',
                    index=False
                )
                
                processed_files[file_type] = str(output_file)
                self.logger.info(f"Saved {len(processed_data)} {file_type} records to {output_file}")
                
            except Exception as e:
                self.logger.error(f"Error processing {file_type} data: {e}")
                raise DataCollectionError(f"Failed to process {file_type} data: {e}")
        
        return processed_files
    
    def _process_yelp_file_chunked(
        self,
        file_path: Path,
        file_type: str,
        chunk_size: int
    ) -> pd.DataFrame:
        """Process Yelp JSON file in chunks"""
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                chunk = []
                
                for line_num, line in enumerate(tqdm(file, desc=f"Reading {file_type}")):
                    try:
                        data = json.loads(line.strip())
                        chunk.append(data)
                        
                        if len(chunk) >= chunk_size:
                            df_chunk = pd.DataFrame(chunk)
                            chunks.append(df_chunk)
                            chunk = []
                            
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON at line {line_num}: {e}")
                        continue
                
                # Process remaining data
                if chunk:
                    df_chunk = pd.DataFrame(chunk)
                    chunks.append(df_chunk)
        
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            raise
        
        if not chunks:
            raise DataCollectionError(f"No valid data found in {file_path}")
        
        # Combine all chunks
        combined_df = pd.concat(chunks, ignore_index=True)
        
        # Apply specific processing based on file type
        if file_type == 'review':
            combined_df = self._process_yelp_reviews(combined_df)
        elif file_type == 'business':
            combined_df = self._process_yelp_businesses(combined_df)
        elif file_type == 'user':
            combined_df = self._process_yelp_users(combined_df)
        
        return combined_df
    
    def _process_yelp_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Yelp review data"""
        # Rename columns to match our schema
        column_mapping = {
            'review_id': 'review_id',
            'user_id': 'user_id',
            'business_id': 'product_id',
            'stars': 'rating',
            'text': 'review_text',
            'date': 'date',
            'useful': 'useful_votes',
            'funny': 'funny_votes',
            'cool': 'cool_votes'
        }
        
        # Select and rename columns
        available_cols = [col for col in column_mapping.keys() if col in df.columns]
        df_processed = df[available_cols].copy()
        df_processed = df_processed.rename(columns=column_mapping)
        
        # Add metadata
        df_processed['source'] = 'yelp_open_dataset'
        df_processed['collected_at'] = datetime.now()
        
        return df_processed
    
    def _process_yelp_businesses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Yelp business data"""
        # Select relevant columns
        relevant_columns = [
            'business_id', 'name', 'city', 'state', 'stars',
            'review_count', 'categories', 'attributes'
        ]
        
        available_cols = [col for col in relevant_columns if col in df.columns]
        df_processed = df[available_cols].copy()
        
        # Add metadata
        df_processed['source'] = 'yelp_open_dataset'
        df_processed['collected_at'] = datetime.now()
        
        return df_processed
    
    def _process_yelp_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Yelp user data"""
        # Select relevant columns
        relevant_columns = [
            'user_id', 'name', 'review_count', 'yelping_since',
            'friends', 'useful', 'funny', 'cool', 'fans',
            'average_stars', 'compliment_hot', 'compliment_more'
        ]
        
        available_cols = [col for col in relevant_columns if col in df.columns]
        df_processed = df[available_cols].copy()
        
        # Add metadata
        df_processed['source'] = 'yelp_open_dataset'
        df_processed['collected_at'] = datetime.now()
        
        return df_processed
    
    def collect_amazon_kaggle_dataset(
        self,
        dataset_name: str,
        files: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Download and process Amazon dataset from Kaggle.
        
        Args:
            dataset_name: Kaggle dataset name (e.g., 'snap-stanford/amazon-fine-food-reviews')
            files: Specific files to download (optional)
        
        Returns:
            Dictionary with paths to processed files
        """
        if not self.kaggle_username or not self.kaggle_key:
            raise DataCollectionError(
                "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables"
            )
        
        self.logger.info(f"Starting Amazon Kaggle dataset collection: {dataset_name}")
        
        try:
            # Import kaggle API
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
            except ImportError:
                raise DataCollectionError(
                    "Kaggle API not installed. Run: pip install kaggle"
                )
            
            # Initialize Kaggle API
            api = KaggleApi()
            api.authenticate()
            
            # Create temporary download directory
            temp_dir = Path("temp_kaggle_download")
            temp_dir.mkdir(exist_ok=True)
            
            try:
                # Download dataset
                self.logger.info(f"Downloading dataset: {dataset_name}")
                api.dataset_download_files(
                    dataset_name,
                    path=str(temp_dir),
                    unzip=True
                )
                
                # Process downloaded files
                processed_files = {}
                download_files = list(temp_dir.glob("*.csv")) + list(temp_dir.glob("*.json"))
                
                for file_path in download_files:
                    if files and file_path.name not in files:
                        continue
                    
                    self.logger.info(f"Processing file: {file_path.name}")
                    
                    # Read file based on extension
                    if file_path.suffix == '.csv':
                        df = pd.read_csv(file_path)
                    elif file_path.suffix == '.json':
                        df = pd.read_json(file_path, lines=True)
                    else:
                        continue
                    
                    # Process Amazon review data
                    df_processed = self._process_amazon_reviews(df)
                    
                    # Save as parquet
                    output_file = self.amazon_raw_path / f"amazon_{file_path.stem}.parquet"
                    df_processed.to_parquet(
                        output_file,
                        compression='snappy',
                        index=False
                    )
                    
                    processed_files[file_path.stem] = str(output_file)
                    self.logger.info(f"Saved {len(df_processed)} records to {output_file}")
                
            finally:
                # Cleanup temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            return processed_files
            
        except Exception as e:
            self.logger.error(f"Error collecting Amazon Kaggle dataset: {e}")
            raise DataCollectionError(f"Failed to collect Amazon dataset: {e}")
    
    def _process_amazon_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Amazon review data"""
        # Common column mappings for Amazon datasets
        column_mappings = {
            # Standard Amazon review columns
            'Id': 'review_id',
            'ProductId': 'product_id',
            'UserId': 'user_id',
            'ProfileName': 'user_name',
            'HelpfulnessNumerator': 'helpful_numerator',
            'HelpfulnessDenominator': 'helpful_denominator',
            'Score': 'rating',
            'Time': 'timestamp',
            'Summary': 'summary',
            'Text': 'review_text',
            # Alternative column names
            'reviewerID': 'user_id',
            'asin': 'product_id',
            'reviewText': 'review_text',
            'overall': 'rating',
            'unixReviewTime': 'timestamp',
            'reviewTime': 'review_date',
            'summary': 'summary'
        }
        
        # Apply column mapping
        df_processed = df.copy()
        for old_col, new_col in column_mappings.items():
            if old_col in df_processed.columns:
                df_processed = df_processed.rename(columns={old_col: new_col})
        
        # Ensure required columns exist
        required_columns = ['review_text', 'rating']
        for col in required_columns:
            if col not in df_processed.columns:
                self.logger.warning(f"Required column '{col}' not found in Amazon dataset")
        
        # Add metadata
        df_processed['source'] = 'amazon_kaggle'
        df_processed['collected_at'] = datetime.now()
        
        # Convert timestamp if needed
        if 'timestamp' in df_processed.columns:
            try:
                df_processed['date'] = pd.to_datetime(df_processed['timestamp'], unit='s')
            except:
                pass
        
        return df_processed
    
    def collect_yelp_fusion_api(
        self,
        locations: List[str],
        categories: Optional[List[str]] = None,
        max_businesses: int = 1000,
        reviews_per_business: int = 3
    ) -> Dict[str, str]:
        """
        Collect data from Yelp Fusion API with rate limiting.
        
        Args:
            locations: List of locations to search (e.g., ['New York, NY', 'Los Angeles, CA'])
            categories: List of business categories (optional)
            max_businesses: Maximum number of businesses to collect per location
            reviews_per_business: Number of reviews to collect per business
        
        Returns:
            Dictionary with paths to collected data files
        """
        if not self.yelp_api_key:
            raise DataCollectionError(
                "Yelp API key not found. Set YELP_API_KEY environment variable"
            )
        
        self.logger.info("Starting Yelp Fusion API data collection")
        
        headers = {
            'Authorization': f'Bearer {self.yelp_api_key}'
        }
        
        businesses_data = []
        reviews_data = []
        
        try:
            for location in locations:
                self.logger.info(f"Collecting businesses in {location}")
                
                # Get businesses
                businesses = self._get_yelp_businesses(
                    location, categories, max_businesses, headers
                )
                businesses_data.extend(businesses)
                
                # Get reviews for each business
                for business in tqdm(businesses, desc=f"Getting reviews for {location}"):
                    try:
                        reviews = self._get_yelp_business_reviews(
                            business['id'], reviews_per_business, headers
                        )
                        
                        # Add business context to reviews
                        for review in reviews:
                            review['business_id'] = business['id']
                            review['business_name'] = business['name']
                            review['business_location'] = location
                        
                        reviews_data.extend(reviews)
                        
                        # Rate limiting
                        time.sleep(0.1)  # 100ms delay between requests
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to get reviews for business {business['id']}: {e}")
                        continue
            
            # Convert to DataFrames and save
            processed_files = {}
            
            if businesses_data:
                businesses_df = pd.DataFrame(businesses_data)
                businesses_df['source'] = 'yelp_fusion_api'
                businesses_df['collected_at'] = datetime.now()
                
                businesses_file = self.yelp_raw_path / "yelp_fusion_businesses.parquet"
                businesses_df.to_parquet(businesses_file, compression='snappy', index=False)
                processed_files['businesses'] = str(businesses_file)
                
                self.logger.info(f"Saved {len(businesses_df)} businesses to {businesses_file}")
            
            if reviews_data:
                reviews_df = pd.DataFrame(reviews_data)
                reviews_df['source'] = 'yelp_fusion_api'
                reviews_df['collected_at'] = datetime.now()
                
                reviews_file = self.yelp_raw_path / "yelp_fusion_reviews.parquet"
                reviews_df.to_parquet(reviews_file, compression='snappy', index=False)
                processed_files['reviews'] = str(reviews_file)
                
                self.logger.info(f"Saved {len(reviews_df)} reviews to {reviews_file}")
            
            return processed_files
            
        except Exception as e:
            self.logger.error(f"Error collecting Yelp Fusion API data: {e}")
            raise DataCollectionError(f"Failed to collect Yelp API data: {e}")
    
    def _get_yelp_businesses(
        self,
        location: str,
        categories: Optional[List[str]],
        max_businesses: int,
        headers: Dict[str, str]
    ) -> List[Dict]:
        """Get businesses from Yelp Fusion API"""
        businesses = []
        offset = 0
        limit = 50  # Yelp API limit per request
        
        while len(businesses) < max_businesses:
            self._check_api_rate_limit()
            
            params = {
                'location': location,
                'limit': min(limit, max_businesses - len(businesses)),
                'offset': offset
            }
            
            if categories:
                params['categories'] = ','.join(categories)
            
            try:
                response = self.session.get(
                    'https://api.yelp.com/v3/businesses/search',
                    headers=headers,
                    params=params,
                    timeout=30
                )
                
                response.raise_for_status()
                self.api_requests_made += 1
                
                data = response.json()
                batch_businesses = data.get('businesses', [])
                
                if not batch_businesses:
                    break
                
                businesses.extend(batch_businesses)
                offset += limit
                
                # Check if we've reached the end
                if len(batch_businesses) < limit:
                    break
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error getting businesses: {e}")
                break
        
        return businesses[:max_businesses]
    
    def _get_yelp_business_reviews(
        self,
        business_id: str,
        max_reviews: int,
        headers: Dict[str, str]
    ) -> List[Dict]:
        """Get reviews for a specific business"""
        self._check_api_rate_limit()
        
        try:
            response = self.session.get(
                f'https://api.yelp.com/v3/businesses/{business_id}/reviews',
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            self.api_requests_made += 1
            
            data = response.json()
            reviews = data.get('reviews', [])
            
            # Process reviews to match our schema
            processed_reviews = []
            for review in reviews[:max_reviews]:
                processed_review = {
                    'review_id': review.get('id'),
                    'user_id': review.get('user', {}).get('id'),
                    'user_name': review.get('user', {}).get('name'),
                    'rating': review.get('rating'),
                    'review_text': review.get('text'),
                    'date': review.get('time_created'),
                    'url': review.get('url')
                }
                processed_reviews.append(processed_review)
            
            return processed_reviews
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error getting reviews for business {business_id}: {e}")
            return []
    
    def validate_data(self, df: pd.DataFrame, data_type: str = "review") -> pd.DataFrame:
        """
        Validate collected data according to quality criteria.
        
        Args:
            df: DataFrame to validate
            data_type: Type of data ('review', 'business', 'user')
        
        Returns:
            Validated and cleaned DataFrame
        """
        self.logger.info(f"Validating {data_type} data: {len(df)} records")
        
        initial_count = len(df)
        
        if data_type == "review":
            df = self._validate_review_data(df)
        elif data_type == "business":
            df = self._validate_business_data(df)
        elif data_type == "user":
            df = self._validate_user_data(df)
        
        final_count = len(df)
        removed_count = initial_count - final_count
        
        if removed_count > 0:
            self.logger.info(f"Validation removed {removed_count} invalid records")
        
        return df
    
    def _validate_review_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate review data"""
        quality_config = self.config.get('data', {}).get('quality', {})
        
        # Remove rows with missing required fields
        required_fields = ['review_text', 'rating']
        df = df.dropna(subset=[col for col in required_fields if col in df.columns])
        
        # Text length validation
        if 'review_text' in df.columns:
            min_length = quality_config.get('min_review_length', 10)
            max_length = quality_config.get('max_review_length', 5000)
            
            df = df[
                (df['review_text'].str.len() >= min_length) &
                (df['review_text'].str.len() <= max_length)
            ]
        
        # Rating validation
        if 'rating' in df.columns:
            min_rating = quality_config.get('min_rating', 1)
            max_rating = quality_config.get('max_rating', 5)
            
            df = df[
                (df['rating'] >= min_rating) &
                (df['rating'] <= max_rating)
            ]
        
        # Remove duplicates
        if 'review_id' in df.columns:
            df = df.drop_duplicates(subset=['review_id'])
        elif 'review_text' in df.columns:
            df = df.drop_duplicates(subset=['review_text'])
        
        return df.reset_index(drop=True)
    
    def _validate_business_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate business data"""
        # Remove duplicates
        if 'business_id' in df.columns:
            df = df.drop_duplicates(subset=['business_id'])
        
        # Remove rows with missing essential fields
        essential_fields = ['business_id', 'name']
        df = df.dropna(subset=[col for col in essential_fields if col in df.columns])
        
        return df.reset_index(drop=True)
    
    def _validate_user_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate user data"""
        # Remove duplicates
        if 'user_id' in df.columns:
            df = df.drop_duplicates(subset=['user_id'])
        
        # Remove rows with missing essential fields
        essential_fields = ['user_id']
        df = df.dropna(subset=[col for col in essential_fields if col in df.columns])
        
        return df.reset_index(drop=True)
    
    def collect_all_data(
        self,
        yelp_open_path: Optional[str] = None,
        amazon_kaggle_dataset: Optional[str] = None,
        yelp_api_locations: Optional[List[str]] = None,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Orchestrator method to collect data from all sources.
        
        Args:
            yelp_open_path: Path to Yelp Open Dataset files
            amazon_kaggle_dataset: Kaggle dataset name for Amazon data
            yelp_api_locations: Locations for Yelp API data collection
            validate: Whether to validate collected data
        
        Returns:
            Summary of collected data
        """
        self.logger.info("Starting comprehensive data collection")
        
        collection_summary = {
            'start_time': datetime.now(),
            'sources': {},
            'total_records': 0,
            'errors': []
        }
        
        # Collect Yelp Open Dataset
        if yelp_open_path:
            try:
                self.logger.info("Collecting Yelp Open Dataset")
                yelp_files = self.collect_yelp_open_dataset(yelp_open_path)
                
                # Validate if requested
                if validate:
                    for file_type, file_path in yelp_files.items():
                        df = pd.read_parquet(file_path)
                        df_validated = self.validate_data(df, file_type)
                        df_validated.to_parquet(file_path, compression='snappy', index=False)
                
                collection_summary['sources']['yelp_open'] = {
                    'files': yelp_files,
                    'status': 'success'
                }
                
            except Exception as e:
                error_msg = f"Yelp Open Dataset collection failed: {e}"
                self.logger.error(error_msg)
                collection_summary['errors'].append(error_msg)
                collection_summary['sources']['yelp_open'] = {'status': 'failed', 'error': str(e)}
        
        # Collect Amazon Kaggle Dataset
        if amazon_kaggle_dataset:
            try:
                self.logger.info("Collecting Amazon Kaggle Dataset")
                amazon_files = self.collect_amazon_kaggle_dataset(amazon_kaggle_dataset)
                
                # Validate if requested
                if validate:
                    for file_name, file_path in amazon_files.items():
                        df = pd.read_parquet(file_path)
                        df_validated = self.validate_data(df, 'review')
                        df_validated.to_parquet(file_path, compression='snappy', index=False)
                
                collection_summary['sources']['amazon_kaggle'] = {
                    'files': amazon_files,
                    'status': 'success'
                }
                
            except Exception as e:
                error_msg = f"Amazon Kaggle dataset collection failed: {e}"
                self.logger.error(error_msg)
                collection_summary['errors'].append(error_msg)
                collection_summary['sources']['amazon_kaggle'] = {'status': 'failed', 'error': str(e)}
        
        # Collect Yelp Fusion API data
        if yelp_api_locations:
            try:
                self.logger.info("Collecting Yelp Fusion API data")
                yelp_api_files = self.collect_yelp_fusion_api(yelp_api_locations)
                
                # Validate if requested
                if validate:
                    for data_type, file_path in yelp_api_files.items():
                        df = pd.read_parquet(file_path)
                        df_validated = self.validate_data(df, 'review' if 'review' in data_type else 'business')
                        df_validated.to_parquet(file_path, compression='snappy', index=False)
                
                collection_summary['sources']['yelp_fusion_api'] = {
                    'files': yelp_api_files,
                    'status': 'success'
                }
                
            except Exception as e:
                error_msg = f"Yelp Fusion API collection failed: {e}"
                self.logger.error(error_msg)
                collection_summary['errors'].append(error_msg)
                collection_summary['sources']['yelp_fusion_api'] = {'status': 'failed', 'error': str(e)}
        
        # Calculate total records
        try:
            total_records = 0
            for source_data in collection_summary['sources'].values():
                if source_data['status'] == 'success':
                    for file_path in source_data['files'].values():
                        df = pd.read_parquet(file_path)
                        total_records += len(df)
            
            collection_summary['total_records'] = total_records
        except Exception as e:
            self.logger.warning(f"Could not calculate total records: {e}")
        
        collection_summary['end_time'] = datetime.now()
        collection_summary['duration'] = collection_summary['end_time'] - collection_summary['start_time']
        
        # Save collection summary
        summary_file = self.raw_data_path / 'collection_summary.json'
        with open(summary_file, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            summary_copy = collection_summary.copy()
            summary_copy['start_time'] = collection_summary['start_time'].isoformat()
            summary_copy['end_time'] = collection_summary['end_time'].isoformat()
            summary_copy['duration'] = str(collection_summary['duration'])
            
            json.dump(summary_copy, f, indent=2)
        
        self.logger.info(f"Data collection completed. Total records: {collection_summary['total_records']}")
        self.logger.info(f"Collection summary saved to: {summary_file}")
        
        return collection_summary


if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    
    # Example comprehensive data collection
    summary = collector.collect_all_data(
        # yelp_open_path="/path/to/yelp_dataset",
        # amazon_kaggle_dataset="snap-stanford/amazon-fine-food-reviews",
        # yelp_api_locations=["New York, NY", "Los Angeles, CA"],
        validate=True
    )
    
    print("Collection Summary:")
    print(f"Total Records: {summary['total_records']}")
    print(f"Duration: {summary['duration']}")
    print(f"Sources: {list(summary['sources'].keys())}")
    
    if summary['errors']:
        print(f"Errors: {summary['errors']}")
