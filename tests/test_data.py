"""
Unit tests for the data collection module.

This module tests data collection functionality including Yelp Open Dataset processing,
Amazon Kaggle dataset handling, Yelp Fusion API integration, data validation,
and comprehensive data collection orchestration.
Uses small synthetic fixtures and validates data shapes and quality.
"""

import unittest
import tempfile
import json
import sys
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection import DataCollector, DataCollectionError


class TestDataCollectorInitialization(unittest.TestCase):
    """Test DataCollector initialization and configuration."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('src.data_collection.yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    def test_initialization_with_config(self, mock_file, mock_yaml):
        """Test DataCollector initialization with configuration file."""
        # Mock configuration
        test_config = {
            'paths': {
                'raw_data': self.test_dir,
                'logs': 'logs'
            },
            'logging': {'level': 'INFO'},
            'data': {
                'quality': {
                    'min_review_length': 10,
                    'max_review_length': 5000,
                    'min_rating': 1,
                    'max_rating': 5
                }
            }
        }
        mock_yaml.return_value = test_config
        
        # Initialize collector
        collector = DataCollector(config_path="test_config.yaml")
        
        # Verify configuration
        self.assertEqual(collector.config['paths']['raw_data'], self.test_dir)
        self.assertEqual(collector.api_rate_limit, 5000)
        self.assertTrue(collector.raw_data_path.exists())
    
    def test_initialization_without_config(self):
        """Test DataCollector initialization with fallback configuration."""
        with patch('builtins.open', side_effect=FileNotFoundError):
            collector = DataCollector(config_path="nonexistent.yaml")
            
            # Should use fallback configuration
            self.assertIn('paths', collector.config)
            self.assertIn('raw_data', collector.config['paths'])


class TestYelpOpenDataset(unittest.TestCase):
    """Test Yelp Open Dataset processing functionality."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.collector = DataCollector()
        self.collector.raw_data_path = Path(self.test_dir)
        self.collector.yelp_raw_path = Path(self.test_dir) / 'yelp'
        self.collector.yelp_raw_path.mkdir(exist_ok=True)
        
        # Create synthetic Yelp data
        self.sample_reviews = [
            {
                "review_id": "rev1",
                "user_id": "user1", 
                "business_id": "biz1",
                "stars": 5,
                "text": "Great restaurant with amazing food!",
                "date": "2023-01-01",
                "useful": 1,
                "funny": 0,
                "cool": 2
            },
            {
                "review_id": "rev2",
                "user_id": "user2",
                "business_id": "biz1", 
                "stars": 2,
                "text": "Service was slow and food was cold.",
                "date": "2023-01-02",
                "useful": 0,
                "funny": 0,
                "cool": 0
            }
        ]
        
        self.sample_businesses = [
            {
                "business_id": "biz1",
                "name": "Test Restaurant",
                "city": "Test City",
                "state": "TC",
                "stars": 4.0,
                "review_count": 100,
                "categories": "Restaurants, American",
                "attributes": {"WiFi": "free"}
            }
        ]
        
        self.sample_users = [
            {
                "user_id": "user1",
                "name": "Test User",
                "review_count": 50,
                "yelping_since": "2020-01-01",
                "friends": ["user2"],
                "useful": 100,
                "funny": 50,
                "cool": 25,
                "fans": 10,
                "average_stars": 4.2
            }
        ]
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_mock_yelp_file(self, data_list, filename):
        """Create a mock Yelp JSON file with sample data."""
        file_path = Path(self.test_dir) / filename
        with open(file_path, 'w') as f:
            for item in data_list:
                f.write(json.dumps(item) + '\n')
        return file_path
    
    def test_process_yelp_reviews(self):
        """Test Yelp review data processing."""
        df = pd.DataFrame(self.sample_reviews)
        processed_df = self.collector._process_yelp_reviews(df)
        
        # Verify shape and columns
        self.assertEqual(len(processed_df), len(self.sample_reviews))
        self.assertIn('product_id', processed_df.columns)  # business_id renamed
        self.assertIn('rating', processed_df.columns)      # stars renamed
        self.assertIn('review_text', processed_df.columns) # text renamed
        self.assertIn('source', processed_df.columns)
        
        # Verify data transformation
        self.assertEqual(processed_df.iloc[0]['product_id'], 'biz1')
        self.assertEqual(processed_df.iloc[0]['rating'], 5)
        self.assertEqual(processed_df.iloc[0]['source'], 'yelp_open_dataset')
    
    def test_process_yelp_businesses(self):
        """Test Yelp business data processing."""
        df = pd.DataFrame(self.sample_businesses)
        processed_df = self.collector._process_yelp_businesses(df)
        
        # Verify basic structure
        self.assertEqual(len(processed_df), len(self.sample_businesses))
        self.assertIn('business_id', processed_df.columns)
        self.assertIn('name', processed_df.columns)
        self.assertIn('source', processed_df.columns)
        self.assertEqual(processed_df.iloc[0]['source'], 'yelp_open_dataset')
    
    def test_process_yelp_users(self):
        """Test Yelp user data processing."""
        df = pd.DataFrame(self.sample_users)
        processed_df = self.collector._process_yelp_users(df)
        
        # Verify basic structure
        self.assertEqual(len(processed_df), len(self.sample_users))
        self.assertIn('user_id', processed_df.columns)
        self.assertIn('source', processed_df.columns)
        self.assertEqual(processed_df.iloc[0]['source'], 'yelp_open_dataset')
    
    @patch('pandas.DataFrame.to_parquet')
    def test_collect_yelp_open_dataset_success(self, mock_to_parquet):
        """Test successful Yelp Open Dataset collection."""
        # Create mock data files
        review_file = self._create_mock_yelp_file(self.sample_reviews, 'yelp_academic_dataset_review.json')
        business_file = self._create_mock_yelp_file(self.sample_businesses, 'yelp_academic_dataset_business.json')
        
        # Run collection
        result = self.collector.collect_yelp_open_dataset(self.test_dir, chunk_size=5)
        
        # Verify results
        self.assertIn('review', result)
        self.assertIn('business', result)
        
        # Verify parquet save was called
        self.assertTrue(mock_to_parquet.called)
    
    def test_collect_yelp_open_dataset_missing_file(self):
        """Test Yelp collection with missing files."""
        # Try to collect from directory with no files
        empty_dir = Path(self.test_dir) / 'empty'
        empty_dir.mkdir()
        
        result = self.collector.collect_yelp_open_dataset(str(empty_dir))
        
        # Should return empty result since no files found
        self.assertEqual(len(result), 0)
    
    def test_process_yelp_file_chunked(self):
        """Test chunked file processing."""
        # Create larger dataset for chunk testing
        large_reviews = self.sample_reviews * 10  # 20 reviews
        review_file = self._create_mock_yelp_file(large_reviews, 'test_reviews.json')
        
        # Process with small chunk size
        result_df = self.collector._process_yelp_file_chunked(review_file, 'review', chunk_size=5)
        
        # Verify all data was processed
        self.assertEqual(len(result_df), 20)
        self.assertIn('source', result_df.columns)


class TestAmazonKaggleDataset(unittest.TestCase):
    """Test Amazon Kaggle dataset processing functionality."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.collector = DataCollector()
        self.collector.amazon_raw_path = Path(self.test_dir) / 'amazon'
        self.collector.amazon_raw_path.mkdir(exist_ok=True)
        
        # Sample Amazon review data
        self.sample_amazon_data = pd.DataFrame({
            'Id': [1, 2, 3],
            'ProductId': ['B001', 'B002', 'B001'],
            'UserId': ['U001', 'U002', 'U003'],
            'Score': [5, 1, 4],
            'Text': [
                'Great product, highly recommend!',
                'Terrible quality, waste of money.',
                'Good value for the price.'
            ],
            'Summary': ['Excellent', 'Poor', 'Good value']
        })
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_process_amazon_reviews(self):
        """Test Amazon review data processing."""
        processed_df = self.collector._process_amazon_reviews(self.sample_amazon_data)
        
        # Verify column mapping
        self.assertIn('review_id', processed_df.columns)
        self.assertIn('product_id', processed_df.columns)
        self.assertIn('user_id', processed_df.columns)
        self.assertIn('rating', processed_df.columns)
        self.assertIn('review_text', processed_df.columns)
        self.assertIn('source', processed_df.columns)
        
        # Verify data mapping
        self.assertEqual(processed_df.iloc[0]['review_id'], 1)
        self.assertEqual(processed_df.iloc[0]['product_id'], 'B001')
        self.assertEqual(processed_df.iloc[0]['rating'], 5)
        self.assertEqual(processed_df.iloc[0]['source'], 'amazon_kaggle')
    
    @patch('src.data_collection.KaggleApi')
    @patch('pandas.read_csv')
    @patch('pandas.DataFrame.to_parquet')
    def test_collect_amazon_kaggle_dataset_success(self, mock_to_parquet, mock_read_csv, mock_kaggle_api):
        """Test successful Amazon Kaggle dataset collection."""
        # Mock environment variables
        with patch.dict('os.environ', {'KAGGLE_USERNAME': 'test_user', 'KAGGLE_KEY': 'test_key'}):
            # Mock Kaggle API
            mock_api_instance = MagicMock()
            mock_kaggle_api.return_value = mock_api_instance
            
            # Mock CSV reading
            mock_read_csv.return_value = self.sample_amazon_data
            
            # Mock file system
            temp_csv = Path(self.test_dir) / 'temp_kaggle_download' / 'reviews.csv'
            temp_csv.parent.mkdir(exist_ok=True)
            temp_csv.write_text("dummy,csv,content")
            
            with patch('pathlib.Path.glob', return_value=[temp_csv]):
                # Run collection
                result = self.collector.collect_amazon_kaggle_dataset('test/dataset')
                
                # Verify API was called
                mock_api_instance.authenticate.assert_called_once()
                mock_api_instance.dataset_download_files.assert_called_once()
    
    def test_collect_amazon_kaggle_dataset_missing_credentials(self):
        """Test Amazon collection without Kaggle credentials."""
        # Ensure credentials are not set
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(DataCollectionError) as context:
                self.collector.collect_amazon_kaggle_dataset('test/dataset')
            
            self.assertIn('Kaggle credentials not found', str(context.exception))


class TestYelpFusionAPI(unittest.TestCase):
    """Test Yelp Fusion API functionality."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.collector = DataCollector()
        self.collector.yelp_raw_path = Path(self.test_dir) / 'yelp'
        self.collector.yelp_raw_path.mkdir(exist_ok=True)
        self.collector.yelp_api_key = 'test_api_key'
        
        # Sample API response data
        self.sample_businesses_response = {
            'businesses': [
                {
                    'id': 'biz1',
                    'name': 'Test Restaurant',
                    'location': {'city': 'Test City'},
                    'rating': 4.5,
                    'review_count': 100
                },
                {
                    'id': 'biz2', 
                    'name': 'Another Restaurant',
                    'location': {'city': 'Test City'},
                    'rating': 3.8,
                    'review_count': 50
                }
            ]
        }
        
        self.sample_reviews_response = {
            'reviews': [
                {
                    'id': 'rev1',
                    'user': {'id': 'user1', 'name': 'Test User'},
                    'rating': 5,
                    'text': 'Great food and service!',
                    'time_created': '2023-01-01T12:00:00',
                    'url': 'https://yelp.com/review/rev1'
                }
            ]
        }
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_collect_yelp_fusion_api_missing_key(self):
        """Test Yelp API collection without API key."""
        collector = DataCollector()
        collector.yelp_api_key = None
        
        with self.assertRaises(DataCollectionError) as context:
            collector.collect_yelp_fusion_api(['New York, NY'])
        
        self.assertIn('Yelp API key not found', str(context.exception))
    
    @patch('requests.Session.get')
    @patch('pandas.DataFrame.to_parquet')
    def test_get_yelp_businesses(self, mock_to_parquet, mock_get):
        """Test getting businesses from Yelp API."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = self.sample_businesses_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        headers = {'Authorization': 'Bearer test_api_key'}
        
        # Test the method
        businesses = self.collector._get_yelp_businesses(
            'New York, NY', None, 50, headers
        )
        
        # Verify results
        self.assertEqual(len(businesses), 2)
        self.assertEqual(businesses[0]['id'], 'biz1')
        self.assertEqual(businesses[1]['id'], 'biz2')
    
    @patch('requests.Session.get')
    def test_get_yelp_business_reviews(self, mock_get):
        """Test getting reviews for a business."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = self.sample_reviews_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        headers = {'Authorization': 'Bearer test_api_key'}
        
        # Test the method
        reviews = self.collector._get_yelp_business_reviews('biz1', 3, headers)
        
        # Verify results
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]['review_id'], 'rev1')
        self.assertEqual(reviews[0]['user_id'], 'user1')
        self.assertEqual(reviews[0]['rating'], 5)
    
    @patch('requests.Session.get')
    def test_api_rate_limiting(self, mock_get):
        """Test API rate limiting functionality."""
        # Set up rate limiting
        self.collector.api_rate_limit = 2
        self.collector.api_requests_made = 2  # At limit
        
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {'businesses': []}
        mock_get.return_value = mock_response
        
        # This should trigger rate limiting check
        headers = {'Authorization': 'Bearer test_api_key'}
        
        # Rate limiting check should be called (we can't easily test the sleep)
        businesses = self.collector._get_yelp_businesses('NYC', None, 1, headers)
        self.assertIsInstance(businesses, list)


class TestDataValidation(unittest.TestCase):
    """Test data validation functionality."""
    
    def setUp(self):
        self.collector = DataCollector()
        
        # Sample valid review data
        self.valid_reviews = pd.DataFrame({
            'review_id': ['r1', 'r2', 'r3'],
            'review_text': [
                'This is a great product with excellent quality.',
                'Good value for money, would recommend.',
                'Amazing service and fast delivery.'
            ],
            'rating': [5, 4, 5],
            'user_id': ['u1', 'u2', 'u3'],
            'product_id': ['p1', 'p1', 'p2']
        })
        
        # Sample invalid review data
        self.invalid_reviews = pd.DataFrame({
            'review_id': ['r1', 'r2', 'r3', 'r4'],
            'review_text': [
                'Great!',  # Too short
                'This is a very long review that exceeds the maximum length limit. ' * 100,  # Too long
                'Good product.',  # Valid
                ''  # Empty
            ],
            'rating': [5, 10, 3, 1],  # One invalid rating (10)
            'user_id': ['u1', 'u2', 'u3', 'u4']
        })
    
    def test_validate_review_data_valid(self):
        """Test validation of valid review data."""
        validated = self.collector._validate_review_data(self.valid_reviews)
        
        # Should keep all valid rows
        self.assertEqual(len(validated), 3)
        self.assertTrue(all(validated['rating'].between(1, 5)))
    
    def test_validate_review_data_invalid(self):
        """Test validation removes invalid review data."""
        validated = self.collector._validate_review_data(self.invalid_reviews)
        
        # Should remove invalid rows (length and rating issues)
        self.assertLess(len(validated), len(self.invalid_reviews))
        
        # All remaining rows should have valid ratings
        self.assertTrue(all(validated['rating'].between(1, 5)))
        
        # All remaining reviews should meet length requirements
        min_length = self.collector.config.get('data', {}).get('quality', {}).get('min_review_length', 10)
        self.assertTrue(all(validated['review_text'].str.len() >= min_length))
    
    def test_validate_business_data(self):
        """Test business data validation."""
        business_data = pd.DataFrame({
            'business_id': ['b1', 'b2', None, 'b3'],  # One missing ID
            'name': ['Restaurant 1', None, 'Restaurant 3', 'Restaurant 4'],  # One missing name
            'city': ['City 1', 'City 2', 'City 3', 'City 4']
        })
        
        validated = self.collector._validate_business_data(business_data)
        
        # Should remove rows with missing essential fields
        self.assertLessEqual(len(validated), len(business_data))
    
    def test_validate_user_data(self):
        """Test user data validation."""
        user_data = pd.DataFrame({
            'user_id': ['u1', 'u2', None, 'u3'],  # One missing ID
            'name': ['User 1', 'User 2', 'User 3', 'User 4'],
            'review_count': [10, 20, 5, 30]
        })
        
        validated = self.collector._validate_user_data(user_data)
        
        # Should remove rows with missing user_id
        self.assertEqual(len(validated), 3)
        self.assertTrue(all(validated['user_id'].notna()))
    
    def test_validate_data_orchestrator(self):
        """Test main validate_data method."""
        # Test with review data
        validated_reviews = self.collector.validate_data(self.valid_reviews, 'review')
        self.assertIsInstance(validated_reviews, pd.DataFrame)
        
        # Test with business data
        business_data = pd.DataFrame({'business_id': ['b1'], 'name': ['Business']})
        validated_business = self.collector.validate_data(business_data, 'business')
        self.assertIsInstance(validated_business, pd.DataFrame)


class TestComprehensiveDataCollection(unittest.TestCase):
    """Test comprehensive data collection orchestration."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.collector = DataCollector()
        self.collector.raw_data_path = Path(self.test_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('pandas.read_parquet')
    @patch('src.data_collection.DataCollector.collect_yelp_open_dataset')
    def test_collect_all_data_success(self, mock_yelp_collection, mock_read_parquet):
        """Test successful comprehensive data collection."""
        # Mock Yelp collection
        mock_yelp_collection.return_value = {
            'review': str(Path(self.test_dir) / 'yelp_reviews.parquet')
        }
        
        # Mock DataFrame for validation
        mock_df = pd.DataFrame({
            'review_text': ['Great product!', 'Not bad.'],
            'rating': [5, 3]
        })
        mock_read_parquet.return_value = mock_df
        
        # Run collection
        summary = self.collector.collect_all_data(
            yelp_open_path=self.test_dir,
            validate=True
        )
        
        # Verify summary structure
        self.assertIn('start_time', summary)
        self.assertIn('end_time', summary)
        self.assertIn('sources', summary)
        self.assertIn('total_records', summary)
        self.assertIn('errors', summary)
        
        # Verify Yelp source was processed
        self.assertIn('yelp_open', summary['sources'])
        self.assertEqual(summary['sources']['yelp_open']['status'], 'success')
    
    @patch('src.data_collection.DataCollector.collect_yelp_open_dataset')
    def test_collect_all_data_with_errors(self, mock_yelp_collection):
        """Test data collection handling errors gracefully."""
        # Mock failure
        mock_yelp_collection.side_effect = Exception("Test error")
        
        # Run collection
        summary = self.collector.collect_all_data(yelp_open_path=self.test_dir)
        
        # Should handle errors gracefully
        self.assertIn('errors', summary)
        self.assertIn('yelp_open', summary['sources'])
        self.assertEqual(summary['sources']['yelp_open']['status'], 'failed')
    
    def test_collect_all_data_no_sources(self):
        """Test data collection with no sources specified."""
        summary = self.collector.collect_all_data()
        
        # Should return valid summary even with no sources
        self.assertIn('start_time', summary)
        self.assertIn('sources', summary)
        self.assertEqual(summary['total_records'], 0)


class TestDataCollectionError(unittest.TestCase):
    """Test custom DataCollectionError exception."""
    
    def test_data_collection_error(self):
        """Test DataCollectionError exception handling."""
        with self.assertRaises(DataCollectionError):
            raise DataCollectionError("Test error message")
        
        try:
            raise DataCollectionError("Test error message")
        except DataCollectionError as e:
            self.assertEqual(str(e), "Test error message")


if __name__ == '__main__':
    unittest.main(verbosity=2)
