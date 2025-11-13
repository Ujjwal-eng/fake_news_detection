"""
Unit tests for the model module.
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.model import FakeNewsClassifier, ModelComparison
    from src.data_processing import DataLoader, TextPreprocessor, FeatureExtractor
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all source files are created first.")


class TestFakeNewsClassifier(unittest.TestCase):
    """Test cases for FakeNewsClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = FakeNewsClassifier("naive_bayes")
    
    def test_model_creation(self):
        """Test model creation with different algorithms."""
        models = ["naive_bayes", "random_forest", "logistic_regression"]
        
        for model_name in models:
            with self.subTest(model=model_name):
                classifier = FakeNewsClassifier(model_name)
                self.assertEqual(classifier.model_name, model_name)
                self.assertIsNotNone(classifier.model)
    
    def test_invalid_model_name(self):
        """Test creation with invalid model name."""
        with self.assertRaises(ValueError):
            FakeNewsClassifier("invalid_model")
    
    def test_prediction_without_training(self):
        """Test that prediction fails without training."""
        with self.assertRaises(ValueError):
            self.classifier.predict(np.array([[1, 2, 3]]))


class TestTextPreprocessor(unittest.TestCase):
    """Test cases for TextPreprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = "Check this out! https://example.com @user #hashtag 123"
        cleaned = self.preprocessor.clean_text(dirty_text)
        
        # Should remove URLs, mentions, hashtags, and numbers
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("@user", cleaned)
        self.assertNotIn("#hashtag", cleaned)
        self.assertNotIn("123", cleaned)
    
    def test_preprocess_text(self):
        """Test complete preprocessing pipeline."""
        text = "This is a TEST sentence with STOPWORDS!"
        processed = self.preprocessor.preprocess_text(text)
        
        # Should be lowercase and processed
        self.assertIsInstance(processed, str)
        self.assertTrue(len(processed) > 0)


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader."""
    
    def test_create_sample_data(self):
        """Test sample data creation."""
        loader = DataLoader()
        df = loader.create_sample_data()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(len(df) > 0)
        self.assertIn('text', df.columns)
        self.assertIn('label', df.columns)
        
        # Check labels are binary
        unique_labels = df['label'].unique()
        self.assertTrue(all(label in [0, 1] for label in unique_labels))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)