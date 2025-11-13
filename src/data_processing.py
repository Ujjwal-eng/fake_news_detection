"""
Text Processing Module for Fake News Detection

This module contains all the essential text processing functions including:
- Tokenization
- Stopword removal  
- Vectorization (Count and TF-IDF)
- Complete preprocessing pipeline

Author: AI Assistant
Date: October 2025
"""

import re
import pandas as pd
import numpy as np
from typing import List, Tuple, Union, Optional
from collections import Counter

# NLTK imports
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Scikit-learn imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

# Download required NLTK data (run once)
def download_nltk_data():
    """Download required NLTK datasets"""
    nltk_downloads = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
    for item in nltk_downloads:
        try:
            nltk.download(item, quiet=True)
        except:
            pass

# Initialize NLTK components
download_nltk_data()


class TextTokenizer:
    """
    Text tokenization utility class
    """
    
    @staticmethod
    def simple_tokenize(text: str) -> List[str]:
        """
        Simple tokenization using split()
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tokens
        """
        return text.split()
    
    @staticmethod
    def nltk_tokenize(text: str, lowercase: bool = True) -> List[str]:
        """
        NLTK word tokenization
        
        Args:
            text (str): Input text
            lowercase (bool): Convert to lowercase
            
        Returns:
            list: List of tokens
        """
        if lowercase:
            text = text.lower()
        return word_tokenize(text)
    
    @staticmethod
    def regex_tokenize(text: str, pattern: str = r'\b\w+\b', lowercase: bool = True) -> List[str]:
        """
        Regular expression tokenization
        
        Args:
            text (str): Input text
            pattern (str): Regex pattern for tokenization
            lowercase (bool): Convert to lowercase
            
        Returns:
            list: List of tokens
        """
        if lowercase:
            text = text.lower()
        return re.findall(pattern, text)
    
    @staticmethod
    def sentence_tokenize(text: str) -> List[str]:
        """
        Sentence tokenization
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of sentences
        """
        return sent_tokenize(text)


class StopwordRemover:
    """
    Stopword removal utility class
    """
    
    def __init__(self, language: str = 'english', custom_stopwords: Optional[List[str]] = None):
        """
        Initialize stopword remover
        
        Args:
            language (str): Language for stopwords
            custom_stopwords (list): Additional custom stopwords
        """
        self.stop_words = set(stopwords.words(language))
        
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
    
    def remove_stopwords(self, tokens: List[str], filter_alpha: bool = True) -> List[str]:
        """
        Remove stopwords from token list
        
        Args:
            tokens (list): List of tokens
            filter_alpha (bool): Keep only alphabetic tokens
            
        Returns:
            list: Filtered tokens
        """
        filtered_tokens = []
        for token in tokens:
            if token.lower() not in self.stop_words:
                if not filter_alpha or token.isalpha():
                    filtered_tokens.append(token)
        return filtered_tokens
    
    def add_stopwords(self, words: List[str]):
        """Add custom stopwords"""
        self.stop_words.update(words)
    
    def remove_stopwords_text(self, text: str) -> str:
        """
        Remove stopwords from text and return processed text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with stopwords removed
        """
        tokens = word_tokenize(text.lower())
        filtered_tokens = self.remove_stopwords(tokens)
        return ' '.join(filtered_tokens)


class TextCleaner:
    """
    Text cleaning utility class
    """
    
    @staticmethod
    def basic_clean(text: str) -> str:
        """
        Basic text cleaning
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def advanced_clean(text: str) -> str:
        """
        Advanced text cleaning with more preprocessing
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class TextNormalizer:
    """
    Text normalization using stemming and lemmatization
    """
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply stemming to tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Apply lemmatization to tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def stem_text(self, text: str) -> str:
        """Apply stemming to text"""
        tokens = word_tokenize(text.lower())
        stemmed_tokens = self.stem_tokens(tokens)
        return ' '.join(stemmed_tokens)
    
    def lemmatize_text(self, text: str) -> str:
        """Apply lemmatization to text"""
        tokens = word_tokenize(text.lower())
        lemmatized_tokens = self.lemmatize_tokens(tokens)
        return ' '.join(lemmatized_tokens)


class TextVectorizer:
    """
    Text vectorization utility class
    """
    
    def __init__(self, vectorizer_type: str = 'tfidf', **kwargs):
        """
        Initialize vectorizer
        
        Args:
            vectorizer_type (str): 'count' or 'tfidf'
            **kwargs: Additional arguments for vectorizer
        """
        self.vectorizer_type = vectorizer_type
        self.vectorizer = None
        
        # Default parameters
        default_params = {
            'lowercase': True,
            'stop_words': 'english',
            'max_features': 5000,
            'ngram_range': (1, 2),
            'min_df': 1,        # Changed from 2 to 1 for small datasets
            'max_df': 0.9       # Changed from 0.8 to 0.9 for small datasets
        }
        
        # Update with user parameters
        default_params.update(kwargs)
        
        if vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(**default_params)
        elif vectorizer_type == 'tfidf':
            # Add TF-IDF specific parameters
            if 'sublinear_tf' not in default_params:
                default_params['sublinear_tf'] = True
            self.vectorizer = TfidfVectorizer(**default_params)
        else:
            raise ValueError("vectorizer_type must be 'count' or 'tfidf'")
    
    def fit_transform(self, texts: List[str]):
        """Fit vectorizer and transform texts"""
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts: List[str]):
        """Transform texts using fitted vectorizer"""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted")
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get feature names"""
        return self.vectorizer.get_feature_names_out()
    
    def get_vocabulary_size(self):
        """Get vocabulary size"""
        return len(self.vectorizer.vocabulary_)


class TextProcessor(BaseEstimator, TransformerMixin):
    """
    Complete text processing pipeline for fake news detection
    """
    
    def __init__(self, 
                 vectorizer_type: str = 'tfidf',
                 ngram_range: Tuple[int, int] = (1, 2),
                 max_features: int = 5000,
                 remove_stopwords: bool = True,
                 use_lemmatization: bool = True,
                 use_stemming: bool = False,
                 custom_stopwords: Optional[List[str]] = None,
                 min_df: int = 1,
                 max_df: float = 0.9):
        """
        Initialize complete text processor
        
        Args:
            vectorizer_type (str): 'count' or 'tfidf'
            ngram_range (tuple): N-gram range for vectorization
            max_features (int): Maximum number of features
            remove_stopwords (bool): Whether to remove stopwords
            use_lemmatization (bool): Whether to use lemmatization
            use_stemming (bool): Whether to use stemming
            custom_stopwords (list): Additional stopwords
            min_df (int): Minimum document frequency
            max_df (float): Maximum document frequency
        """
        self.vectorizer_type = vectorizer_type
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        self.use_stemming = use_stemming
        self.min_df = min_df
        self.max_df = max_df
        
        # Initialize components
        self.cleaner = TextCleaner()
        self.tokenizer = TextTokenizer()
        self.stopword_remover = StopwordRemover(custom_stopwords=custom_stopwords)
        self.normalizer = TextNormalizer()
        self.vectorizer = None
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess individual text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Clean text
        text = self.cleaner.basic_clean(text)
        
        # Tokenize
        tokens = self.tokenizer.nltk_tokenize(text, lowercase=True)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.stopword_remover.remove_stopwords(tokens)
        
        # Apply normalization
        if self.use_stemming:
            tokens = self.normalizer.stem_tokens(tokens)
        elif self.use_lemmatization:
            tokens = self.normalizer.lemmatize_tokens(tokens)
        
        return ' '.join(tokens)
    
    def fit(self, X, y=None):
        """
        Fit the text processor
        
        Args:
            X (list): List of texts
            y: Ignored (for sklearn compatibility)
            
        Returns:
            self: Fitted processor
        """
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in X]
        
        # Initialize vectorizer
        vectorizer_params = {
            'ngram_range': self.ngram_range,
            'max_features': self.max_features,
            'stop_words': 'english',
            'min_df': self.min_df,
            'max_df': self.max_df
        }
        
        if self.vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(**vectorizer_params)
        else:
            vectorizer_params['sublinear_tf'] = True
            self.vectorizer = TfidfVectorizer(**vectorizer_params)
        
        # Fit vectorizer
        self.vectorizer.fit(processed_texts)
        
        return self
    
    def transform(self, X):
        """
        Transform texts to feature matrix
        
        Args:
            X (list): List of texts
            
        Returns:
            sparse matrix: Feature matrix
        """
        if self.vectorizer is None:
            raise ValueError("Processor not fitted. Call fit() first.")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in X]
        
        # Vectorize
        return self.vectorizer.transform(processed_texts)
    
    def fit_transform(self, X, y=None):
        """
        Fit processor and transform texts
        
        Args:
            X (list): List of texts
            y: Ignored (for sklearn compatibility)
            
        Returns:
            sparse matrix: Feature matrix
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self):
        """Get feature names from vectorizer"""
        if self.vectorizer is None:
            raise ValueError("Processor not fitted")
        return self.vectorizer.get_feature_names_out()
    
    def get_vocabulary_size(self):
        """Get vocabulary size"""
        if self.vectorizer is None:
            raise ValueError("Processor not fitted")
        return len(self.vectorizer.vocabulary_)


def analyze_text_statistics(texts: List[str]) -> dict:
    """
    Analyze basic text statistics
    
    Args:
        texts (list): List of texts
        
    Returns:
        dict: Text statistics
    """
    stats = {
        'num_texts': len(texts),
        'avg_length': np.mean([len(text) for text in texts]),
        'avg_words': np.mean([len(text.split()) for text in texts]),
        'min_length': min([len(text) for text in texts]),
        'max_length': max([len(text) for text in texts]),
    }
    
    return stats


def get_most_common_words(texts: List[str], n: int = 20, preprocessed: bool = False) -> List[Tuple[str, int]]:
    """
    Get most common words in text corpus
    
    Args:
        texts (list): List of texts
        n (int): Number of top words to return
        preprocessed (bool): Whether texts are already preprocessed
        
    Returns:
        list: List of (word, frequency) tuples
    """
    if not preprocessed:
        processor = TextProcessor()
        texts = [processor.preprocess_text(text) for text in texts]
    
    # Combine all texts
    all_words = ' '.join(texts).split()
    
    # Count frequencies
    word_freq = Counter(all_words)
    
    return word_freq.most_common(n)


def compare_vocabularies(texts1: List[str], texts2: List[str], 
                        labels: Tuple[str, str] = ('Group 1', 'Group 2'),
                        n: int = 20) -> dict:
    """
    Compare vocabularies between two text groups
    
    Args:
        texts1 (list): First group of texts
        texts2 (list): Second group of texts
        labels (tuple): Labels for the groups
        n (int): Number of top words to analyze
        
    Returns:
        dict: Comparison results
    """
    processor = TextProcessor()
    
    # Preprocess texts
    processed1 = [processor.preprocess_text(text) for text in texts1]
    processed2 = [processor.preprocess_text(text) for text in texts2]
    
    # Get word frequencies
    words1 = Counter(' '.join(processed1).split())
    words2 = Counter(' '.join(processed2).split())
    
    # Get top words for each group
    top_words1 = words1.most_common(n)
    top_words2 = words2.most_common(n)
    
    # Find unique words
    vocab1 = set(words1.keys())
    vocab2 = set(words2.keys())
    
    unique_to_1 = vocab1 - vocab2
    unique_to_2 = vocab2 - vocab1
    common_words = vocab1 & vocab2
    
    return {
        'group1_label': labels[0],
        'group2_label': labels[1],
        'top_words_group1': top_words1,
        'top_words_group2': top_words2,
        'unique_to_group1': list(unique_to_1)[:20],
        'unique_to_group2': list(unique_to_2)[:20],
        'common_words_count': len(common_words),
        'group1_vocab_size': len(vocab1),
        'group2_vocab_size': len(vocab2)
    }


# Example usage and testing
if __name__ == "__main__":
    # Sample texts for testing
    sample_texts = [
        "Scientists at MIT have developed a breakthrough quantum computer.",
        "SHOCKING: This one weird trick will help you lose weight fast!",
        "The Federal Reserve announced a new interest rate policy.",
        "BREAKING: Aliens have landed and the government is hiding it!"
    ]
    
    # Test individual components
    print("=== Testing Text Processing Components ===\n")
    
    # Test tokenization
    tokenizer = TextTokenizer()
    tokens = tokenizer.nltk_tokenize(sample_texts[0])
    print(f"Tokens: {tokens[:10]}...\n")
    
    # Test stopword removal
    stopword_remover = StopwordRemover()
    filtered_tokens = stopword_remover.remove_stopwords(tokens)
    print(f"After stopword removal: {filtered_tokens}\n")
    
    # Test complete pipeline
    processor = TextProcessor(vectorizer_type='tfidf', max_features=100)
    X = processor.fit_transform(sample_texts)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Feature names (first 10): {processor.get_feature_names()[:10]}")
    
    # Test text statistics
    stats = analyze_text_statistics(sample_texts)
    print(f"\nText statistics: {stats}")
