"""
Machine learning models for fake news detection.
This module implements various ML algorithms and provides training/prediction functionality.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import joblib
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_processing import prepare_data, DataLoader, FeatureExtractor
import config


class FakeNewsClassifier:
    """Main classifier for fake news detection."""
    
    def __init__(self, model_name: str = "naive_bayes"):
        """
        Initialize classifier with specified model.
        
        Args:
            model_name (str): Name of the model to use
        """
        self.model_name = model_name
        self.model = self._create_model(model_name)
        self.feature_extractor = None
        self.is_trained = False
    
    def _create_model(self, model_name: str) -> Any:
        """
        Create ML model based on name.
        
        Args:
            model_name (str): Model name
            
        Returns:
            Sklearn model object
        """
        models = {
            "naive_bayes": MultinomialNB(alpha=1.0),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                random_state=config.RANDOM_STATE,
                max_depth=10,
                min_samples_split=5
            ),
            "logistic_regression": LogisticRegression(
                random_state=config.RANDOM_STATE,
                max_iter=1000,
                C=1.0
            )
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
        
        return models[model_name]
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              feature_extractor: FeatureExtractor) -> None:
        """
        Train the model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            feature_extractor (FeatureExtractor): Fitted feature extractor
        """
        print(f"Training {self.model_name} model...")
        
        self.model.fit(X_train, y_train)
        self.feature_extractor = feature_extractor
        self.is_trained = True
        
        print(f"Model training completed!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def predict_text(self, text: str) -> Tuple[int, float]:
        """
        Predict single text article.
        
        Args:
            text (str): News article text
            
        Returns:
            Tuple[int, float]: (prediction, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Transform text to features
        X = self.feature_extractor.transform_texts([text])
        
        # Get prediction and probability
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        return prediction, confidence
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics
    
    def get_classification_report(self, X_test: np.ndarray, y_test: np.ndarray) -> str:
        """
        Get detailed classification report.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            str: Classification report
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.predict(X_test)
        target_names = ['Real News', 'Fake News']
        return classification_report(y_test, y_pred, target_names=target_names)
    
    def plot_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Plot confusion matrix.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real News', 'Fake News'],
                   yticklabels=['Real News', 'Fake News'])
        plt.title(f'Confusion Matrix - {self.model_name.title()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_path: str, vectorizer_path: str) -> None:
        """
        Save trained model and vectorizer.
        
        Args:
            model_path (str): Path to save model
            vectorizer_path (str): Path to save vectorizer
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save untrained model.")
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
        
        # Save model and vectorizer
        joblib.dump(self.model, model_path)
        joblib.dump(self.feature_extractor.vectorizer, vectorizer_path)
        
        print(f"Model saved to: {model_path}")
        print(f"Vectorizer saved to: {vectorizer_path}")
    
    def load_model(self, model_path: str, vectorizer_path: str) -> None:
        """
        Load trained model and vectorizer.
        
        Args:
            model_path (str): Path to model file
            vectorizer_path (str): Path to vectorizer file
        """
        try:
            self.model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            
            # Create feature extractor with loaded vectorizer
            self.feature_extractor = FeatureExtractor()
            self.feature_extractor.vectorizer = vectorizer
            
            self.is_trained = True
            print(f"Model loaded from: {model_path}")
            print(f"Vectorizer loaded from: {vectorizer_path}")
            
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
        except Exception as e:
            print(f"Error loading model: {e}")


class ModelComparison:
    """Compare multiple models and select the best one."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        feature_extractor: FeatureExtractor) -> None:
        """
        Train all available models and evaluate them.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            feature_extractor: Fitted feature extractor
        """
        print("=== Training and Comparing All Models ===\n")
        
        for model_name in config.MODEL_NAMES:
            print(f"Training {model_name}...")
            
            # Create and train model
            classifier = FakeNewsClassifier(model_name)
            classifier.train(X_train, y_train, feature_extractor)
            
            # Evaluate model
            metrics = classifier.evaluate(X_test, y_test)
            
            # Store results
            self.models[model_name] = classifier
            self.results[model_name] = metrics
            
            print(f"✓ {model_name} - Accuracy: {metrics['accuracy']:.4f}")
            print()
    
    def get_best_model(self) -> Tuple[str, FakeNewsClassifier]:
        """
        Get the best performing model based on F1-score.
        
        Returns:
            Tuple[str, FakeNewsClassifier]: (model_name, classifier)
        """
        if not self.results:
            raise ValueError("No models trained. Call train_all_models() first.")
        
        best_model_name = max(self.results.keys(),
                             key=lambda x: self.results[x]['f1_score'])
        
        return best_model_name, self.models[best_model_name]
    
    def print_comparison(self) -> None:
        """Print comparison of all models."""
        if not self.results:
            print("No models to compare.")
            return
        
        print("=== Model Comparison Results ===")
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 70)
        
        for model_name, metrics in self.results.items():
            print(f"{model_name:<20} {metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                  f"{metrics['f1_score']:<10.4f}")
        
        best_model, _ = self.get_best_model()
        print(f"\n🏆 Best model: {best_model}")


def train_and_save_best_model(data_file: Optional[str] = None) -> str:
    """
    Train all models, select the best one, and save it.
    
    Args:
        data_file (str, optional): Path to dataset file
        
    Returns:
        str: Name of the best model
    """
    # Load data
    loader = DataLoader()
    
    if data_file and os.path.exists(data_file):
        df = loader.load_data(data_file)
    else:
        print("Using sample data for demonstration...")
        df = loader.create_sample_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_extractor = prepare_data(df)
    
    # Compare all models
    comparison = ModelComparison()
    comparison.train_all_models(X_train, y_train, X_test, y_test, feature_extractor)
    comparison.print_comparison()
    
    # Get and save best model
    best_model_name, best_classifier = comparison.get_best_model()
    
    # Save the best model
    model_path = os.path.join(config.MODELS_DIR, "model.pkl")
    vectorizer_path = os.path.join(config.MODELS_DIR, "vectorizer.pkl")
    
    best_classifier.save_model(model_path, vectorizer_path)
    
    return best_model_name


if __name__ == "__main__":
    # Demo the model training pipeline
    print("=== Fake News Detection - Model Training Demo ===\n")
    
    try:
        # Train and save the best model
        best_model = train_and_save_best_model()
        print(f"\n🎉 Training completed! Best model: {best_model}")
        
        # Demo prediction
        print("\n=== Testing Predictions ===")
        
        # Load the saved model
        classifier = FakeNewsClassifier()
        model_path = os.path.join(config.MODELS_DIR, "model.pkl")
        vectorizer_path = os.path.join(config.MODELS_DIR, "vectorizer.pkl")
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            classifier.load_model(model_path, vectorizer_path)
            
            # Test with sample texts
            test_texts = [
                "Scientists at MIT have developed a new breakthrough in quantum computing technology.",
                "SHOCKING: This miracle cure will solve all your problems instantly!"
            ]
            
            for text in test_texts:
                prediction, confidence = classifier.predict_text(text)
                label = "FAKE" if prediction == 1 else "REAL"
                print(f"\nText: {text[:50]}...")
                print(f"Prediction: {label} (Confidence: {confidence:.2f})")
        
    except Exception as e:
        print(f"Error in model training: {e}")