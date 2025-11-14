"""
Quick Model Retraining Script
This will retrain your models with better data
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_processing import TextPreprocessor

def retrain_all_models(data_path='data/training_data.csv'):
    """Retrain all models with proper data"""
    
    print("=" * 60)
    print("RETRAINING FAKE NEWS DETECTION MODELS")
    print("=" * 60)
    
    # Check if training data already exists (skip creating new dataset if it does)
    if not os.path.exists(data_path):
        print("\nStep 1: Creating training dataset...")
        exec(open('download_dataset.py').read())
    else:
        # Check dataset size
        df_check = pd.read_csv(data_path)
        if len(df_check) < 100:
            print(f"\nStep 1: Dataset too small ({len(df_check)} samples), creating larger one...")
            exec(open('download_dataset.py').read())
        else:
            print(f"\nStep 1: Using existing dataset with {len(df_check)} samples")
    
    # Load the data
    print("\nStep 2: Loading training data...")
    if not os.path.exists(data_path):
        fallback_path = 'data/training_data.csv' if data_path != 'data/training_data.csv' else 'data/sample_data.csv'
        if os.path.exists(fallback_path):
            data_path = fallback_path
            print(f"⚠️  Using fallback: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples from {data_path}")
    print(f"  - Real news: {len(df[df['label']==0])} samples")
    print(f"  - Fake news: {len(df[df['label']==1])} samples")
    
    # Preprocess text
    print("\n🔤 Step 3: Preprocessing text...")
    preprocessor = TextPreprocessor()
    df['processed_text'] = df['text'].apply(lambda x: preprocessor.preprocess(x))
    
    # Split data
    X = df['processed_text'].values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # Vectorize
    print("\n🔢 Step 4: Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"✓ Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Define models
    models = {
        'Naive Bayes': MultinomialNB(alpha=0.1),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=20),
        'SVM': SVC(kernel='linear', probability=True, random_state=42)
    }
    
    print("\n🤖 Step 5: Training models...")
    print("-" * 60)
    
    results = []
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Train
        model.fit(X_train_vec, y_train)
        
        # Predict
        y_train_pred = model.predict(X_train_vec)
        y_test_pred = model.predict(X_test_vec)
        
        # Calculate accuracy
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        print(f"  ✓ Train Accuracy: {train_acc:.3f}")
        print(f"  ✓ Test Accuracy:  {test_acc:.3f}")
        
        # Classification report
        print("\n  Classification Report:")
        report = classification_report(y_test, y_test_pred, 
                                      target_names=['Real', 'Fake'],
                                      digits=3)
        print(report)
        
        results.append({
            'model': model_name,
            'train_acc': train_acc,
            'test_acc': test_acc
        })
        
        # Save model
        model_filename = model_name.lower().replace(' ', '_') + '_model.joblib'
        model_path = os.path.join('models', model_filename)
        joblib.dump(model, model_path)
        print(f"  ✓ Saved to: {model_path}")
    
    # Save vectorizer
    vectorizer_path = os.path.join('models', 'vectorizer.joblib')
    joblib.dump(vectorizer, vectorizer_path)
    print(f"\n✓ Saved vectorizer to: {vectorizer_path}")
    
    # Save training results
    print("\n" + "=" * 60)
    print("📊 TRAINING SUMMARY")
    print("=" * 60)
    for result in results:
        print(f"{result['model']:20} | Train: {result['train_acc']:.3f} | Test: {result['test_acc']:.3f}")
    
    # Save results to file
    with open('models/training_results.txt', 'w') as f:
        f.write("FAKE NEWS DETECTION - MODEL TRAINING RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training Data: {len(df)} articles\n")
        f.write(f"Train/Test Split: {len(X_train)}/{len(X_test)}\n\n")
        f.write("MODEL PERFORMANCE:\n")
        f.write("-" * 60 + "\n")
        for result in results:
            f.write(f"\n{result['model'].upper().replace(' ', '_')}:\n")
            f.write(f"  Train Accuracy: {result['train_acc']:.3f}\n")
            f.write(f"  Test Accuracy: {result['test_acc']:.3f}\n")
    
    print("\n✅ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("\n🚀 Next step: Restart your Flask app to use the new models")
    print("   Run: .venv\\Scripts\\python.exe app.py")

if __name__ == "__main__":
    retrain_all_models()
