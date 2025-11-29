"""
Quick model retraining script - uses cached processed data if available
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from src.data_processing import TextPreprocessor, TextVectorizer
import pickle

print("="*80)
print("QUICK MODEL RETRAINING")
print("="*80)

# Check for cached processed data
cache_file = "data/processed_cache.pkl"

if os.path.exists(cache_file):
    print("\n✓ Found cached processed data, loading...")
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    X_train_vec = cache['X_train_vec']
    X_test_vec = cache['X_test_vec']
    y_train = cache['y_train']
    y_test = cache['y_test']
    vectorizer = cache['vectorizer']
    print(f"  Loaded training set: {X_train_vec.shape[0]} samples")
    print(f"  Loaded test set: {X_test_vec.shape[0]} samples")
else:
    # Full preprocessing
    print("\n1. Loading training data...")
    data_file = "data/training_data.csv"
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        exit(1)
    
    df = pd.read_csv(data_file)
    print(f"   ✓ Loaded {len(df)} articles")
    
    print("\n2. Preprocessing texts (this may take a few minutes)...")
    preprocessor = TextPreprocessor()
    
    # Process in batches with progress
    batch_size = 1000
    processed_texts = []
    for i in range(0, len(df), batch_size):
        batch = df['text'].iloc[i:i+batch_size]
        processed_batch = batch.apply(preprocessor.preprocess)
        processed_texts.extend(processed_batch.tolist())
        print(f"   Processed {min(i+batch_size, len(df))}/{len(df)} articles...")
    
    df['processed_text'] = processed_texts
    print("   ✓ Text preprocessing complete")
    
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    print("\n4. Vectorizing text...")
    vectorizer = TextVectorizer(
        vectorizer_type='tfidf',
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9
    )
    
    X_train_vec = vectorizer.fit_transform(X_train.tolist())
    X_test_vec = vectorizer.transform(X_test.tolist())
    print(f"   ✓ Feature matrix shape: {X_train_vec.shape}")
    
    # Cache the processed data
    print("\n5. Caching processed data for future use...")
    os.makedirs("data", exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'X_train_vec': X_train_vec,
            'X_test_vec': X_test_vec,
            'y_train': y_train,
            'y_test': y_test,
            'vectorizer': vectorizer.vectorizer
        }, f)
    print(f"   ✓ Cached to {cache_file}")

# Save vectorizer
print("\nSaving vectorizer...")
vectorizer_path = "models/vectorizer.joblib"
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer if isinstance(vectorizer, object) and hasattr(vectorizer, 'vectorizer') else vectorizer, vectorizer_path)
print(f"✓ Vectorizer saved to {vectorizer_path}")

# Define models
models = {
    'Naive Bayes': MultinomialNB(alpha=1.0),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        max_depth=10,
        n_jobs=-1
    ),
    'Logistic Regression': LogisticRegression(
        random_state=42, 
        max_iter=1000, 
        C=1.0,
        n_jobs=-1
    ),
    'SVM': SVC(
        kernel='linear', 
        random_state=42, 
        probability=True,
        C=1.0
    )
}

# Train and save models
print("\nTraining models...")
print("="*80)

results = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Train
    model.fit(X_train_vec, y_train)
    
    # Predict
    y_pred = model.predict(X_test_vec)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    # Save model
    model_filename = model_name.lower().replace(' ', '_') + '_model.joblib'
    model_path = os.path.join('models', model_filename)
    joblib.dump(model, model_path)
    
    print(f"✓ {model_name}: {accuracy*100:.1f}% accuracy - Saved to {model_path}")

# Summary
print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save results
results_file = "models/training_results.txt"
with open(results_file, 'w') as f:
    f.write("FAKE NEWS DETECTION - MODEL TRAINING RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(results_df.to_string(index=False))

print(f"\n✓ Results saved to {results_file}")
print("\n✅ All models retrained successfully!")
print("   Version warnings should now be completely resolved.")
