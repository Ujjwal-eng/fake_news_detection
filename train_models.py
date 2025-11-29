"""
Simple script to retrain all models with current scikit-learn version
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

print("="*80)
print("RETRAINING MODELS WITH CURRENT SCIKIT-LEARN VERSION")
print("="*80)

# Load training data
print("\n1. Loading training data...")
data_file = "data/training_data.csv"

if not os.path.exists(data_file):
    print(f"Error: {data_file} not found!")
    print("Please ensure training_data.csv exists in the data folder.")
    exit(1)

df = pd.read_csv(data_file)
print(f"   ✓ Loaded {len(df)} articles")
print(f"   - Real news: {len(df[df['label'] == 0])}")
print(f"   - Fake news: {len(df[df['label'] == 1])}")

# Preprocess texts
print("\n2. Preprocessing texts...")
preprocessor = TextPreprocessor()
df['processed_text'] = df['text'].apply(preprocessor.preprocess)
print("   ✓ Text preprocessing complete")

# Split data
print("\n3. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], 
    df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']
)
print(f"   ✓ Training set: {len(X_train)} samples")
print(f"   ✓ Test set: {len(X_test)} samples")

# Vectorize
print("\n4. Vectorizing text (TF-IDF)...")
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

# Save vectorizer
vectorizer_path = "models/vectorizer.joblib"
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer.vectorizer, vectorizer_path)
print(f"   ✓ Vectorizer saved to {vectorizer_path}")

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

# Train and evaluate each model
print("\n5. Training models...")
print("="*80)

results = []

for model_name, model in models.items():
    print(f"\n   Training {model_name}...")
    
    # Train
    model.fit(X_train_vec, y_train)
    
    # Predict
    y_pred = model.predict(X_test_vec)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Store results
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
    
    print(f"   ✓ {model_name} trained!")
    print(f"     - Accuracy:  {accuracy*100:.1f}%")
    print(f"     - Precision: {precision*100:.1f}%")
    print(f"     - Recall:    {recall*100:.1f}%")
    print(f"     - F1-Score:  {f1*100:.1f}%")
    print(f"     - Saved to: {model_path}")

# Print summary
print("\n" + "="*80)
print("TRAINING COMPLETE - SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save results
results_file = "models/training_results.txt"
with open(results_file, 'w') as f:
    f.write("FAKE NEWS DETECTION - MODEL TRAINING RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Training Data: {len(df)} articles\n")
    f.write(f"Training Set: {len(X_train)} samples\n")
    f.write(f"Test Set: {len(X_test)} samples\n")
    f.write(f"Features: {X_train_vec.shape[1]}\n\n")
    f.write(results_df.to_string(index=False))

print(f"\n✓ Results saved to {results_file}")
print("\nAll models retrained successfully with current scikit-learn version!")
print("Version mismatch warnings should now be resolved.")
