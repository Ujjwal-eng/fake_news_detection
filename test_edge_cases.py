"""
Test script to check how the model handles edge cases:
- URLs/Links
- Non-English text
- Special characters
- Numbers
- Empty/very short text
"""

import joblib
import os
from src.data_processing import TextPreprocessor

# Load models and vectorizer
print("Loading models...")
vectorizer = joblib.load('models/vectorizer.joblib')
nb_model = joblib.load('models/naive_bayes_model.joblib')
preprocessor = TextPreprocessor()

# Test cases
test_cases = {
    "URL Only": "https://www.example.com/fake-news-article",
    "URL in Text": "Check this article https://example.com/news it says something shocking!",
    "Multiple URLs": "Visit https://news.com and www.fakenews.org for more info",
    
    "Hindi Text": "यह एक नकली खबर है जो वायरल हो रही है",
    "Mixed Language": "This is fake news यह झूठी खबर है completely false",
    "Arabic Text": "هذا خبر مزيف ينتشر بسرعة",
    
    "Only Numbers": "12345 67890 999 2025",
    "Numbers in Text": "The government will give 500 rupees to 100 million people by 2025",
    
    "Special Characters": "!!! $$$ @@@ ### *** %%%",
    "Mixed Special Chars": "BREAKING!!! Government announces $$$ 1000% tax increase ***URGENT***",
    
    "Emojis": "😱😱😱 SHOCKING NEWS 🔥🔥🔥 Share before deleted!!!",
    
    "Very Short": "Fake",
    "Empty Spaces": "   ",
    "Single Word": "BREAKING",
    
    "All Caps": "SHARE THIS MESSAGE IMMEDIATELY OR YOUR ACCOUNT WILL BE DELETED",
    
    "Email": "contact@fakenews.com has leaked government secrets",
    
    "Legitimate News": "The Reserve Bank of India announced a 25 basis point increase in the repo rate to 6.5% during its monetary policy meeting today.",
    
    "Typical Fake Pattern": "URGENT!!! WhatsApp will charge Rs 500 from tomorrow! Forward to 20 people to avoid charges! My friend already paid! Share now!"
}

print("\n" + "="*80)
print("TESTING EDGE CASES FOR FAKE NEWS DETECTION MODEL")
print("="*80)

for test_name, test_text in test_cases.items():
    print(f"\n{'─'*80}")
    print(f"Test Case: {test_name}")
    print(f"{'─'*80}")
    print(f"Original Input:\n  {test_text[:100]}{'...' if len(test_text) > 100 else ''}")
    
    # Preprocess
    preprocessed = preprocessor.preprocess(test_text)
    print(f"\nAfter Preprocessing:\n  '{preprocessed[:100]}{'...' if len(preprocessed) > 100 else ''}'")
    print(f"  Length: {len(preprocessed)} characters")
    
    # Vectorize
    vectorized = vectorizer.transform([preprocessed])
    print(f"\nVectorized Shape: {vectorized.shape}")
    print(f"  Non-zero features: {vectorized.nnz}")
    
    # Predict
    try:
        prediction = nb_model.predict(vectorized)[0]
        probabilities = nb_model.predict_proba(vectorized)[0]
        
        result = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
        confidence = max(probabilities) * 100
        
        print(f"\n✓ Prediction: {result}")
        print(f"  Confidence: {confidence:.2f}%")
        print(f"  Probabilities: Real={probabilities[0]:.4f}, Fake={probabilities[1]:.4f}")
    except Exception as e:
        print(f"\n✗ Error during prediction: {str(e)}")

print("\n" + "="*80)
print("TEST COMPLETED")
print("="*80)
