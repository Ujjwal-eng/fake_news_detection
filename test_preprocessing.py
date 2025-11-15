"""
Simple test to show how text preprocessing handles edge cases
"""

from src.data_processing import TextPreprocessor
import re

preprocessor = TextPreprocessor()

# Test cases
test_cases = {
    "1. URL Only": "https://www.example.com/fake-news-article",
    "2. URL in Text": "Check this article https://example.com/news it says something shocking!",
    "3. Hindi Text": "यह एक नकली खबर है जो वायरल हो रही है",
    "4. Mixed Language": "This is fake news यह झूठी खबर है completely false",
    "5. Only Numbers": "12345 67890 999 2025",
    "6. Numbers in Text": "The government will give 500 rupees to 100 million people by 2025",
    "7. Special Characters Only": "!!! $$$ @@@ ### *** %%%",
    "8. Mixed Special Chars": "BREAKING!!! Government announces $$$ 1000% tax increase ***URGENT***",
    "9. Very Short": "Fake",
    "10. Empty Spaces": "   ",
    "11. Single Word": "BREAKING",
    "12. Email": "contact@fakenews.com has leaked government secrets",
    "13. Legitimate News": "The Reserve Bank of India announced a 25 basis point increase in the repo rate to 6.5% during its monetary policy meeting today.",
    "14. Typical Fake Pattern": "URGENT!!! WhatsApp will charge Rs 500 from tomorrow! Forward to 20 people to avoid charges!"
}

print("\n" + "="*90)
print(" EDGE CASE TESTING: How Text Preprocessing Handles Different Inputs")
print("="*90)

for test_name, test_text in test_cases.items():
    print(f"\n{'─'*90}")
    print(f"Test Case {test_name}")
    print(f"{'─'*90}")
    print(f"Original Input ({len(test_text)} chars):")
    print(f"  '{test_text}'")
    
    # Preprocess
    preprocessed = preprocessor.preprocess(test_text)
    
    print(f"\nAfter Preprocessing ({len(preprocessed)} chars):")
    if preprocessed:
        print(f"  '{preprocessed}'")
        print(f"  Word count: {len(preprocessed.split())}")
    else:
        print(f"  [EMPTY STRING - All content removed!]")
    
    # Show what was removed
    if not preprocessed:
        print(f"\n⚠️  WARNING: Preprocessing removed ALL content!")
        print(f"  → Model will receive empty/minimal features")
        print(f"  → Prediction will be unreliable or based on default bias")

print("\n" + "="*90)
print(" SUMMARY OF PREPROCESSING BEHAVIOR")
print("="*90)
print("""
The TextPreprocessor does the following (in order):
1. Convert to lowercase
2. Remove URLs (http://, https://, www.)
3. Remove special characters and digits [^a-zA-Z\\s]
4. Tokenize into words
5. Remove English stopwords (the, is, a, etc.)
6. Apply stemming (running → run)

IMPLICATIONS FOR EDGE CASES:
─────────────────────────────────────────────────────────────────────────────
• URLs/Links:        Completely removed → empty result
• Non-English text:  All non-English characters removed → empty result
• Numbers only:      All numbers removed → empty result
• Special chars:     All removed → empty result
• Very short text:   May become empty after stopword removal
• Empty input:       Stays empty

WHAT THE MODEL WILL DO:
─────────────────────────────────────────────────────────────────────────────
When the preprocessed text is empty or very short:
• The vectorizer will produce a sparse vector (mostly zeros)
• The model will make a prediction based on its DEFAULT BIAS
• Typically predicts the majority class from training (likely FAKE in your case)
• Confidence may be around 50-70% (weak prediction)

RECOMMENDATION:
─────────────────────────────────────────────────────────────────────────────
Add input validation in your Flask app BEFORE preprocessing:
1. Check minimum text length (e.g., at least 20 characters)
2. Check if text contains mostly non-English characters
3. Check if text is mostly URLs/numbers/special characters
4. Return a user-friendly error message for invalid inputs
""")
print("="*90 + "\n")
