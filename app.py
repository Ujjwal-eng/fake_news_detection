"""
Fake News Detection Web Application
A Flask-based web interface for detecting fake news using machine learning.
"""

from flask import Flask, render_template, request, jsonify
import os
import joblib
import numpy as np
from src.data_processing import TextPreprocessor
import config

app = Flask(__name__)

# Global variables for model and vectorizer
model = None
vectorizer = None
model_name = None

def load_model():
    """Load the trained model and vectorizer"""
    global model, vectorizer, model_name
    
    try:
        # Try to load trained models
        model_path = os.path.join(config.MODELS_DIR, 'naive_bayes_model.joblib')
        vectorizer_path = os.path.join(config.MODELS_DIR, 'vectorizer.joblib')
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            model_name = "Naive Bayes"
            print("✓ Model loaded successfully!")
            return True
        else:
            print("⚠ No trained model found. Please train a model first.")
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load model at startup
model_loaded = load_model()

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', model_loaded=model_loaded, model_name=model_name)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if not model_loaded:
            return jsonify({
                'error': 'Model not loaded. Please train a model first.',
                'success': False
            })
        
        # Get text from request
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'error': 'Please enter some text to analyze.',
                'success': False
            })
        
        # Preprocess the text
        preprocessor = TextPreprocessor()
        processed_text = preprocessor.preprocess(text)
        
        # Vectorize the text
        text_vectorized = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        prediction_proba = model.predict_proba(text_vectorized)[0]
        
        # Prepare response
        result = {
            'success': True,
            'prediction': 'FAKE NEWS' if prediction == 1 else 'REAL NEWS',
            'confidence': float(max(prediction_proba) * 100),
            'fake_probability': float(prediction_proba[1] * 100),
            'real_probability': float(prediction_proba[0] * 100),
            'model_used': model_name
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'success': False
        })

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'model_name': model_name
    })

if __name__ == '__main__':
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
