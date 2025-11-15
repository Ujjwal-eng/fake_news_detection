"""
Fake News Detection Web Application
A Flask-based web interface for detecting fake news using ensemble machine learning.

Dataset Coverage: 2016-2023
- 11,632 professionally labeled articles
- 50% Real News (5,816 articles)
- 50% Fake News (5,816 articles)

Models: Naive Bayes (83.2%), Logistic Regression (90.0%), Random Forest (87.8%), SVM (89.7%)
Ensemble: Majority voting with confidence-based tie-breaking
"""

from flask import Flask, render_template, request, jsonify
import os
import re
import joblib
import numpy as np
from src.data_processing import TextPreprocessor
import config

# Try to import FactChecker (optional - requires spacy)
try:
    from src.fact_checker import FactChecker
    FACT_CHECKER_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"⚠️ Fact Checker not available: {e}")
    print("   Install spacy with: pip install spacy && python -m spacy download en_core_web_sm")
    FactChecker = None
    FACT_CHECKER_AVAILABLE = False

app = Flask(__name__)

# Global variables for models and vectorizer
models = {}
vectorizer = None
fact_checker = None
current_model_name = "Naive Bayes"

# Sample news datasets for testing - 2025 news with detectable patterns
SAMPLE_NEWS = {
    "real_1": {
        "title": "India Launches Gaganyaan Mission Successfully",
        "text": "The Indian Space Research Organisation successfully launched the Gaganyaan-1 unmanned test flight from Sriharikota on Monday morning. The mission marks a crucial milestone in India's human spaceflight program, testing critical systems including the crew module and escape mechanisms. ISRO Chairman Dr. S Somanath confirmed all systems performed as expected during the 15-minute suborbital flight. Prime Minister congratulated the ISRO team, stating this brings India closer to joining the elite group of nations with human spaceflight capability. The first crewed mission is planned for late 2025."
    },
    "real_2": {
        "title": "Mumbai Metro Line 3 Opens After Nine Years of Construction",
        "text": "The Mumbai Metro Rail Corporation inaugurated the 33.5-kilometer underground Metro Line 3 connecting Colaba to SEEPZ on Sunday. Chief Minister Eknath Shinde and Union Railway Minister flagged off the first train from the Bandra-Kurla Complex station. The Aqua Line features 27 stations, including India's deepest metro station at 32 meters below ground level. Officials estimate the line will reduce travel time between South Mumbai and the suburbs by over an hour, serving approximately 1.7 million passengers daily. Commuters praised the modern amenities and air-conditioned coaches."
    },
    "real_3": {
        "title": "Indian Women's Cricket Team Wins T20 World Cup",
        "text": "India defeated Australia by 8 wickets in the final at Dubai International Stadium to claim their first ICC Women's T20 World Cup title on Sunday. Smriti Mandhana scored an unbeaten 72 off 43 balls while chasing a target of 157 runs. Captain Harmanpreet Kaur praised the team's consistency throughout the tournament, where India remained undefeated in eight matches. The victory sparked celebrations across India, with cricketers receiving congratulatory messages from the Prime Minister and sports minister. The BCCI announced a prize money of 5 crore rupees for the winning team."
    },
    "real_4": {
        "title": "India Signs Free Trade Agreement with European Union",
        "text": "India and the European Union signed a comprehensive free trade agreement in Brussels on Thursday after negotiations spanning over a decade. The deal will eliminate tariffs on 95 percent of goods traded between India and the 27-member bloc over the next 10 years. External Affairs Minister Dr. S Jaishankar described the agreement as a landmark achievement that will boost bilateral trade to $200 billion annually. The pact covers goods, services, investment, and includes commitments on labor rights and environmental standards. Parliament will ratify the agreement in the next session."
    },
    "real_5": {
        "title": "Bangalore Airport Ranks Third Best in Asia-Pacific",
        "text": "Kempegowda International Airport in Bangalore has been ranked third-best airport in the Asia-Pacific region by Skytrax in their 2025 World Airport Awards. The airport scored high marks for cleanliness, customer service, and terminal facilities, moving up from seventh place last year. Airport CEO Hari Marar attributed the recognition to infrastructure upgrades and staff training programs. The airport handled 37 million passengers in 2024, making it India's third-busiest after Delhi and Mumbai. A new terminal expansion project worth 13,000 crores is underway to increase capacity to 65 million passengers annually."
    },
    "fake_1": {
        "title": "SHOCKING: Government Announces 200% Tax on All Bank Deposits",
        "text": "BREAKING NEWS! Finance Ministry has announced a shocking 200 percent tax on all bank savings and fixed deposits starting next week! This means if you have 1 lakh rupees in your account, you will have to pay 2 lakh rupees as tax! Banking experts are calling this the WORST decision in Indian history! People are rushing to withdraw cash before the deadline! Share this immediately before it's too late! Your hard-earned money is at risk! The mainstream media is not covering this story because banks are pressuring them! Wake up India!"
    },
    "fake_2": {
        "title": "Doctors Reveal: Drinking Hot Water Cures Cancer, Diabetes, Heart Disease",
        "text": "MEDICAL BREAKTHROUGH that Big Pharma doesn't want you to know! Top doctors have discovered that drinking hot water on empty stomach can cure cancer, diabetes, and all heart diseases within 30 days! This ancient secret has been hidden from the public for decades! Pharmaceutical companies are trying to ban this information because they will lose billions! My uncle had stage 4 cancer and was completely cured by this simple remedy! Share this with everyone you know before this post gets deleted! No need for expensive treatments anymore! God bless!"
    },
    "fake_3": {
        "title": "WhatsApp Will Start Charging Rs 500 Per Month From Tomorrow",
        "text": "URGENT MESSAGE: WhatsApp will start charging Rs 500 per month from tomorrow! To avoid the charges, forward this message to 20 contacts and your profile will turn blue, meaning you're exempt! If you don't forward, your account will be charged automatically! This is confirmed by WhatsApp CEO! Already 10 lakh people have been charged! My friend didn't forward and got charged Rs 5000! Please forward immediately to save your money! This is not a joke! Copy and paste this message to all groups! Do it now before midnight deadline!"
    },
    "fake_4": {
        "title": "Scientists Discover Humans Can Live 500 Years By Eating This",
        "text": "UNBELIEVABLE DISCOVERY! Scientists at Harvard University have found that eating this one common Indian spice can make you live up to 500 years! The research was kept secret for 20 years but has now been leaked! Billionaires have been using this secret formula! Doctors are shocked! This ancient Ayurvedic remedy reverses aging completely! My grandmother is 250 years old and still looks 40 because of this! The government wants to ban this information! Click here to learn the secret ingredient before this video is removed! Only 24 hours left! Share urgently!"
    },
    "fake_5": {
        "title": "Modi Announces Free iPhone 15 For Every Indian Citizen",
        "text": "BREAKING: Prime Minister has announced that every Indian citizen will receive a FREE iPhone 15 as part of Digital India initiative! The government has ordered 1 billion units from Apple! To register, click the link below and enter your Aadhaar number, bank details, and OTP! First come first served! My neighbor already registered and received the iPhone yesterday! Only genuine citizens will get this opportunity! Registration closes in 2 hours! Forward this to all WhatsApp groups immediately! Those who don't register will miss this golden opportunity forever! Act fast!"
    }
}

def load_all_models():
    """Load all available trained models"""
    global models, vectorizer, current_model_name
    
    try:
        # Load vectorizer (shared across all models)
        vectorizer_path = os.path.join(config.MODELS_DIR, 'vectorizer.joblib')
        if os.path.exists(vectorizer_path):
            vectorizer = joblib.load(vectorizer_path)
        else:
            print("⚠ No vectorizer found.")
            return False
        
        # Define available models
        available_models = {
            'Naive Bayes': 'naive_bayes_model.joblib',
            'Random Forest': 'random_forest_model.joblib',
            'SVM': 'svm_model.joblib',
            'Logistic Regression': 'logistic_regression_model.joblib'
        }
        
        # Load all available models
        models_loaded = 0
        for model_name, model_file in available_models.items():
            model_path = os.path.join(config.MODELS_DIR, model_file)
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
                models_loaded += 1
                print(f"✓ {model_name} model loaded successfully!")
        
        if models_loaded > 0:
            print(f"✓ Total {models_loaded} models loaded!")
            
            # Initialize fact checker (optional feature)
            global fact_checker
            if FACT_CHECKER_AVAILABLE and FactChecker is not None:
                try:
                    fact_checker = FactChecker()
                    print("✓ Fact Checker initialized successfully!")
                except Exception as e:
                    print(f"⚠ Fact Checker initialization failed: {e}")
                    fact_checker = None
            else:
                print("⚠ Fact Checker not available (spacy not installed)")
                fact_checker = None
            
            return True
        else:
            print("⚠ No trained models found. Please train models first.")
            return False
            
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# Load all models at startup
model_loaded = load_all_models()

@app.route('/')
def home():
    """Render the professional UI as default home page"""
    return render_template('index_professional.html',
                         model_loaded=len(models) > 0,
                         available_models=list(models.keys()),
                         sample_news=SAMPLE_NEWS)

def validate_input(text: str, preprocessor) -> tuple:
    """
    Validate user input before processing
    Returns: (is_valid: bool, error_message: str, warning_message: str)
    """
    # Check if empty
    if not text or not text.strip():
        return False, "Please enter some text to analyze.", ""
    
    # Check minimum length
    if len(text.strip()) < 20:
        return False, "Text is too short. Please enter at least 20 characters of article text.", ""
    
    # Check if it's just a URL
    url_pattern = r'^(https?://|www\.)\S+$'
    if re.match(url_pattern, text.strip()):
        return False, "Please enter the article text content, not just the URL. Copy the full article text instead.", ""
    
    # Check if mostly non-English characters
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    total_chars = len(re.sub(r'\s', '', text))
    if total_chars > 0 and english_chars / total_chars < 0.3:
        return False, "This model only supports English text. Text appears to be in a non-English language.", ""
    
    # Check if preprocessed text will be too short
    preprocessed = preprocessor.preprocess(text)
    if len(preprocessed.strip()) < 10:
        return False, "Text contains insufficient analyzable content after preprocessing. Please provide more meaningful English text.", ""
    
    # Warning for very short preprocessed text
    warning = ""
    if len(preprocessed.split()) < 5:
        warning = "⚠️ Very short text detected. Prediction confidence may be low."
    
    return True, "", warning


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests using ensemble voting from all models"""
    
    try:
        if not model_loaded:
            return jsonify({
                'error': 'Model not loaded. Please train a model first.',
                'success': False
            })
        
        # Get text from request
        data = request.get_json()
        text = data.get('text', '').strip()
        use_ensemble = data.get('ensemble', True)  # Default to ensemble mode
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        
        # Validate input
        is_valid, error_msg, warning_msg = validate_input(text, preprocessor)
        if not is_valid:
            return jsonify({
                'error': error_msg,
                'success': False
            })
        
        # Preprocess the text
        processed_text = preprocessor.preprocess(text)
        
        # Vectorize the text
        text_vectorized = vectorizer.transform([processed_text])
        
        # Get predictions from all models
        all_predictions = {}
        fake_votes = 0
        real_votes = 0
        sum_of_confidences = 0.0  # Sum of winning confidences only
        all_fake_probs = []
        all_real_probs = []
        
        for model_name, model in models.items():
            # Make prediction
            prediction = model.predict(text_vectorized)[0]
            prediction_proba = model.predict_proba(text_vectorized)[0]
            
            # prediction_proba is [prob_class_0, prob_class_1]
            # where class 0 = REAL, class 1 = FAKE
            
            # FIX: Some models (Random Forest) may return unnormalized values
            # Normalize if values are not in 0-1 range
            if prediction_proba[0] > 1.0 or prediction_proba[1] > 1.0:
                total = prediction_proba[0] + prediction_proba[1]
                prediction_proba = prediction_proba / total if total > 0 else prediction_proba
            
            real_prob = float(prediction_proba[0] * 100)
            fake_prob = float(prediction_proba[1] * 100)
            
            # Collect all probabilities
            all_fake_probs.append(fake_prob)
            all_real_probs.append(real_prob)
            
            # Store individual model result
            all_predictions[model_name] = {
                'prediction': 'FAKE NEWS' if prediction == 1 else 'REAL NEWS',
                'confidence': fake_prob if prediction == 1 else real_prob,
                'fake_probability': fake_prob,
                'real_probability': real_prob
            }
            
            # Count votes and sum winning confidences
            if prediction == 1:  # Fake
                fake_votes += 1
                sum_of_confidences += fake_prob
            else:  # Real
                real_votes += 1
                sum_of_confidences += real_prob
        
        # Calculate average confidences for each class
        num_models = len(models)
        avg_fake_confidence = sum(all_fake_probs) / num_models
        avg_real_confidence = sum(all_real_probs) / num_models
        
        # Calculate ensemble confidence (average of winning models' confidences)
        if fake_votes > 0 and fake_votes == real_votes:
            # In case of tie, use the class with higher average
            ensemble_confidence = max(avg_fake_confidence, avg_real_confidence)
        elif fake_votes > real_votes:
            # Majority FAKE: average confidence of models that predicted FAKE
            ensemble_confidence = sum_of_confidences / fake_votes
        else:
            # Majority REAL: average confidence of models that predicted REAL
            ensemble_confidence = sum_of_confidences / real_votes
        
        # Ensemble decision logic
        if fake_votes > real_votes:
            # Majority says FAKE
            final_prediction = 'FAKE NEWS'
            final_confidence = ensemble_confidence
            decision_type = 'Majority Vote'
        elif real_votes > fake_votes:
            # Majority says REAL
            final_prediction = 'REAL NEWS'
            final_confidence = ensemble_confidence
            decision_type = 'Majority Vote'
        else:
            # TIE (2-2): Use average confidence as tie-breaker
            if avg_fake_confidence > avg_real_confidence:
                final_prediction = 'FAKE NEWS'
                final_confidence = avg_fake_confidence
                decision_type = 'Confidence Tie-Breaker (2-2 split)'
            else:
                final_prediction = 'REAL NEWS'
                final_confidence = avg_real_confidence
                decision_type = 'Confidence Tie-Breaker (2-2 split)'
        
        # Prepare response
        result = {
            'success': True,
            'prediction': final_prediction,
            'confidence': round(final_confidence, 2),
            'fake_votes': fake_votes,
            'real_votes': real_votes,
            'decision_type': decision_type,
            'fake_probability': round(avg_fake_confidence, 2),
            'real_probability': round(avg_real_confidence, 2),
            'individual_results': all_predictions,
            'total_models': num_models
        }
        
        # Debug: Print the confidence being sent
        print(f"DEBUG: Sending confidence = {result['confidence']}%")
        print(f"DEBUG: final_confidence = {final_confidence}")
        print(f"DEBUG: ensemble_confidence = {ensemble_confidence}")
        
        # Add warning if text was very short
        if warning_msg:
            result['warning'] = warning_msg
        
        # Add fact-checking if available
        if fact_checker:
            try:
                # Analyze text for factual claims
                fact_check_result = fact_checker.analyze(text)
                
                # Get final verdict combining ML and fact-checking
                is_fake = (final_prediction == 'FAKE NEWS')
                final_verdict = fact_checker.get_verdict(
                    ml_prediction=is_fake,
                    ml_confidence=final_confidence,
                    fact_check_result=fact_check_result
                )
                
                # Add fact-checking data to result
                result['fact_check'] = {
                    'entities_found': fact_check_result['entities'],
                    'numerical_issues': fact_check_result['numerical_issues'],
                    'verification_results': fact_check_result['verification_results'],
                    'warnings': fact_check_result['warnings'],
                    'confidence_adjustment': fact_check_result['confidence_adjustment']
                }
                
                # Update final prediction and confidence with fact-check verdict
                result['final_verdict'] = final_verdict['verdict']
                result['adjusted_confidence'] = final_verdict['adjusted_confidence']
                result['fact_check_reason'] = final_verdict['reason']
                
            except Exception as fact_check_error:
                print(f"Fact-checking error: {fact_check_error}")
                # Continue without fact-checking if it fails
                result['fact_check'] = None
        
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
        'available_models': list(models.keys()),
        'current_model': current_model_name
    })

if __name__ == '__main__':
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
