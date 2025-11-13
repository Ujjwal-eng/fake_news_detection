# 🔍 Fake News Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An AI-powered web application that uses Natural Language Processing (NLP) and Machine Learning to detect fake news articles with high accuracy. This project demonstrates end-to-end ML model development, from data preprocessing to web deployment.

## 🌟 Features

- **🤖 Multiple ML Models**: Naive Bayes, Random Forest, Logistic Regression, and SVM
- **🔍 Advanced NLP**: Text preprocessing with tokenization, stemming, and TF-IDF vectorization
- **🎨 Modern Web Interface**: Beautiful, responsive Flask-based UI
- **⚡ Real-time Predictions**: Instant classification with confidence scores
- **📊 Detailed Analytics**: Probability distributions and model performance metrics
- **🧪 Comprehensive Testing**: Unit tests and model evaluation suite

## 📁 Project Structure

```
fake-news-detection/
├── app.py                 # Flask web application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore rules
├── templates/            # HTML templates
│   ├── index.html        # Main page
│   └── about.html        # About page
├── src/                  # Source code modules
│   ├── __init__.py
│   ├── data_processing.py # NLP preprocessing utilities
│   ├── model.py          # ML model implementation
│   └── utils.py          # Helper functions
├── data/                 # Dataset storage
│   ├── raw/              # Raw datasets
│   ├── processed/        # Processed datasets
│   └── sample_data.csv   # Sample dataset
├── models/               # Trained models
│   ├── naive_bayes_model.joblib
│   ├── vectorizer.joblib
│   └── training_results.txt
├── notebooks/            # Jupyter notebooks
│   ├── data_exploration.ipynb
│   └── text_processing_basics.ipynb
└── tests/                # Unit tests
    ├── __init__.py
    └── test_model.py
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Ujjwal-eng/fake_news_detection.git
cd fake_news_detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Required NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## 💻 Usage

### Running the Web Application
```bash
python app.py
```
Then open your browser and navigate to: `http://localhost:5000`

### Training Models
To train new models with your own dataset:
```bash
# Place your dataset in data/raw/
# Then run the training script
python -m src.model
```

### Using Jupyter Notebooks
For experimentation and analysis:
```bash
jupyter notebook
# Open notebooks/ directory
```

## 📊 Dataset

The system works with news datasets in CSV format:

### Required Format
```csv
text,label
"News article text here...",0
"Another news article...",1
```

- **text**: The news article content (string)
- **label**: 0 for real news, 1 for fake news (integer)

### Sample Dataset
A sample dataset is included in `data/sample_data.csv` for testing purposes.

## 🎯 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | ~85% | ~84% | ~86% | ~85% |
| Random Forest | ~88% | ~87% | ~89% | ~88% |
| Logistic Regression | ~86% | ~85% | ~87% | ~86% |
| SVM | ~87% | ~86% | ~88% | ~87% |

*Results may vary based on training data quality and quantity*

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Flask**: Web framework for the application
- **scikit-learn**: Machine learning models and evaluation
- **NLTK**: Natural language processing toolkit
- **Pandas & NumPy**: Data manipulation and analysis
- **Joblib**: Model serialization

### Machine Learning
- Naive Bayes Classifier
- Random Forest Classifier
- Logistic Regression
- Support Vector Machine (SVM)
- TF-IDF Vectorization

### Frontend
- HTML5 & CSS3
- JavaScript (Vanilla)
- Responsive Design

## 🧠 How It Works

### 1. Text Preprocessing Pipeline
```
Raw Text → Lowercasing → Tokenization → Stop Word Removal → Stemming → Clean Text
```

### 2. Feature Extraction
- **TF-IDF Vectorization**: Converts text to numerical features
- **N-gram Analysis**: Captures word patterns (unigrams, bigrams)
- **Feature Selection**: Identifies most informative features

### 3. Classification
- Multiple ML models trained on labeled datasets
- Ensemble predictions for improved accuracy
- Probability estimation for confidence scoring

### 4. Web Interface
- User submits news text
- Backend processes and vectorizes text
- Model predicts and returns result with confidence

## 🎓 Key Learnings & Skills Demonstrated

- ✅ End-to-end ML project development
- ✅ Natural Language Processing techniques
- ✅ Model training, evaluation, and optimization
- ✅ Web application development with Flask
- ✅ RESTful API design
- ✅ Version control with Git/GitHub
- ✅ Code organization and best practices

## 📸 Screenshots

### Home Page
Clean and intuitive interface for news analysis

### Results Page
Detailed prediction with confidence scores and probabilities

## 🔮 Future Enhancements

- [ ] Deep learning models (LSTM, BERT)
- [ ] Multi-language support
- [ ] Source credibility analysis
- [ ] Real-time news monitoring
- [ ] Browser extension
- [ ] Mobile application
- [ ] Integration with fact-checking APIs
- [ ] User authentication and history

## ⚠️ Disclaimer

This system is designed as an educational tool and should not be used as the sole method for verifying news authenticity. Always:
- Cross-reference with multiple reliable sources
- Check the original source's credibility
- Consider the context and date of publication
- Consult professional fact-checkers for important decisions

## 👨‍💻 Author

**Ujjwal Bansal**
- GitHub: [@Ujjwal-eng](https://github.com/Ujjwal-eng)
- Project: [Fake News Detection](https://github.com/Ujjwal-eng/fake_news_detection)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- scikit-learn team for excellent ML libraries
- NLTK developers for NLP tools
- Flask community for the lightweight web framework
- Kaggle for providing datasets
- Research papers on fake news detection for inspiration

## 📞 Contact & Support

If you have questions or suggestions:
- Open an issue on GitHub
- Fork the project and submit a pull request
- Star ⭐ the repository if you find it helpful!

---

**Made with ❤️ and Python**