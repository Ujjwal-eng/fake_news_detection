# Fake News Detection App

A machine learning-based application to detect fake news articles using Natural Language Processing and various ML algorithms.

## Features

- **Text Analysis**: Advanced NLP preprocessing and feature extraction
- **Multiple ML Models**: Naive Bayes, Random Forest, and Logistic Regression
- **Interactive Web Interface**: User-friendly Streamlit app
- **Real-time Predictions**: Instant fake news classification
- **Model Performance Metrics**: Accuracy, precision, recall, and F1-score visualization

## Project Structure

```
fake-news-detection/
├── app.py                 # Main Streamlit web application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore rules
├── src/                  # Source code modules
│   ├── __init__.py
│   ├── data_processing.py # Data preprocessing utilities
│   ├── model.py          # ML model implementation
│   └── utils.py          # Helper functions
├── data/                 # Dataset storage
│   ├── raw/              # Raw datasets
│   ├── processed/        # Processed datasets
│   └── sample_data.csv   # Sample dataset for testing
├── models/               # Trained model storage
│   └── .gitkeep
├── notebooks/            # Jupyter notebooks for experimentation
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
└── tests/                # Unit tests
    ├── __init__.py
    └── test_model.py
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download
If you have this project locally, navigate to the project directory.

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Required NLTK Data
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Usage

### Running the Web Application
```bash
streamlit run app.py
```

### Training New Models
```bash
python src/model.py
```

### Using in Jupyter Notebooks
Start Jupyter and open the notebooks in the `notebooks/` directory:
```bash
jupyter notebook
```

## Dataset

The app works with news datasets containing:
- **Text**: The news article content
- **Label**: 0 for real news, 1 for fake news

### Sample Data Format
```csv
text,label
"Breaking: Scientists discover new...",0
"SHOCKING: Celebrity reveals secret...",1
```

## Model Performance

The app includes multiple ML algorithms:
- **Naive Bayes**: Fast, good baseline performance
- **Random Forest**: Robust, handles feature interactions well  
- **Logistic Regression**: Interpretable, good for text classification

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset sources and ML libraries used
- Inspiration from fake news detection research papers
- Streamlit community for the excellent web framework