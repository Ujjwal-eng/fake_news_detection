# Configuration settings for Fake News Detection App

import os

# Data paths
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = "models"

# Model settings
MODEL_NAMES = ["naive_bayes", "random_forest", "logistic_regression"]
DEFAULT_MODEL = "naive_bayes"

# Text preprocessing settings
MAX_FEATURES = 10000
MIN_DF = 2
MAX_DF = 0.8
NGRAM_RANGE = (1, 2)

# Model hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Streamlit app settings
APP_TITLE = "Fake News Detection"
APP_ICON = "🔍"
SIDEBAR_TITLE = "Settings"

# File paths
SAMPLE_DATA_FILE = os.path.join(DATA_DIR, "sample_data.csv")
VECTORIZER_FILE = os.path.join(MODELS_DIR, "vectorizer.pkl")
MODEL_FILE = os.path.join(MODELS_DIR, "model.pkl")