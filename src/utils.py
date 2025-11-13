"""
Utility functions for the fake news detection app.
This module provides helper functions for visualization, text analysis, and general utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from wordcloud import WordCloud
from collections import Counter
import re
from textblob import TextBlob


def create_performance_chart(metrics: Dict[str, Dict[str, float]]) -> go.Figure:
    """
    Create an interactive performance comparison chart.
    
    Args:
        metrics (Dict): Model performance metrics
        
    Returns:
        plotly.graph_objects.Figure: Interactive chart
    """
    models = list(metrics.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig = go.Figure()
    
    for metric in metric_names:
        values = [metrics[model][metric] for model in models]
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=models,
            y=values,
            text=[f"{v:.3f}" for v in values],
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        barmode='group',
        height=500,
        showlegend=True
    )
    
    return fig


def create_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, 
                                model_name: str) -> go.Figure:
    """
    Create an interactive confusion matrix plot.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        model_name (str): Name of the model
        
    Returns:
        plotly.graph_objects.Figure: Interactive confusion matrix
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Real News', 'Fake News'],
        y=['Real News', 'Fake News'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20},
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Confusion Matrix - {model_name}",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=500,
        width=500
    )
    
    return fig


def generate_wordcloud(texts: List[str], title: str = "Word Cloud") -> plt.Figure:
    """
    Generate a word cloud from text data.
    
    Args:
        texts (List[str]): List of texts
        title (str): Title for the word cloud
        
    Returns:
        matplotlib.pyplot.Figure: Word cloud figure
    """
    # Combine all texts
    combined_text = ' '.join(texts)
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(combined_text)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig


def analyze_text_features(texts: List[str], labels: List[int]) -> Dict[str, any]:
    """
    Analyze various features of the text data.
    
    Args:
        texts (List[str]): List of news texts
        labels (List[int]): Corresponding labels (0=real, 1=fake)
        
    Returns:
        Dict: Analysis results
    """
    df = pd.DataFrame({'text': texts, 'label': labels})
    
    # Calculate text features
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['char_count'] = df['text'].apply(len)
    df['exclamation_count'] = df['text'].apply(lambda x: x.count('!'))
    df['question_count'] = df['text'].apply(lambda x: x.count('?'))
    df['upper_case_count'] = df['text'].apply(lambda x: sum(1 for c in x if c.isupper()))
    df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Group by label
    real_news = df[df['label'] == 0]
    fake_news = df[df['label'] == 1]
    
    analysis = {
        'total_articles': len(df),
        'real_news_count': len(real_news),
        'fake_news_count': len(fake_news),
        'avg_word_count_real': real_news['word_count'].mean(),
        'avg_word_count_fake': fake_news['word_count'].mean(),
        'avg_exclamation_real': real_news['exclamation_count'].mean(),
        'avg_exclamation_fake': fake_news['exclamation_count'].mean(),
        'avg_sentiment_real': real_news['sentiment'].mean(),
        'avg_sentiment_fake': fake_news['sentiment'].mean(),
        'dataframe': df
    }
    
    return analysis


def create_feature_distribution_plot(analysis: Dict) -> go.Figure:
    """
    Create feature distribution plots comparing real vs fake news.
    
    Args:
        analysis (Dict): Text analysis results
        
    Returns:
        plotly.graph_objects.Figure: Distribution plots
    """
    df = analysis['dataframe']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Word Count', 'Exclamation Marks', 'Sentiment Score', 'Character Count'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    features = ['word_count', 'exclamation_count', 'sentiment', 'char_count']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for feature, (row, col) in zip(features, positions):
        real_data = df[df['label'] == 0][feature]
        fake_data = df[df['label'] == 1][feature]
        
        fig.add_trace(
            go.Histogram(x=real_data, name='Real News', opacity=0.7, nbinsx=20),
            row=row, col=col
        )
        fig.add_trace(
            go.Histogram(x=fake_data, name='Fake News', opacity=0.7, nbinsx=20),
            row=row, col=col
        )
    
    fig.update_layout(
        title="Feature Distribution: Real vs Fake News",
        height=700,
        showlegend=True,
        barmode='overlay'
    )
    
    return fig


def extract_key_phrases(texts: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
    """
    Extract key phrases from texts using simple n-gram analysis.
    
    Args:
        texts (List[str]): List of texts
        top_n (int): Number of top phrases to return
        
    Returns:
        List[Tuple[str, int]]: List of (phrase, frequency) tuples
    """
    # Simple bigram extraction
    all_bigrams = []
    
    for text in texts:
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Generate bigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        all_bigrams.extend(bigrams)
    
    # Count frequencies
    phrase_freq = Counter(all_bigrams)
    return phrase_freq.most_common(top_n)


def calculate_readability_score(text: str) -> float:
    """
    Calculate a simple readability score based on sentence and word length.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        float: Readability score (higher = more complex)
    """
    sentences = text.split('.')
    words = text.split()
    
    if len(sentences) == 0 or len(words) == 0:
        return 0.0
    
    avg_sentence_length = len(words) / len(sentences)
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    # Simple readability formula
    readability = (avg_sentence_length * 0.5) + (avg_word_length * 0.5)
    return readability


def create_prediction_confidence_chart(predictions: List[Tuple[str, float]]) -> go.Figure:
    """
    Create a chart showing prediction confidence distribution.
    
    Args:
        predictions (List[Tuple[str, float]]): List of (prediction, confidence) tuples
        
    Returns:
        plotly.graph_objects.Figure: Confidence distribution chart
    """
    df = pd.DataFrame(predictions, columns=['prediction', 'confidence'])
    
    fig = px.histogram(
        df, 
        x='confidence', 
        color='prediction',
        title="Prediction Confidence Distribution",
        labels={'confidence': 'Confidence Score', 'count': 'Number of Predictions'},
        nbins=20
    )
    
    fig.update_layout(height=500)
    return fig


def format_prediction_result(prediction: int, confidence: float) -> Dict[str, any]:
    """
    Format prediction result for display.
    
    Args:
        prediction (int): Model prediction (0=real, 1=fake)
        confidence (float): Prediction confidence
        
    Returns:
        Dict: Formatted result
    """
    label = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
    color = "red" if prediction == 1 else "green"
    
    # Confidence level description
    if confidence >= 0.9:
        confidence_level = "Very High"
    elif confidence >= 0.8:
        confidence_level = "High"
    elif confidence >= 0.7:
        confidence_level = "Moderate"
    elif confidence >= 0.6:
        confidence_level = "Low"
    else:
        confidence_level = "Very Low"
    
    return {
        'label': label,
        'prediction': prediction,
        'confidence': confidence,
        'confidence_percentage': f"{confidence*100:.1f}%",
        'confidence_level': confidence_level,
        'color': color,
        'icon': "🚨" if prediction == 1 else "✅"
    }


def clean_filename(filename: str) -> str:
    """
    Clean filename for safe file operations.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Cleaned filename
    """
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    
    # Limit length
    if len(filename) > 100:
        filename = filename[:100]
    
    return filename


def get_model_info() -> Dict[str, str]:
    """
    Get information about available models.
    
    Returns:
        Dict[str, str]: Model information
    """
    return {
        'naive_bayes': 'Multinomial Naive Bayes - Fast, good baseline performance',
        'random_forest': 'Random Forest - Robust, handles feature interactions well',
        'logistic_regression': 'Logistic Regression - Interpretable, good for text classification'
    }


def create_sample_news_data() -> pd.DataFrame:
    """
    Create extended sample news data for demonstration.
    
    Returns:
        pd.DataFrame: Sample dataset
    """
    sample_data = [
        # Real news examples
        ("The World Health Organization announced new guidelines for global health emergency preparedness following extensive consultation with member countries.", 0),
        ("Researchers at Stanford University published a peer-reviewed study showing the effects of climate change on ocean temperatures over the past decade.", 0),
        ("The Federal Reserve announced a 0.25% interest rate adjustment following their monthly economic policy meeting in Washington D.C.", 0),
        ("NASA's James Webb Space Telescope captured detailed images of distant galaxies, providing new insights into early universe formation.", 0),
        ("The European Union reached a consensus on new renewable energy targets during their climate summit in Brussels this week.", 0),
        
        # Fake news examples
        ("BREAKING: Scientists discover that drinking coffee backwards cures all diseases! Doctors are FURIOUS about this one simple trick!", 1),
        ("SHOCKING REVELATION: Ancient aliens built the pyramids using this one weird technology that NASA doesn't want you to know!", 1),
        ("URGENT: Government secretly replacing all birds with surveillance drones! Here's the PROOF they don't want you to see!", 1),
        ("MIRACLE CURE: Local grandmother discovers how to lose 100 pounds in 3 days using this kitchen ingredient!", 1),
        ("EXPOSED: The REAL reason why celebrities look young revealed! The beauty industry HATES this secret method!", 1),
        
        # Additional mixed examples
        ("Local university receives grant funding for renewable energy research project focusing on solar panel efficiency improvements.", 0),
        ("UNBELIEVABLE: This simple household item can predict the future with 100% accuracy! Fortune tellers DON'T want you to know this!", 1),
        ("The Department of Transportation announced new safety regulations for autonomous vehicles following comprehensive testing and public hearings.", 0),
        ("MIND-BLOWING: Eating pizza for breakfast makes you immortal according to this SUPPRESSED scientific study!", 1),
        ("Economic analysts report steady growth in the technology sector with particular strength in artificial intelligence and machine learning companies.", 0)
    ]
    
    return pd.DataFrame(sample_data, columns=['text', 'label'])


if __name__ == "__main__":
    # Demo utility functions
    print("=== Fake News Detection - Utilities Demo ===\n")
    
    # Create sample data
    df = create_sample_news_data()
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # Analyze text features
    analysis = analyze_text_features(texts, labels)
    
    print("Text Analysis Results:")
    print(f"Total articles: {analysis['total_articles']}")
    print(f"Real news: {analysis['real_news_count']}")
    print(f"Fake news: {analysis['fake_news_count']}")
    print(f"Avg words (real): {analysis['avg_word_count_real']:.1f}")
    print(f"Avg words (fake): {analysis['avg_word_count_fake']:.1f}")
    print(f"Avg sentiment (real): {analysis['avg_sentiment_real']:.2f}")
    print(f"Avg sentiment (fake): {analysis['avg_sentiment_fake']:.2f}")
    
    # Extract key phrases
    real_texts = df[df['label'] == 0]['text'].tolist()
    fake_texts = df[df['label'] == 1]['text'].tolist()
    
    print(f"\nTop phrases in real news:")
    real_phrases = extract_key_phrases(real_texts, 5)
    for phrase, freq in real_phrases:
        print(f"  '{phrase}': {freq}")
    
    print(f"\nTop phrases in fake news:")
    fake_phrases = extract_key_phrases(fake_texts, 5)
    for phrase, freq in fake_phrases:
        print(f"  '{phrase}': {freq}")
    
    # Demo prediction formatting
    print(f"\n=== Prediction Formatting Demo ===")
    sample_predictions = [(0, 0.85), (1, 0.92), (0, 0.67), (1, 0.78)]
    
    for pred, conf in sample_predictions:
        result = format_prediction_result(pred, conf)
        print(f"{result['icon']} {result['label']} - Confidence: {result['confidence_percentage']} ({result['confidence_level']})")