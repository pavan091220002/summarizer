import requests
from newsapi import NewsApiClient
from transformers import pipeline
from gtts import gTTS
import json
from typing import List, Dict
import torch

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key="6c81b327f6054da4adc9360a13844f75")

# Topic extraction function (title content only)
def extract_topics(text: str) -> List[str]:
    """Use the full title content as a single topic."""
    return [text] if text else ["General"]

# News Extraction using NewsAPI
def extract_news(company: str, num_articles: int = 10) -> List[Dict]:
    """Extract news articles for a given company using NewsAPI."""
    try:
        response = newsapi.get_everything(
            q=company,
            language="en",
            sort_by="publishedAt",
            page_size=num_articles
        )
        articles = response["articles"]
        return [
            {
                "title": article["title"],
                "summary": article["description"] or "No summary available",
                "url": article["url"],
                "topics": extract_topics(article["title"])  # Use title content only
            }
            for article in articles[:num_articles]
        ]
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

# Sentiment Analysis (PyTorch backend)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")

def analyze_sentiment(text: str) -> str:
    """Perform sentiment analysis on text."""
    result = sentiment_analyzer(text)[0]
    label = result["label"].capitalize()
    return "Neutral" if result["score"] < 0.6 else label

# Comparative Analysis
def comparative_analysis(articles: List[Dict]) -> Dict:
    """Compare sentiment and topics across articles."""
    sentiment_dist = {"Positive": 0, "Negative": 0, "Neutral": 0}
    topics = set()
    comparisons = []

    for article in articles:
        sentiment_dist[article["sentiment"]] += 1
        topics.update(article["topics"])

    for i in range(len(articles) - 1):
        comp = {
            "Comparison": f"Article {i+1} ({articles[i]['sentiment']}) vs Article {i+2} ({articles[i+1]['sentiment']})",
            "Impact": f"Article {i+1} focuses on {', '.join(articles[i]['topics'])}, while Article {i+2} highlights {', '.join(articles[i+1]['topics'])}."
        }
        comparisons.append(comp)

    return {
        "Sentiment Distribution": sentiment_dist,
        "Coverage Differences": comparisons,
        "Topic Overlap": {"Common Topics": list(topics)}
    }

# TTS Generation
def generate_tts(text: str, filename: str = "output.mp3") -> str:
    """Generate Hindi TTS from text."""
    tts = gTTS(text=text, lang="hi", slow=False)
    tts.save(filename)
    return filename

def process_company(company: str) -> Dict:
    """Main processing function."""
    articles = extract_news(company)
    if not articles:
        return {"error": "No articles found for this company."}

    for article in articles:
        article["sentiment"] = analyze_sentiment(article["summary"])
    
    comparative = comparative_analysis(articles)
    final_analysis_en = f"{company}'s latest news coverage is mostly {max(comparative['Sentiment Distribution'], key=comparative['Sentiment Distribution'].get)}."
    final_analysis_hi = f"{company} की नवीनतम समाचार कवरेज ज्यादातर {translate_sentiment(comparative['Sentiment Distribution'])} है।"
    audio_file = generate_tts(final_analysis_hi, "output.mp3")

    return {
        "Company": company,
        "Articles": articles,
        "Comparative Sentiment Score": comparative,
        "Final Sentiment Analysis": final_analysis_en,
        "Audio": audio_file
    }

def translate_sentiment(sentiment_dist: Dict) -> str:
    """Translate the dominant sentiment to Hindi."""
    dominant = max(sentiment_dist, key=sentiment_dist.get)
    translation = {"Positive": "सकारात्मक", "Negative": "नकारात्मक", "Neutral": "तटस्थ"}
    return translation.get(dominant, "तटस्थ")
