import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from gtts import gTTS
import json
from typing import List, Dict
import re

# Force PyTorch as the backend
import torch

# News Extraction
def extract_news(company: str, num_articles: int = 10) -> List[Dict]:
    """Extract news articles for a given company using Google search."""
    query = f"{company} news site:*.edu | site:*.org | site:*.gov -inurl:(signup | login)"
    url = f"https://www.google.com/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    articles = []
    for link in soup.select("a[href^='http']")[:num_articles]:
        href = link.get("href")
        if "google.com" not in href and href.endswith((".html", ".htm", "/")):
            try:
                article = scrape_article(href)
                if article:
                    articles.append(article)
            except Exception as e:
                print(f"Error scraping {href}: {e}")
    return articles[:num_articles]

def scrape_article(url: str) -> Dict:
    """Scrape title and summary from a news article."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    
    title = soup.find("h1") or soup.find("title")
    title = title.text.strip() if title else "No Title"
    
    paragraphs = soup.find_all("p")
    summary = " ".join(p.text.strip() for p in paragraphs[:3]) or "No summary available"
    summary = re.sub(r'\s+', ' ', summary)[:200] + "..."  # Limit summary length
    
    return {"title": title, "summary": summary, "url": url}

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
    for article in articles:
        article["sentiment"] = analyze_sentiment(article["summary"])
        article["topics"] = ["Company News"]  # Placeholder; enhance with topic modeling if needed

    comparative = comparative_analysis(articles)
    final_analysis = f"{company}'s latest news coverage is mostly {max(comparative['Sentiment Distribution'], key=comparative['Sentiment Distribution'].get)}."
    summary_text = " ".join([a["summary"] for a in articles])
    audio_file = generate_tts(final_analysis, "output.mp3")

    return {
        "Company": company,
        "Articles": articles,
        "Comparative Sentiment Score": comparative,
        "Final Sentiment Analysis": final_analysis,
        "Audio": audio_file
    }