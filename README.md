# News Summarization & TTS Application

A Python-based application that fetches news articles for a given company, performs sentiment analysis, generates a comparative report, and provides a text-to-speech (TTS) output in Hindi. The application features a Gradio web interface and a FastAPI endpoint for easy interaction.

**Features**

1. **News Extraction:** Fetches the latest news articles about a company using the NewsAPI.
2. **Sentiment Analysis:** Analyzes the sentiment of article summaries using a pre-trained transformer model.
3. **Comparative Analysis:** Compares sentiment and topics across articles.
4. **Text-to-Speech (TTS):** Generates an audio summary in Hindi using gTTS.
5. **Web Interface:** Interactive UI built with Gradio.
6. **API:** FastAPI endpoint for programmatic access.

**Prerequisites**
1. Python 3.8 or higher
2. A NewsAPI key (sign up at NewsAPI.org to get one)
   
**Installation**

**1. Clone the Repository**

```bash
   # Clone the repository
   git clone https://github.com/your-username/news-summarizer.git

   # Navigate into the directory
   cd news-summarizer
```
**2. Install Dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure NewsAPI Key**

**Replace the api_key in utils.py with your own NewsAPI key:**

**But i already mentioned my api**

```bash
newsapi = NewsApiClient(api_key="YOUR_API_KEY_HERE")
```
Save and exit.

**Usage**

**Running the Gradio Interface**

**Launch the web interface to analyze companies interactively:**

```bash
python app.py
```

1. Open your browser and go to http://0.0.0.0:7860.
2. Enter a company name (e.g., "Tesla") and click "Analyze" to see the sentiment report and hear the Hindi TTS output.


**Using the FastAPI Endpoint**

**Start the API server:**

```bash
uvicorn api:app --reload
```
1. Access the endpoint at http://localhost:8000/analyze/{company} (e.g., http://localhost:8000/analyze/Tesla).
2. The API returns a JSON response with the analysis results.

**Example Output**

For input "Tesla":

1. Sentiment Report: A JSON-formatted report with article details, sentiment distribution, and comparative analysis.
2. Audio: An MP3 file with a Hindi summary (e.g., "Tesla की नवीनतम समाचार कवरेज ज्यादातर सकारात्मक है।").


**Project Structure**
```bash
news-summarizer/
├── api.py              # FastAPI endpoint
├── app.py              # Gradio interface
├── utils.py            # Core logic (news extraction, sentiment analysis, TTS)
├── requirements.txt    # Dependencies
└── README.md           # This file
```
## Deployment by using Hugging space
1. Create a new space
2. upload your files
3. Build the application
   
**Check out my Link: https://scary29-summarizer.hf.space/**
```bash


https://scary29-summarizer.hf.space/
```
## Dependencies
- `requests`: For HTTP requests.
- `beautifulsoup4`: For web scraping (not used in current code but included).
- `transformers` & `torch`: For sentiment analysis and summarization.
- `gtts`: For text-to-speech generation.
- `gradio`: For the web interface.
- `fastapi` & `uvicorn`: For the API server.
- `newsapi-python`: For fetching news articles.
- `numpy`: For numerical operations.

See `requirements.txt` for version details.

## Notes
- The application uses `distilbert-base-uncased-finetuned-sst-2-english` for sentiment analysis and `facebook/bart-large-cnn` for summarization.
- The TTS output is in Hindi, but the sentiment report is in English.
- If no articles are found, an error message is returned.

## Limitations
- Requires an active internet connection for NewsAPI and model downloads.
- The NewsAPI free tier has rate limits (e.g., 100 requests/day).
- Sentiment analysis accuracy depends on the pre-trained model and article summaries.

## Contributing
Feel free to fork this repository, submit issues, or create pull requests to improve the project!
