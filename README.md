# Sentiment Analysis - Movie Reviews

A comprehensive end-to-end machine learning project for sentiment analysis of movie reviews with a FastAPI backend and Streamlit UI.

## Project Structure

```
sentiment-analysis-movie-reviews/
├── app/                    # UI Layer (Streamlit)
├── backend/                # API Layer (FastAPI)
├── models/                 # Trained ML models
├── notebooks/              # EDA + experiments
├── src/                    # Core ML pipeline
├── data/                   # Dataset (optional or sample)
├── tests/                  # Unit testing
├── config/                 # Config files
├── artifacts/              # Saved outputs (vectorizer, model)
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the backend:
   ```bash
   uvicorn backend.main:app --reload
   ```
4. Run the frontend:
   ```bash
   streamlit run app/main.py
   ```

## Features

- Sentiment classification (Positive/Negative/Neutral)
- REST API endpoints
- Interactive Streamlit dashboard
- Model evaluation metrics
- Data preprocessing pipeline

## License

MIT
