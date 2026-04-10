# 🎬 Sentiment Analysis - Movie Reviews

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chandkund-sentiment-analysis-movie-reviews-appmain-utlo7o.streamlit.app/)

**🚀 Live Demo**: [https://chandkund-sentiment-analysis-movie-reviews-appmain-utlo7o.streamlit.app/](https://chandkund-sentiment-analysis-movie-reviews-appmain-utlo7o.streamlit.app/)

A comprehensive end-to-end machine learning project for sentiment analysis of movie reviews. Built with FastAPI backend, Streamlit frontend, and scikit-learn models achieving 85.6% accuracy.

## ✨ Features

- 🎭 **Real-time Sentiment Analysis** - Classify movie reviews as Positive or Negative
- 📊 **Interactive Dashboard** - Beautiful Streamlit UI with confidence visualizations
- 🔌 **REST API** - FastAPI backend with automatic documentation
- 🤖 **ML Pipeline** - Complete workflow from data preprocessing to model deployment
- 📈 **Model Performance** - 85.6% accuracy on test dataset
- 🐳 **Docker Support** - Containerized deployment ready
- ☁️ **Cloud Deployment** - Live on Streamlit Cloud

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **ML**: scikit-learn, NLTK, pandas, numpy
- **Visualization**: Matplotlib
- **Deployment**: Docker, Streamlit Cloud
- **Version Control**: Git, GitHub

## 🚀 Quick Start

### Online Demo
Visit the [live application](https://chandkund-sentiment-analysis-movie-reviews-appmain-utlo7o.streamlit.app/) to try it out instantly!

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/chandkund/sentiment-analysis-movie-reviews.git
   cd sentiment-analysis-movie-reviews
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**

   **Option A: Run both services**
   ```bash
   # Terminal 1 - Start FastAPI backend
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

   # Terminal 2 - Start Streamlit frontend
   python -m streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
   ```

   **Option B: Run with Docker**
   ```bash
   docker build -t sentiment-analysis .
   docker run -p 8000:8000 -p 8501:8501 sentiment-analysis
   ```

5. **Access the application**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## 📊 Usage

### Web Interface
1. Open http://localhost:8501 in your browser
2. Enter a movie review in the text area
3. Click "🚀 Analyze Sentiment"
4. View the prediction with confidence score and chart

### API Usage
```python
import requests

# Single prediction
response = requests.post("http://localhost:8000/predict",
    json={"text": "This movie was absolutely amazing!"})

print(response.json())
# {"sentiment": 1, "confidence": 0.92, "label": "Positive"}
```

## 🏗️ Project Structure

```
sentiment-analysis-movie-reviews/
├── app/                    # Streamlit web application
│   ├── components/         # Reusable UI components
│   ├── main.py            # Main Streamlit app
│   └── utils.py           # Helper functions
├── backend/                # FastAPI backend
│   ├── main.py            # FastAPI application
│   ├── routes/            # API endpoints
│   ├── schemas/           # Pydantic models
│   └── services/          # Business logic
├── models/                 # Trained ML models
│   ├── model.pkl          # LogisticRegression model
│   ├── vectorizer.pkl     # TF-IDF vectorizer
│   └── label_encoder.pkl  # Label encoder
├── src/                    # ML pipeline source code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── inference.py
├── data/                   # Dataset and raw data
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── train.py              # Model training script
└── README.md
```

## 🔬 Model Details

- **Algorithm**: LogisticRegression
- **Features**: TF-IDF vectorization (max_features=5000)
- **Preprocessing**: Text cleaning, stopword removal, stemming
- **Training Data**: 5,000+ movie reviews
- **Accuracy**: 85.6% on test set
- **Classes**: Binary classification (Positive/Negative)

## 📚 API Documentation

### Endpoints

- `GET /` - Health check
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions

### Request/Response Examples

**Single Prediction:**
```json
POST /predict
{
  "text": "This movie was fantastic!"
}

Response:
{
  "sentiment": 1,
  "confidence": 0.89,
  "label": "Positive"
}
```

**Batch Prediction:**
```json
POST /predict/batch
{
  "texts": [
    "Amazing movie!",
    "Terrible plot"
  ]
}

Response:
{
  "predictions": [
    {"sentiment": 1, "confidence": 0.92, "label": "Positive"},
    {"sentiment": 0, "confidence": 0.78, "label": "Negative"}
  ]
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Movie review dataset from [source]
- Built with ❤️ by Chandan Kundan
- Inspired by various sentiment analysis tutorials

---

**⭐ Star this repo if you found it helpful!**

**📧 Contact**: chandanstudy02@gmail.com
