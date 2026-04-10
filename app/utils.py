import pickle
import os

def load_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "model.pkl")
    vectorizer_path = os.path.join(base_dir, "models", "vectorizer.pkl")
    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
    return model, vectorizer

def predict(text, model, vectorizer):
    vec = vectorizer.transform([text])
    
    try:
        prob = model.predict_proba(vec)[0]
        confidence = max(prob)
    except:
        confidence = 0.9  # fallback for SVM

    prediction = model.predict(vec)[0]
    return prediction, confidence