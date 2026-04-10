import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.data_preprocessing import clean_text
from src.feature_engineering import get_vectorizer
from src.model_evaluation import evaluate_model
from sklearn.model_selection import train_test_split

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

def get_models():
    """Get dictionary of ML models for training."""
    return {
        "logistic": LogisticRegression(max_iter=1000),
        "svm": LinearSVC(max_iter=1000)
    }

# Load data (sample for faster training)
print("Loading data...")
df = pd.read_csv("data/movie_reviews.csv")
df = df.sample(n=5000, random_state=42)  # Use smaller sample for faster training
print(f"Using {len(df)} samples")

# Preprocess
print("Preprocessing text...")
df['review'] = df['review'].apply(clean_text)

X = df['review']
y = df['sentiment']

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Vectorize
print("Vectorizing text...")
vectorizer = get_vectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train models
print("Training models...")
models = get_models()

best_model = None
best_score = 0

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    metrics = evaluate_model(y_test, preds)
    print(f"{name}: {metrics}")

    if metrics['f1'] > best_score:
        best_score = metrics['f1']
        best_model = model

# Save model
print("Saving best model...")
pickle.dump(best_model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
pickle.dump(label_encoder, open("models/label_encoder.pkl", "wb"))

print("✅ Best model and encoders saved!")