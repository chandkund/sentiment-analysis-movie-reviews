import streamlit as st
import matplotlib.pyplot as plt
from utils import load_model, predict

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Load model
model, vectorizer = load_model()

# Header
st.markdown("""
# 🎬  Movie Sentiment Analyzer
### Analyze movie reviews using Machine Learning
""")

# Sidebar
st.sidebar.header("⚙️ Options")
use_example = st.sidebar.button("Use Example Review")

# Example text
if use_example:
    review = "This movie was absolutely amazing, शानदार and full of emotions!"
else:
    review = ""

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    review = st.text_area(
        "✍️ Enter your movie review:",
        value=review,
        height=200,
        placeholder="Type your review here..."
    )

    analyze = st.button("🚀 Analyze Sentiment")

with col2:
    st.markdown("### 📊 About")
    st.info("""
    This app uses Machine Learning to classify reviews as:
    - 😊 Positive  
    - 😡 Negative  
    """)

# Prediction
if analyze:
    if review.strip() == "":
        st.warning("⚠️ Please enter a review")
    else:
        sentiment, confidence = predict(review, model, vectorizer)

        st.markdown("## 🔍 Result")

        col3, col4 = st.columns(2)

        with col3:
            if sentiment == 1:
                st.success(f"😊 Positive Sentiment")
            else:
                st.error(f"😡 Negative Sentiment")

            st.metric("Confidence", f"{confidence:.2f}")

            # Progress bar
            st.progress(int(confidence * 100))

        with col4:
            # Chart
            labels = ["Negative", "Positive"]
            values = [1 - confidence, confidence]

            fig, ax = plt.subplots()
            ax.bar(labels, values)
            ax.set_title("Prediction Confidence")
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Chandan | ML Project")