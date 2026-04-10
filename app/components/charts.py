import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd


def render_charts():
    """
    Render analytics charts and visualizations.
    """
    col1, col2 = st.columns(2)
    
    # Generate sample data
    dates = pd.date_range(start='2026-04-01', periods=30, freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'positive': [30 + i % 20 for i in range(30)],
        'negative': [15 + i % 15 for i in range(30)],
        'neutral': [10 + i % 8 for i in range(30)]
    })
    
    with col1:
        st.subheader("Sentiment Trend")
        fig = px.line(
            sample_data,
            x='date',
            y=['positive', 'negative', 'neutral'],
            labels={'value': 'Count', 'date': 'Date'},
            title='Sentiment Distribution Over Time'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sentiment Distribution")
        sentiment_counts = {
            'Positive': 65,
            'Negative': 20,
            'Neutral': 15
        }
        fig = go.Figure(data=[
            go.Pie(
                labels=list(sentiment_counts.keys()),
                values=list(sentiment_counts.values()),
                marker=dict(colors=['#2ecc71', '#e74c3c', '#95a5a6'])
            )
        ])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", 100)
    
    with col2:
        st.metric("Avg Confidence", "87%")
    
    with col3:
        st.metric("Most Common", "Positive")
    
    with col4:
        st.metric("Processing Time", "0.42s")
