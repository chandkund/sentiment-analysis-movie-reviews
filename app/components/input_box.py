import streamlit as st


def render_input_box():
    """
    Render the input text box for sentiment analysis.
    
    Returns:
        str: The review text entered by the user
    """
    review_text = st.text_area(
        "Enter a movie review:",
        placeholder="Write your movie review here...",
        height=200,
        key="review_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        analyze_button = st.button("🔍 Analyze", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    if analyze_button and not review_text:
        st.warning("⚠️ Please enter a review to analyze")
        return None
    
    return review_text if analyze_button else None
