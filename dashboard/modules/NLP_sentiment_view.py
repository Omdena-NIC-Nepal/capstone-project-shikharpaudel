import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from modules.sentiment_utils import analyze_sentiments

def show_sentiment_analysis():
    st.title("Sentiment Analysis of Climate-related Text")

    user_input = st.text_area("Enter climate-related text (one per line):", height=200)
    if st.button("Analyze Sentiment"):
        text_list = [line.strip() for line in user_input.strip().split('\n') if line.strip()]
        
        if text_list:
            result_df = analyze_sentiments(text_list)

            st.subheader("Sentiment Results")
            st.dataframe(result_df)

            # Bar chart
            sentiment_counts = result_df['Sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']

            fig, ax = plt.subplots()
            sns.barplot(data=sentiment_counts, x='Sentiment', y='Count', palette='pastel', ax=ax)
            ax.set_title("Sentiment Score Distribution")
            st.pyplot(fig)
        else:
            st.warning("Please enter at least one line of text.")
