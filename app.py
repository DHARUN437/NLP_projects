import streamlit as st
import joblib
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Load NLTK stopwords
nltk.download('stopwords')

# Load the model, vectorizer, and dataset
model = joblib.load('model.pkl')
vectorizer = joblib.load('count_v_res')
data = pd.read_csv('Restaurant_Reviews.tsv', sep='\t')  # Adjust path as needed

# Initialize PorterStemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Define a preprocessing function
def preprocess_text(text):
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

# Preprocess the dataset column if not already processed
if 'processed_text' not in data.columns:
    data['processed_text'] = data['Review'].apply(preprocess_text)

# Define a function to find the most similar review and get its sentiment
def find_most_similar_review(user_text):
    preprocessed_text = preprocess_text(user_text)
    
    # Vectorize the user input and dataset
    user_vector = vectorizer.transform([preprocessed_text])
    dataset_vectors = vectorizer.transform(data['processed_text'])
    
    # Compute cosine similarity
    similarities = cosine_similarity(user_vector, dataset_vectors)
    
    # Find the index of the most similar review
    most_similar_idx = similarities.argmax()
    
    # Get the most similar review's text and sentiment
    most_similar_review = data.iloc[most_similar_idx]
    return most_similar_review['processed_text'], most_similar_review['Liked']

# Define a function for sentiment analysis using VADER
def analyze_token_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)['compound']
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

# CSS for background colors, fonts, and themes
page_bg_style = """
<style>
body {
    background-color: #f0e6d6;
}
.stApp {
    background-color: #fdfcfb;
    color: #333;
    font-family: 'Arial', sans-serif;
}
.stTextInput>div>textarea {
    background-color: #f7f0ea;
    border: 1px solid #c5a880;
    color: #333;
    font-size: 16px;
}
.stButton>button {
    background-color: #ff6f61;
    border-radius: 10px;
    color: white;
}
.stMarkdown h1 {
    color: #ff6f61;
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
}
.stMarkdown p {
    color: #795548;
    font-size: 1.25rem;
}
</style>
"""

# Streamlit app
def main():
    # Apply background colors and custom CSS
    st.markdown(page_bg_style, unsafe_allow_html=True)
    
    # Title and intro
    st.markdown("# 🍽️ Restaurant Review Sentiment Analysis")
    st.markdown("Welcome! Analyze your restaurant reviews.")

    # Layout for input and results
    with st.form(key='nlpForm'):
        raw_text = st.text_area("📝 Enter Your Review Here")
        submit_button = st.form_submit_button(label='Analyze Review')


    if submit_button:
                    # Analyze sentiment using TextBlob
            blob_sentiment = TextBlob(raw_text).sentiment.polarity
            if blob_sentiment > 0:
                st.markdown("### Sentiment: Positive 😊")
            elif blob_sentiment < 0:
                st.markdown("### Sentiment: Negative 😠")
            else:
                st.markdown("### Sentiment: Neutral 😐")

    
            # Analyze sentiment using VADER
           # st.info("Comparing with similar reviews from our dataset...")
            # Find and display most similar review
            similar_review, similar_sentiment = find_most_similar_review(raw_text)
            #st.write(f"**Most Similar Review from Dataset:** {similar_review}")
            #st.write(f"**Sentiment from Dataset:** {'Positive' if similar_sentiment == 1 else 'Negative'}")

   

if __name__ == '__main__':
    main()