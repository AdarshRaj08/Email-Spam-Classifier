import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB


# Load the trained pipeline
pipeline = joblib.load('abc.pkl')

# Function to predict class
def predict_class(text):
    prediction = pipeline.predict([text])
    return prediction[0]

# Streamlit UI

st.markdown("""
    <div style='background-color:#f9f9f9;padding:10px;border-radius:10px'>
    <h2 style='text-align:center;color:#dc143c;'>Welcome to Email Spam Classifier</h2>
    <p style='text-align:justify;color:#5f6368;'>Email Spam Classifier" is a machine learning project aimed at automatically detecting and classifying spam emails. Using techniques like TF-IDF vectorization and BernoulliNB , the system learns to distinguish between legitimate emails and spam messages. With high accuracy, it sorts incoming emails, saving users' time and enhancing their email security..</p>
    </div>
    """, unsafe_allow_html=True)

st.write('## Classify Email')
email = st.text_area('Enter the email here:')
if st.button('Submit'):
    if email:
        prediction = predict_class(email)
        if prediction == 1:
            st.success('The email is Spam.')
        else:
            st.error('Not Spam')
    else:
        st.warning('Please enter email to classify.')
