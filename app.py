import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize the PorterStemmer
ps = PorterStemmer()

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))  # Load the TfidfVectorizer
model = pickle.load(open('model.pkl', 'rb'))       # Load the MultinomialNB model

# Function to preprocess the text
def transform_text(text):
    text = text.lower()  # Lowercase the text
    text = nltk.word_tokenize(text)  # Tokenization

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    # Remove all non-alphanumeric characters

    text = y[:]  # Clone the cleaned list
    y.clear()  # Empty the list

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    # Remove stopwords and punctuation

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # Stemming

    return " ".join(y)  # Return the processed text as a string

# Streamlit app title
st.title('Email/SMS Spam Classifier')

# Input field
input_sms = st.text_area('Enter the message')

if st.button('Predict'):
    # 1. Preprocess the input
    transform_sms = transform_text(input_sms)

    # 2. Vectorize the input
    vector_input = tfidf.transform([transform_sms])

    # 3. Predict using the model
    result = model.predict(vector_input)[0]

    # 4. Display the result
    if result == 1:
        st.header('SPAM')
    else:
        st.header('NOT SPAM')
