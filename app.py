import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only display errors

import streamlit as st
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
# Load the trained model and tokenizer
loaded_model = pickle.load(open("trainedmodel.sav", "rb"))
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))  # Assuming tokenizer was saved as a pickle file
#tokenizer = pickle.load('tokenizer.pkl')
max_len = 100  # Update this to the max_len used in training

# Title of the Streamlit app
st.title("Sentiment Analysis")

# Text input for the user to type a sentence
test_sentence = st.text_input("Enter a sentence to test sentiment:")

# Check if the sentence is not empty
if st.button("Submit"):
    # Tokenize and pad the test sentence
    test_sentence_seq = tokenizer.texts_to_sequences([test_sentence])
    test_sentence_padded = pad_sequences(test_sentence_seq, maxlen=max_len)
    
    # Make a prediction
    prediction = loaded_model.predict(test_sentence_padded)
    # Interpret and display the prediction
    if prediction[0][0] > 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    # Display results
    st.write("Predicted Sentiment:", sentiment)
    st.write("Prediction Probability:", prediction[0][0])
