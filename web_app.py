import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Function to load the model and tokenizer
@st.cache(allow_output_mutation=True)  # Cache the model and tokenizer to avoid reloading
def load_model_and_tokenizer():
    try:
        # Load your model (If you saved it previously)
        model = load_model("C:/Users/KARTHIK M/Documents/College works/M.Sc Data Science/deep learning/holiday project/rnn_model.keras")
        
        # Load the tokenizer (if saved separately, adjust paths)
        with open("C:/Users/KARTHIK M/Documents/College works/M.Sc Data Science/deep learning/holiday project/tokenizer.pickle", "rb") as handle:
            tokenizer = pickle.load(handle)
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Function to preprocess the input text
def preprocess_text(text):
    if text:
        # Tokenize and pad the input text
        text_seq = tokenizer.texts_to_sequences([text])
        text_padded = pad_sequences(text_seq, maxlen=200)
        
        # Check if the result is valid
        if text_padded is not None and text_padded.size > 0:
            return text_padded
        else:
            st.warning("Preprocessing failed. Please check the input text.")
            return None
    return None

# Streamlit app layout
st.title("Sentiment Analysis with LSTM")
st.markdown("Enter a sentence to analyze its sentiment")

# Input field for the sentence
user_input = st.text_area("Enter Text", "Type something here")

# Button to trigger sentiment prediction
if st.button("Analyze Sentiment"):
    if user_input:
        if model and tokenizer:  # Ensure model and tokenizer are loaded
            try:
                # Preprocess the input text
                processed_text = preprocess_text(user_input)
                
                # Ensure that preprocessing returns a valid padded sequence
                if processed_text is not None:
                    # Make a prediction
                    prediction = model.predict(processed_text)

                    # Check if prediction is valid
                    if prediction is None or prediction.size == 0:
                        st.error("Model prediction failed. Please check the model.")
                    else:
                        # Interpret the prediction
                        sentiment = "Positive" if prediction > 0.5 else "Negative"
                        
                        # Display the result
                        st.subheader(f"Predicted Sentiment: {sentiment}")
                        st.write(f"Prediction Probability: {prediction[0][0]:.4f}")
                else:
                    st.warning("Unable to preprocess the text. Please check the input.")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Model or tokenizer not loaded correctly. Please try again later.")
    else:
        st.write("Please enter some text for analysis.")
