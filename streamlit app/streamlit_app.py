import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import numpy as np

# Load the TF-IDF Vectorizer
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    loaded_vect = pickle.load(f)

# Load the Keras model
model_path = 'models/toxic_comment_model.keras'
loaded_model = load_model(model_path, custom_objects=None)

# Streamlit app
st.title('Toxic Comment Classification')

# Input text area for user to enter comments
user_input = st.text_area("Enter comments to classify:", "Type your comments here...")

if st.button('Classify'):
    # Process input
    if user_input:
        # Split the input text into separate comments if there are multiple
        comments = [comment.strip() for comment in user_input.split('\n') if comment.strip()]
        
        if comments:
            # Preprocess comments using the loaded TF-IDF Vectorizer
            comments_vectors = loaded_vect.transform(comments)

            # Predict using the loaded model
            predictions = (loaded_model.predict(comments_vectors) > 0.5).astype(int)
            
            # Display predictions
            for comment, prediction in zip(comments, predictions):
                toxicity = 'Toxic' if bool(prediction[0]) else 'Not Toxic'
                st.write(f'Comment: {comment}')
                st.write(f'Toxic: {toxicity}')
        else:
            st.write("Please enter at least one comment.")
    else:
        st.write("Please enter some text.")
