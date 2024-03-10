import streamlit as st
import joblib
import pandas as pd

# Load CSS styles from assets folder
with open('assests/style.css', 'r') as f:
    css = f.read()
st.markdown('<style>{}</style>'.format(css), unsafe_allow_html=True)

# Load the trained model
trained_model = joblib.load('trained_model.pkl')

# Define the class labels
labels = ['Commenting', 'Ogling/Facial Expressions/Staring', 'Touching /Groping']

# Function to predict probabilities for each class
def predict_probabilities(text, model):
    probabilities = model.predict_proba([text])[0]
    return probabilities

# Streamlit app layout and functionality
st.title('Explainable Sexual Harassment Categorization')

# Text input widget
text_input = st.text_input('Enter the text:', '')

# Button to trigger prediction
if st.button('Predict'):
    if text_input:
        # Predict probabilities
        probabilities = predict_probabilities(text_input, trained_model)
        
        # Display probabilities with labels
        st.write('Predicted Probabilities:')
        for label, probability in zip(labels, probabilities):
            st.write(f'{label}: {probability:.2f}')
        
        # Find the category with the highest probability
        max_prob_idx = probabilities.argmax()
        predicted_category = labels[max_prob_idx]

        # Display predicted category with highest probability
        st.write(f'Predicted Category with Highest Probability: {predicted_category}')

    else:
        st.warning('Please enter some text.')

# Additional styling
st.markdown("---")
st.markdown("#### About")
st.markdown("This Streamlit app is made with love by the team Deception Detectors.")
