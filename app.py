import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
pipe_lr = tf.keras.models.load_model('./models/saved_model/my_model')
tokenizer = joblib.load("./models/tokenizer1.0.1.pkl")

sentiment_labels = ['negative', 'positive']

# Function that gives you the prediction 'positive' or 'negative'
def predict_emotions(docx):
    sequences = tokenizer.texts_to_sequences([docx])
    padded_sequences = pad_sequences(sequences, maxlen=100)
    results = pipe_lr.predict(padded_sequences)
    # Check if predictions exist
    if results.shape[0] > 0:
        # Classify the sentiment based on the prediction probabilities
        classified_sentiment = sentiment_labels[np.argmax(results[0])]
    else:
        # Unable to classify se ntiment
        classified_sentiment = "Unknown"
    return classified_sentiment

# Function that gives you the prediction probability for 'positive' and 'negative'

def get_prediction_proba(docx):
    sequences = tokenizer.texts_to_sequences([docx])
    padded_sequences = pad_sequences(sequences, maxlen=100)
    results = pipe_lr.predict(padded_sequences)
    class_probabilities = results
    return class_probabilities

emotions_emoji_dict = {"positive": "🤗", "negative": "😠"}

# Main function
def main():
    st.title("Emotion Classifier App")
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("_Home-Emotion In Text_")

        with st.form(key="emotion_clf_form"):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label="Analyze")

        if submit_text:
            col1, col2 = st.columns(2)

            # Apply functions here
            prediction = predict_emotions(raw_text)
            emoji_icon = emotions_emoji_dict[prediction]
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                st.write("{}: {}".format(prediction, emoji_icon))
            with col2:
                st.success("Prediction Probability")
                class_labels = ['negative', 'positive']
                proba_df = pd.DataFrame(probability, columns=class_labels)
                st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ['emotions', 'probability']

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x="emotions", y=alt.Y(
                    "probability", axis=alt.Axis(format='%')), color='emotions')
                st.altair_chart(fig, use_container_width=True)
    else:
        st.subheader("About")


if __name__ == '__main__':
    main()
