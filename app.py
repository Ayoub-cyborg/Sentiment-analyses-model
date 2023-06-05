import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Load the model and tokenizer
pipe_lr = tf.keras.models.load_model('./models/saved_model/my_model')
tokenizer = joblib.load("./models/tokenizer1.0.1.pkl")

sentiment_labels = ['negative', 'positive']
emotions_emoji_dict = {"positive": "🤗", "negative": "😠"}


# Function that gives you the prediction 'positive' or 'negative'
def predict_emotions(docx):
    sequences = tokenizer.texts_to_sequences([docx])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
    results = pipe_lr.predict(padded_sequences)
    # Check if predictions exist
    if results.shape[0] > 0:
        # Classify the sentiment based on the prediction probabilities
        classified_sentiment = sentiment_labels[np.argmax(results)]
    else:
        # Unable to classify se ntiment
        classified_sentiment = "Unknown"
    return classified_sentiment

# Function that gives you the prediction probability for 'positive' and 'negative'

def get_prediction_proba(docx):
    sequences = tokenizer.texts_to_sequences([docx])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
    results = pipe_lr.predict(padded_sequences)
    class_probabilities = results
    return class_probabilities


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
            
        with st.form(key="emotion_form"):
            file = st.file_uploader("Upload CSV", type="csv")
            submit_file = st.form_submit_button(label="Analyze")
            
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
           
            
            # add to analyze csv file and show result in fig
        if submit_file and file is not None:
            cols = st.columns(1)
            with cols[0]:
                df = pd.read_csv(file)
                df['Prediction'] = df['Text'].apply(predict_emotions)
                df['Emoji'] = df['Prediction'].map(emotions_emoji_dict)

                # Count positive and negative predictions
                prediction_counts = df['Prediction'].value_counts().reset_index()
                prediction_counts.columns = ['Emotion', 'Count']
                total_count = prediction_counts['Count'].sum()
                prediction_counts['Percentage'] = prediction_counts['Count'] / total_count * 100

                st.write(prediction_counts)  # Ensure the DataFrame has data

                fig = alt.Chart(prediction_counts).mark_bar().encode(
                    x='Emotion:N',
                    y=alt.Y('Percentage:Q', axis=alt.Axis(format='.0f', title='Percentage'), scale=alt.Scale(domain=(0, 100))),
                    color='Emotion:N',
                    tooltip=['Emotion', 'Count', alt.Tooltip('Percentage:Q', format='.2f')]
                ).properties(
                    width=500,
                    height=300
                ).interactive()

                st.altair_chart(fig, use_container_width=True)

    else:
        st.subheader("About")


if __name__ == '__main__':
    main()