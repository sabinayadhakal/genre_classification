import streamlit as st
import pickle

with open('genre_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('overview_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('genre_binarizer.pkl', 'rb') as f:
    mlb = pickle.load(f)

st.title("🎬 Movie Genre Predictor")
st.markdown("Enter a short overview of your movie to predict its genres.")

overview_input = st.text_area("Movie Overview", placeholder="A group of intergalactic criminals must pull together to stop a fanatical warrior...")

if st.button("Predict Genre"):
    if overview_input.strip() == "":
        st.warning("Please enter a movie overview.")
    else:
        X_input = vectorizer.transform([overview_input]).toarray()

        y_pred = model.predict(X_input)

        predicted_genres = mlb.inverse_transform(y_pred)

        if predicted_genres and predicted_genres[0]:
            st.success("Predicted Genres:")
            for genre in predicted_genres[0]:
                st.markdown(f"- {genre}")
        else:
            st.info("No genre could be confidently predicted.")
