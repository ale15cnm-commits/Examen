import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")


st.write("# Predicción de categoría de Premio Nobel")

st.header("Texto")


def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"[^a-z\s]", "", texto)

    stop_words = set(stopwords.words("english"))

    texto = " ".join(
        palabra for palabra in texto.split()
        if palabra not in stop_words
    )

    return texto


texto = st.text_input("Introduce el texto a evaluar")


nobel = pd.read_csv("nobel.csv", encoding="utf-8")

X = nobel["Motivation"].apply(limpiar_texto)
y = nobel["Category"]


vect = CountVectorizer()
X_dtm = vect.fit_transform(X)


nb = MultinomialNB()
nb.fit(X_dtm, y)


if texto:
    texto_limpio = limpiar_texto(texto)
    texto_dtm = vect.transform([texto_limpio])

    prediction = nb.predict(texto_dtm)

    st.subheader("Predicción")
    st.write(prediction[0])
