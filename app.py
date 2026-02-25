# ===============================================================
# 🧠 AI Universal Studio — Versão PRO (Interview Edition)
# ===============================================================
# Multimodal: Texto + Imagem
# NLP: TF-IDF (estável, explicável e cloud-safe)
# ===============================================================

import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
import torch

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from deep_translator import GoogleTranslator

# ===============================================================
# ⚙️ Configuração da Página
# ===============================================================
st.set_page_config(
    page_title="AI Universal Studio — Interview Edition",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI Universal Studio — Interview Edition")
st.info(
    """
Projeto demonstrativo de **IA aplicada**, combinando **texto + imagem**  
com foco em **estabilidade, explicabilidade e deploy em nuvem**.
"""
)

# ===============================================================
# 📦 Modelo de Imagem (BLIP)
# ===============================================================
@st.cache_resource
def load_caption_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model.eval()
    return processor, model


caption_processor, caption_model = load_caption_model()

# ===============================================================
# 💾 Funções auxiliares
# ===============================================================
def gerar_caption_imagem(image):
    inputs = caption_processor(image, return_tensors="pt")
    with torch.no_grad():
        out = caption_model.generate(**inputs)

    caption_en = caption_processor.decode(
        out[0], skip_special_tokens=True
    )

    caption_pt = GoogleTranslator(
        source="en", target="pt"
    ).translate(caption_en)

    return caption_pt


def salvar_modelo(modelo, encoder, vectorizer):
    joblib.dump(modelo, "modelo_rf.pkl")
    joblib.dump(encoder, "encoder.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")


def carregar_modelo():
    if all(
        os.path.exists(f)
        for f in ["modelo_rf.pkl", "encoder.pkl", "vectorizer.pkl"]
    ):
        return (
            joblib.load("modelo_rf.pkl"),
            joblib.load("encoder.pkl"),
            joblib.load("vectorizer.pkl"),
        )
    return None, None, None


# ===============================================================
# 🔁 Sessão
# ===============================================================
for var, default in {
    "textos": [],
    "labels": [],
    "modelo": None,
    "encoder": None,
    "vectorizer": None,
}.items():
    if var not in st.session_state:
        st.session_state[var] = default

# ===============================================================
# 🧭 Abas
# ===============================================================
tabs = st.tabs(
    [
        "🧩 Base de Treinamento",
        "⚙️ Treinar Modelo",
        "🔮 Previsão",
    ]
)

# ===============================================================
# 1️⃣ BASE DE TREINAMENTO
# ===============================================================
with tabs[0]:
    st.header("🧩 Base de Treinamento")
    st.write("Insira exemplos de texto e classe para ensinar o modelo.")

    entradas = []
    for i in range(3):
        col1, col2 = st.columns([3, 1])
        texto = col1.text_input(f"Texto {i+1}", key=f"t_{i}")
        classe = col2.selectbox(
            "Classe", ["Baixo", "Moderado", "Alto"], key=f"c_{i}"
        )

        if texto:
            entradas.append({"texto": texto, "classe": classe})

    if entradas and st.button("💾 Salvar base"):
        st.session_state.textos = [e["texto"] for e in entradas]
        st.session_state.labels = [e["classe"] for e in entradas]
        st.success("Base salva!")
        st.dataframe(pd.DataFrame(entradas), use_container_width=True)

# ===============================================================
# 2️⃣ TREINAR MODELO
# ===============================================================
with tabs[1]:
    st.header("⚙️ Treinar Modelo")

    if not st.session_state.textos:
        st.warning("Crie a base primeiro.")
    else:
        if st.button("🚀 Treinar"):
            vectorizer = TfidfVectorizer(
                max_features=500,
                ngram_range=(1, 2),
                stop_words="portuguese",
            )

            X = vectorizer.fit_transform(
                st.session_state.textos
            ).toarray()

            encoder = LabelEncoder()
            y = encoder.fit_transform(st.session_state.labels)

            modelo = RandomForestClassifier(random_state=42)
            modelo.fit(X, y)

            salvar_modelo(modelo, encoder, vectorizer)

            st.session_state.modelo = modelo
            st.session_state.encoder = encoder
            st.session_state.vectorizer = vectorizer

            st.success("Modelo treinado com sucesso!")

        modelo, enc, vec = carregar_modelo()
        if modelo:
            st.session_state.modelo = modelo
            st.session_state.encoder = enc
            st.session_state.vectorizer = vec
            st.info("Modelo salvo carregado automaticamente.")

# ===============================================================
# 3️⃣ PREVISÃO MULTIMODAL
# ===============================================================
with tabs[2]:
    st.header("🔮 Previsão")

    img = st.file_uploader("Imagem (opcional)", type=["jpg", "png", "jpeg"])
    texto = st.text_area("Texto descritivo")

    desc_img = ""
    if img:
        image = Image.open(img).convert("RGB")
        st.image(image, use_container_width=True)

        with st.spinner("Analisando imagem..."):
            desc_img = gerar_caption_imagem(image)
            st.caption(f"Descrição da imagem: {desc_img}")

    entrada = f"{desc_img} {texto}".strip()
    st.text_area("Entrada final", entrada, height=100)

    if st.button("🔍 Prever"):
        if not st.session_state.modelo:
            st.warning("Treine o modelo primeiro.")
        elif not entrada:
            st.warning("Informe texto ou imagem.")
        else:
            X = st.session_state.vectorizer.transform([entrada]).toarray()
            pred = st.session_state.modelo.predict(X)[0]
            classe = st.session_state.encoder.inverse_transform([pred])[0]

            cores = {"Baixo": "green", "Moderado": "orange", "Alto": "red"}

            st.markdown(
                f"""
                <div style='padding:20px;border-radius:12px;background:#f0f2f6;text-align:center;'>
                    <h2 style='color:{cores[classe]};'>Resultado: {classe}</h2>
                    <p>Classificação baseada em texto e imagem</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
