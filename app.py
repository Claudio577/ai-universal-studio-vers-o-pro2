# ===============================================================
# 🧠 AI Universal Studio — Versão PRO++ (FINAL / CLOUD READY)
# ===============================================================
# Sistema multimodal: Texto + Imagem + Áudio
# ===============================================================

import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
import torch

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel

# ===============================================================
# ⚙️ Configuração da Página
# ===============================================================
st.set_page_config(
    page_title="AI Universal Studio PRO++",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI Universal Studio — Versão PRO++")
st.info(
    """
Sistema **Multimodal Inteligente** que aprende com **texto**, **imagem** e **voz**  
para gerar previsões personalizadas usando **IA de ponta** 🔥
"""
)

# ===============================================================
# 📦 Carregamento dos Modelos (CACHE)
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


@st.cache_resource
def load_text_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu")


caption_processor, caption_model = load_caption_model()
text_embedder = load_text_model()
whisper = load_whisper_model()

# ===============================================================
# 💾 Funções auxiliares
# ===============================================================
def salvar_modelo(modelo, encoder):
    joblib.dump(modelo, "modelo_rf.pkl")
    joblib.dump(encoder, "encoder.pkl")


def carregar_modelo():
    if os.path.exists("modelo_rf.pkl") and os.path.exists("encoder.pkl"):
        return joblib.load("modelo_rf.pkl"), joblib.load("encoder.pkl")
    return None, None


def gerar_embedding_texto(texto):
    return text_embedder.encode([texto])[0]


def transcrever_audio(arquivo):
    try:
        segments, _ = whisper.transcribe(arquivo)
        texto = " ".join(seg.text for seg in segments)
        return texto
    except Exception as e:
        return f"[Erro ao processar áudio: {e}]"


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


# ===============================================================
# 🔁 Sessão Compartilhada
# ===============================================================
for var, default in {
    "base_textos": [],
    "base_labels": [],
    "modelo_rf": None,
    "encoder": None,
}.items():
    if var not in st.session_state:
        st.session_state[var] = default

# ===============================================================
# 🧭 Abas
# ===============================================================
aba = st.tabs(
    [
        "🧩 Etapa 1 - Base de Treinamento",
        "⚙️ Etapa 2 - Treinar Modelo",
        "🔮 Etapa 3 - Fazer Previsão",
    ]
)

# ===============================================================
# 1️⃣ ETAPA 1 — Base de Treinamento
# ===============================================================
with aba[0]:
    st.header("🧩 Etapa 1 – Criar base de aprendizado")
    st.write("Adicione exemplos (texto + categoria) para ensinar o modelo.")

    entradas = []
    for i in range(3):
        col1, col2 = st.columns([3, 1])
        texto = col1.text_input(f"📝 Exemplo {i+1}", key=f"texto_{i}")
        categoria = col2.selectbox(
            "🎯 Categoria",
            ["Baixo", "Moderado", "Alto"],
            index=1,
            key=f"cat_{i}",
        )
        if texto:
            entradas.append({"texto": texto, "categoria": categoria})

    if entradas and st.button("💾 Salvar base"):
        st.session_state.base_textos = [e["texto"] for e in entradas]
        st.session_state.base_labels = [e["categoria"] for e in entradas]
        st.success("✅ Base salva com sucesso!")
        st.dataframe(pd.DataFrame(entradas), use_container_width=True)

# ===============================================================
# 2️⃣ ETAPA 2 — Treinar Modelo
# ===============================================================
with aba[1]:
    st.header("⚙️ Etapa 2 – Treinar modelo")

    if not st.session_state.base_textos:
        st.warning("⚠️ Nenhum dado disponível. Vá para a Etapa 1.")
    else:
        if st.button("🚀 Treinar modelo"):
            X = np.array(
                [gerar_embedding_texto(t) for t in st.session_state.base_textos]
            )

            encoder = LabelEncoder()
            y = encoder.fit_transform(st.session_state.base_labels)

            modelo = RandomForestClassifier(random_state=42)
            modelo.fit(X, y)

            st.session_state.modelo_rf = modelo
            st.session_state.encoder = encoder
            salvar_modelo(modelo, encoder)

            st.success("✅ Modelo treinado e salvo!")

        modelo_salvo, encoder_salvo = carregar_modelo()
        if modelo_salvo:
            st.session_state.modelo_rf = modelo_salvo
            st.session_state.encoder = encoder_salvo
            st.info("💾 Modelo salvo carregado automaticamente.")

# ===============================================================
# 3️⃣ ETAPA 3 — Previsão Multimodal
# ===============================================================
with aba[2]:
    st.header("🔮 Etapa 3 – Fazer previsão")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_img = st.file_uploader(
            "📷 Imagem (opcional)", type=["jpg", "jpeg", "png"]
        )
    with col2:
        uploaded_audio = st.file_uploader(
            "🎤 Áudio (opcional)", type=["mp3", "wav", "m4a"]
        )

    texto_input = st.text_area("💬 Texto descritivo (opcional)")

    desc_img = ""
    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="📸 Imagem enviada", use_container_width=True)
        with st.spinner("🔍 Analisando imagem..."):
            desc_img = gerar_caption_imagem(image)
            st.markdown(f"*Descrição da imagem:* {desc_img}")

    desc_audio = ""
    if uploaded_audio:
        st.audio(uploaded_audio)
        with st.spinner("🎧 Transcrevendo áudio..."):
            desc_audio = transcrever_audio(uploaded_audio)
            st.markdown(f"*Transcrição do áudio:* {desc_audio}")

    entrada = f"{desc_img} {desc_audio} {texto_input}".strip()
    st.text_area("🧩 Entrada combinada", value=entrada, height=120)

    if st.button("🔍 Fazer previsão"):
        if not st.session_state.modelo_rf or not st.session_state.encoder:
            st.warning("⚠️ Treine o modelo antes.")
        elif not entrada:
            st.warning("⚠️ Insira imagem, áudio ou texto.")
        else:
            emb = gerar_embedding_texto(entrada).reshape(1, -1)
            pred = st.session_state.modelo_rf.predict(emb)[0]
            classe = st.session_state.encoder.inverse_transform([pred])[0]

            cor = {"Baixo": "green", "Moderado": "orange", "Alto": "red"}[classe]

            st.markdown(
                f"""
                <div style='background:#f0f2f6;padding:20px;border-radius:12px;text-align:center;'>
                    <h3>🧠 Previsão da IA:
                        <span style='color:{cor};'>{classe}</span>
                    </h3>
                    <p style='color:gray;'>Baseado em texto, imagem e voz</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
