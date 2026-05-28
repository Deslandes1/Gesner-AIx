import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time
import hashlib
import re
import base64
import csv
import io
import os
import shutil
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from PIL import Image
from transformers import pipeline

# =========================
# DATA DIRECTORY
# =========================
DATA_DIR = ".gesner_data"
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_FILE = os.path.join(DATA_DIR, "training_data.json")
DICT_FILE = os.path.join(DATA_DIR, "dictionaries.json")
VOICE_FILE = os.path.join(DATA_DIR, "voice_cache.json")
COGNITIVE_FILE = os.path.join(DATA_DIR, "cognitive_examples.json")

# =========================
# PERSISTENCE
# =========================
def save_training_data():
    with open(TRAINING_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.training_data, f, ensure_ascii=False, indent=2)

def load_training_data():
    if os.path.exists(TRAINING_FILE):
        with open(TRAINING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_dictionaries():
    with open(DICT_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.dictionaries, f, ensure_ascii=False, indent=2)

def load_dictionaries():
    if os.path.exists(DICT_FILE):
        with open(DICT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"ht": {}, "fr": {}, "en": {}}

def load_cognitive_examples():
    if os.path.exists(COGNITIVE_FILE):
        with open(COGNITIVE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_cognitive_examples():
    with open(COGNITIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.cognitive_examples, f, indent=2)

# =========================
# KNOWLEDGE BASE
# =========================
HAITIAN_KNOWLEDGE_FACTS = [
    "Kristòf Kolon te dekouvri Ayiti an 1492.",
    "Pòtoprens se kapital Ayiti.",
    "Ayiti endepandan depi 1804.",
    "Tousen Louverture se yon lidè revolisyon.",
    "Jan Jak Desalin se papa endepandans Ayiti."
]

# =========================
# INDEX SYSTEM
# =========================
def rebuild_index():
    if st.session_state.training_data:
        st.session_state.texts = [x["text"] for x in st.session_state.training_data]
        embeddings = [np.array(x["embedding"], dtype=np.float32) for x in st.session_state.training_data]
        dim = len(embeddings[0])
        st.session_state.index = faiss.IndexFlatL2(dim)
        st.session_state.index.add(np.array(embeddings))
    else:
        st.session_state.index = None
        st.session_state.texts = []

def initialize_default_training():
    if not st.session_state.training_data:
        for fact in HAITIAN_KNOWLEDGE_FACTS:
            emb = st.session_state.embedding_model.encode([fact])[0]
            st.session_state.training_data.append({
                "text": fact,
                "embedding": emb.tolist()
            })
        rebuild_index()
        save_training_data()

# =========================
# SAFE RESPONSE ENGINE
# =========================
def generate_response(user_input, uploaded_image=None):
    try:
        if uploaded_image:
            return "📷 Mwen resevwa imaj la, men mwen pa ka analize li kounye a.", False, False

        if user_input.strip():
            return f"Ou di: {user_input}", False, False

    except Exception:
        pass

    return "Mwen pa gen repons kounye a.", True, False

# =========================
# CHAT INTERFACE (FIXED + COLORFUL UI)
# =========================
def chat_interface(t):
    st.markdown(
        f"""
        <h1 style='text-align:center; color:#00ffd5; text-shadow:0px 0px 10px #00ffd5;'>
            🧠 {t['app_title']}
        </h1>
        """,
        unsafe_allow_html=True
    )

    # CHAT DISPLAY
    chat_text = ""
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            chat_text += "🧑‍💻 " + msg["content"] + "\n\n"
        else:
            chat_text += "🤖 " + msg["content"] + "\n\n"

    st.text_area("Chat history", value=chat_text, height=400, disabled=True)

    col1, col2, col3 = st.columns([6,1,1])

    with col1:
        user_input = st.text_input("Message", placeholder="Ekri kesyon ou...")

    with col2:
        uploaded_file = st.file_uploader("📷", type=["jpg","png","jpeg"])

    with col3:
        send = st.button("📤 Send")

    if st.button("🗑️ Clear Chat"):
        st.session_state.conversation_history = []
        st.rerun()

    if send and user_input.strip():
        img_bytes = uploaded_file.read() if uploaded_file else None

        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        answer, is_fallback, skip_audio = generate_response(user_input, img_bytes)

        # SAFE FIX (prevents crash empty response)
        if not answer or not isinstance(answer, str):
            answer = "Mwen pa gen repons kounye a."

        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()

# =========================
# TRAINING CENTER (RESTORED)
# =========================
def training_center():
    st.markdown(
        "<h2 style='color:#ffcc00;'>📚 Training Center</h2>",
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["➕ Add Fact", "🧠 Cognitive Training"])

    with tab1:
        fact = st.text_area("Nouvo reyalite")
        if st.button("Ajoute"):
            if fact.strip():
                emb = st.session_state.embedding_model.encode([fact])[0]
                st.session_state.training_data.append({
                    "text": fact,
                    "embedding": emb.tolist()
                })
                rebuild_index()
                save_training_data()
                st.success("Ajoute!")

    with tab2:
        st.write("Kognitif training (simple mode)")
        q = st.text_input("Input")
        a = st.text_input("Output")

        if st.button("Save Pattern"):
            if q and a:
                st.session_state.cognitive_examples.append({"q": q, "a": a})
                save_cognitive_examples()
                st.success("Saved!")

# =========================
# SESSION STATE INIT
# =========================
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "embedding_model" not in st.session_state:
    with st.spinner("Loading AI model..."):
        st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.session_state.index = None
    st.session_state.texts = []

if "training_data" not in st.session_state:
    st.session_state.training_data = load_training_data()

if "dictionaries" not in st.session_state:
    st.session_state.dictionaries = load_dictionaries()

if "cognitive_examples" not in st.session_state:
    st.session_state.cognitive_examples = load_cognitive_examples()

# =========================
# COLORFUL UI STYLE
# =========================
st.set_page_config(page_title="Gesner AI", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f0c29,#302b63,#24243e);
    color: white;
}
.stTextInput input {
    background:#111 !important;
    color:white !important;
}
.stTextArea textarea {
    background:#111 !important;
    color:white !important;
}
button {
    background: linear-gradient(90deg,#00ffd5,#0088ff) !important;
    color:black !important;
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LANGUAGE
# =========================
TEXTS = {
    "en": {"app_title": "Gesner AI"},
    "ht": {"app_title": "Gesner AI - Asistan"},
    "fr": {"app_title": "Gesner IA"}
}

# =========================
# MAIN
# =========================
def main():
    rebuild_index()
    initialize_default_training()

    menu = st.sidebar.radio("Menu", ["Chat", "Training Center"])

    if menu == "Chat":
        chat_interface(TEXTS["en"])
    else:
        training_center()

if __name__ == "__main__":
    main()
