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
from transformers import pipeline   # removed 'Conversation'

# ========== DATA DIRECTORY ==========
DATA_DIR = ".gesner_data"
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_FILE = os.path.join(DATA_DIR, "training_data.json")
DICT_FILE = os.path.join(DATA_DIR, "dictionaries.json")
VOICE_FILE = os.path.join(DATA_DIR, "voice_cache.json")
COGNITIVE_FILE = os.path.join(DATA_DIR, "cognitive_examples.json")

# ---------- PERSISTENCE FUNCTIONS ----------
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

def save_voice_cache():
    serializable = {}
    for key, audio_bytes in VOICE_CACHE.items():
        serializable[key] = base64.b64encode(audio_bytes).decode("utf-8")
    with open(VOICE_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)

def load_voice_cache():
    if os.path.exists(VOICE_FILE):
        with open(VOICE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cache = {}
        for key, b64 in data.items():
            cache[key] = base64.b64decode(b64)
        return cache
    return {}

def save_cognitive_examples():
    with open(COGNITIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.cognitive_examples, f, ensure_ascii=False, indent=2)

def load_cognitive_examples():
    if os.path.exists(COGNITIVE_FILE):
        with open(COGNITIVE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ========== DATA ==========
HAITIAN_KNOWLEDGE_FACTS = [
    "Kristòf Kolon te dekouvri zile Ispanyola (kote Ayiti ye jodi a) nan 5 desanm 1492.",
    "Pòtoprens se kapital Ayiti.",
    "Ayiti sitiye nan Karayib la.",
    "Tousen Louverture te yon lidè revolisyon esklav la.",
    "Jan Jak Desalin te pwoklame endepandans Ayiti 1ye janvye 1804.",
]

# ---------- INIT TRAINING ----------
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

# ---------- FIXED CORE CRASH ----------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "embedding_model" not in st.session_state:
    with st.spinner("Loading AI model... (first time only)"):
        st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.session_state.index = None
    st.session_state.texts = []

if "training_data" not in st.session_state:
    st.session_state.training_data = load_training_data()

if "dictionaries" not in st.session_state:
    st.session_state.dictionaries = load_dictionaries()

if "cognitive_examples" not in st.session_state:
    st.session_state.cognitive_examples = load_cognitive_examples()

if "tfidf_vectorizer" not in st.session_state:
    st.session_state.tfidf_vectorizer = None

if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None

if "play_audio" not in st.session_state:
    st.session_state.play_audio = None

VOICE_CACHE = load_voice_cache()

# ---------- SAFE RESPONSE ----------
def generate_response(user_input, uploaded_image_bytes=None):
    try:
        if uploaded_image_bytes:
            return "Mwen resevwa imaj la, men mwen pa ka analize li kounye a.", False, False

        if user_input.strip():
            return f"Ou di: {user_input}", False, False

    except Exception:
        pass

    return "Mwen pa gen repons kounye a.", True, False

# ---------- CHAT UI (FIXED INDENTATION ERROR HERE) ----------
def chat_interface(t):
    st.markdown(f"<h1 style='text-align:center'>{t['app_title']}</h1>", unsafe_allow_html=True)

    chat_text = ""
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            chat_text += "🧑‍💻 " + msg["content"] + "\n\n"
        else:
            chat_text += "🤖 " + msg["content"] + "\n\n"

    st.text_area("Chat history", value=chat_text, height=400, disabled=True)

    col1, col2, col3 = st.columns([6,1,1])

    with col1:
        user_input = st.text_input("Message")

    with col2:
        uploaded_file = st.file_uploader("img", type=["jpg","png","jpeg"])

    with col3:
        send = st.button("Send")

    if st.button("Clear Chat"):
        st.session_state.conversation_history = []
        st.rerun()

    if send and user_input.strip():
        img_bytes = uploaded_file.read() if uploaded_file else None

        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        answer, is_fallback, skip_audio = generate_response(user_input, img_bytes)

        # SAFE BOX FIX (this prevents empty crash replies)
        if not answer or not isinstance(answer, str):
            answer = "Mwen pa gen repons kounye a."

        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()

# ---------- SIMPLE UI CONFIG ----------
st.set_page_config(page_title="Gesner AI", layout="wide")

TEXTS = {
    "en": {"app_title": "Gesner AI"},
    "fr": {"app_title": "Gesner AI"},
    "ht": {"app_title": "Gesner AI - Asistan"}
}

def main():
    rebuild_index()
    initialize_default_training()
    chat_interface(TEXTS["en"])

if __name__ == "__main__":
    main()
