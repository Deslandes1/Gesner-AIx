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
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

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
# HAITIAN KNOWLEDGE FACTS
# =========================
HAITIAN_KNOWLEDGE_FACTS = [
    "Kristòf Kolon te dekouvri zile Ispanyola nan 1492.",
    "Pòtoprens se kapital Ayiti.",
    "Ayiti sitiye nan Karayib la.",
    "Tousen Louverture te yon lidè revolisyon.",
    "Jan Jak Desalin te pwoklame endepandans Ayiti an 1804.",
    "Soup joumou se manje endepandans Ayiti.",
    "Vodou se yon relijyon tradisyonèl Ayiti.",
    "Kanaval Ayiti fèt chak ane.",
]

# =========================
# CORE ANSWERS (TRAINED MEMORY)
# =========================
CORE_ANSWERS = {
    "ki moun ki dekouvri ayiti": "Kristòf Kolon te dekouvri Ayiti an 1492.",
    "kijan ou rele": "Non mwen se Gesner AI, kreye pa Gesner Deslandes.",
    "ki dat ayiti endepandan": "Ayiti vin endepandan 1 janvye 1804.",
    "kisa soup joumou ye": "Soup joumou se soup libète Ayiti.",
    "ki moun ki tousen louverture": "Li te lidè revolisyon esklav Ayiti.",
}

# =========================
# SAVE / LOAD FUNCTIONS
# =========================
def save_training_data():
    with open(TRAINING_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.training_data, f, ensure_ascii=False, indent=2)

def load_training_data():
    if os.path.exists(TRAINING_FILE):
        with open(TRAINING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# =========================
# VOICE CACHE
# =========================
def load_voice_cache():
    if os.path.exists(VOICE_FILE):
        with open(VOICE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            k: base64.b64decode(v) for k, v in data.items()
        }
    return {}

VOICE_CACHE = load_voice_cache()

# =========================
# CORE TRAINING LOADER
# =========================
def load_core_answers_into_training():
    for q, a in CORE_ANSWERS.items():
        text = f"Q: {q} | A: {a}"
        if not any(x["text"] == text for x in st.session_state.training_data):
            emb = st.session_state.embedding_model.encode([text])[0]
            st.session_state.training_data.append({
                "text": text,
                "embedding": emb.tolist()
            })

# =========================
# INITIALIZE TRAINING
# =========================
def initialize_default_training():
    if not st.session_state.training_data:
        for fact in HAITIAN_KNOWLEDGE_FACTS:
            emb = st.session_state.embedding_model.encode([fact])[0]
            st.session_state.training_data.append({
                "text": fact,
                "embedding": emb.tolist()
            })

        load_core_answers_into_training()
        rebuild_index()
        save_training_data()

# =========================
# GROK API (FALLBACK BRAIN)
# =========================
def call_grok_api(prompt):
    api_key = st.secrets.get("GROK_API_KEY", None)
    if not api_key:
        return None

    try:
        res = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "grok-1",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 400
            },
            timeout=5
        )
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
    except:
        pass
    return None

# =========================
# CORE ANSWER ENGINE
# =========================
def get_core_answer(q):
    q = q.lower().strip()
    return CORE_ANSWERS.get(q, None)

# =========================
# RETRIEVAL SYSTEM
# =========================
def retrieve_facts(query, k=5):
    if st.session_state.index is None:
        return []

    q_emb = st.session_state.embedding_model.encode([query])[0]
    D, I = st.session_state.index.search(np.array([q_emb]).astype("float32"), k)

    results = []
    for i in I[0]:
        if i != -1:
            results.append(st.session_state.texts[i])
    return results

# =========================
# RESPONSE ENGINE
# =========================
def generate_response(user_input):
    core = get_core_answer(user_input)
    if core:
        return core

    facts = retrieve_facts(user_input)
    if facts:
        return facts[0]

    grok = call_grok_api(user_input)
    if grok:
        return grok

    return "Mwen pa gen repons sa kounye a, tanpri anseye m nan Training Center."

# =========================
# FAISS INDEX
# =========================
def rebuild_index():
    if not st.session_state.training_data:
        return

    st.session_state.texts = [x["text"] for x in st.session_state.training_data]
    emb = [np.array(x["embedding"], dtype=np.float32) for x in st.session_state.training_data]

    dim = len(emb[0])
    st.session_state.index = faiss.IndexFlatL2(dim)
    st.session_state.index.add(np.array(emb))

# =========================
# SESSION INIT
# =========================
if "training_data" not in st.session_state:
    st.session_state.training_data = load_training_data()

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    st.session_state.index = None
    st.session_state.texts = []

# =========================
# UI CONFIG (DO NOT CHANGE COLOR)
# =========================
st.set_page_config(page_title="Gesner AI", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
}
.stTextArea textarea {
    background-color: #000 !important;
    color: #fff !important;
    font-weight: bold;
}
.stButton button {
    background-color: #e94560 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TRAINING CENTER
# =========================
def training_center():
    st.title("🧠 Training Center")

    new_fact = st.text_area("Add Knowledge")
    if st.button("Add"):
        emb = st.session_state.embedding_model.encode([new_fact])[0]
        st.session_state.training_data.append({
            "text": new_fact,
            "embedding": emb.tolist()
        })
        rebuild_index()
        save_training_data()
        st.success("Added!")

# =========================
# CHAT UI
# =========================
def chat():
    st.title("Gesner AI Chat")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for c in st.session_state.chat:
        st.write(c)

    msg = st.text_input("Ask")
    if st.button("Send"):
        answer = generate_response(msg)
        st.session_state.chat.append(f"🧑 {msg}")
        st.session_state.chat.append(f"🤖 {answer}")
        st.rerun()

# =========================
# MAIN
# =========================
def main():
    initialize_default_training()
    rebuild_index()

    menu = st.sidebar.radio("Menu", ["Chat", "Training Center"])

    if menu == "Chat":
        chat()
    else:
        training_center()

main()
