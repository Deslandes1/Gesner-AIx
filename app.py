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
VOICE_FILE = os.path.join(DATA_DIR, "voice_cache.json")
COGNITIVE_FILE = os.path.join(DATA_DIR, "cognitive_examples.json")

# =========================
# TRAINING FACTS (HARD CORE)
# =========================
HAITIAN_KNOWLEDGE_FACTS = [
    "Kristòf Kolon te dekouvri zile Ispanyola nan 1492.",
    "Pòtoprens se kapital Ayiti.",
    "Ayiti pran endepandans li 1 janvye 1804.",
    "Tousen Louverture se yon lidè revolisyon.",
    "Jan Jak Desalin se papa endepandans Ayiti.",
    "Soup joumou se manje endepandans Ayiti.",
    "Konpa se mizik nasyonal Ayiti.",
    "Diri ak pwa se manje prensipal Ayiti.",
    "Vodou se yon relijyon tradisyonèl Ayiti.",
    "Ayiti sitiye nan Karayib la."
]

# =========================
# SESSION INIT SAFE
# =========================
if "training_data" not in st.session_state:
    st.session_state.training_data = []

if "cognitive_examples" not in st.session_state:
    st.session_state.cognitive_examples = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

if "index" not in st.session_state:
    st.session_state.index = None

if "texts" not in st.session_state:
    st.session_state.texts = []

if "tfidf_vectorizer" not in st.session_state:
    st.session_state.tfidf_vectorizer = None

if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None

VOICE_CACHE = {}

# =========================
# SAVE / LOAD SAFE
# =========================
def load_training():
    if os.path.exists(TRAINING_FILE):
        try:
            return json.load(open(TRAINING_FILE, "r", encoding="utf-8"))
        except:
            return []
    return []

def save_training():
    with open(TRAINING_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.training_data, f, ensure_ascii=False, indent=2)

# =========================
# GROK API (FIXED SAFE ACCESS)
# =========================
def call_grok(prompt):
    api_key = st.secrets.get("GROK_API_KEY", None)
    if not api_key:
        return None

    try:
        r = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "grok-1",
                "messages": [
                    {"role": "system", "content": "You are Gesner AI. Answer in Haitian Creole."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.6
            },
            timeout=4
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except:
        pass

    return None

# =========================
# DIRECT INTELLIGENCE CORE
# =========================
CORE_ANSWERS = {
    "kijan ou rele": "Mwen rele Gesner AI, asistan entèlijan Gesner Deslandes.",
    "ki moun ki dekouvri ayiti": "Kristòf Kolon te dekouvri Ayiti an 1492.",
    "ki dat ayiti pran endepandans": "Ayiti pran endepandans 1 janvye 1804.",
    "kisa kapital ayiti ye": "Pòtoprens se kapital Ayiti."
}

def core_answer(q):
    q = q.lower().strip()
    for k, v in CORE_ANSWERS.items():
        if k in q:
            return v
    return None

# =========================
# TRAINING MATCH
# =========================
def training_match(query):
    for item in st.session_state.training_data:
        if "text" in item:
            if query.lower() in item["text"].lower():
                return item["text"]
    return None

# =========================
# FAISS BUILD SAFE
# =========================
def rebuild_index():
    valid = [x for x in st.session_state.training_data if "embedding" in x]

    if not valid:
        st.session_state.index = None
        return

    try:
        vectors = [np.array(x["embedding"], dtype=np.float32) for x in valid]
        st.session_state.texts = [x["text"] for x in valid]

        dim = len(vectors[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(vectors))

        st.session_state.index = index
    except:
        st.session_state.index = None

# =========================
# INITIALIZE TRAINING
# =========================
def init_training():
    if not st.session_state.training_data:
        for fact in HAITIAN_KNOWLEDGE_FACTS:
            emb = st.session_state.embedding_model.encode([fact])[0]
            st.session_state.training_data.append({
                "text": fact,
                "embedding": emb.tolist()
            })
        save_training()
        rebuild_index()

# =========================
# RESPONSE ENGINE (UPGRADED INTELLIGENCE)
# =========================
def generate_response(q):
    q = q.lower()

    # 1. Core answers (FASTEST)
    ans = core_answer(q)
    if ans:
        return ans

    # 2. Training memory
    t = training_match(q)
    if t:
        return t

    # 3. FAISS semantic search
    if st.session_state.index:
        try:
            emb = st.session_state.embedding_model.encode([q])[0].astype("float32").reshape(1, -1)
            D, I = st.session_state.index.search(emb, 1)
            if I[0][0] != -1:
                return st.session_state.texts[I[0][0]]
        except:
            pass

    # 4. Grok ONLINE intelligence
    g = call_grok(q)
    if g:
        return g

    # 5. FINAL fallback
    return "Mwen pa gen repons sa kounye a. Tanpri anseye m li nan Sant Fòmasyon."

# =========================
# UI
# =========================
st.set_page_config(page_title="Gesner AI", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}
* {
    color: white !important;
}
.stTextInput input {
    background:#111827 !important;
    color:white !important;
}
.stTextArea textarea {
    background:black !important;
    color:white !important;
}
.stButton button {
    background:#e11d48 !important;
    color:white !important;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# MAIN CHAT
# =========================
def chat():
    st.title("🧠 Gesner AI Ultra Intelligence")

    user = st.text_input("Poze kestyon ou:")

    if st.button("Voye") and user:
        reply = generate_response(user)

        st.session_state.conversation_history.append(("You", user))
        st.session_state.conversation_history.append(("AI", reply))

    for role, msg in st.session_state.conversation_history[::-1]:
        st.write(f"**{role}:** {msg}")

# =========================
# TRAINING CENTER
# =========================
def training_center():
    st.subheader("📚 Training Center")

    txt = st.text_area("Anseye Gesner AI nouvo bagay")

    if st.button("Ajoute"):
        emb = st.session_state.embedding_model.encode([txt])[0]
        st.session_state.training_data.append({
            "text": txt,
            "embedding": emb.tolist()
        })
        save_training()
        rebuild_index()
        st.success("Ajoute!")

    st.write("### Done ki deja antrene:")
    for i, t in enumerate(st.session_state.training_data):
        st.write(f"{i+1}. {t['text']}")

# =========================
# ROUTER
# =========================
menu = st.sidebar.radio("Menu", ["Chat", "Training Center"])

init_training()

if menu == "Chat":
    chat()
else:
    training_center()
