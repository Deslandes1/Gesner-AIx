import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
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
from PIL import Image

# =========================
# DATA DIRECTORY
# =========================
DATA_DIR = ".gesner_data"
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_FILE = os.path.join(DATA_DIR, "training_data.json")

# =========================
# GROK CONFIG
# =========================
def get_grok_api_key():
    try:
        return st.secrets["GROK_API_KEY"]
    except:
        return None

def call_grok_api(prompt):
    api_key = get_grok_api_key()
    if not api_key:
        return None

    endpoint = "https://api.x.ai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "grok-1",
        "messages": [
            {"role": "system", "content": "You are Gesner AI. Answer clearly in Haitian Creole."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }

    try:
        r = requests.post(endpoint, headers=headers, json=payload, timeout=5)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except:
        return None

# =========================
# LOAD / SAVE TRAINING
# =========================
def load_training_data():
    if os.path.exists(TRAINING_FILE):
        with open(TRAINING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_training_data():
    with open(TRAINING_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.training_data, f, indent=2)

# =========================
# KNOWLEDGE BASE (KEEPED)
# =========================
HAITIAN_KNOWLEDGE_FACTS = [
    "Kristòf Kolon te dekouvri Ayiti an 1492.",
    "Pòtoprens se kapital Ayiti.",
    "Ayiti endepandan depi 1804.",
    "Tousen Louverture se yon lidè revolisyon.",
    "Jan Jak Desalin se papa endepandans Ayiti."
]

# =========================
# INIT MODEL
# =========================
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

if "training_data" not in st.session_state:
    st.session_state.training_data = load_training_data()

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# =========================
# BUILD INDEX
# =========================
def rebuild_index():
    if st.session_state.training_data:
        embeddings = [np.array(x["embedding"], dtype=np.float32) for x in st.session_state.training_data]
        dim = len(embeddings[0])
        st.session_state.index = faiss.IndexFlatL2(dim)
        st.session_state.index.add(np.array(embeddings))
        st.session_state.texts = [x["text"] for x in st.session_state.training_data]
    else:
        st.session_state.index = None
        st.session_state.texts = []

# =========================
# INITIAL TRAINING (IMPORTANT - KEPT DATA)
# =========================
def initialize_training():
    if not st.session_state.training_data:
        for fact in HAITIAN_KNOWLEDGE_FACTS:
            emb = st.session_state.embedding_model.encode([fact])[0]
            st.session_state.training_data.append({
                "text": fact,
                "embedding": emb.tolist()
            })
        save_training_data()

# =========================
# FIND ANSWER LOCALLY
# =========================
def find_local_answer(query):
    if not st.session_state.training_data:
        return None

    query_emb = st.session_state.embedding_model.encode([query])[0].astype("float32").reshape(1, -1)
    distances, indices = st.session_state.index.search(query_emb, k=3)

    results = []
    for i in indices[0]:
        if i != -1:
            results.append(st.session_state.training_data[i]["text"])

    if results:
        return results[0]

    return None

# =========================
# SAFE RESPONSE ENGINE (WITH GROK FALLBACK)
# =========================
def generate_response(user_input, uploaded_image=None):

    if uploaded_image:
        return "📷 Mwen resevwa imaj la, men mwen pa ka analize li kounye a."

    # 1. LOCAL TRAINING FIRST
    local = find_local_answer(user_input)
    if local:
        return local

    # 2. GROK FALLBACK (ONLINE LOGIC)
    grok_answer = call_grok_api(user_input)
    if grok_answer:
        return grok_answer

    # 3. FINAL FALLBACK
    return "Mwen pa jwenn repons nan sistèm lan."

# =========================
# TRAINING CENTER
# =========================
def training_center():
    st.markdown("<h2 style='color:#FFD700;'>📚 Training Center</h2>", unsafe_allow_html=True)

    new_fact = st.text_area("Ajoute nouvo konesans")

    if st.button("Ajoute"):
        if new_fact.strip():
            emb = st.session_state.embedding_model.encode([new_fact])[0]
            st.session_state.training_data.append({
                "text": new_fact,
                "embedding": emb.tolist()
            })
            save_training_data()
            rebuild_index()
            st.success("Ajoute!")

# =========================
# CHAT UI
# =========================
def chat_interface():

    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg,#0f0c29,#302b63,#24243e);
    }
    .user-msg, .bot-msg {
        color: white !important;
        font-weight: 900 !important;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧠 Gesner AI")

    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'>🧑‍💻 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

    user_input = st.text_input("Ekri mesaj ou")

    uploaded = st.file_uploader("Upload image", type=["png","jpg","jpeg"])

    if st.button("Send"):
        st.session_state.conversation_history.append({"role":"user","content":user_input})

        img = uploaded.read() if uploaded else None
        reply = generate_response(user_input, img)

        # FORCE STRONG WHITE OUTPUT
        if not reply:
            reply = "Mwen pa gen repons kounye a."

        st.session_state.conversation_history.append({"role":"assistant","content":reply})
        st.rerun()

    if st.button("Clear Chat"):
        st.session_state.conversation_history = []
        st.rerun()

# =========================
# INIT SYSTEM
# =========================
initialize_training()
rebuild_index()

menu = st.sidebar.radio("Menu", ["Chat", "Training Center"])

if menu == "Chat":
    chat_interface()
else:
    training_center()
