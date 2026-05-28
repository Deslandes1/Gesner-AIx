import streamlit as st
import json
import numpy as np
import faiss
import os
import re
import requests
from datetime import datetime
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Gesner AI", page_icon="🧠", layout="wide")

DATA_DIR = ".gesner_data"
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_FILE = os.path.join(DATA_DIR, "training.json")

# =========================
# LIGHT UI THEME (UPDATED)
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #f5f7ff;
    color: #111;
}
[data-testid="stSidebar"] {
    background-color: #eef2ff;
}
.stTextInput input, .stTextArea textarea {
    background-color: white !important;
    color: black !important;
    border-radius: 10px;
}
.stButton button {
    background-color: #4f46e5 !important;
    color: white !important;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TRAINING CENTER DATA
# =========================
HAITIAN_KNOWLEDGE_FACTS = [
    "Kristòf Kolon te dekouvri Ayiti nan 1492.",
    "Pòtoprens se kapital Ayiti.",
    "Ayiti sitiye nan Karayib la.",
    "Jan Jak Desalin te pwoklame endepandans 1804.",
    "Tousen Louverture se yon lidè revolisyon ayisyen."
]

CORE_ANSWERS = {
    "kijan ou rele": "Mwen rele Gesner AI.",
    "ki moun ki dekouvri ayiti": "Kristòf Kolon te dekouvri Ayiti nan 1492.",
    "ki kote ayiti ye": "Ayiti sitiye nan Karayib la.",
    "site konbyen let ki genhen nan alfabe kreyol la":
        "Gen 32 lèt nan alfabè kreyòl la."
}

# =========================
# LOAD MODEL
# =========================
if "model" not in st.session_state:
    st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")

if "training_data" not in st.session_state:
    if os.path.exists(TRAINING_FILE):
        st.session_state.training_data = json.load(open(TRAINING_FILE, "r", encoding="utf-8"))
    else:
        st.session_state.training_data = []

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.texts = []

# =========================
# BUILD INDEX SAFE
# =========================
def rebuild_index():
    try:
        texts = []
        vectors = []

        for item in st.session_state.training_data:
            if "text" in item:
                texts.append(item["text"])
                emb = st.session_state.model.encode(item["text"])
                vectors.append(emb)

        if len(vectors) == 0:
            return

        dim = len(vectors[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(vectors).astype("float32"))

        st.session_state.index = index
        st.session_state.texts = texts
    except:
        st.session_state.index = None

# =========================
# ADD TRAINING FACTS
# =========================
def init_training():
    existing = {x["text"] for x in st.session_state.training_data if "text" in x}

    for fact in HAITIAN_KNOWLEDGE_FACTS:
        if fact not in existing:
            st.session_state.training_data.append({"text": fact})

    rebuild_index()

init_training()

# =========================
# CORE ANSWER
# =========================
def get_core_answer(q):
    q = q.lower().strip()
    return CORE_ANSWERS.get(q)

# =========================
# RETRIEVAL
# =========================
def search_memory(query):
    if st.session_state.index is None:
        return []

    q_vec = st.session_state.model.encode(query).astype("float32").reshape(1, -1)
    _, idx = st.session_state.index.search(q_vec, 3)

    results = []
    for i in idx[0]:
        if i != -1 and i < len(st.session_state.texts):
            results.append(st.session_state.texts[i])
    return results

# =========================
# GROK API
# =========================
def call_grok(prompt):
    try:
        key = st.secrets.get("GROK_API_KEY")
        if not key:
            return None

        res = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "grok-1",
                "messages": [
                    {"role": "system", "content": "Answer directly in Haitian Creole only."},
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=5
        )

        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
    except:
        return None

# =========================
# MAIN AI ENGINE
# =========================
def generate_answer(user_input):
    q = user_input.lower().strip()

    # 1. CORE ANSWERS
    if get_core_answer(q):
        return get_core_answer(q)

    # 2. MEMORY SEARCH
    mem = search_memory(user_input)
    if mem:
        return mem[0]

    # 3. GROK FALLBACK
    grok = call_grok(user_input)
    if grok:
        return grok

    # 4. FINAL FALLBACK
    return "Mwen pa gen repons sa kounye a."

# =========================
# CHAT UI (FIXED NO REPEAT)
# =========================
def chat():
    st.title("🧠 Gesner AI")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        role, text = msg
        if role == "user":
            st.write("🧑", text)
        else:
            st.write("🤖", text)

    user = st.text_input("Mande Gesner AI")

    if st.button("Send") and user:
        st.session_state.chat.append(("user", user))

        answer = generate_answer(user)

        # IMPORTANT FIX: no repetition, clean output only
        answer = answer.replace(user, "").strip()

        st.session_state.chat.append(("ai", answer))
        st.rerun()

# =========================
# SIDEBAR (UPDATED)
# =========================
def sidebar():
    st.sidebar.markdown("## 🧠 Gesner AI")
    st.sidebar.markdown("""
**Company:** Globalinternet.py/software  
**Built by:** Gesner Deslandes  
📞 (509)-47385663  
📧 deslandes78@gmail.com  
""")

# =========================
# RUN APP
# =========================
sidebar()
chat()
