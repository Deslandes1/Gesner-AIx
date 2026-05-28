import streamlit as st
import json
import numpy as np
import faiss
import os
import re
import base64
import hashlib
import csv
import io
import requests
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Gesner AI", page_icon="🧠", layout="wide")

DATA_DIR = ".gesner_data"
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_FILE = os.path.join(DATA_DIR, "training_data.json")
VOICE_FILE = os.path.join(DATA_DIR, "voice_cache.json")
DICT_FILE = os.path.join(DATA_DIR, "dictionaries.json")
COGNITIVE_FILE = os.path.join(DATA_DIR, "cognitive.json")

# =============================
# STYLING (LIGHT THEME)
# =============================
st.markdown("""
<style>
.stApp {
    background: #f5f7ff;
}
[data-testid="stSidebar"] {
    background: #eaf0ff;
}
* {
    color: #111 !important;
}
.stTextInput input, .stTextArea textarea {
    background: white !important;
    color: black !important;
    border-radius: 10px;
}
.stButton button {
    background: #4f7cff !important;
    color: white !important;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =============================
# DATA LOAD/SAVE
# =============================
def load_json(path, default):
    if os.path.exists(path):
        return json.load(open(path, "r", encoding="utf-8"))
    return default

def save_json(path, data):
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# =============================
# HAITIAN KNOWLEDGE BASE
# =============================
HAITIAN_KNOWLEDGE_FACTS = [
    "Kristòf Kolon te dekouvri Ayiti nan 1492.",
    "Pòtoprens se kapital Ayiti.",
    "Ayiti sitiye nan Karayib la.",
    "Jan Jak Desalin te pwoklame endepandans 1 janvye 1804.",
    "Tousen Louverture te yon lidè revolisyon.",
    "Soup joumou se manje endepandans Ayiti.",
    "Diri ak pwa se manje prensipal Ayiti.",
    "Vodou se yon relijyon tradisyonèl Ayiti."
]

CORE_ANSWERS = {
    "kijan ou rele": "Mwen rele Gesner AI.",
    "ki moun ki dekouvri ayiti": "Kristòf Kolon te dekouvri Ayiti.",
    "ki dat ayiti endepandan": "1 janvye 1804."
}

# =============================
# INIT SESSION
# =============================
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "training_data" not in st.session_state:
    st.session_state.training_data = load_json(TRAINING_FILE, [])

if "dicts" not in st.session_state:
    st.session_state.dicts = load_json(DICT_FILE, {})

if "cognitive" not in st.session_state:
    st.session_state.cognitive = load_json(COGNITIVE_FILE, [])

if "model" not in st.session_state:
    st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")

if "index" not in st.session_state:
    st.session_state.index = None

if "texts" not in st.session_state:
    st.session_state.texts = []

# =============================
# SAFE INDEX BUILD
# =============================
def rebuild_index():
    if not st.session_state.training_data:
        st.session_state.index = None
        st.session_state.texts = []
        return

    st.session_state.texts = [x["text"] for x in st.session_state.training_data]

    vectors = []
    for x in st.session_state.training_data:
        if "embedding" in x:
            vectors.append(np.array(x["embedding"], dtype=np.float32))
        else:
            emb = st.session_state.model.encode(x["text"])
            vectors.append(np.array(emb, dtype=np.float32))

    if len(vectors) == 0:
        return

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))

    st.session_state.index = index

# =============================
# SAVE TRAINING
# =============================
def save_training():
    save_json(TRAINING_FILE, st.session_state.training_data)

# =============================
# GROK API
# =============================
def call_grok(prompt):
    try:
        key = st.secrets.get("GROK_API_KEY")
        if not key:
            return None

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "grok-1",
            "messages": [
                {"role": "system", "content": "You are Gesner AI. Answer in Haitian Creole only. Do not repeat the question."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.4
        }

        r = requests.post("https://api.x.ai/v1/chat/completions", json=payload, headers=headers, timeout=6)

        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except:
        pass
    return None

# =============================
# CLEAN OUTPUT (NO QUESTION REPEAT)
# =============================
def clean_answer(text, question):
    if not text:
        return text
    q = question.lower().strip()
    t = text.strip()

    # remove echo
    if q in t.lower():
        t = re.sub(q, "", t, flags=re.IGNORECASE).strip()

    return t

# =============================
# ANSWER ENGINE
# =============================
def answer(question):
    q = question.lower().strip()

    # CORE
    if q in CORE_ANSWERS:
        return CORE_ANSWERS[q]

    for k, v in CORE_ANSWERS.items():
        if k in q:
            return v

    # LOCAL TRAINING MATCH
    if st.session_state.index:
        emb = st.session_state.model.encode([question]).astype(np.float32)
        D, I = st.session_state.index.search(emb, 1)

        if I[0][0] != -1:
            return st.session_state.texts[I[0][0]]

    # GROK FALLBACK (IMPORTANT)
    grok = call_grok(question)
    if grok:
        return clean_answer(grok, question)

    return "Mwen pa jwenn repons sa kounye a."

# =============================
# ADD TRAINING
# =============================
def add_fact(text):
    emb = st.session_state.model.encode(text)
    st.session_state.training_data.append({
        "text": text,
        "embedding": emb.tolist()
    })
    save_training()
    rebuild_index()

# =============================
# UI SIDEBAR
# =============================
def sidebar():
    with st.sidebar:
        st.markdown("## 🧠 Gesner AI")
        st.markdown("""
**Company:** Globalinternet.py/software  
**Founder:** Gesner Deslandes  
📞 (509)-47385663  
📧 deslandes78@gmail.com  
""")

        return st.radio("Menu", ["Chat", "Training Center"])

# =============================
# CHAT UI
# =============================
def chat():
    st.title("Gesner AI")

    for c in st.session_state.conversation:
        if c["role"] == "user":
            st.markdown("🧑 " + c["text"])
        else:
            st.markdown("🤖 " + c["text"])

    msg = st.text_input("Ask something")

    if st.button("Send"):
        if msg:
            st.session_state.conversation.append({"role": "user", "text": msg})

            res = answer(msg)

            st.session_state.conversation.append({"role": "ai", "text": res})

            st.rerun()

# =============================
# TRAINING CENTER
# =============================
def training():
    st.title("Training Center")

    txt = st.text_area("Add knowledge")

    if st.button("Add"):
        if txt:
            add_fact(txt)
            st.success("Saved!")

    st.markdown("### Existing Data")
    for i, t in enumerate(st.session_state.training_data):
        st.write(f"{i+1}. {t['text']}")

# =============================
# INIT
# =============================
rebuild_index()

menu = sidebar()

if menu == "Chat":
    chat()
else:
    training()
