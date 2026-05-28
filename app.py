import streamlit as st
import json
import numpy as np
import faiss
import os
import re
import requests
from sentence_transformers import SentenceTransformer
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Gesner AI", page_icon="🧠", layout="wide")

DATA_DIR = ".gesner_data"
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_FILE = os.path.join(DATA_DIR, "training.json")

# =========================
# LIGHT UI
# =========================
st.markdown("""
<style>
.stApp { background: #f4f7ff; }
[data-testid="stSidebar"] { background: #e9efff; }

* { color: #111 !important; }

.stTextInput input, .stTextArea textarea {
    background: #fff !important;
    color: #000 !important;
    border-radius: 10px;
}

.stButton button {
    background: #3b6cff !important;
    color: white !important;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD / SAVE
# =========================
def load_training():
    if os.path.exists(TRAINING_FILE):
        return json.load(open(TRAINING_FILE, "r", encoding="utf-8"))
    return []

def save_training():
    json.dump(st.session_state.training, open(TRAINING_FILE, "w", encoding="utf-8"), indent=2)

# =========================
# INITIALIZATION
# =========================
if "training" not in st.session_state:
    st.session_state.training = load_training()

if "chat" not in st.session_state:
    st.session_state.chat = []

if "model" not in st.session_state:
    st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")

if "index" not in st.session_state:
    st.session_state.index = None

if "texts" not in st.session_state:
    st.session_state.texts = []

# =========================
# HAITIAN KNOWLEDGE (SAFE)
# =========================
BASE_FACTS = [
    "Kristòf Kolon te rive Ayiti an 1492.",
    "Pòtoprens se kapital Ayiti.",
    "Ayiti sitiye nan Karayib la.",
    "Endepandans Ayiti se 1 janvye 1804."
]

CORE_ANSWERS = {
    "kijan ou rele": "Mwen rele Gesner AI.",
    "ki moun ki dekouvri ayiti": "Kristòf Kolon te dekouvri Ayiti.",
    "ki dat ayiti endepandan": "1 janvye 1804."
}

# =========================
# BUILD INDEX (SAFE)
# =========================
def rebuild_index():
    texts = [x["text"] for x in st.session_state.training]

    # merge base facts if empty
    if not texts:
        texts = BASE_FACTS

    embeddings = []
    for t in texts:
        emb = st.session_state.model.encode(t)
        embeddings.append(np.array(emb, dtype=np.float32))

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    st.session_state.index = index
    st.session_state.texts = texts

# =========================
# GROK API (IMPORTANT FIX)
# =========================
def ask_grok(question):
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
                {
                    "role": "system",
                    "content": "You are Gesner AI. Answer ONLY in Haitian Creole. Do not repeat the question."
                },
                {"role": "user", "content": question}
            ],
            "temperature": 0.2
        }

        r = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=8
        )

        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]

    except:
        return None

    return None

# =========================
# CLEAN OUTPUT
# =========================
def clean(text, question):
    if not text:
        return text

    q = question.lower().strip()
    t = text.strip()

    if q in t.lower():
        t = t.replace(q, "")

    return t.strip()

# =========================
# MAIN ANSWER ENGINE (FIXED)
# =========================
def answer(question):
    q = question.lower().strip()

    # 1. CORE ANSWERS
    if q in CORE_ANSWERS:
        return CORE_ANSWERS[q]

    for k, v in CORE_ANSWERS.items():
        if k in q:
            return v

    # 2. FAISS ONLY IF STRONG MATCH
    if st.session_state.index is not None:
        emb = st.session_state.model.encode(question)
        emb = np.array([emb], dtype=np.float32)

        D, I = st.session_state.index.search(emb, 1)

        distance = D[0][0]
        idx = I[0][0]

        # IMPORTANT FIX: strict threshold
        if idx != -1 and distance < 0.85:
            return st.session_state.texts[idx]

    # 3. GROK ALWAYS FALLBACK (MAIN INTELLIGENCE)
    grok = ask_grok(question)
    if grok:
        return clean(grok, question)

    # 4. FINAL FALLBACK
    return "Mwen pa jwenn repons sa kounye a."

# =========================
# ADD TRAINING
# =========================
def add_training(text):
    emb = st.session_state.model.encode(text)
    st.session_state.training.append({
        "text": text,
        "embedding": emb.tolist()
    })
    save_training()
    rebuild_index()

# =========================
# SIDEBAR
# =========================
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

# =========================
# CHAT UI
# =========================
def chat_ui():
    st.title("Gesner AI Chat")

    for m in st.session_state.chat:
        if m["role"] == "user":
            st.markdown("🧑 " + m["text"])
        else:
            st.markdown("🤖 " + m["text"])

    msg = st.text_input("Ask something")

    if st.button("Send"):
        if msg:
            st.session_state.chat.append({"role": "user", "text": msg})

            res = answer(msg)

            st.session_state.chat.append({"role": "ai", "text": res})

            st.rerun()

# =========================
# TRAINING CENTER
# =========================
def training_ui():
    st.title("Training Center")

    txt = st.text_area("Add knowledge")

    if st.button("Save"):
        if txt.strip():
            add_training(txt)
            st.success("Saved!")

    st.markdown("### Data")
    for i, t in enumerate(st.session_state.training):
        st.write(f"{i+1}. {t['text']}")

# =========================
# INIT
# =========================
rebuild_index()

menu = sidebar()

if menu == "Chat":
    chat_ui()
else:
    training_ui()
