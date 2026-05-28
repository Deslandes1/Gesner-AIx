import streamlit as st
import json
import numpy as np
import faiss
import os
import re
import requests
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Gesner AI", page_icon="🧠", layout="wide")

DATA_DIR = ".gesner_data"
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_FILE = os.path.join(DATA_DIR, "training.json")

# =========================
# GROK API
# =========================
def get_grok_key():
    try:
        return st.secrets["GROK_API_KEY"]
    except:
        return None

def call_grok(prompt):
    key = get_grok_key()
    if not key:
        return None

    try:
        r = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "grok-1",
                "messages": [
                    {"role": "system", "content": "You are Gesner AI. Answer directly, do NOT repeat the question."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 400
            },
            timeout=6
        )

        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except:
        return None

# =========================
# TRAINING DATA
# =========================
DEFAULT_FACTS = [
    "Pòtoprens se kapital Ayiti.",
    "Ayiti nan Karayib la.",
    "Ayiti pran endepandans an 1804."
]

def load_training():
    if os.path.exists(TRAINING_FILE):
        return json.load(open(TRAINING_FILE, "r", encoding="utf-8"))
    return [{"text": x} for x in DEFAULT_FACTS]

def save_training():
    json.dump(st.session_state.training, open(TRAINING_FILE, "w", encoding="utf-8"), indent=2)

# =========================
# INIT STATE
# =========================
if "training" not in st.session_state:
    st.session_state.training = load_training()

if "model" not in st.session_state:
    st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")

if "chat" not in st.session_state:
    st.session_state.chat = []

if "index" not in st.session_state:
    st.session_state.index = None

if "texts" not in st.session_state:
    st.session_state.texts = []

# =========================
# BUILD INDEX
# =========================
def rebuild():
    texts = [x["text"] for x in st.session_state.training]
    st.session_state.texts = texts

    if len(texts) == 0:
        st.session_state.index = None
        return

    emb = st.session_state.model.encode(texts).astype("float32")
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)

    st.session_state.index = index

rebuild()

# =========================
# CLEAN OUTPUT
# =========================
def clean(text):
    if not text:
        return text
    return text.replace("?", "").strip()

# =========================
# SEARCH TRAINING MEMORY
# =========================
def search_training(query):
    if st.session_state.index is None:
        return None

    q = st.session_state.model.encode([query]).astype("float32")
    D, I = st.session_state.index.search(q, 1)

    idx = I[0][0]
    if idx != -1:
        return st.session_state.texts[idx]
    return None

# =========================
# AI ENGINE (GROK FIRST)
# =========================
def generate_answer(user_input):
    user_input = user_input.strip()

    # 1. GROK FIRST
    grok = call_grok(user_input)
    if grok:
        return clean(grok)

    # 2. TRAINING MEMORY
    memory = search_training(user_input)
    if memory:
        return clean(memory)

    # 3. FINAL FALLBACK
    return "Mwen pa gen repons sa kounye a."

# =========================
# LIGHT UI
# =========================
st.markdown("""
<style>
.stApp { background: #f6f8fc; }

[data-testid="stSidebar"] {
    background: #e9eef7;
}

h1,h2,h3,p,div,label {
    color: #111 !important;
}

.stTextInput input {
    background: white !important;
    color: black !important;
}

.stButton button {
    background: #4a6cff !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 🧠 Gesner AI")

    st.markdown("""
**Globalinternet.py/software**  
Built by Gesner Deslandes  

📞 (509)-47385663  
📧 deslandes78@gmail.com  
""")

    if st.button("Rebuild AI"):
        rebuild()
        st.success("Updated")

# =========================
# CHAT UI (FIXED CRASH)
# =========================
st.title("🧠 Gesner AI")

for c in st.session_state.chat:
    role = c.get("role", "bot")
    text = c.get("text", "")

    if role == "user":
        st.markdown("🧑 " + text)
    else:
        st.markdown("🤖 " + text)

msg = st.text_input("Ask something...")

if st.button("Send"):
    if msg and msg.strip():

        st.session_state.chat.append({
            "role": "user",
            "text": msg.strip()
        })

        answer = generate_answer(msg)

        st.session_state.chat.append({
            "role": "bot",
            "text": answer
        })

        st.rerun()

# =========================
# TRAINING CENTER
# =========================
st.markdown("---")
st.subheader("🧠 Training Center")

new_fact = st.text_input("Add knowledge")

if st.button("Add"):
    if new_fact and new_fact.strip():
        st.session_state.training.append({"text": new_fact.strip()})
        save_training()
        rebuild()
        st.success("Saved")
