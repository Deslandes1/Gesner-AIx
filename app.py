import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import requests
import base64
import re
from datetime import datetime

# =========================
# DATA DIRECTORY
# =========================
DATA_DIR = ".gesner_data"
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_FILE = os.path.join(DATA_DIR, "training_data.json")

# =========================
# HAITIAN KNOWLEDGE BASE
# =========================
HAITIAN_KNOWLEDGE_FACTS = [
    "Kristòf Kolon te dekouvri Ayiti an 1492.",
    "Pòtoprens se kapital Ayiti.",
    "Ayiti sitiye nan Karayib la.",
    "Tousen Louverture te yon lidè revolisyon esklav.",
    "Jan Jak Desalin te pwoklame endepandans 1804.",
    "Soup joumou se manje endepandans Ayiti.",
]

# =========================
# CORE TRAINED ANSWERS
# =========================
CORE_ANSWERS = {
    "ki moun ki dekouvri ayiti": "Kristòf Kolon te dekouvri Ayiti an 1492.",
    "ki dat ayiti endepandan": "Ayiti vin endepandan 1 janvye 1804.",
    "kijan ou rele": "Mwen se Gesner AI, kreye pa Gesner Deslandes.",
    "kisa soup joumou ye": "Soup joumou se manje libète Ayiti.",
}

# =========================
# SAVE / LOAD
# =========================
def load_training_data():
    if os.path.exists(TRAINING_FILE):
        with open(TRAINING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_training_data():
    with open(TRAINING_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.training_data, f, ensure_ascii=False, indent=2)

# =========================
# FIX OLD DATA (CRASH FIX)
# =========================
def fix_training_data():
    fixed = []

    for item in st.session_state.training_data:
        if isinstance(item, dict) and "text" in item:
            if "embedding" not in item:
                emb = st.session_state.embedding_model.encode([item["text"]])[0]
                item["embedding"] = emb.tolist()
            fixed.append(item)

    st.session_state.training_data = fixed
    save_training_data()

# =========================
# INIT DEFAULT TRAINING
# =========================
def initialize_training():
    if not st.session_state.training_data:
        for fact in HAITIAN_KNOWLEDGE_FACTS:
            emb = st.session_state.embedding_model.encode([fact])[0]
            st.session_state.training_data.append({
                "text": fact,
                "embedding": emb.tolist()
            })

        load_core_into_training()
        save_training_data()

# =========================
# LOAD CORE ANSWERS INTO AI MEMORY
# =========================
def load_core_into_training():
    for q, a in CORE_ANSWERS.items():
        text = f"{q} => {a}"
        if not any(x["text"] == text for x in st.session_state.training_data):
            emb = st.session_state.embedding_model.encode([text])[0]
            st.session_state.training_data.append({
                "text": text,
                "embedding": emb.tolist()
            })

# =========================
# CORE ANSWER ENGINE
# =========================
def get_core_answer(q):
    return CORE_ANSWERS.get(q.lower().strip(), None)

# =========================
# GROK FALLBACK
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
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 300
            },
            timeout=5
        )

        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except:
        pass

    return None

# =========================
# RESPONSE ENGINE
# =========================
def generate_response(q):
    core = get_core_answer(q)
    if core:
        return core

    # search training memory
    for item in st.session_state.training_data:
        if q.lower() in item["text"].lower():
            return item["text"]

    grok = call_grok(q)
    if grok:
        return grok

    return "Mwen pa gen repons sa. Anseye m nan Training Center."

# =========================
# FAISS INDEX
# =========================
def rebuild_index():
    if not st.session_state.training_data:
        return

    texts = []
    vectors = []

    for x in st.session_state.training_data:
        if "text" in x and "embedding" in x:
            texts.append(x["text"])
            vectors.append(np.array(x["embedding"], dtype=np.float32))

    if not vectors:
        return

    dim = len(vectors[0])
    st.session_state.index = faiss.IndexFlatL2(dim)
    st.session_state.index.add(np.array(vectors))

    st.session_state.texts = texts

# =========================
# CHAT UI
# =========================
def chat():
    st.title("🧠 Gesner AI Chat")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for m in st.session_state.chat:
        st.write(m)

    msg = st.text_input("Ask Gesner AI")

    if st.button("Send"):
        answer = generate_response(msg)

        st.session_state.chat.append(f"🧑 {msg}")
        st.session_state.chat.append(f"🤖 {answer}")

        st.rerun()

# =========================
# TRAINING CENTER
# =========================
def training_center():
    st.title("🧠 Training Center")

    new_fact = st.text_area("Teach Gesner AI")

    if st.button("Add"):
        emb = st.session_state.embedding_model.encode([new_fact])[0]

        st.session_state.training_data.append({
            "text": new_fact,
            "embedding": emb.tolist()
        })

        rebuild_index()
        save_training_data()
        st.success("Saved!")

# =========================
# SESSION INIT
# =========================
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

if "training_data" not in st.session_state:
    st.session_state.training_data = load_training_data()

if "index" not in st.session_state:
    st.session_state.index = None

if "texts" not in st.session_state:
    st.session_state.texts = []

# =========================
# UI DESIGN (UNCHANGED COLOR)
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
# MAIN APP
# =========================
def main():
    initialize_training()
    fix_training_data()
    rebuild_index()

    menu = st.sidebar.radio("Menu", ["Chat", "Training Center"])

    if menu == "Chat":
        chat()
    else:
        training_center()

main()
