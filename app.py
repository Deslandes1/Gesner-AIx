import streamlit as st
import json
import numpy as np
import faiss
import os
import requests
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Gesner AI", page_icon="🧠", layout="wide")

DATA_DIR = ".gesner_data"
os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# UI (LIGHT MODE)
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #f5f7ff;
}
[data-testid="stSidebar"] {
    background-color: #eef2ff;
}
.stTextInput input {
    background-color: white !important;
    color: black !important;
}
.stButton button {
    background-color: #4f46e5 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# CORE KNOWLEDGE (ONLY SAFE FACTS)
# =========================
CORE_ANSWERS = {
    "kijan ou rele": "Mwen rele Gesner AI.",
    "ki kote ayiti ye": "Ayiti sitiye nan Karayib la.",
    "ki moun ki dekouvri ayiti": "Kristòf Kolon te dekouvri Ayiti nan 1492."
}

# =========================
# MODEL (FOR OPTIONAL MEMORY ONLY)
# =========================
if "model" not in st.session_state:
    st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")

if "training_data" not in st.session_state:
    st.session_state.training_data = []

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.texts = []

# =========================
# SAFE FAISS BUILD
# =========================
def rebuild_index():
    texts = []
    vectors = []

    for item in st.session_state.training_data:
        if "text" in item:
            texts.append(item["text"])
            vec = st.session_state.model.encode(item["text"])
            vectors.append(vec)

    if not vectors:
        return

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))

    st.session_state.index = index
    st.session_state.texts = texts

# =========================
# CORE ANSWER CHECK
# =========================
def core_answer(q):
    return CORE_ANSWERS.get(q.lower().strip())

# =========================
# GROK (ONLINE BRAIN)
# =========================
def grok_call(prompt):
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
                    {
                        "role": "system",
                        "content": "You are Gesner AI. Always answer in Haitian Creole. Be direct and accurate."
                    },
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=8
        )

        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
    except:
        return None

# =========================
# MAIN ENGINE (GROK-FIRST LOGIC)
# =========================
def generate_answer(user_input):
    q = user_input.lower().strip()

    # 1. CORE LOCAL ANSWERS
    if core_answer(q):
        return core_answer(q)

    # 2. ALWAYS USE GROK FOR UNKNOWN QUESTIONS
    grok = grok_call(user_input)
    if grok:
        return grok

    # 3. FINAL FALLBACK
    return "Mwen pa gen repons sa kounye a."

# =========================
# CHAT UI
# =========================
def chat():
    st.title("🧠 Gesner AI (Grok Powered)")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, msg in st.session_state.chat:
        st.write("🧑" if role == "user" else "🤖", msg)

    user = st.text_input("Poze kestyon")

    if st.button("Send") and user:
        st.session_state.chat.append(("user", user))

        answer = generate_answer(user)

        # clean output
        answer = answer.replace(user, "").strip()

        st.session_state.chat.append(("ai", answer))
        st.rerun()

# =========================
# SIDEBAR
# =========================
def sidebar():
    st.sidebar.markdown("## 🧠 Gesner AI")
    st.sidebar.markdown("""
**Globalinternet.py/software**  
Built by Gesner Deslandes  
📞 (509)-47385663  
📧 deslandes78@gmail.com  
""")

# =========================
# RUN
# =========================
sidebar()
chat()
