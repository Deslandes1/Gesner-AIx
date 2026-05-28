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
# LIGHT UI
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #f5f7ff;
}
[data-testid="stSidebar"] {
    background-color: #eef2ff;
}
.stTextInput input, .stTextArea textarea {
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
# KNOWLEDGE BASE
# =========================
HAITIAN_KNOWLEDGE_FACTS = [
    "Kristòf Kolon te dekouvri Ayiti nan 1492.",
    "Pòtoprens se kapital Ayiti.",
    "Ayiti sitiye nan Karayib la.",
    "Jan Jak Desalin te pwoklame endepandans 1804."
]

CORE_ANSWERS = {
    "kijan ou rele": "Mwen rele Gesner AI.",
    "ki kote ayiti ye": "Ayiti sitiye nan Karayib la.",
    "ki moun ki dekouvri ayiti": "Kristòf Kolon te dekouvri Ayiti nan 1492.",
    "soup joumou": "Soup joumou se manje endepandans Ayiti."
}

# =========================
# MODEL
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
# SAFE INDEX
# =========================
def rebuild_index():
    texts = []
    vectors = []

    for item in st.session_state.training_data:
        if "text" in item:
            texts.append(item["text"])
            vec = st.session_state.model.encode(item["text"])
            vectors.append(vec)

    if len(vectors) == 0:
        return

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))

    st.session_state.index = index
    st.session_state.texts = texts

# =========================
# INIT TRAINING
# =========================
def init_training():
    existing = {x["text"] for x in st.session_state.training_data if "text" in x}

    for fact in HAITIAN_KNOWLEDGE_FACTS:
        if fact not in existing:
            st.session_state.training_data.append({"text": fact})

    rebuild_index()

init_training()

# =========================
# CORE ANSWERS
# =========================
def core_answer(q):
    return CORE_ANSWERS.get(q.lower().strip())

# =========================
# INTENT ROUTER
# =========================
def intent_router(q):
    q = q.lower()

    if "kijan ou rele" in q:
        return "Mwen rele Gesner AI."

    if "ki kote ayiti ye" in q:
        return "Ayiti sitiye nan Karayib la."

    if "ki moun ki dekouvri ayiti" in q:
        return "Kristòf Kolon te dekouvri Ayiti nan 1492."

    if "soup joumou" in q:
        return "Soup joumou se manje endepandans Ayiti."

    return None

# =========================
# MEMORY SEARCH
# =========================
def search_memory(query):
    if st.session_state.index is None:
        return []

    vec = st.session_state.model.encode(query).astype("float32").reshape(1, -1)
    _, idx = st.session_state.index.search(vec, 3)

    results = []
    for i in idx[0]:
        if i != -1 and i < len(st.session_state.texts):
            results.append(st.session_state.texts[i])

    return results

# =========================
# GROK API (ONLINE BRAIN)
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
                    {"role": "system", "content": "Answer ONLY in Haitian Creole. Be direct and concise."},
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=6
        )

        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
    except:
        return None

# =========================
# MAIN INTELLIGENCE ENGINE (FIXED)
# =========================
def generate_answer(user_input):
    q = user_input.lower().strip()

    # 1. CORE
    ca = core_answer(q)
    if ca:
        return ca

    # 2. INTENT
    intent = intent_router(q)
    if intent:
        return intent

    # 3. MEMORY (ONLY SAFE MATCH)
    mem = search_memory(user_input)
    if mem:
        best = mem[0]
        if any(w in best.lower() for w in q.split()[:2]):
            return best

    # 4. GROK = PRIMARY ONLINE BRAIN FOR UNKNOWN QUESTIONS
    grok = grok_call(user_input)
    if grok:
        return grok

    # 5. FINAL FALLBACK
    return "Mwen pa gen repons sa kounye a."

# =========================
# CHAT UI
# =========================
def chat():
    st.title("🧠 Gesner AI")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for r, t in st.session_state.chat:
        st.write("🧑" if r == "user" else "🤖", t)

    user = st.text_input("Poze kestyon")

    if st.button("Send") and user:
        st.session_state.chat.append(("user", user))

        answer = generate_answer(user)

        # CLEAN OUTPUT (no repetition)
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
# RUN APP
# =========================
sidebar()
chat()
