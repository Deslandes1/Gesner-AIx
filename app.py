import streamlit as st
import numpy as np
import faiss
import json
import os
import requests
import hashlib
from sentence_transformers import SentenceTransformer
from datetime import datetime

# =========================
# DATA
# =========================
DATA_DIR = ".gesner_data"
os.makedirs(DATA_DIR, exist_ok=True)
TRAIN_FILE = os.path.join(DATA_DIR, "training.json")

# =========================
# HAITIAN KNOWLEDGE
# =========================
HAITIAN_KNOWLEDGE = [
    "Kristòf Kolon te dekouvri Ayiti an 1492.",
    "Pòtoprens se kapital Ayiti.",
    "Ayiti sitiye nan Karayib la.",
    "Tousen Louverture te yon lidè revolisyon.",
    "Jan Jak Desalin te pwoklame endepandans 1804.",
    "Soup joumou se manje endepandans.",
]

# =========================
# CORE INTELLIGENCE
# =========================
CORE = {
    "kijan ou rele": "Mwen se Gesner AI, kreye pa Gesner Deslandes.",
    "ki moun ki dekouvri ayiti": "Kristòf Kolon te dekouvri Ayiti an 1492.",
    "ki dat ayiti endepandan": "Ayiti vin endepandan 1 janvye 1804.",
}

# =========================
# LOAD / SAVE
# =========================
def load_data():
    if os.path.exists(TRAIN_FILE):
        with open(TRAIN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_data():
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.data, f, ensure_ascii=False, indent=2)

# =========================
# FIX OLD DATA
# =========================
def fix_data():
    fixed = []
    for item in st.session_state.data:
        if isinstance(item, dict) and "text" in item:
            if "embedding" not in item:
                emb = st.session_state.model.encode([item["text"]])[0]
                item["embedding"] = emb.tolist()
            fixed.append(item)
    st.session_state.data = fixed
    save_data()

# =========================
# INIT TRAINING
# =========================
def init_training():
    if not st.session_state.data:
        for f in HAITIAN_KNOWLEDGE:
            emb = st.session_state.model.encode([f])[0]
            st.session_state.data.append({
                "text": f,
                "embedding": emb.tolist()
            })
        save_data()

# =========================
# CORE ANSWER
# =========================
def core_answer(q):
    return CORE.get(q.lower().strip())

# =========================
# TRAINING ANSWER (FIXED INTELLIGENCE)
# =========================
def training_answer(q):
    q = q.lower().strip()

    for item in st.session_state.data:
        text = item["text"].lower()

        if q in text:
            if "=>" in text:
                return text.split("=>")[1].strip()
            return text

    return None

# =========================
# FAISS SEARCH (REAL AI MEMORY)
# =========================
def search_memory(q):
    if st.session_state.index is None:
        return None

    q_emb = st.session_state.model.encode([q])[0]
    D, I = st.session_state.index.search(np.array([q_emb]).astype("float32"), 3)

    for i in I[0]:
        if i != -1:
            return st.session_state.texts[i]
    return None

# =========================
# GROK FALLBACK
# =========================
def grok(q):
    key = st.secrets.get("GROK_API_KEY")
    if not key:
        return None

    try:
        r = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "grok-1",
                "messages": [{"role": "user", "content": q}],
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
# MAIN AI BRAIN (FULL INTELLIGENCE PIPELINE)
# =========================
def brain(q):
    c = core_answer(q)
    if c:
        return c

    t = training_answer(q)
    if t:
        return t

    m = search_memory(q)
    if m:
        return m

    g = grok(q)
    if g:
        return g

    return "Mwen pa konn sa. Anseye m li nan Training Center."

# =========================
# AUDIO (SIMPLE CACHE)
# =========================
VOICE_CACHE = {}

def voice_key(text):
    return hashlib.md5(text.encode()).hexdigest()

# =========================
# FAISS BUILD
# =========================
def rebuild():
    texts = []
    vectors = []

    for x in st.session_state.data:
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
# SESSION INIT
# =========================
if "model" not in st.session_state:
    st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")

if "data" not in st.session_state:
    st.session_state.data = load_data()

if "index" not in st.session_state:
    st.session_state.index = None

if "texts" not in st.session_state:
    st.session_state.texts = []

# =========================
# UI STYLE (UNCHANGED)
# =========================
st.set_page_config(page_title="Gesner AI", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
}
.stTextArea textarea {
    background:black;
    color:white;
    font-weight:bold;
}
.stButton button {
    background:#e94560;
    color:white;
}
</style>
""", unsafe_allow_html=True)

# =========================
# CHAT
# =========================
def chat():
    st.title("🧠 Gesner AI Ultra Intelligence")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for c in st.session_state.chat:
        st.write(c)

    msg = st.text_input("Ask")

    if st.button("Send"):
        answer = brain(msg)

        st.session_state.chat.append(f"🧑 {msg}")
        st.session_state.chat.append(f"🤖 {answer}")

        # AUTO LEARN (optional intelligence growth)
        emb = st.session_state.model.encode([msg + " => " + answer])[0]
        st.session_state.data.append({
            "text": msg + " => " + answer,
            "embedding": emb.tolist()
        })

        save_data()
        rebuild()

        st.rerun()

# =========================
# TRAINING CENTER
# =========================
def training():
    st.title("🧠 Training Center")

    t = st.text_area("Teach AI (text or Q => A)")

    if st.button("Add"):
        emb = st.session_state.model.encode([t])[0]
        st.session_state.data.append({
            "text": t,
            "embedding": emb.tolist()
        })
        save_data()
        rebuild()
        st.success("Learned!")

# =========================
# MAIN
# =========================
def main():
    init_training()
    fix_data()
    rebuild()

    menu = st.sidebar.radio("Menu", ["Chat", "Training Center"])

    if menu == "Chat":
        chat()
    else:
        training()

main()
