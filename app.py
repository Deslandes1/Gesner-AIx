import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
import os
import requests

# =========================
# DATA
# =========================
DATA_DIR = ".gesner_data"
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_FILE = os.path.join(DATA_DIR, "training.json")

# =========================
# LOAD / SAVE
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
        json.dump(st.session_state.training_data, f, indent=2, ensure_ascii=False)

# =========================
# SESSION INIT
# =========================
if "training_data" not in st.session_state:
    st.session_state.training_data = load_training()

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "model" not in st.session_state:
    st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")

if "index" not in st.session_state:
    st.session_state.index = None

if "texts" not in st.session_state:
    st.session_state.texts = []

# =========================
# CORE ANSWERS
# =========================
CORE = {
    "kijan ou rele": "Mwen rele Gesner AI, asistan Gesner Deslandes.",
    "ki dekouvri ayiti": "Kristòf Kolon te dekouvri Ayiti an 1492.",
    "ki dat endepandans ayiti": "Ayiti pran endepandans 1 janvye 1804.",
    "kisa kapital ayiti ye": "Pòtoprens se kapital Ayiti."
}

def normalize(q):
    q = q.lower().strip()
    q = re.sub(r"\s+", " ", q)
    q = q.replace("konbyen", "ki kantite")
    q = q.replace("alfabe", "alfabè")
    q = q.replace("let", "lèt")
    q = q.replace("genhen", "gen")
    return q

def core_answer(q):
    qn = normalize(q)

    if "alfabè" in qn and "lèt" in qn:
        return "Nan alfabè kreyòl la gen 32 lèt."

    for k, v in CORE.items():
        if k in qn:
            return v

    return None

# =========================
# TRAINING MATCH
# =========================
def training_match(q):
    q = q.lower()
    best = None
    best_score = 0

    for item in st.session_state.training_data:
        text = item.get("text", "").lower()
        score = sum(1 for w in q.split() if w in text)

        if score > best_score:
            best_score = score
            best = item["text"]

    return best if best_score >= 2 else None

# =========================
# FAISS
# =========================
def rebuild_index():
    valid = [x for x in st.session_state.training_data if "embedding" in x]

    if not valid:
        st.session_state.index = None
        return

    vectors = []
    texts = []

    for x in valid:
        try:
            vectors.append(np.array(x["embedding"], dtype=np.float32))
            texts.append(x["text"])
        except:
            continue

    if not vectors:
        return

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))

    st.session_state.index = index
    st.session_state.texts = texts

# =========================
# GROK (OPTIONAL)
# =========================
def call_grok(q):
    key = st.secrets.get("GROK_API_KEY", None)
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
                    {"role": "system", "content": "Answer in Haitian Creole only."},
                    {"role": "user", "content": q}
                ]
            },
            timeout=4
        )

        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except:
        pass

    return None

# =========================
# INTELLIGENCE ENGINE
# =========================
def generate_response(q):

    # 1. CORE
    ans = core_answer(q)
    if ans:
        return ans

    # 2. TRAINING
    t = training_match(q)
    if t:
        return t

    # 3. FAISS
    if st.session_state.index:
        try:
            emb = st.session_state.model.encode([q])[0].astype("float32").reshape(1, -1)
            D, I = st.session_state.index.search(emb, 1)

            if I[0][0] != -1:
                return st.session_state.texts[I[0][0]]
        except:
            pass

    # 4. GROK
    g = call_grok(q)
    if g:
        return g

    return "Mwen pa gen repons sa kounye a."

# =========================
# TRAINING CENTER
# =========================
def training_center():
    st.subheader("📚 Training Center")

    txt = st.text_area("Anseye AI nouvo reyalite")

    if st.button("Ajoute"):
        emb = st.session_state.model.encode([txt])[0]
        st.session_state.training_data.append({
            "text": txt,
            "embedding": emb.tolist()
        })
        save_training()
        rebuild_index()
        st.success("Ajoute!")

    for i, t in enumerate(st.session_state.training_data):
        st.write(f"{i+1}. {t['text']}")

# =========================
# CHAT (FIX: NO QUESTION REPEAT)
# =========================
def chat():
    st.title("🧠 Gesner AI Ultra")

    q = st.text_input("Poze kestyon:")

    if st.button("Voye") and q:
        r = generate_response(q)

        # FIX: no repetition of question in AI memory
        clean_answer = r.strip()

        st.session_state.conversation.append(clean_answer)

    for msg in st.session_state.conversation[::-1]:
        st.write(f"🤖 {msg}")

# =========================
# SIDEBAR (UPDATED + SAME STYLE)
# =========================
def sidebar():
    st.sidebar.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #0f172a, #1e293b);
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("🧠 Gesner AI")

    st.sidebar.markdown("""
**Company:** Globalinternet.py / Software  
**Built by:** Gesner Deslandes  
**Phone:** (509) 4738-5663  
**Email:** deslandes78@gmail.com  
""")

# =========================
# START
# =========================
st.set_page_config(page_title="Gesner AI", layout="wide")

sidebar()

menu = st.sidebar.radio("Menu", ["Chat", "Training Center"])

if not st.session_state.index:
    rebuild_index()

if menu == "Chat":
    chat()
else:
    training_center()
