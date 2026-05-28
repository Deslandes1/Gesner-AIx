import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

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
# INIT SESSION
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
# HAITIAN CORE KNOWLEDGE
# =========================
CORE = {
    "kijan ou rele": "Mwen rele Gesner AI, asistan Gesner Deslandes.",
    "ki dekouvri ayiti": "Kristòf Kolon te dekouvri Ayiti an 1492.",
    "ki dat endepandans ayiti": "Ayiti pran endepandans 1 janvye 1804.",
    "kisa kapital ayiti ye": "Pòtoprens se kapital Ayiti."
}

# =========================
# NORMALIZATION FIX
# =========================
def normalize(q):
    q = q.lower().strip()
    q = re.sub(r"\s+", " ", q)

    q = q.replace("site", "ki")
    q = q.replace("konbyen", "ki kantite")
    q = q.replace("let", "lèt")
    q = q.replace("alfabe", "alfabè")
    q = q.replace("genhen", "gen")

    return q

# =========================
# CORE ANSWER (FIXED)
# =========================
def core_answer(q):
    qn = normalize(q)

    if "alfabè" in qn and "lèt" in qn:
        return "Nan alfabè kreyòl la gen 32 lèt."

    for k, v in CORE.items():
        if k in qn:
            return v

    return None

# =========================
# TRAINING MATCH (UPGRADED)
# =========================
def training_match(q):
    q = q.lower()
    best = None
    score_best = 0

    for item in st.session_state.training_data:
        text = item.get("text", "").lower()
        score = sum(1 for w in q.split() if w in text)

        if score > score_best:
            score_best = score
            best = item["text"]

    return best if score_best >= 2 else None

# =========================
# FAISS BUILD SAFE
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
# INIT DEFAULT TRAINING
# =========================
DEFAULTS = [
    "Pòtoprens se kapital Ayiti.",
    "Ayiti pran endepandans 1 janvye 1804.",
    "Konpa se mizik Ayiti."
]

def init_training():
    if not st.session_state.training_data:
        for fact in DEFAULTS:
            emb = st.session_state.model.encode([fact])[0]
            st.session_state.training_data.append({
                "text": fact,
                "embedding": emb.tolist()
            })

        save_training()
        rebuild_index()

# =========================
# GROK (LAST RESORT ONLY)
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
                    {"role": "system", "content": "Answer in Haitian Creole."},
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
# INTELLIGENCE ENGINE (FIXED ORDER)
# =========================
def generate_response(q):

    # 1. CORE (MANDATORY FIRST)
    ans = core_answer(q)
    if ans:
        return ans

    # 2. TRAINING CENTER
    t = training_match(q)
    if t:
        return t

    # 3. FAISS SEARCH
    if st.session_state.index:
        try:
            emb = st.session_state.model.encode([q])[0].astype("float32").reshape(1, -1)
            D, I = st.session_state.index.search(emb, 1)

            if I[0][0] != -1:
                return st.session_state.texts[I[0][0]]
        except:
            pass

    # 4. GROK (LAST)
    g = call_grok(q)
    if g:
        return g

    # 5. FALLBACK
    return "Mwen pa gen repons sa kounye a. Eseye anseye m li nan Training Center."

# =========================
# UI
# =========================
st.set_page_config(page_title="Gesner AI", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}
* {
    color: white !important;
}
.stTextInput input {
    background:black !important;
    color:white !important;
}
.stTextArea textarea {
    background:black !important;
    color:white !important;
}
.stButton button {
    background:#e11d48 !important;
    color:white !important;
}
</style>
""", unsafe_allow_html=True)

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

    st.write("### Done:")
    for i, t in enumerate(st.session_state.training_data):
        st.write(f"{i+1}. {t['text']}")

# =========================
# CHAT
# =========================
def chat():
    st.title("🧠 Gesner AI Ultra")

    q = st.text_input("Poze kestyon:")

    if st.button("Voye") and q:
        r = generate_response(q)
        st.session_state.conversation.append(("You", q))
        st.session_state.conversation.append(("AI", r))

    for r in st.session_state.conversation[::-1]:
        st.write(f"**{r[0]}:** {r[1]}")

# =========================
# START
# =========================
menu = st.sidebar.radio("Menu", ["Chat", "Training Center"])

init_training()
rebuild_index()

if menu == "Chat":
    chat()
else:
    training_center()
