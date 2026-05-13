import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import hashlib
import re
import base64
import csv
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Gesner AI", page_icon="🧠", layout="wide")

# =========================
# SESSION INIT (SAFE)
# =========================
def init():
    defaults = {
        "conversation_history": [],
        "training_data": [],
        "texts": [],
        "index": None,
        "tfidf_vectorizer": None,
        "tfidf_matrix": None,
        "training_access": False,
        "chat_language": "en",
        "ui_language": "en",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()

# =========================
# MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =========================
# INDEX BUILDER (SAFE)
# =========================
def rebuild_index():
    data = st.session_state.training_data

    if not data:
        st.session_state.index = None
        st.session_state.texts = []
        return

    texts = [x["text"] for x in data]
    embeddings = np.array([x["embedding"] for x in data]).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    st.session_state.index = index
    st.session_state.texts = texts

    vec = TfidfVectorizer()
    st.session_state.tfidf_vectorizer = vec
    st.session_state.tfidf_matrix = vec.fit_transform(texts)

# =========================
# TRAINING
# =========================
def add_training(text):
    text = text.strip()
    if not text:
        return

    emb = model.encode([text])[0].tolist()

    st.session_state.training_data.append({
        "text": text,
        "embedding": emb
    })

    rebuild_index()

# =========================
# RETRIEVAL
# =========================
def retrieve(query):
    if st.session_state.index is None:
        return None

    q_emb = model.encode([query])[0].astype("float32").reshape(1, -1)

    D, I = st.session_state.index.search(q_emb, 1)

    if I[0][0] != -1 and D[0][0] < 1.2:
        return st.session_state.texts[I[0][0]]

    vec = st.session_state.tfidf_vectorizer
    mat = st.session_state.tfidf_matrix

    if vec is not None:
        q = vec.transform([query])
        scores = cosine_similarity(q, mat).flatten()
        idx = np.argmax(scores)

        if scores[idx] > 0.2:
            return st.session_state.texts[idx]

    return None

# =========================
# SIMPLE LOGIC ENGINE
# =========================
def logic(q):
    q = q.lower()

    m = re.search(r"(\d+)\s*([\+\-\*\/])\s*(\d+)", q)
    if m:
        return str(eval(f"{m[1]}{m[2]}{m[3]}"))

    if "time" in q:
        return datetime.now().strftime("%H:%M")

    return None

# =========================
# RESPONSE
# =========================
def answer(q):
    r = retrieve(q)
    if r:
        return r

    l = logic(q)
    if l:
        return l

    return "I don't know yet. Train me."

# =========================
# CHARACTER PICKER (TRAINING ONLY)
# =========================
def character_picker(key_prefix):
    if not st.session_state.training_access:
        return

    chars = ["e", "è", "o", "ò", "an", "en", "on"]

    cols = st.columns(len(chars))

    for i, ch in enumerate(chars):
        with cols[i]:
            if st.button(ch, key=f"char_{key_prefix}_{ch}"):
                idx = key_prefix.split("_")[1]
                key = f"edit_text_{idx}"
                st.session_state[key] = st.session_state.get(key, "") + ch

# =========================
# UI HEADER
# =========================
st.title("💬 Gesner AI")

# =========================
# CHAT DISPLAY
# =========================
for msg in st.session_state.conversation_history:
    if msg["role"] == "user":
        st.markdown(f"🧑 {msg['content']}")
    else:
        st.markdown(f"🤖 {msg['content']}")

# =========================
# INPUT
# =========================
q = st.text_input("Ask something")

if st.button("Send"):
    if q:
        st.session_state.conversation_history.append({"role": "user", "content": q})
        a = answer(q)
        st.session_state.conversation_history.append({"role": "assistant", "content": a})
        st.rerun()

# =========================
# SIDEBAR TRAINING
# =========================
st.sidebar.title("Training")

train_text = st.sidebar.text_input("Teach me")

if st.sidebar.button("Train"):
    if train_text:
        add_training(train_text)
        st.sidebar.success("Learned!")

# =========================
# MANAGE TRAINING DATA
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("Manage Data")

for i, item in enumerate(st.session_state.training_data):
    with st.sidebar.expander(f"Fact {i+1}"):

        key = f"edit_{i}"

        # CHARACTER PICKER ONLY HERE (TRAINING ONLY)
        character_picker(key)

        new_text = st.text_area(
            "Edit",
            value=item["text"],
            key=f"edit_text_{i}"
        )

        if st.button("Save", key=f"save_{i}"):
            emb = model.encode([new_text])[0].tolist()
            st.session_state.training_data[i] = {
                "text": new_text,
                "embedding": emb
            }
            rebuild_index()
            st.rerun()

        if st.button("Delete", key=f"del_{i}"):
            st.session_state.training_data.pop(i)
            rebuild_index()
            st.rerun()

# =========================
# CLEAR CHAT
# =========================
if st.sidebar.button("Clear Chat"):
    st.session_state.conversation_history = []
    st.rerun()
