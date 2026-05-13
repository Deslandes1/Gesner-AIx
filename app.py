import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time
import hashlib
import re
import base64
import csv
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

st.set_page_config(
    page_title="Gesner AI",
    page_icon="🧠",
    layout="wide"
)

# ---------- SESSION STATE ----------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "embedding_model" not in st.session_state:
    with st.spinner("Loading AI model... (first time only)"):
        st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

if "training_data" not in st.session_state:
    st.session_state.training_data = []

if "index" not in st.session_state:
    st.session_state.index = None

if "texts" not in st.session_state:
    st.session_state.texts = []

if "training_access" not in st.session_state:
    st.session_state.training_access = False

if "chat_language" not in st.session_state:
    st.session_state.chat_language = "en"

if "tfidf_vectorizer" not in st.session_state:
    st.session_state.tfidf_vectorizer = None

if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None


# ---------- CORE FIXED ENGINE ----------

def build_tfidf():
    if len(st.session_state.texts) > 0:
        st.session_state.tfidf_vectorizer = TfidfVectorizer()
        st.session_state.tfidf_matrix = st.session_state.tfidf_vectorizer.fit_transform(st.session_state.texts)


def rebuild_index():
    if len(st.session_state.training_data) == 0:
        st.session_state.index = None
        st.session_state.texts = []
        return

    texts = []
    embeddings = []

    for item in st.session_state.training_data:
        texts.append(item["text"])
        embeddings.append(np.array(item["embedding"], dtype=np.float32))

    embeddings = np.vstack(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    st.session_state.index = index
    st.session_state.texts = texts

    build_tfidf()


def add_to_training(text):
    if not text.strip():
        return

    embedding = st.session_state.embedding_model.encode([text])[0].astype(np.float32)

    st.session_state.training_data.append({
        "text": text,
        "embedding": embedding.tolist()
    })

    rebuild_index()


# ---------- RESPONSE SYSTEM ----------

def retrieve(query):
    if st.session_state.index is None:
        return None

    query_vec = st.session_state.embedding_model.encode([query])[0].astype(np.float32).reshape(1, -1)

    D, I = st.session_state.index.search(query_vec, 1)

    if I[0][0] != -1 and D[0][0] < 1.2:
        return st.session_state.texts[I[0][0]]

    # TFIDF fallback
    if st.session_state.tfidf_vectorizer:
        q = st.session_state.tfidf_vectorizer.transform([query])
        sims = cosine_similarity(q, st.session_state.tfidf_matrix).flatten()
        idx = sims.argmax()
        if sims[idx] > 0.2:
            return st.session_state.texts[idx]

    return None


def generate_response(q):
    q_lower = q.lower()

    # simple reasoning
    math = re.search(r"(\d+)\s*([\+\-\*\/])\s*(\d+)", q_lower)
    if math:
        a, op, b = int(math.group(1)), math.group(2), int(math.group(3))
        return str(eval(f"{a}{op}{b}"))

    if "who created you" in q_lower:
        return "I was created by Gesner Deslandes."

    result = retrieve(q)

    if result:
        return result

    return "I don’t know yet. Train me in the Training Center."


# ---------- UI ----------

def chat():
    st.title("💬 Gesner AI Chat")

    for msg in st.session_state.conversation_history:
        role = "🧑" if msg["role"] == "user" else "🤖"
        st.write(f"{role}: {msg['content']}")

    user = st.text_input("Ask something")

    if st.button("Send"):
        if user:
            st.session_state.conversation_history.append({"role": "user", "content": user})
            reply = generate_response(user)
            st.session_state.conversation_history.append({"role": "assistant", "content": reply})
            st.rerun()

    if st.button("Clear Chat"):
        st.session_state.conversation_history = []
        st.rerun()


def training():
    st.title("🔧 Training Center")

    text = st.text_area("Teach something")

    if st.button("Train"):
        if text:
            add_to_training(text)
            st.success("Trained!")

    st.markdown("### 📚 Learned Data")
    for i, item in enumerate(st.session_state.training_data):
        st.write(f"{i+1}. {item['text']}")


# ---------- SIDEBAR ----------

def sidebar():
    st.sidebar.title("Gesner AI")

    if not st.session_state.training_access:
        key = st.sidebar.text_input("API Key", type="password")
        if st.sidebar.button("Unlock"):
            if key == "GESNER-UNLOCK":
                st.session_state.training_access = True
                st.rerun()
            else:
                st.sidebar.error("Wrong key")
    else:
        st.sidebar.success("Training unlocked")
        if st.sidebar.button("Lock"):
            st.session_state.training_access = False
            st.rerun()


# ---------- MAIN ----------

def main():
    sidebar()

    if st.session_state.training_access:
        mode = st.radio("Mode", ["Chat", "Training"])

        if mode == "Chat":
            chat()
        else:
            training()
    else:
        chat()


if __name__ == "__main__":
    main()
