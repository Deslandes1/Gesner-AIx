import os
import pickle
import hashlib
import time
import base64
import json
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch

# -------------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------------
KNOWLEDGE_DIR = Path("./knowledge")
INDEX_PATH = Path("./faiss_index")
DOCS_PATH = Path("./documents.pkl")
VOICE_CACHE_PATH = Path("./voice_cache.json")
DICTIONARY_PATH = Path("./dictionary.json")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Use a small, fast model that always replies
LLM_MODEL = "google/flan-t5-small"   # ~300MB, fast generation

# -------------------------------------------------------------------
# 2. Session State
# -------------------------------------------------------------------
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "voice_cache" not in st.session_state:
        st.session_state.voice_cache = load_voice_cache()
    if "dictionary" not in st.session_state:
        st.session_state.dictionary = load_dictionary()
    if "index" not in st.session_state:
        st.session_state.index = None
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "last_response" not in st.session_state:
        st.session_state.last_response = ""

# -------------------------------------------------------------------
# 3. Knowledge Base & FAISS
# -------------------------------------------------------------------
def load_documents():
    docs = []
    if KNOWLEDGE_DIR.exists():
        for file in KNOWLEDGE_DIR.glob("*.txt"):
            with open(file, "r", encoding="utf-8") as f:
                docs.append({"filename": file.name, "text": f.read()})
    return docs

def rebuild_index():
    st.info("Rebuilding knowledge index...")
    docs = load_documents()
    if not docs:
        st.warning("No .txt files found in ./knowledge/")
        st.session_state.documents = []
        st.session_state.index = None
        return
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(INDEX_PATH))
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)
    st.session_state.index = index
    st.session_state.documents = docs
    st.success(f"Index rebuilt with {len(docs)} docs.")

def load_index():
    if INDEX_PATH.exists() and DOCS_PATH.exists():
        st.session_state.index = faiss.read_index(str(INDEX_PATH))
        with open(DOCS_PATH, "rb") as f:
            st.session_state.documents = pickle.load(f)
        return True
    return False

def retrieve_context(query, top_k=3):
    if st.session_state.index is None or not st.session_state.documents:
        return ""
    model = SentenceTransformer(EMBEDDING_MODEL)
    q_emb = model.encode([query])
    dist, idxs = st.session_state.index.search(q_emb.astype(np.float32), top_k)
    contexts = [st.session_state.documents[i]["text"] for i in idxs[0] if i != -1]
    return "\n\n".join(contexts)

# -------------------------------------------------------------------
# 4. LLM Response (fast, always replies)
# -------------------------------------------------------------------
@st.cache_resource
def load_llm():
    # Use a small T5 model – fast and reliable
    return pipeline("text2text-generation", model=LLM_MODEL, device=-1)  # CPU

def get_ai_response(user_input):
    context = retrieve_context(user_input)
    prompt = f"""Answer the user's question based on the context below. If the context is empty, answer using general knowledge.

Context:
{context}

Question: {user_input}
Answer:"""
    try:
        llm = load_llm()
        result = llm(prompt, max_length=200, do_sample=True, temperature=0.7)[0]["generated_text"]
        if not result.strip():
            return "I'm not sure how to answer that. Could you rephrase?"
        return result.strip()
    except Exception as e:
        st.error(f"LLM error: {e}")
        return "Sorry, I encountered an error. Please try again."

# -------------------------------------------------------------------
# 5. Voice Training
# -------------------------------------------------------------------
def load_voice_cache():
    if VOICE_CACHE_PATH.exists():
        with open(VOICE_CACHE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_voice_cache():
    with open(VOICE_CACHE_PATH, "w") as f:
        json.dump(st.session_state.voice_cache, f, indent=2)

def train_voice(name, audio_bytes):
    key = hashlib.md5(audio_bytes).hexdigest()
    st.session_state.voice_cache[key] = {
        "name": name,
        "audio_b64": base64.b64encode(audio_bytes).decode("utf-8"),
        "timestamp": time.time()
    }
    save_voice_cache()
    st.success(f"Voice '{name}' trained!")

# -------------------------------------------------------------------
# 6. Dictionary
# -------------------------------------------------------------------
def load_dictionary():
    if DICTIONARY_PATH.exists():
        with open(DICTIONARY_PATH, "r") as f:
            return json.load(f)
    return {}

def save_dictionary():
    with open(DICTIONARY_PATH, "w") as f:
        json.dump(st.session_state.dictionary, f, indent=2)

def add_term(term, definition):
    st.session_state.dictionary[term] = definition
    save_dictionary()

def delete_term(term):
    if term in st.session_state.dictionary:
        del st.session_state.dictionary[term]
        save_dictionary()

# -------------------------------------------------------------------
# 7. UI Pages (with your original styling)
# -------------------------------------------------------------------
def chat_interface(t):
    st.markdown("### 💬 Chat with Gesner AI")
    # Display chat history
    chat_display = "\n".join(st.session_state.messages)
    st.text_area("Conversation", value=chat_display, height=400,
                 key="chat_display", disabled=True, label_visibility="collapsed")

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Message", key="user_msg",
                                   placeholder=t['chat_input'],
                                   label_visibility="collapsed")
        submitted = st.form_submit_button("Send", use_container_width=True)
    if submitted and user_input:
        st.session_state.messages.append(f"You: {user_input}")
        with st.spinner("Gesner is thinking..."):
            reply = get_ai_response(user_input)
        st.session_state.messages.append(f"🧠 Gesner AI: {reply}")
        st.rerun()

def dictionary_manager(t):
    st.markdown("### 📖 Dictionary")
    with st.form("add_dict"):
        term = st.text_input("Term")
        defi = st.text_area("Definition")
        if st.form_submit_button("Add/Update"):
            if term and defi:
                add_term(term, defi)
                st.rerun()
    st.markdown("#### Existing entries")
    for term, defi in list(st.session_state.dictionary.items()):
        col1, col2, col3 = st.columns([2, 3, 1])
        col1.write(f"**{term}**")
        col2.write(defi)
        if col3.button("❌", key=f"del_{term}"):
            delete_term(term)
            st.rerun()

def voice_training_center(t):
    st.markdown("### 🎤 Voice Training")
    name = st.text_input("Voice name")
    audio = st.file_uploader("Upload audio (WAV/MP3)", type=["wav", "mp3"])
    if st.button("Train") and name and audio:
        train_voice(name, audio.read())
    st.markdown("#### Trained voices")
    for key, data in st.session_state.voice_cache.items():
        st.write(f"- {data['name']} ({time.ctime(data['timestamp'])})")

def training_center(t):
    st.markdown("### 🧠 Training Center")
    if st.button("🔄 Rebuild Knowledge Index"):
        rebuild_index()
        st.rerun()
    if st.button("🎤 Clear Voice Cache"):
        st.session_state.voice_cache = {}
        save_voice_cache()
        st.success("Voice cache cleared")
        st.rerun()

def show_sidebar():
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=Gesner+AI", use_container_width=False)
        st.markdown("### Navigation")
        menu = st.radio("Go to", ["Chat", "Dictionary", "Voice Training", "Training Center"],
                        label_visibility="collapsed")
        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.caption("Gesner AI")
    t = {
        "chat_input": "Ask me anything...",
        "chat_interface_label": "Chat",
        "dictionary": "Dictionary",
        "voice_training": "Voice Training",
        "training_center": "Training Center"
    }
    return menu, t

# -------------------------------------------------------------------
# 8. Main
# -------------------------------------------------------------------
def main():
    # Original custom CSS (from your logs, adapted)
    st.markdown("""
    <style>
        /* Keep the original dark background */
        .stApp {
            background-color: #0e1117;
        }
        /* Sidebar styling as in your logs */
        [data-testid="stSidebar"] {
            background-color: #1e1e2f;
        }
        [data-testid="stSidebar"] .stSelectbox {
            background-color: #000000 !important;
            border-radius: 12px !important;
        }
        [data-testid="stSidebar"] .stSelectbox svg {
            fill: #e94560 !important;
            stroke: #e94560 !important;
        }
        div[data-baseweb="popover"] ul {
            background-color: #000000 !important;
        }
        /* Chat area */
        [data-testid="stTextArea"] textarea {
            background-color: #1e1e2f !important;
            color: white !important;
            border-radius: 12px;
        }
        /* Buttons */
        .stButton button {
            background-color: #e94560;
            color: white;
            border-radius: 20px;
        }
        .stButton button:hover {
            background-color: #ff6b8b;
        }
        /* Input field */
        [data-testid="stForm"] input {
            background-color: #2a2c3a;
            color: white;
            border-radius: 25px;
            border: 1px solid #e94560;
        }
    </style>
    """, unsafe_allow_html=True)

    st.set_page_config(page_title="Gesner AI", page_icon="🧠", layout="wide")
    init_session_state()

    if not load_index():
        if KNOWLEDGE_DIR.exists() and any(KNOWLEDGE_DIR.glob("*.txt")):
            rebuild_index()
        else:
            st.info("No knowledge base found. Place .txt files in ./knowledge/")

    menu, t = show_sidebar()
    if menu == t["chat_interface_label"]:
        chat_interface(t)
    elif menu == t["dictionary"]:
        dictionary_manager(t)
    elif menu == t["voice_training"]:
        voice_training_center(t)
    elif menu == t["training_center"]:
        training_center(t)

if __name__ == "__main__":
    main()
