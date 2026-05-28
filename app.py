import os
import pickle
import hashlib
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline   # <-- removed 'Conversation'
import json
import base64
import time

# -------------------------------------------------------------------
# 1. Configuration & Constants
# -------------------------------------------------------------------
KNOWLEDGE_DIR = Path("./knowledge")
INDEX_PATH = Path("./faiss_index")
DOCS_PATH = Path("./documents.pkl")
VOICE_CACHE_PATH = Path("./voice_cache.json")
DICTIONARY_PATH = Path("./dictionary.json")

MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "microsoft/DialoGPT-medium"   # or any text generation model

# -------------------------------------------------------------------
# 2. Session State Initialization
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

# -------------------------------------------------------------------
# 3. Knowledge Base & FAISS Index
# -------------------------------------------------------------------
def load_documents():
    """Load all .txt files from KNOWLEDGE_DIR."""
    docs = []
    if KNOWLEDGE_DIR.exists():
        for file in KNOWLEDGE_DIR.glob("*.txt"):
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
                docs.append({"filename": file.name, "text": text})
    return docs

def rebuild_index():
    """Build or load the FAISS index from knowledge documents."""
    st.info("Rebuilding knowledge index...")
    docs = load_documents()
    if not docs:
        st.warning("No knowledge documents found. Place .txt files in ./knowledge/")
        st.session_state.documents = []
        st.session_state.index = None
        return

    model = SentenceTransformer(MODEL_NAME)
    texts = [doc["text"] for doc in docs]
    embeddings = model.encode(texts, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    # Save for future runs
    faiss.write_index(index, str(INDEX_PATH))
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)

    st.session_state.index = index
    st.session_state.documents = docs
    st.success(f"Index rebuilt with {len(docs)} documents.")

def load_index():
    """Load existing FAISS index and documents if available."""
    if INDEX_PATH.exists() and DOCS_PATH.exists():
        st.session_state.index = faiss.read_index(str(INDEX_PATH))
        with open(DOCS_PATH, "rb") as f:
            st.session_state.documents = pickle.load(f)
        return True
    return False

def retrieve_context(query, top_k=3):
    """Retrieve top_k relevant text chunks from the knowledge base."""
    if st.session_state.index is None or not st.session_state.documents:
        return ""
    model = SentenceTransformer(MODEL_NAME)
    query_emb = model.encode([query])
    distances, indices = st.session_state.index.search(query_emb.astype(np.float32), top_k)
    contexts = []
    for idx in indices[0]:
        if idx != -1:
            contexts.append(st.session_state.documents[idx]["text"])
    return "\n\n".join(contexts)

# -------------------------------------------------------------------
# 4. LLM Response Generation (with RAG)
# -------------------------------------------------------------------
@st.cache_resource
def load_generator():
    # Using a text generation pipeline
    return pipeline("text-generation", model=LLM_MODEL_NAME)

def get_ai_response(user_input):
    """Generate response using LLM + retrieved context."""
    context = retrieve_context(user_input)
    prompt = f"""You are Gesner AI, a helpful assistant. Use the following context to answer the user's question. If the context is empty, use your general knowledge.

Context:
{context}

User: {user_input}
Gesner AI:"""
    
    generator = load_generator()
    result = generator(prompt, do_sample=True, max_new_tokens=150, temperature=0.7)[0]["generated_text"]
    # Extract only the assistant's answer (after "Gesner AI:")
    answer = result.split("Gesner AI:")[-1].strip()
    return answer

# -------------------------------------------------------------------
# 5. Voice Training & Caching
# -------------------------------------------------------------------
def load_voice_cache():
    if VOICE_CACHE_PATH.exists():
        with open(VOICE_CACHE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_voice_cache():
    with open(VOICE_CACHE_PATH, "w") as f:
        json.dump(st.session_state.voice_cache, f, indent=2)

def train_voice(voice_name, audio_bytes):
    """Store audio bytes as base64 in cache (simulate training)."""
    key = hashlib.md5(audio_bytes).hexdigest()
    st.session_state.voice_cache[key] = {
        "name": voice_name,
        "audio_b64": base64.b64encode(audio_bytes).decode("utf-8"),
        "timestamp": time.time()
    }
    save_voice_cache()
    st.success(f"Voice '{voice_name}' trained successfully!")

# -------------------------------------------------------------------
# 6. Dictionary Manager
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
# 7. UI Pages
# -------------------------------------------------------------------
def chat_interface(t):
    st.markdown("## 💬 Gesner AI Chat")
    
    # Display conversation
    conversation_text = "\n".join(st.session_state.messages)
    st.text_area("Chat history", value=conversation_text, height=400,
                 key="chat_display", disabled=True, label_visibility="hidden")
    
    # Input form – automatically clears after submit
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Your message", key="user_input",
                                   placeholder=t.get("chat_input", "Type your question here..."),
                                   label_visibility="hidden")
        col1, col2 = st.columns([1, 5])
        with col1:
            submitted = st.form_submit_button("📤 Send", use_container_width=True)
        with col2:
            st.markdown("")  # spacer
    
    if submitted and user_input:
        # Append user message
        st.session_state.messages.append(f"You: {user_input}")
        # Get AI reply
        with st.spinner("Gesner AI is thinking..."):
            reply = get_ai_response(user_input)
        st.session_state.messages.append(f"🧠 Gesner AI: {reply}")
        st.rerun()

def dictionary_manager(t):
    st.markdown("## 📖 Dictionary Manager")
    with st.form("add_term_form"):
        term = st.text_input("Term")
        definition = st.text_area("Definition")
        if st.form_submit_button("Add / Update"):
            if term and definition:
                add_term(term, definition)
                st.success(f"Added '{term}'")
                st.rerun()
    
    st.markdown("### Existing Terms")
    if st.session_state.dictionary:
        for term, defi in list(st.session_state.dictionary.items()):
            col1, col2, col3 = st.columns([2, 3, 1])
            col1.write(f"**{term}**")
            col2.write(defi)
            if col3.button("❌", key=f"del_{term}"):
                delete_term(term)
                st.rerun()
    else:
        st.info("No terms yet. Add one above.")

def voice_training_center(t):
    st.markdown("## 🎤 Voice Training Center")
    voice_name = st.text_input("Voice name (e.g., 'John')")
    uploaded_file = st.file_uploader("Upload an audio sample (WAV/MP3)", type=["wav", "mp3"])
    if st.button("Train Voice") and voice_name and uploaded_file:
        audio_bytes = uploaded_file.read()
        train_voice(voice_name, audio_bytes)
    
    st.markdown("### Trained Voices")
    if st.session_state.voice_cache:
        for key, data in list(st.session_state.voice_cache.items()):
            st.write(f"- {data['name']} (trained {time.ctime(data['timestamp'])})")
    else:
        st.info("No voices trained yet.")

def training_center(t):
    st.markdown("## 🧠 Training Center")
    st.write("Rebuild the knowledge base index from documents in `./knowledge/`.")
    if st.button("🔄 Rebuild Knowledge Index"):
        rebuild_index()
        st.rerun()
    if st.button("🎤 Rebuild Voice Cache"):
        st.session_state.voice_cache = {}
        save_voice_cache()
        st.success("Voice cache cleared.")
        st.rerun()

def show_sidebar():
    """Return (selected_menu, translations_dict)."""
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
        st.caption("Gesner AI v1.0")
    
    # Translations (simplified)
    t = {
        "chat_input": "Ask me anything...",
        "chat_interface_label": "Chat",
        "dictionary": "Dictionary",
        "voice_training": "Voice Training",
        "training_center": "Training Center"
    }
    return menu, t

# -------------------------------------------------------------------
# 8. Main App
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Gesner AI", page_icon="🧠", layout="wide")
    
    # Custom CSS for professional look
    st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #0e1117;
        }
        /* Chat display area */
        [data-testid="stTextArea"] textarea {
            background-color: #1e1e2f !important;
            color: #ffffff !important;
            border-radius: 12px;
            font-family: monospace;
            font-size: 14px;
        }
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #1a1c23;
        }
        /* Buttons */
        .stButton button {
            background-color: #e94560;
            color: white;
            border-radius: 20px;
            transition: 0.2s;
        }
        .stButton button:hover {
            background-color: #ff6b8b;
            color: white;
        }
        /* Input field */
        [data-testid="stForm"] input {
            border-radius: 25px;
            background-color: #2a2c3a;
            color: white;
            border: 1px solid #e94560;
        }
        /* Radio labels */
        .stRadio label {
            font-weight: 500;
        }
    </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    
    # Load or rebuild index
    if not load_index():
        if KNOWLEDGE_DIR.exists() and any(KNOWLEDGE_DIR.glob("*.txt")):
            rebuild_index()
        else:
            st.info("No knowledge base found. Place .txt files in ./knowledge/ to enable RAG.")
    
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
