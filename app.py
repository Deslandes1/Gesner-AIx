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
import os
import shutil
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from PIL import Image
from transformers import pipeline   # removed 'Conversation'

# ========== DATA DIRECTORY ==========
DATA_DIR = ".gesner_data"
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_FILE = os.path.join(DATA_DIR, "training_data.json")
DICT_FILE = os.path.join(DATA_DIR, "dictionaries.json")
VOICE_FILE = os.path.join(DATA_DIR, "voice_cache.json")
COGNITIVE_FILE = os.path.join(DATA_DIR, "cognitive_examples.json")

# ---------- PERSISTENCE FUNCTIONS ----------
def save_training_data():
    with open(TRAINING_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.training_data, f, ensure_ascii=False, indent=2)

def load_training_data():
    if os.path.exists(TRAINING_FILE):
        with open(TRAINING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_dictionaries():
    with open(DICT_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.dictionaries, f, ensure_ascii=False, indent=2)

def load_dictionaries():
    if os.path.exists(DICT_FILE):
        with open(DICT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"ht": {}, "fr": {}, "en": {}}

def save_voice_cache():
    serializable = {}
    for key, audio_bytes in VOICE_CACHE.items():
        serializable[key] = base64.b64encode(audio_bytes).decode("utf-8")
    with open(VOICE_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)

def load_voice_cache():
    if os.path.exists(VOICE_FILE):
        with open(VOICE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cache = {}
        for key, b64 in data.items():
            cache[key] = base64.b64decode(b64)
        return cache
    return {}

def save_cognitive_examples():
    with open(COGNITIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.cognitive_examples, f, ensure_ascii=False, indent=2)

def load_cognitive_examples():
    if os.path.exists(COGNITIVE_FILE):
        with open(COGNITIVE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ========== HAITIAN KNOWLEDGE BASE ==========
HAITIAN_KNOWLEDGE_FACTS = [
    "Kristòf Kolon te dekouvri zile Ispanyola (kote Ayiti ye jodi a) nan 5 desanm 1492.",
    "Kolon te rele zile a 'La Isla Española'. Pita fransè yo te rele l 'Saint-Domingue'.",
    "Anvan Kolon, Endyen Taino yo te rete sou zile a depi anviwon 300 anvan epòk nou an.",
    "Pòtoprens se kapital Ayiti. Li sou kòt lwès peyi a.",
    "Ayiti sitiye nan Karayib la, sou zile Ispanyola ki gen tou Repiblik Dominikèn.",
    "Ayiti gen yon sipèfisi 27,750 kilomèt kare. Li se twazyèm pi gwo peyi Karayib la.",
    "Gwo rivyè Ayiti yo se Latibonit, Lakay, ak Ladesdèyè.",
    "Toupatou nan Ayiti gen bèl plaj, sitou nan Kokoye (Labade) ak Jakmèl.",
    "Tousen Louverture te yon lidè enpòtan nan revolisyon esklav la.",
    "Jan Jak Desalin te pwoklame endepandans Ayiti 1ye janvye 1804.",
    "Anri Kristòf te bati Sitadèl Laferyè a, youn nan pi gwo fò nan Amerik yo.",
    "Etazini te okipe Ayiti soti 1915 rive 1934.",
    "François Duvalier (Papa Dok) te dirije 1957-1971.",
    "Jean-Claude Duvalier (Bébé Dok) te dirije 1971-1986.",
    "Jean-Bertrand Aristide te premye prezidan demokratik eli an 1990.",
    "Tranblemanntè 12 janvye 2010 te fè gwo ravaj nan Pòtoprens.",
    "Jovenel Moïse te asasine 7 jiyè 2021.",
    "Vodou se yon relijyon ki fèt nan melanj tradisyon Afriken ak Krisyanis.",
    "Kanaval Ayiti fèt chak ane anvan Karèm.",
    "Kompas (conpa) se yon dans ak mizik popilè an Ayiti.",
    "Diri ak pwa se manje nasyonal Ayiti.",
    "Soup joumou se manje senbolik pou 1ye janvye, jou endepandans.",
    "Alfabè kreyòl la gen 32 lèt.",
    "Pwonon pèsonèl an Kreyòl: Mwen, ou, li, nou, yo.",
    "Salitasyon debaz: Bonjou, Bonswa, Kijan ou rele?, Mwen rele...",
    "Tan pase: yo itilize 'te' devan vèb. Egzanp: Mwen te manje.",
    "Tan kap vini: yo itilize 'ap' oswa 'pral'. Egzanp: Mwen ap manje.",
]

def get_default_training_facts():
    return HAITIAN_KNOWLEDGE_FACTS

def initialize_default_training():
    if not st.session_state.training_data:
        for fact in HAITIAN_KNOWLEDGE_FACTS:
            embedding = st.session_state.embedding_model.encode([fact])[0]
            st.session_state.training_data.append({"text": fact, "embedding": embedding.tolist()})
        rebuild_index()
        save_training_data()

# ---------- CORE FIX: SAFE CHAT CLEAR STATE ----------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# ---------- SESSION STATE ----------
if "embedding_model" not in st.session_state:
    with st.spinner("Loading AI model..."):
        st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.session_state.index = None
    st.session_state.texts = []

if "training_data" not in st.session_state:
    st.session_state.training_data = load_training_data()

if "dictionaries" not in st.session_state:
    st.session_state.dictionaries = load_dictionaries()

if "cognitive_examples" not in st.session_state:
    st.session_state.cognitive_examples = load_cognitive_examples()

if "tfidf_vectorizer" not in st.session_state:
    st.session_state.tfidf_vectorizer = None

if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None

if "play_audio" not in st.session_state:
    st.session_state.play_audio = None

VOICE_CACHE = load_voice_cache()

# ---------- INDEX ----------
def rebuild_index():
    if st.session_state.training_data:
        st.session_state.texts = [x["text"] for x in st.session_state.training_data]
        embeddings = [np.array(x["embedding"], dtype=np.float32) for x in st.session_state.training_data]
        dim = len(embeddings[0])
        st.session_state.index = faiss.IndexFlatL2(dim)
        st.session_state.index.add(np.array(embeddings))
    else:
        st.session_state.index = None
        st.session_state.texts = []

# ---------- FIXED CHAT RESPONSE ----------
def generate_response(user_input):
    return f"Ou te di: {user_input}"

# ---------- CHAT UI ----------
def chat_interface():
    st.title("Gesner AI")

    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            st.write("🧑", msg["content"])
        else:
            st.write("🤖", msg["content"])

    col1, col2 = st.columns([4,1])

    with col1:
        user_input = st.text_input("Message")

    with col2:
        send = st.button("Send")

    if st.button("Clear Chat"):
        st.session_state.conversation_history = []
        st.rerun()

    if send and user_input.strip():
        st.session_state.conversation_history.append({"role":"user","content":user_input})

        answer = generate_response(user_input)

        if not answer:
            answer = "Mwen pa gen repons kounye a."

        st.session_state.conversation_history.append({"role":"assistant","content":answer})
        st.rerun()

# ---------- RUN ----------
rebuild_index()
initialize_default_training()
chat_interface()
