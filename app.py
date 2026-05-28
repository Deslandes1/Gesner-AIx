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
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from PIL import Image

# ================= DATA DIRECTORY =================
DATA_DIR = ".gesner_data"
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_FILE = os.path.join(DATA_DIR, "training_data.json")
DICT_FILE = os.path.join(DATA_DIR, "dictionaries.json")
VOICE_FILE = os.path.join(DATA_DIR, "voice_cache.json")
COGNITIVE_FILE = os.path.join(DATA_DIR, "cognitive_examples.json")

# ================= PERSISTENCE =================
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

def load_voice_cache():
    if os.path.exists(VOICE_FILE):
        with open(VOICE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cache = {}
        for k, v in data.items():
            cache[k] = base64.b64decode(v)
        return cache
    return {}

VOICE_CACHE = load_voice_cache()

# ================= HAITIAN KNOWLEDGE =================
HAITIAN_KNOWLEDGE_FACTS = [
    "Kristòf Kolon te dekouvri zile Ispanyola (kote Ayiti ye jodi a) nan 5 desanm 1492.",
    "Kolon te rele zile a 'La Isla Española'. Pita fransè yo te rele l 'Saint-Domingue'.",
    "Anvan Kolon, Endyen Taino yo te rete sou zile a depi anviwon 300 anvan epòk nou an.",
    "Pòtoprens se kapital Ayiti.",
    "Ayiti sitiye nan Karayib la sou zile Ispanyola.",
    "Ayiti te vin endepandan 1ye janvye 1804.",
    "Tousen Louverture se yon lidè revolisyon ayisyen.",
    "Jan Jak Desalin te pwoklame endepandans Ayiti."
]

# ================= CORE ANSWERS =================
CORE_ANSWERS = {
    "kijan ou rele": "Mwen se Gesner AI.",
    "ki moun ki dekouvri ayiti": "Kristòf Kolon te dekouvri Ayiti an 1492.",
    "ki dat ayiti endepandan": "Ayiti te vin endepandan 1ye janvye 1804.",
    "ki kote ayiti ye": "Ayiti sitiye nan Karayib la."
}

def get_core_answer(q):
    return CORE_ANSWERS.get(q.lower().strip())

# ================= GROK API =================
def get_grok_key():
    try:
        return st.secrets["GROK_API_KEY"]
    except:
        return None

def call_grok(prompt):
    key = get_grok_key()
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
                    {"role": "system", "content": "You are Gesner AI. Reply directly and briefly in Haitian Creole."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.6,
                "max_tokens": 400
            },
            timeout=5
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except:
        pass
    return None

# ================= TRAINING INIT =================
def init_training():
    if not st.session_state.training_data:
        st.session_state.training_data = [{"text": f} for f in HAITIAN_KNOWLEDGE_FACTS]
        save_training_data()

# ================= RETRIEVAL =================
def retrieve(query):
    q = query.lower()
    results = []
    for item in st.session_state.training_data:
        if q in item["text"].lower():
            results.append(item["text"])
    return results[:5]

# ================= RESPONSE ENGINE =================
def generate_response(user_input):
    core = get_core_answer(user_input)
    if core:
        return core

    facts = retrieve(user_input)
    if facts:
        return facts[0]

    grok = call_grok(user_input)
    if grok:
        return grok

    return "Mwen pa gen repons sa kounye a."

# ================= UI =================
st.set_page_config(page_title="Gesner AI", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
}
.stTextInput input {
    background:#0f3460;
    color:white;
}
div[data-testid="stTextArea"] textarea {
    background:black !important;
    color:white !important;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

if "chat" not in st.session_state:
    st.session_state.chat = []

if "training_data" not in st.session_state:
    st.session_state.training_data = []

init_training()

# ================= CHAT =================
st.title("Gesner AI")

for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.markdown(f"<p style='color:white;font-weight:bold;'>🧑 {msg['text']}</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='color:white;font-weight:bold;'>🤖 {msg['text']}</p>", unsafe_allow_html=True)

user_input = st.text_input("", placeholder="Poze kesyon ou...")

col1, col2 = st.columns(2)

with col1:
    send = st.button("Voye")

with col2:
    clear = st.button("Efase")

if clear:
    st.session_state.chat = []
    st.rerun()

if send and user_input:
    st.session_state.chat.append({"role": "user", "text": user_input})
    reply = generate_response(user_input)
    st.session_state.chat.append({"role": "assistant", "text": reply})
    st.rerun()

# ================= TRAINING CENTER =================
st.sidebar.title("Menu")
page = st.sidebar.selectbox("Chwazi", ["Chat", "Training Center"])

if page == "Training Center":
    st.subheader("Training Center")

    new_fact = st.text_area("Ajoute konesans")
    if st.button("Sove"):
        if new_fact.strip():
            st.session_state.training_data.append({"text": new_fact})
            save_training_data()
            st.success("Sove")

    for i, f in enumerate(st.session_state.training_data):
        st.write(f["text"])
