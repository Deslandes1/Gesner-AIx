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
from transformers import pipeline

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

# ================= HAITIAN KNOWLEDGE =================
HAITIAN_KNOWLEDGE_FACTS = [
    "Kristòf Kolon te dekouvri zile Ispanyola (kote Ayiti ye jodi a) nan 5 desanm 1492.",
    "Kolon te rele zile a 'La Isla Española'. Pita fransè yo te rele l 'Saint-Domingue'.",
    "Anvan Kolon, Endyen Taino yo te rete sou zile a depi anviwon 300 anvan epòk nou an.",
    "Pòtoprens se kapital Ayiti. Li sou kòt lwès peyi a.",
    "Ayiti sitiye nan Karayib la, sou zile Ispanyola ki gen tou Repiblik Dominikèn.",
    "Ayiti gen yon sipèfisi 27,750 kilomèt kare. Li se twazyèm pi gwo peyi Karayib la.",
    "Tousen Louverture te yon lidè enpòtan nan revolisyon esklav la.",
    "Jan Jak Desalin te pwoklame endepandans Ayiti 1ye janvye 1804.",
    "Anri Kristòf te bati Sitadèl Laferyè a.",
    "Tranblemanntè 12 janvye 2010 te fè gwo ravaj nan Pòtoprens.",
    "Vodou se yon relijyon ki fèt nan melanj tradisyon Afriken ak Krisyanis.",
    "Diri ak pwa se manje nasyonal Ayiti.",
    "Soup joumou se manje endepandans Ayiti."
]

# ================= CORE ANSWERS =================
CORE_ANSWERS = {
    "kijan ou rele": "Non mwen se Gesner AI, kreye pa Gesner Deslandes.",
    "ki moun ki dekouvri ayiti": "Kristòf Kolon te dekouvri Ayiti an 1492.",
    "ki kote ayiti ye": "Ayiti sitiye nan Karayib la sou zile Ispanyola.",
    "ki dat ayiti endepandan": "Ayiti te vin endepandan 1ye janvye 1804.",
    "kisa soup joumou ye": "Soup joumou se manje endepandans Ayiti."
}

def get_core_answer(q):
    q = q.lower().strip()
    return CORE_ANSWERS.get(q)

# ================= GROK API =================
def get_grok_api_key():
    try:
        return st.secrets["GROK_API_KEY"]
    except:
        return None

def call_grok_api(prompt):
    api_key = get_grok_api_key()
    if not api_key:
        return None

    endpoint = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "grok-1",
        "messages": [
            {"role": "system", "content": "You are Gesner AI. Respond in Haitian Creole."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }

    try:
        r = requests.post(endpoint, headers=headers, json=payload, timeout=5)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except:
        pass
    return None

# ================= RETRIEVAL =================
def retrieve_facts(query):
    results = []
    q = query.lower()
    for f in st.session_state.training_data:
        if q in f["text"].lower():
            results.append(f["text"])
    return results[:5]

# ================= RESPONSE ENGINE =================
def generate_response(user_input):
    core = get_core_answer(user_input)
    if core:
        return core

    facts = retrieve_facts(user_input)
    if facts:
        return facts[0]

    grok = call_grok_api(user_input)
    if grok:
        return grok

    return "Mwen pa jwenn repons sa. Eseye aprann mwen li nan Training Center."

# ================= TRAINING INIT =================
def initialize_training():
    if not st.session_state.training_data:
        for f in HAITIAN_KNOWLEDGE_FACTS:
            st.session_state.training_data.append({"text": f})
        save_training_data()

# ================= UI =================
def chat_interface():
    st.markdown("<h1 style='text-align:center;color:white;'>Gesner AI</h1>", unsafe_allow_html=True)

    for msg in st.session_state.chat:
        if msg["role"] == "user":
            st.markdown(f"<div style='color:white;font-weight:bold;'>🧑 {msg['text']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:white;font-weight:bold;'>🤖 {msg['text']}</div>", unsafe_allow_html=True)

    user_input = st.text_input("Message", label_visibility="collapsed")

    if st.button("Send"):
        if user_input:
            st.session_state.chat.append({"role": "user", "text": user_input})
            reply = generate_response(user_input)
            st.session_state.chat.append({"role": "bot", "text": reply})
            st.rerun()

    if st.button("Clear Chat"):
        st.session_state.chat = []
        st.rerun()

def training_center():
    st.subheader("Training Center")

    new_fact = st.text_area("Add Knowledge")
    if st.button("Add"):
        if new_fact:
            st.session_state.training_data.append({"text": new_fact})
            save_training_data()
            st.success("Saved")

    st.markdown("Existing Data:")
    for i, f in enumerate(st.session_state.training_data):
        st.write(f["text"])

# ================= SESSION INIT =================
if "chat" not in st.session_state:
    st.session_state.chat = []

if "training_data" not in st.session_state:
    st.session_state.training_data = load_training_data()

if "dictionaries" not in st.session_state:
    st.session_state.dictionaries = load_dictionaries()

VOICE_CACHE = load_voice_cache()

initialize_training()

# ================= APP =================
st.set_page_config(page_title="Gesner AI", layout="wide")

menu = st.sidebar.selectbox("Menu", ["Chat", "Training Center"])

if menu == "Chat":
    chat_interface()
else:
    training_center()
