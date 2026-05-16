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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ========== DATA DIRECTORY (NO FORCED RESET) ==========
DATA_DIR = ".gesner_data"
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_FILE = os.path.join(DATA_DIR, "training_data.json")
DICT_FILE = os.path.join(DATA_DIR, "dictionaries.json")
VOICE_FILE = os.path.join(DATA_DIR, "voice_cache.json")

# ---------- PERSISTENCE FUNCTIONS ----------
# ... (same as before, omitted for brevity)

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

# ---------- DEFAULT TRAINING (same as previous) ----------
def get_default_training_facts():
    # ... (keep all previous facts)
    pass

def initialize_default_training():
    if not st.session_state.training_data:
        default_facts = get_default_training_facts()
        for fact in default_facts:
            if fact.strip():
                embedding = st.session_state.embedding_model.encode([fact])[0]
                st.session_state.training_data.append({"text": fact, "embedding": embedding.tolist()})
        rebuild_index()
        save_training_data()

# ---------- STREAMLIT PAGE CONFIG ----------
st.set_page_config(
    page_title="Gesner AI",
    page_icon="🧠",
    layout="wide"
)

# ---------- CSS (dark theme + spinning globe) ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #1a1a2e 100%);
        border-right: 2px solid #e94560;
    }
    .stMarkdown, .stTextInput label, .stTextArea label, .stSelectbox label, .stButton button, .stCaption,
    h1, h2, h3, h4, h5, h6, p, li, div, span, strong, em, .footer,
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stSelectbox {
        background-color: #000000 !important;
        border-radius: 12px !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: #000000 !important;
        border: 1px solid #e94560 !important;
        border-radius: 12px !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background-color: #000000 !important;
        color: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox svg {
        fill: #e94560 !important;
        stroke: #e94560 !important;
    }
    div[data-baseweb="popover"] ul {
        background-color: #000000 !important;
        border: 1px solid #e94560 !important;
    }
    div[data-baseweb="popover"] li {
        color: white !important;
        background-color: #000000 !important;
    }
    div[data-baseweb="popover"] li:hover {
        background-color: #e94560 !important;
        color: white !important;
    }
    .stButton button {
        background-color: #e94560 !important;
        color: white !important;
        border-radius: 30px !important;
        font-weight: bold !important;
        width: 100%;
        border: none;
    }
    .stButton button:hover {
        background-color: #ff6b6b !important;
        transform: scale(1.02);
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #0f3460 !important;
        color: white !important;
        border-radius: 12px;
        border: 1px solid #e94560;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .user-message {
        background: linear-gradient(135deg, #e94560, #ff6b6b);
        color: white;
    }
    .assistant-message {
        background: linear-gradient(135deg, #0f3460, #1a4a7a);
        color: white;
    }
    .speak-btn {
        background-color: #ffaa33;
        border: none;
        border-radius: 30px;
        padding: 5px 12px;
        margin-left: 12px;
        cursor: pointer;
        font-size: 1rem;
        transition: 0.2s;
    }
    .speak-btn:hover {
        background-color: #ffcc66;
        transform: scale(1.05);
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #e94560;
    }
    .char-picker {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .char-btn {
        background-color: #2a5298;
        border: none;
        border-radius: 20px;
        padding: 5px 12px;
        color: white;
        cursor: pointer;
        font-size: 1rem;
        transition: 0.2s;
        margin-right: 5px;
    }
    .char-btn:hover {
        background-color: #e94560;
    }
    /* Spinning globe animation */
    @keyframes spin-globe {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .spinning-globe {
        animation: spin-globe 4s linear infinite;
        display: inline-block;
        font-size: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- LANGUAGES AND TEXTS (same as before) ----------
LANGUAGES = {
    "English": "en",
    "Français": "fr",
    "Kreyòl Ayisyen": "ht",
    "Español": "es"
}

TEXTS = {
    # ... (same full TEXTS as previous version, omitted for brevity)
}

# ---------- SESSION STATE ----------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "embedding_model" not in st.session_state:
    with st.spinner("Loading AI model... (first time only)"):
        st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.session_state.index = None
    st.session_state.texts = []
if "training_data" not in st.session_state:
    st.session_state.training_data = load_training_data()
if "dictionaries" not in st.session_state:
    st.session_state.dictionaries = load_dictionaries()
if "training_access" not in st.session_state:
    st.session_state.training_access = False
if "ui_language" not in st.session_state:
    st.session_state.ui_language = "en"
if "tfidf_vectorizer" not in st.session_state:
    st.session_state.tfidf_vectorizer = None
if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None

VOICE_CACHE = load_voice_cache()

# ========== PRE‑DEFINED VOICE MAPPING ==========
PREDEFINED_VOICES = {
    "kijan ou rele": "https://raw.githubusercontent.com/Deslandes1/Gesner-AIx/main/recording.wav",
    "site konbyen let ki genhen nan alfabe kreyol la": "https://raw.githubusercontent.com/Deslandes1/Gesner-AIx/main/recording%20(1).wav",
    "konbyen let ki gehen nan alfabe kreyol la": "https://raw.githubusercontent.com/Deslandes1/Gesner-AIx/main/recording%20(3).wav"
}

def normalize_text(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

def get_predefined_voice_url(user_question):
    norm_q = normalize_text(user_question)
    for key, url in PREDEFINED_VOICES.items():
        if key in norm_q or norm_q.startswith(key):
            return url
    return None

# ---------- HELPER FUNCTIONS ----------
def save_all():
    save_training_data()
    save_dictionaries()
    save_voice_cache()

def build_tfidf():
    if st.session_state.texts:
        st.session_state.tfidf_vectorizer = TfidfVectorizer(stop_words=None)
        st.session_state.tfidf_matrix = st.session_state.tfidf_vectorizer.fit_transform(st.session_state.texts)

def rebuild_index():
    if st.session_state.training_data:
        st.session_state.texts = [item["text"] for item in st.session_state.training_data]
        embeddings = [np.array(item["embedding"], dtype=np.float32) for item in st.session_state.training_data]
        dim = len(embeddings[0])
        st.session_state.index = faiss.IndexFlatL2(dim)
        st.session_state.index.add(np.array(embeddings))
        build_tfidf()
    else:
        st.session_state.index = None
        st.session_state.texts = []
        st.session_state.tfidf_vectorizer = None
        st.session_state.tfidf_matrix = None

def add_to_training(text):
    if not text.strip():
        return False
    embedding = st.session_state.embedding_model.encode([text])[0]
    st.session_state.training_data.append({"text": text, "embedding": embedding.tolist()})
    rebuild_index()
    save_training_data()
    return True

def update_training_item(idx, new_text):
    if not new_text.strip():
        return False
    embedding = st.session_state.embedding_model.encode([new_text])[0]
    st.session_state.training_data[idx] = {"text": new_text, "embedding": embedding.tolist()}
    rebuild_index()
    save_training_data()
    return True

def delete_training_item(idx):
    st.session_state.training_data.pop(idx)
    rebuild_index()
    save_training_data()

def get_voice_filename(text):
    norm = text.strip().lower()
    h = hashlib.md5(norm.encode()).hexdigest()
    return h

def save_voice_for_text(text, audio_bytes):
    global VOICE_CACHE
    key = get_voice_filename(text)
    VOICE_CACHE[key] = audio_bytes
    save_voice_cache()

def get_voice_for_text(text):
    if not text:
        return None
    key = get_voice_filename(text)
    return VOICE_CACHE.get(key)

def character_picker(key_prefix, label="Insert Kreyòl characters:"):
    chars = ["e","è","E","È","o","ò","O","Ò","an","An","AN","en","En","EN","on","On","ON","oun","Oun","OUN"]
    st.markdown(f"**{label}**")
    cols = st.columns(len(chars))
    for i, ch in enumerate(chars):
        with cols[i]:
            if st.button(ch, key=f"char_{key_prefix}_{ch}"):
                if key_prefix.startswith("edit_"):
                    idx = key_prefix.split("_")[1]
                    key = f"edit_text_{idx}"
                    current = st.session_state.get(key, "")
                    st.session_state[key] = current + ch
                st.rerun()

def retrieve_facts_hybrid(query, k=3):
    if st.session_state.index is None or st.session_state.index.ntotal == 0:
        return []
    query_embedding = st.session_state.embedding_model.encode([query])[0].astype(np.float32).reshape(1, -1)
    distances, indices = st.session_state.index.search(query_embedding, k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and idx < len(st.session_state.texts) and distances[0][i] < 1.2:
            results.append(st.session_state.texts[idx])
    if st.session_state.tfidf_vectorizer is not None and st.session_state.tfidf_matrix is not None:
        q_vec = st.session_state.tfidf_vectorizer.transform([query])
        scores = cosine_similarity(q_vec, st.session_state.tfidf_matrix).flatten()
        top_indices = scores.argsort()[-k:][::-1]
        for idx in top_indices:
            if scores[idx] > 0.1 and st.session_state.texts[idx] not in results:
                results.append(st.session_state.texts[idx])
    return results[:k]

def direct_keyword_answer(query):
    q_lower = query.lower().strip()
    if any(q in q_lower for q in ["kijan ou rele", "kiyès ou ye", "kisa ou ye", "ki moun ou ye", "what is your name", "who are you"]):
        return "Non pa mwen se Gesner L’IA, kreyatè mwen an se Gesner Deslandes nan GlobalInternet.py."
    if any(q in q_lower for q in ["kiyès ki kreye ou", "ki moun ki fè ou", "who created you", "ki moun ki devlope ou", "kiyès ki te kreye ou"]):
        return "Mwen te kreye pa Gesner Deslandes, fondatè GlobalInternet.py. Li se yon enjenyè ki renmen edike Ayiti."
    if q_lower in ["bonjou","bonswa","hello","hi","salut"]:
        return "Bonjou! Kijan ou ye? Mwen la pou reponn kesyon ou."
    return None

def reason_about_question(query):
    q = query.lower().strip()
    math_match = re.search(r"(\d+)\s*([\+\-\*\/])\s*(\d+)", q)
    if math_match:
        try:
            a, op, b = int(math_match.group(1)), math_match.group(2), int(math_match.group(3))
            if op == '+': res = a + b
            elif op == '-': res = a - b
            elif op == '*': res = a * b
            elif op == '/': res = a / b
            else: res = None
            if res is not None:
                if isinstance(res, float) and res.is_integer():
                    res = int(res)
                return f"Repons lan se {res}."
        except: pass
    if "kapital" in q or "capital" in q:
        capitals = {"france":"Paris","ayiti":"Pòtoprens","haiti":"Port‑au‑Prince","etazini":"Washington, D.C.","usa":"Washington, D.C.","kanada":"Ottawa","brezil":"Brasília","alman":"Bèlen","itali":"Wòm","espay":"Madrid","angle":"Londr","japon":"Tokiyo"}
        for country, cap in capitals.items():
            if country in q:
                return f"Kapital {country.title()} se {cap}."
    if "ki lè li ye" in q or "what time" in q:
        now = datetime.now().strftime("%H:%M")
        return f"Kounye a li {now}."
    return None

def reason_answer(query, retrieved_facts):
    if not retrieved_facts:
        return None
    if len(retrieved_facts) == 1:
        return retrieved_facts[0]
    q_lower = query.lower()
    if any(word in q_lower for word in ["raconte", "rakonte", "istwa", "history", "histoire"]):
        history_facts = [f for f in retrieved_facts if any(kw in f.lower() for kw in ["endepandan", "revolisyon", "duvalier", "tranblemanntè", "1804", "1915", "1957", "bwa kayiman"])]
        if history_facts:
            combined = ". ".join(history_facts[:3])
            return combined + "."
        else:
            return retrieved_facts[0]
    return retrieved_facts[0]

# ========== FIXED: SPECIAL CASE FOR MISSPELLED ALPHABET QUESTION (INCLUDES QUESTION TEXT, NO AUDIO) ==========
def generate_response(user_input):
    normalized = user_input.strip().lower()
    patterns = [
        "site konbyen let ki genhen nan alfabe kreyol la",
        "site konbyen let ki genhen nan alfabe kreyol",
        "site konbyen let ki genhen nan alfabe kreyòl la",
        "site konbyen let ki genhen nan alfabe kreyòl",
        "konbyen let ki gehen nan alfabe kreyol la",
        "site konbyen let ki genhen"
    ]
    for pat in patterns:
        if pat in normalized:
            # Return the question followed by the alphabet list, and flag to skip audio
            answer = "Site konbyen let ki genhen nan alfabe kreyol la ?: A, AN, B, CH, D, E, È, EN, F, G, H, I, J, K, L, M, N, NG, O, Ò, ON, OU, OUN, P, R, S, T, UI, V, W, Y, Z"
            return answer, False, True  # third value: skip_audio

    with st.spinner("🧠 Gesner AI ap reflechi... (thinking...)"):
        time.sleep(0.8)
        math_result = reason_about_question(user_input)
        if math_result and ("+" in user_input or "-" in user_input or "*" in user_input or "/" in user_input):
            return math_result, False, False
        direct = direct_keyword_answer(user_input)
        if direct:
            return direct, False, False
        facts = retrieve_facts_hybrid(user_input, k=5)
        if facts:
            reasoned = reason_answer(user_input, facts)
            return reasoned, False, False
        logic = reason_about_question(user_input)
        if logic:
            return logic, False, False
    return "Mwen poko konn sa. Tanpri anseye m nan Sant Fòmasyon.", True, False

def play_voice_button(text, user_question, button_label="🔊", key_suffix=""):
    predefined_url = get_predefined_voice_url(user_question) if user_question else None
    if predefined_url:
        html = f"""
        <button class="speak-btn" id="voiceBtn_{key_suffix}" style="background-color:#ffaa33; border:none; border-radius:30px; padding:5px 12px; margin-left:12px; cursor:pointer;">{button_label}</button>
        <audio id="customAudio_{key_suffix}" style="display:none;"></audio>
        <script>
            (function() {{
                const btn = document.getElementById('voiceBtn_{key_suffix}');
                const audioEl = document.getElementById('customAudio_{key_suffix}');
                audioEl.src = "{predefined_url}";
                btn.onclick = () => audioEl.play();
            }})();
        </script>
        """
        return html
    voice_bytes = get_voice_for_text(text)
    if voice_bytes:
        audio_b64 = base64.b64encode(voice_bytes).decode()
        mime = "audio/wav"
        html = f"""
        <button class="speak-btn" id="voiceBtn_{key_suffix}" style="background-color:#ffaa33; border:none; border-radius:30px; padding:5px 12px; margin-left:12px; cursor:pointer;">{button_label}</button>
        <audio id="customAudio_{key_suffix}" style="display:none;"></audio>
        <script>
            (function() {{
                const audioData = "{audio_b64}";
                const binaryStr = atob(audioData);
                const bytes = new Uint8Array(binaryStr.length);
                for (let i = 0; i < binaryStr.length; i++) bytes[i] = binaryStr.charCodeAt(i);
                const audioBlob = new Blob([bytes], {{ type: '{mime}' }});
                const audioUrl = URL.createObjectURL(audioBlob);
                const audioEl = document.getElementById('customAudio_{key_suffix}');
                audioEl.src = audioUrl;
                document.getElementById('voiceBtn_{key_suffix}').onclick = () => audioEl.play();
            }})();
        </script>
        """
        return html
    else:
        return ""

def play_fallback_audio_french():
    html = """
    <button id="fallbackFrenchBtn" style="background-color:#ffaa33; border:none; border-radius:30px; padding:5px 12px; margin-left:12px; cursor:pointer;">🔊</button>
    <script>
        (function() {
            const btn = document.getElementById('fallbackFrenchBtn');
            if (!btn) return;
            btn.onclick = function() {
                let utterance = new SpeechSynthesisUtterance("Gesner AI réfléchit à votre question. Veuillez patienter.");
                utterance.lang = "fr-FR";
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(utterance);
            };
        })();
    </script>
    """
    return html

# ---------- UI COMPONENTS (unchanged except chat_interface to handle skip_audio) ----------
def dictionary_manager(t):
    # ... (same as before)
    pass

def voice_training(t):
    # ... (same)
    pass

def bulk_training(t):
    # ... (same)
    pass

def manage_trained_facts(t):
    # ... (same)
    pass

def test_training_section(t):
    # ... (same)
    pass

def training_center(t):
    # ... (same)
    pass

def chat_interface(t):
    st.markdown(f"<h1 style='text-align:center; color:#ffd966;'>{t['app_title']}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Mwen reponn sèlman an Kreyòl. Poze m kesyon sou alfabè, gramè, istwa Ayiti, oswa nenpòt bagay ou te anseye m.</p>", unsafe_allow_html=True)
    for idx, msg in enumerate(st.session_state.conversation_history):
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">🧑‍💻 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([10,1])
            with col1:
                st.markdown(f'<div class="chat-message assistant-message" style="width:100%;">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            with col2:
                # Check if this message has skip_audio flag
                if not msg.get("skip_audio", False):
                    user_q = st.session_state.conversation_history[idx-1]["content"] if idx > 0 else ""
                    btn_html = play_voice_button(msg["content"], user_q, "🔊", f"chat_{idx}")
                    if btn_html:
                        st.components.v1.html(btn_html, height=50)
                    else:
                        if msg.get("is_fallback", False) and st.session_state.ui_language == "fr":
                            fallback_html = play_fallback_audio_french()
                            st.components.v1.html(fallback_html, height=50)
    user_input = st.text_input(t['chat_input'], key="chat_input")
    if st.button(t['send'], use_container_width=True, key="send_btn"):
        if user_input.strip():
            answer, is_fallback, skip_audio = generate_response(user_input)
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": answer,
                "is_fallback": is_fallback,
                "skip_audio": skip_audio
            })
            st.rerun()
    if st.button(t['clear'], use_container_width=True, key="clear_btn"):
        st.session_state.conversation_history = []
        st.rerun()

def show_sidebar():
    # ... (same as before, spinning globe)
    pass

def main():
    # ... (same)
    pass

if __name__ == "__main__":
    main()
