import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time
import hashlib
import re
import base64
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

st.set_page_config(
    page_title="Gesner AI",
    page_icon="🧠",
    layout="wide"
)

# ---------- CSS (dark theme) ----------
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
    .stMarkdown, .stTextInput label, .stSelectbox label, .stButton button, .stCaption,
    h1, h2, h3, h4, h5, h6, p, li, div, span, strong, em, .footer,
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
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
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- LANGUAGES ----------
LANGUAGES = {
    "English": "en",
    "Français": "fr",
    "Kreyòl Ayisyen": "ht",
    "Español": "es"
}

TEXTS = {
    "en": {
        "app_title": "💬 Gesner AI Chat",
        "chat_input": "Ask me anything...",
        "send": "Send",
        "clear": "Clear Chat",
        "sidebar_company": "GlobalInternet.py",
        "sidebar_product": "Gesner AI – Your Personal Assistant",
        "built_by": "Gesner Deslandes – Coder in Chief",
        "phone": "📞 (509)-47385663",
        "email": "✉️ deslandes78@gmail.com",
        "website_label": "🌐 Website:",
        "website_link": "https://globalinternetsitepy-abh7v6tnmskxxnuplrdcgk.streamlit.app/",
        "pricing_title": "💰 Licensing",
        "pricing_table": "| License | Price (one‑time) |\n|---------|------------------|\n| **Personal** | $49 |\n| **Business** | $299 |\n| **Enterprise / Source** | $999 |\n",
        "unlock_training": "🔐 Unlock Training Center",
        "api_key_label": "Enter API Key",
        "training_section": "🔧 Training Center (Dictionaries & Voice)",
        "dict_title": "📖 Dictionaries",
        "dict_ht": "Kreyòl Ayisyen",
        "dict_fr": "Français",
        "dict_en": "English",
        "dict_word": "Word",
        "dict_meaning": "Meaning",
        "dict_add": "Add Entry",
        "dict_delete": "Delete",
        "train_entry_button": "Train AI with this entry",
        "trained_entry_success": "✅ Trained: {word} → {meaning}",
        "voice_training_title": "🎙️ Voice Training (Kreyòl only)",
        "voice_upload": "Upload voice (WAV/MP3)",
        "voice_transcribed_text": "Text spoken in the audio (exact transcript)",
        "voice_train": "Train voice + text",
        "voice_success": "Voice and text stored!",
        "record_btn": "🔴 Record",
        "stop_btn": "⏹️ Stop",
        "download_btn": "💾 Download",
        "footer": "© GlobalInternet.py – Gesner AI | Public chat always free, training protected by API key"
    },
    "fr": {
        "app_title": "💬 Gesner IA Chat",
        "chat_input": "Demandez‑moi n'importe quoi...",
        "send": "Envoyer",
        "clear": "Effacer l'historique",
        "sidebar_company": "GlobalInternet.py",
        "sidebar_product": "Gesner IA – Votre assistant personnel",
        "built_by": "Gesner Deslandes – Ingénieur en chef",
        "phone": "📞 (509)-47385663",
        "email": "✉️ deslandes78@gmail.com",
        "website_label": "🌐 Site web :",
        "website_link": "https://globalinternetsitepy-abh7v6tnmskxxnuplrdcgk.streamlit.app/",
        "pricing_title": "💰 Licence",
        "pricing_table": "| Licence | Prix (unique) |\n|---------|---------------|\n| **Personnelle** | 49 $ |\n| **Entreprise** | 299 $ |\n| **Entreprise / Code source** | 999 $ |\n",
        "unlock_training": "🔐 Déverrouiller le centre d'entraînement",
        "api_key_label": "Entrez la clé API",
        "training_section": "🔧 Centre d'entraînement (Dictionnaires & Voix)",
        "dict_title": "📖 Dictionnaires",
        "dict_ht": "Kreyòl Ayisyen",
        "dict_fr": "Français",
        "dict_en": "English",
        "dict_word": "Mot",
        "dict_meaning": "Signification",
        "dict_add": "Ajouter",
        "dict_delete": "Supprimer",
        "train_entry_button": "Entraîner l'IA avec cette entrée",
        "trained_entry_success": "✅ Entraîné : {word} → {meaning}",
        "voice_training_title": "🎙️ Entraînement vocal (Kreyòl seulement)",
        "voice_upload": "Télécharger voix (WAV/MP3)",
        "voice_transcribed_text": "Texte parlé dans l'audio",
        "voice_train": "Entraîner voix + texte",
        "voice_success": "Voix et texte enregistrés !",
        "record_btn": "🔴 Enregistrer",
        "stop_btn": "⏹️ Arrêter",
        "download_btn": "💾 Télécharger",
        "footer": "© GlobalInternet.py – Gesner IA | Chat public toujours gratuit, entraînement protégé par clé API"
    },
    "ht": {
        "app_title": "💬 Gesner AI Chat",
        "chat_input": "Pose yon kesyon...",
        "send": "Voye",
        "clear": "Efase listorik",
        "sidebar_company": "GlobalInternet.py",
        "sidebar_product": "Gesner AI – Asistan pèsonèl ou",
        "built_by": "Gesner Deslandes – Enjenyè anchèf",
        "phone": "📞 (509)-47385663",
        "email": "✉️ deslandes78@gmail.com",
        "website_label": "🌐 Sitwèb :",
        "website_link": "https://globalinternetsitepy-abh7v6tnmskxxnuplrdcgk.streamlit.app/",
        "pricing_title": "💰 Pri",
        "pricing_table": "| Lisans | Pri (yon fwa) |\n|--------|---------------|\n| **Pèsonèl** | $49 |\n| **Biznis** | $299 |\n| **Antrepriz / Kòd sous** | $999 |\n",
        "unlock_training": "🔐 Débloke sant fòmasyon",
        "api_key_label": "Antre kle API",
        "training_section": "🔧 Sant Fòmasyon (Diksyonè & Vwa)",
        "dict_title": "📖 Diksyonè",
        "dict_ht": "Kreyòl Ayisyen",
        "dict_fr": "Français",
        "dict_en": "English",
        "dict_word": "Mo",
        "dict_meaning": "Siyifikasyon",
        "dict_add": "Ajoute",
        "dict_delete": "Efase",
        "train_entry_button": "Antrene AI ak antre sa a",
        "trained_entry_success": "✅ Antrene : {word} → {meaning}",
        "voice_training_title": "🎙️ Fòmasyon vwa (Kreyòl sèlman)",
        "voice_upload": "Chaje vwa (WAV/MP3)",
        "voice_transcribed_text": "Tèks ki nan odyo a",
        "voice_train": "Antrene vwa + tèks",
        "voice_success": "Vwa ak tèks sove!",
        "record_btn": "🔴 Anrejistre",
        "stop_btn": "⏹️ Sispann",
        "download_btn": "💾 Telechaje",
        "footer": "© GlobalInternet.py – Gesner AI | Chat piblik tou gratis, fòmasyon pwoteje pa kle API"
    },
    "es": {
        "app_title": "💬 Gesner AI Chat",
        "chat_input": "Pregúntame cualquier cosa...",
        "send": "Enviar",
        "clear": "Borrar historial",
        "sidebar_company": "GlobalInternet.py",
        "sidebar_product": "Gesner AI – Tu asistente personal",
        "built_by": "Gesner Deslandes – Codificador Jefe",
        "phone": "📞 (509)-47385663",
        "email": "✉️ deslandes78@gmail.com",
        "website_label": "🌐 Sitio web:",
        "website_link": "https://globalinternetsitepy-abh7v6tnmskxxnuplrdcgk.streamlit.app/",
        "pricing_title": "💰 Licencia",
        "pricing_table": "| Licencia | Precio (único) |\n|----------|----------------|\n| **Personal** | $49 |\n| **Negocios** | $299 |\n| **Empresa / Código fuente** | $999 |\n",
        "unlock_training": "🔐 Desbloquear centro de entrenamiento",
        "api_key_label": "Ingrese la clave API",
        "training_section": "🔧 Centro de Entrenamiento (Diccionarios & Voz)",
        "dict_title": "📖 Diccionarios",
        "dict_ht": "Kreyòl Ayisyen",
        "dict_fr": "Français",
        "dict_en": "English",
        "dict_word": "Palabra",
        "dict_meaning": "Significado",
        "dict_add": "Añadir",
        "dict_delete": "Eliminar",
        "train_entry_button": "Entrenar IA con esta entrada",
        "trained_entry_success": "✅ Entrenado: {word} → {meaning}",
        "voice_training_title": "🎙️ Entrenamiento de voz (solo Kreyòl)",
        "voice_upload": "Subir voz (WAV/MP3)",
        "voice_transcribed_text": "Texto hablado en el audio",
        "voice_train": "Entrenar voz + texto",
        "voice_success": "¡Voz y texto guardados!",
        "record_btn": "🔴 Grabar",
        "stop_btn": "⏹️ Detener",
        "download_btn": "💾 Descargar",
        "footer": "© GlobalInternet.py – Gesner AI | Chat público siempre gratuito, entrenamiento protegido por clave API"
    }
}

# ---------- SESSION STATE ----------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "embedding_model" not in st.session_state:
    with st.spinner("Loading AI model... (first time only)"):
        st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.session_state.index = None
    st.session_state.texts = []
if "dictionaries" not in st.session_state:
    st.session_state.dictionaries = {"ht": {}, "fr": {}, "en": {}}
if "training_access" not in st.session_state:
    st.session_state.training_access = False
if "chat_language" not in st.session_state:
    st.session_state.chat_language = "ht"
if "tfidf_vectorizer" not in st.session_state:
    st.session_state.tfidf_vectorizer = None
if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None

# ---------- VOICE CACHE (in memory) ----------
VOICE_CACHE = {}

def get_voice_filename(text):
    norm = text.strip().lower()
    h = hashlib.md5(norm.encode()).hexdigest()
    return h

def save_voice_for_text(text, audio_bytes):
    key = get_voice_filename(text)
    VOICE_CACHE[key] = audio_bytes
    return key

def get_voice_for_text(text):
    key = get_voice_filename(text)
    return VOICE_CACHE.get(key)

# ---------- HYBRID RETRIEVAL ----------
def build_tfidf():
    if st.session_state.texts:
        st.session_state.tfidf_vectorizer = TfidfVectorizer(stop_words=None)
        st.session_state.tfidf_matrix = st.session_state.tfidf_vectorizer.fit_transform(st.session_state.texts)

def retrieve_facts_hybrid(query, k=3):
    semantic_results = retrieve_relevant_facts(query, k=k, threshold=1.2)
    if not semantic_results:
        semantic_results = []
    keyword_results = []
    if st.session_state.tfidf_vectorizer is not None and st.session_state.tfidf_matrix is not None:
        q_vec = st.session_state.tfidf_vectorizer.transform([query])
        scores = cosine_similarity(q_vec, st.session_state.tfidf_matrix).flatten()
        top_indices = scores.argsort()[-k:][::-1]
        for idx in top_indices:
            if scores[idx] > 0.1:
                keyword_results.append(st.session_state.texts[idx])
    combined = list(dict.fromkeys(semantic_results + keyword_results))
    return combined[:k]

def retrieve_relevant_facts(query, k=3, threshold=1.2):
    if st.session_state.index is None or st.session_state.index.ntotal == 0:
        return []
    query_embedding = st.session_state.embedding_model.encode([query])[0].astype(np.float32).reshape(1, -1)
    distances, indices = st.session_state.index.search(query_embedding, k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and idx < len(st.session_state.texts) and distances[0][i] < threshold:
            results.append(st.session_state.texts[idx])
    return results

def add_to_facts(text):
    if not text.strip():
        return False
    embedding = st.session_state.embedding_model.encode([text])[0]
    if st.session_state.index is None:
        dim = len(embedding)
        st.session_state.index = faiss.IndexFlatL2(dim)
        st.session_state.texts = []
    st.session_state.index.add(np.array([embedding], dtype=np.float32))
    st.session_state.texts.append(text)
    build_tfidf()
    return True

# ---------- DIRECT KEYWORD ANSWERS ----------
def direct_keyword_answer(query, lang):
    q_lower = query.lower().strip()
    
    # Identity
    identity_queries = [
        "kijan ou rele", "kiyès ou ye", "kisa ou ye",
        "ki moun ou ye", "what is your name", "who are you"
    ]
    if any(q in q_lower for q in identity_queries):
        return "Non pa mw se Gesner L’IA, kreyatè mw an se Gesner Deslandes nan GlobalInternet.py."
    
    # Creator
    creator_queries = [
        "kiyès ki kreye ou", "ki moun ki fè ou", "who created you",
        "ki moun ki devlope ou", "kiyès ki te kreye ou"
    ]
    if any(q in q_lower for q in creator_queries):
        return "Mwen te kreye pa Gesner Deslandes, fondatè GlobalInternet.py. Li se yon enjenyè ki renmen edike Ayiti."
    
    # Greetings
    if q_lower in ["bonjou", "bonswa", "hello", "hi", "salut"]:
        return "Bonjou! Kijan ou ye? Mwen la pou reponn kesyon ou."
    
    return None

# ---------- LOGICAL REASONING ----------
def reason_about_question(query, lang):
    q = query.lower().strip()
    
    # Simple arithmetic
    math_match = re.search(r"(\d+)\s*([\+\-\*\/])\s*(\d+)", q)
    if math_match:
        try:
            a, op, b = int(math_match.group(1)), math_match.group(2), int(math_match.group(3))
            if op == '+':
                res = a + b
            elif op == '-':
                res = a - b
            elif op == '*':
                res = a * b
            elif op == '/':
                res = a / b
            else:
                res = None
            if res is not None:
                if lang == "ht":
                    return f"Repons lan se {res}."
                elif lang == "fr":
                    return f"La réponse est {res}."
                elif lang == "es":
                    return f"La respuesta es {res}."
                else:
                    return f"The answer is {res}."
        except:
            pass
    
    # Capital cities
    if "kapital" in q or "capital" in q:
        capitals = {
            "france": "Paris",
            "ayiti": "Pòtoprens",
            "haiti": "Port‑au‑Prince",
            "etazini": "Washington, D.C.",
            "usa": "Washington, D.C.",
            "kanada": "Ottawa",
            "brezil": "Brasília",
            "alman": "Bèlen",
            "itali": "Wòm",
            "espay": "Madrid",
            "angle": "Londr",
            "japon": "Tokiyo",
        }
        for country, cap in capitals.items():
            if country in q:
                if lang == "ht":
                    return f"Kapital {country.title()} se {cap}."
                elif lang == "fr":
                    return f"La capitale de {country.title()} est {cap}."
                elif lang == "es":
                    return f"La capital de {country.title()} es {cap}."
                else:
                    return f"The capital of {country.title()} is {cap}."
    
    # Current time
    if "ki lè li ye" in q or "what time" in q:
        now = datetime.now().strftime("%H:%M")
        if lang == "ht":
            return f"Kounye a li {now}."
        elif lang == "fr":
            return f"Il est {now}."
        elif lang == "es":
            return f"Son las {now}."
        else:
            return f"It is {now}."
    
    return None

# ---------- RESPONSE ----------
def generate_response(user_input, target_lang):
    # Phase 1: direct keywords
    direct = direct_keyword_answer(user_input, target_lang)
    if direct:
        return direct, False, None
    
    # Phase 2: trained facts (from dictionaries and voice training)
    facts = retrieve_facts_hybrid(user_input, k=3)
    if facts:
        return facts[0], False, None
    
    # Phase 3: logical reasoning
    logic = reason_about_question(user_input, target_lang)
    if logic:
        return logic, False, None
    
    # Phase 4: fallback
    fallbacks = {
        "en": "I don't know that yet. Please teach me using the Training Center (dictionaries or voice training).",
        "fr": "Je ne connais pas encore cela. Enseignez‑moi via le Centre d'entraînement (dictionnaires ou voix).",
        "ht": "Mwen poko konn sa. Tanpri anseye m nan Sant Fòmasyon (diksyonè oswa vwa).",
        "es": "Todavía no lo sé. Por favor enséñame en el Centro de Entrenamiento (diccionarios o voz)."
    }
    return fallbacks.get(target_lang, fallbacks["en"]), True, target_lang

def play_voice_button(text, is_fallback, fallback_audio_lang, button_label="🔊", key_suffix=""):
    if is_fallback:
        lang_map = {"en": "en-US", "fr": "fr-FR", "ht": "fr-FR", "es": "es-ES"}
        tts_lang = lang_map.get(fallback_audio_lang, "en-US")
        safe_text = json.dumps(text)
        html = f"""
        <button class="speak-btn" id="ttsBtn_{key_suffix}" style="background-color:#ffaa33; border:none; border-radius:30px; padding:5px 12px; margin-left:12px; cursor:pointer;">{button_label}</button>
        <script>
            (function() {{
                const btn = document.getElementById('ttsBtn_{key_suffix}');
                let utterance = null;
                function speakWithVoice() {{
                    if (utterance) window.speechSynthesis.cancel();
                    utterance = new SpeechSynthesisUtterance({safe_text});
                    utterance.lang = '{tts_lang}';
                    let voices = window.speechSynthesis.getVoices();
                    if (voices.length === 0) {{
                        window.speechSynthesis.onvoiceschanged = function() {{
                            voices = window.speechSynthesis.getVoices();
                            selectBestVoice(voices, utterance);
                            window.speechSynthesis.speak(utterance);
                        }};
                        return;
                    }}
                    selectBestVoice(voices, utterance);
                    window.speechSynthesis.speak(utterance);
                }}
                function selectBestVoice(voices, utterance) {{
                    let langCode = '{tts_lang}';
                    let priorityNames = [];
                    if (langCode === 'fr-FR') priorityNames = ['Google français', 'Microsoft Hortense', 'Microsoft Denis', 'Samantha', 'Thomas'];
                    if (langCode === 'en-US') priorityNames = ['Google US English', 'Microsoft David', 'Microsoft Zira', 'Samantha'];
                    if (langCode === 'es-ES') priorityNames = ['Google español', 'Microsoft Helena', 'Microsoft Pablo', 'Monica'];
                    let selected = null;
                    for (let name of priorityNames) {{
                        selected = voices.find(v => v.lang === langCode && v.name.includes(name));
                        if (selected) break;
                    }}
                    if (!selected) selected = voices.find(v => v.lang === langCode);
                    if (selected) utterance.voice = selected;
                }}
                btn.onclick = speakWithVoice;
            }})();
        </script>
        """
        return html
    else:
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

# ---------- DICTIONARY MANAGER ----------
def dictionary_manager(t):
    st.markdown(f"## {t['dict_title']}")
    col1, col2, col3 = st.columns(3)
    
    def display_dict(lang_code, lang_label, dict_data):
        st.markdown(f"### {lang_label}")
        w = st.text_input(f"{t['dict_word']} ({lang_code.upper()})", key=f"{lang_code}_word")
        m = st.text_input(f"{t['dict_meaning']} ({lang_code.upper()})", key=f"{lang_code}_meaning")
        if st.button(t['dict_add'], key=f"add_{lang_code}"):
            if w and m:
                dict_data[w] = m
                # Also train the AI with this fact
                fact = f"{w} means {m}"
                add_to_facts(fact)
                st.success(t['trained_entry_success'].format(word=w, meaning=m))
                st.rerun()
        for word, meaning in list(dict_data.items()):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.text(f"{word}: {meaning}")
            with col_b:
                if st.button(f"{t['dict_delete']}", key=f"del_{lang_code}_{word}"):
                    del dict_data[word]
                    st.rerun()
    
    with col1:
        display_dict("ht", t['dict_ht'], st.session_state.dictionaries["ht"])
    with col2:
        display_dict("fr", t['dict_fr'], st.session_state.dictionaries["fr"])
    with col3:
        display_dict("en", t['dict_en'], st.session_state.dictionaries["en"])

# ---------- VOICE TRAINING ----------
def voice_training(t):
    st.markdown(f"## {t['voice_training_title']}")
    st.info("🎙️ Upload your voice for Kreyòl phrases. Gesner AI will use your exact voice when answering those sentences.")
    recorder_html = f"""
    <div id="recorder-container">
        <button id="recordBtn" style="background-color:#e94560; border:none; border-radius:30px; padding:8px 16px; color:white;">{t['record_btn']}</button>
        <button id="stopBtn" disabled style="background-color:#555; border:none; border-radius:30px; padding:8px 16px;">{t['stop_btn']}</button>
        <p id="recordingStatus"></p>
        <audio id="audioPlayback" controls style="width:100%; margin-top:10px;"></audio>
        <a id="downloadLink" style="display:block; margin-top:10px; color:#ffaa66;">{t['download_btn']}</a>
    </div>
    <script>
        let mediaRecorder; let audioChunks = [];
        const recordBtn = document.getElementById('recordBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusP = document.getElementById('recordingStatus');
        const audioPlayback = document.getElementById('audioPlayback');
        const downloadLink = document.getElementById('downloadLink');
        recordBtn.onclick = async () => {{
            try {{
                const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
                mediaRecorder.onstop = () => {{
                    const audioBlob = new Blob(audioChunks, {{ type: 'audio/wav' }});
                    const audioUrl = URL.createObjectURL(audioBlob);
                    audioPlayback.src = audioUrl;
                    downloadLink.href = audioUrl;
                    downloadLink.download = 'recording.wav';
                    downloadLink.style.display = 'block';
                    audioChunks = [];
                    statusP.innerText = '';
                }};
                mediaRecorder.start();
                recordBtn.disabled = true;
                stopBtn.disabled = false;
                statusP.innerText = 'Recording...';
            }} catch (err) {{
                statusP.innerText = 'Microphone access denied or error: ' + err.message;
            }}
        }};
        stopBtn.onclick = () => {{
            if (mediaRecorder && mediaRecorder.state === 'recording') {{
                mediaRecorder.stop();
                recordBtn.disabled = false;
                stopBtn.disabled = true;
                statusP.innerText = 'Stopped. Click Download to save file, then upload below.';
            }}
        }};
    </script>
    """
    st.components.v1.html(recorder_html, height=200)
    st.markdown(f"### 📂 {t['voice_upload']}")
    uploaded_file = st.file_uploader(t['voice_upload'], type=["wav", "mp3"], key="voice_upload")
    transcript = st.text_area(t['voice_transcribed_text'], key="voice_transcript")
    if uploaded_file and transcript.strip():
        if st.button(t['voice_train'], use_container_width=True):
            audio_bytes = uploaded_file.read()
            save_voice_for_text(transcript.strip(), audio_bytes)
            add_to_facts(transcript.strip())
            st.success(t['voice_success'])

# ---------- SIDEBAR ----------
def show_sidebar():
    lang_names = list(LANGUAGES.keys())
    selected_lang_name = st.sidebar.selectbox("🌐 Language", lang_names, key="main_lang_selector")
    selected_lang_code = LANGUAGES[selected_lang_name]
    st.session_state.ui_language = selected_lang_code
    st.session_state.chat_language = selected_lang_code
    t = TEXTS.get(st.session_state.ui_language, TEXTS["en"])
    
    st.sidebar.markdown(
        f"""
        <div style="text-align: center;">
            <div style="font-size:80px; animation:spin 4s linear infinite; display:inline-block;">🌍</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown(f"## **{t['sidebar_company']}**")
    st.sidebar.markdown(f"### {t['sidebar_product']}")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{t['built_by']}**")
    st.sidebar.markdown(t['phone'])
    st.sidebar.markdown(t['email'])
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"{t['website_label']} [{t['website_link']}]({t['website_link']})")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {t['pricing_title']}")
    st.sidebar.markdown(t['pricing_table'])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {t['unlock_training']}")
    if not st.session_state.training_access:
        api_key_input = st.sidebar.text_input(t['api_key_label'], type="password", key="api_key_input")
        if st.sidebar.button("Unlock Training Center"):
            # API key is fixed – use the same as before
            if api_key_input == "PNL_fJC4L5QNjA0GJbc4N8TzIXBjdfIXfgcLv1yZ8Yc":
                st.session_state.training_access = True
                st.sidebar.success("Access granted!")
                st.rerun()
            else:
                st.sidebar.error("Invalid API Key")
    else:
        st.sidebar.success("✅ Training mode active")
        if st.sidebar.button("Lock Training Center"):
            st.session_state.training_access = False
            st.rerun()
    
    st.sidebar.markdown("---")
    if st.sidebar.button(t['clear'], key="clear_sidebar"):
        st.session_state.conversation_history = []
        st.rerun()

# ---------- MAIN CHAT INTERFACE ----------
def chat_interface(t):
    st.markdown(f"<h1 style='text-align:center; color:#ffd966;'>{t['app_title']}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Ask me anything. I learn from dictionaries and voice training.</p>", unsafe_allow_html=True)
    
    # Chat history
    for idx, msg in enumerate(st.session_state.conversation_history):
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">🧑‍💻 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([10, 1])
            with col1:
                st.markdown(f'<div class="chat-message assistant-message" style="width:100%;">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            with col2:
                btn_html = play_voice_button(
                    msg["content"],
                    msg.get("is_fallback", False),
                    msg.get("fallback_lang"),
                    "🔊",
                    f"chat_{idx}"
                )
                if btn_html:
                    st.components.v1.html(btn_html, height=50)
    
    # Input
    user_input = st.text_input(t['chat_input'], key="chat_input")
    if st.button(t['send'], use_container_width=True, key="send_btn"):
        if user_input.strip():
            target_lang = st.session_state.chat_language
            answer, is_fallback, fallback_lang = generate_response(user_input, target_lang)
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": answer,
                "is_fallback": is_fallback,
                "fallback_lang": fallback_lang
            })
            st.rerun()
    
    if st.button(t['clear'], use_container_width=True, key="clear_btn"):
        st.session_state.conversation_history = []
        st.rerun()

# ---------- TRAINING CENTER (dictionaries + voice) ----------
def training_center(t):
    st.markdown(f"<h1 style='text-align:center;'>🔧 {t['training_section']}</h1>", unsafe_allow_html=True)
    dictionary_manager(t)
    st.markdown("---")
    voice_training(t)

# ---------- MAIN ----------
def main():
    # Set default UI language
    if "ui_language" not in st.session_state:
        st.session_state.ui_language = "en"
    if "chat_language" not in st.session_state:
        st.session_state.chat_language = "en"
    
    show_sidebar()
    t = TEXTS.get(st.session_state.ui_language, TEXTS["en"])
    
    if st.session_state.training_access:
        mode = st.radio("Select mode", ["💬 Chat Mode", "🔧 Training Center"], horizontal=True)
        if mode == "💬 Chat Mode":
            chat_interface(t)
        else:
            training_center(t)
    else:
        chat_interface(t)
    
    st.markdown(f'<div class="footer">{t["footer"]}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
