import streamlit as st
import json
import os
import numpy as np
import time
import hashlib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Gesner AI – Fast Chat", page_icon="🧠", layout="wide")

# ---------- CSS: sidebar same gradient as main ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-right: 2px solid #e94560;
    }
    .stMarkdown, .stTextInput label, .stButton button, h1, h2, p, div, span {
        color: #ffffff !important;
    }
    .stButton button {
        background-color: #e94560 !important;
        border-radius: 30px !important;
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #0f3460 !important;
        border-radius: 12px;
        border: 1px solid #e94560;
        color: white;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 20px;
        margin-bottom: 1rem;
    }
    .user-message {
        background: linear-gradient(135deg, #e94560, #ff6b6b);
        color: white;
    }
    .assistant-message {
        background: linear-gradient(135deg, #0f3460, #1a4a7a);
        color: white;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #e94560;
        color: white;
    }
    .char-picker {
        background: rgba(15,52,96,0.6);
        border-radius: 12px;
        padding: 10px;
        margin: 10px 0;
    }
    .char-btn {
        background-color: #2a5298;
        border: none;
        border-radius: 20px;
        padding: 5px 12px;
        color: white;
        cursor: pointer;
        font-size: 1rem;
        margin: 3px;
        transition: 0.2s;
    }
    .char-btn:hover {
        background-color: #e94560;
    }
    </style>
    """, unsafe_allow_html=True
)

# ---------- Language texts (including Haitian Creole) ----------
LANGUAGES = {
    "English": "en",
    "Français": "fr",
    "Kreyòl Ayisyen": "ht",
    "Español": "es"
}

TEXTS = {
    "en": {
        "chat_title": "💬 Gesner AI Chat",
        "user_prefix": "🧑‍💻 You: ",
        "assistant_prefix": "🤖 Gesner AI: ",
        "send_button": "Send",
        "chat_input_placeholder": "Ask me anything...",
        "training_title": "📚 Teach me something new",
        "fact_label": "Enter a fact, sentence, or Q&A pair:",
        "voice_upload_label": "Optional: upload your voice for this text",
        "learn_button": "Learn this",
        "question_list_title": "📋 Choose a trained question:",
        "ask_button": "Ask this",
        "clear_chat": "🗑️ Clear chat history",
        "reset_all": "🔥 Reset all knowledge",
        "footer": "© GlobalInternet.py – Gesner AI | Fast, lightweight, always learning",
        "no_facts_answer": "I don't know that yet. Please teach me in the training section (API key required).",
        "training_locked": "🔒 Training is locked. Enter the API key in the sidebar to teach me new facts.",
        "api_key_label": "Enter API Key to teach me",
        "unlock_button": "Unlock Training",
        "lock_button": "Lock Training",
        "training_active": "Training mode active",
        "invalid_key": "Invalid API Key",
        "char_picker_label": "Insert Kreyòl characters (click to add):"
    },
    "fr": {
        "chat_title": "💬 Gesner IA Chat",
        "user_prefix": "🧑‍💻 Vous : ",
        "assistant_prefix": "🤖 Gesner IA : ",
        "send_button": "Envoyer",
        "chat_input_placeholder": "Demandez-moi n'importe quoi...",
        "training_title": "📚 Enseignez-moi quelque chose",
        "fact_label": "Entrez un fait, une phrase ou une paire Q/R :",
        "voice_upload_label": "Optionnel : téléchargez votre voix pour ce texte",
        "learn_button": "Apprendre",
        "question_list_title": "📋 Choisissez une question entraînée :",
        "ask_button": "Poser",
        "clear_chat": "🗑️ Effacer l'historique",
        "reset_all": "🔥 Tout réinitialiser",
        "footer": "© GlobalInternet.py – Gesner IA | Rapide, léger, toujours en apprentissage",
        "no_facts_answer": "Je ne connais pas encore cela. Veuillez m'enseigner dans la section d'entraînement (clé API requise).",
        "training_locked": "🔒 L'entraînement est verrouillé. Entrez la clé API dans la barre latérale pour m'enseigner.",
        "api_key_label": "Entrez la clé API pour m'enseigner",
        "unlock_button": "Déverrouiller",
        "lock_button": "Verrouiller",
        "training_active": "Mode entraînement actif",
        "invalid_key": "Clé API invalide",
        "char_picker_label": "Insérer des caractères kreyòl (cliquez pour ajouter) :"
    },
    "ht": {
        "chat_title": "💬 Gesner AI Chat",
        "user_prefix": "🧑‍💻 Ou : ",
        "assistant_prefix": "🤖 Gesner AI : ",
        "send_button": "Voye",
        "chat_input_placeholder": "Pose yon kesyon...",
        "training_title": "📚 Anseye m yon bagay nouvo",
        "fact_label": "Antre yon reyalite, yon fraz, oswa yon kesyon/repons :",
        "voice_upload_label": "Opsyonèl: chaje vwa ou pou tèks sa a",
        "learn_button": "Aprann sa",
        "question_list_title": "📋 Chwazi yon kesyon antrene :",
        "ask_button": "Mande sa",
        "clear_chat": "🗑️ Efase listorik chat la",
        "reset_all": "🔥 Efase tout konesans",
        "footer": "© GlobalInternet.py – Gesner AI | Rapid, lejè, toujou ap aprann",
        "no_facts_answer": "Mwen poko konnen sa. Tanpri anseye m nan seksyon fòmasyon (kle API obligatwa).",
        "training_locked": "🔒 Fòmasyon an bloke. Antre kle API a nan ba lateral la pou anseye m nouvo reyalite.",
        "api_key_label": "Antre kle API pou anseye m",
        "unlock_button": "Dekloke Fòmasyon",
        "lock_button": "Bloke Fòmasyon",
        "training_active": "Mòd fòmasyon aktif",
        "invalid_key": "Kle API pa bon",
        "char_picker_label": "Antre karaktè kreyòl (klike pou ajoute) :"
    },
    "es": {
        "chat_title": "💬 Gesner AI Chat",
        "user_prefix": "🧑‍💻 Tú: ",
        "assistant_prefix": "🤖 Gesner AI: ",
        "send_button": "Enviar",
        "chat_input_placeholder": "Pregúntame cualquier cosa...",
        "training_title": "📚 Enséñame algo nuevo",
        "fact_label": "Ingrese un hecho, frase o par pregunta/respuesta:",
        "voice_upload_label": "Opcional: sube tu voz para este texto",
        "learn_button": "Aprender",
        "question_list_title": "📋 Elige una pregunta entrenada:",
        "ask_button": "Preguntar",
        "clear_chat": "🗑️ Borrar historial",
        "reset_all": "🔥 Reiniciar todo",
        "footer": "© GlobalInternet.py – Gesner AI | Rápido, ligero, siempre aprendiendo",
        "no_facts_answer": "Todavía no sé eso. Por favor enséñame en la sección de entrenamiento (se requiere clave API).",
        "training_locked": "🔒 El entrenamiento está bloqueado. Ingrese la clave API en la barra lateral para enseñarme.",
        "api_key_label": "Ingrese la clave API para enseñarme",
        "unlock_button": "Desbloquear",
        "lock_button": "Bloquear",
        "training_active": "Modo entrenamiento activo",
        "invalid_key": "Clave API inválida",
        "char_picker_label": "Insertar caracteres kreyòl (haga clic para agregar):"
    }
}

# ---------- session state ----------
if "training_data" not in st.session_state:
    st.session_state.training_data = []
if "texts" not in st.session_state:
    st.session_state.texts = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "tfidf_vectorizer" not in st.session_state:
    st.session_state.tfidf_vectorizer = None
if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None
if "training_access" not in st.session_state:
    st.session_state.training_access = False
if "ui_language" not in st.session_state:
    st.session_state.ui_language = "en"
if "train_text_area" not in st.session_state:
    st.session_state.train_text_area = ""

# ---------- API key (same as before) ----------
REQUIRED_API_KEY = "PNL_fJC4L5QNjA0GJbc4N8TzIXBjdfIXfgcLv1yZ8Yc"

# ---------- voice cache ----------
VOICE_CACHE_DIR = "voice_cache"
os.makedirs(VOICE_CACHE_DIR, exist_ok=True)

def get_voice_filename(text):
    norm = text.strip().lower()
    h = hashlib.md5(norm.encode()).hexdigest()
    return os.path.join(VOICE_CACHE_DIR, f"{h}.wav")

def save_voice_for_text(text, audio_bytes):
    with open(get_voice_filename(text), "wb") as f:
        f.write(audio_bytes)

def get_voice_for_text(text):
    fname = get_voice_filename(text)
    if os.path.exists(fname):
        with open(fname, "rb") as f:
            return f.read()
    return None

# ---------- TF‑IDF knowledge base ----------
def build_tfidf():
    if st.session_state.texts:
        st.session_state.tfidf_vectorizer = TfidfVectorizer(stop_words=None)
        st.session_state.tfidf_matrix = st.session_state.tfidf_vectorizer.fit_transform(st.session_state.texts)

def retrieve_relevant_facts(query, k=3):
    if not st.session_state.texts or st.session_state.tfidf_vectorizer is None:
        return []
    q_vec = st.session_state.tfidf_vectorizer.transform([query])
    scores = cosine_similarity(q_vec, st.session_state.tfidf_matrix).flatten()
    top_indices = scores.argsort()[-k:][::-1]
    return [st.session_state.texts[i] for i in top_indices if scores[i] > 0.1]

def add_to_training(text):
    if not text.strip():
        st.warning("Please enter some text.")
        return False
    st.session_state.training_data.append({"text": text, "embedding": []})
    st.session_state.texts.append(text)
    with open("training_data.json", "w") as f:
        json.dump(st.session_state.training_data, f)
    build_tfidf()
    st.success(f"✅ Learned: {text[:100]}...")
    return True

def load_previous_training():
    if os.path.exists("training_data.json"):
        try:
            with open("training_data.json") as f:
                data = json.load(f)
            if isinstance(data, list):
                st.session_state.training_data = data
                st.session_state.texts = [item["text"] for item in data]
                build_tfidf()
        except Exception:
            pass

# ---------- built‑in answers (Kreyòl / multilingual) ----------
def direct_keyword_answer(query, lang):
    q = query.lower().strip()
    if any(w in q for w in ["konbyen vwayèl", "vwayel"]):
        return "Alfabè kreyòl la gen 8 vwayèl: A, E, È, I, O, Ò, OU, UI."
    if "konbyen konsòn" in q:
        return "Alfabè kreyòl la gen 24 konsòn."
    if any(w in q for w in ["konbyen lèt", "konbyen let"]):
        return "Alfabè kreyòl la gen 32 lèt: A, AN, B, CH, D, E, È, EN, F, G, H, I, J, K, L, M, N, NG, O, Ò, ON, OU, OUN, P, R, S, T, UI, V, W, Y, Z."
    if any(p in q for p in ["kijan ou rele", "kiyès ou ye", "what is your name", "who are you"]):
        return "Non pa mw se Gesner L’IA, kreyatè mw an se Gesner Deslandes nan GlobalInternet.py."
    if any(p in q for p in ["bonjou", "bonswa", "hello", "hi"]):
        return "Bonjou! Kijan ou ye? Mwen la pou reponn kesyon ou."
    return None

def reason_about_question(query, lang):
    q = query.lower().strip()
    m = re.search(r"(\d+)\s*([\+\-\*\/])\s*(\d+)", q)
    if m:
        a, op, b = int(m[1]), m[2], int(m[3])
        if op == '+': return f"Repons lan se {a+b}."
        if op == '-': return f"Repons lan se {a-b}."
        if op == '*': return f"Repons lan se {a*b}."
        if op == '/': return f"Repons lan se {a/b}."
    if "kapital" in q or "capital" in q:
        caps = {"france":"Paris","ayiti":"Pòtoprens","haiti":"Port‑au‑Prince","etazini":"Washington, D.C."}
        for ctry, cap in caps.items():
            if ctry in q:
                return f"Kapital {ctry.title()} se {cap}."
    if "ki lè li ye" in q or "what time" in q:
        return f"Kounye a li {time.strftime('%H:%M')}."
    return None

def generate_answer(query, lang):
    direct = direct_keyword_answer(query, lang)
    if direct:
        return direct, False
    facts = retrieve_relevant_facts(query, k=3)
    if facts:
        return facts[0], False
    logic = reason_about_question(query, lang)
    if logic:
        return logic, False
    fallbacks = {
        "en": "I don't know that yet. Please teach me in the training section (API key required).",
        "fr": "Je ne connais pas encore cela. Veuillez m'enseigner dans la section d'entraînement (clé API requise).",
        "ht": "Mwen poko konnen sa. Tanpri anseye m nan seksyon fòmasyon (kle API obligatwa).",
        "es": "Todavía no sé eso. Por favor enséñame en la sección de entrenamiento (se requiere clave API)."
    }
    return fallbacks.get(lang, fallbacks["en"]), True

# ---------- Voice button with language fallback ----------
def play_voice_button(text, is_fallback, lang, key_suffix=""):
    import base64
    if not is_fallback:
        voice_bytes = get_voice_for_text(text)
        if voice_bytes:
            b64 = base64.b64encode(voice_bytes).decode()
            html = f"""
            <button id="vb{key_suffix}" style="background:#ffaa33; border:none; border-radius:30px; padding:5px 12px;">🔊</button>
            <audio id="ad{key_suffix}" style="display:none;"></audio>
            <script>
                (function(){{
                    const btn = document.getElementById('vb{key_suffix}');
                    const aud = document.getElementById('ad{key_suffix}');
                    const b64 = "{b64}";
                    const binary = atob(b64);
                    const bytes = new Uint8Array(binary.length);
                    for(let i=0;i<binary.length;i++) bytes[i]=binary.charCodeAt(i);
                    const blob = new Blob([bytes], {{type:'audio/wav'}});
                    aud.src = URL.createObjectURL(blob);
                    btn.onclick = () => aud.play();
                }})();
            </script>
            """
            return html
    # Fallback TTS
    tts_lang_map = {"en": "en-US", "fr": "fr-FR", "ht": "fr-FR", "es": "es-ES"}
    tts_lang = tts_lang_map.get(lang, "fr-FR")
    safe_text = json.dumps(text)
    return f"""
    <button id="tts{key_suffix}" style="background:#ffaa33; border:none; border-radius:30px; padding:5px 12px;">🔊</button>
    <script>
        document.getElementById('tts{key_suffix}').onclick = () => {{
            var utterance = new SpeechSynthesisUtterance({safe_text});
            utterance.lang = '{tts_lang}';
            window.speechSynthesis.speak(utterance);
        }};
    </script>
    """

# ---------- Character picker component ----------
def character_picker(target_key):
    """Display buttons for Kreyòl letters. When clicked, append to the text area stored in session_state[target_key]."""
    chars_lower = ["a", "an", "b", "ch", "d", "e", "è", "en", "f", "g", "h", "i", "j", "k", "l", "m", "n", "ng", "o", "ò", "on", "ou", "oun", "p", "r", "s", "t", "ui", "v", "w", "y", "z"]
    chars_upper = [c.upper() for c in chars_lower]
    # Combine both cases in a readable order
    all_chars = []
    for lc, uc in zip(chars_lower, chars_upper):
        all_chars.extend([lc, uc])
    # Remove duplicates that might appear (e.g., 'a' and 'A' are fine)
    # Create a row of buttons
    cols = st.columns(len(all_chars))
    for i, ch in enumerate(all_chars):
        with cols[i]:
            if st.button(ch, key=f"char_{target_key}_{ch}"):
                current = st.session_state.get(target_key, "")
                st.session_state[target_key] = current + ch
                st.rerun()

# ---------- main UI ----------
load_previous_training()

# Language selection in sidebar
with st.sidebar:
    lang_name = st.selectbox("🌐 Language", list(LANGUAGES.keys()), key="lang_selector")
    st.session_state.ui_language = LANGUAGES[lang_name]

t = TEXTS[st.session_state.ui_language]

st.markdown(f"<h1 style='text-align:center;'>{t['chat_title']}</h1>", unsafe_allow_html=True)

# Display chat history
for idx, msg in enumerate(st.session_state.conversation_history):
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-message user-message">{t["user_prefix"]}{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        col1, col2 = st.columns([10,1])
        with col1:
            st.markdown(f'<div class="chat-message assistant-message">{t["assistant_prefix"]}{msg["content"]}</div>', unsafe_allow_html=True)
        with col2:
            btn = play_voice_button(msg["content"], msg.get("is_fallback", False), st.session_state.ui_language, f"chat{idx}")
            if btn:
                st.components.v1.html(btn, height=50)

# ---- Question list from training (selectbox) ----
if st.session_state.texts:
    st.markdown(f"### {t['question_list_title']}")
    options = [f"{i+1}: {fact[:80]}{'...' if len(fact)>80 else ''}" for i, fact in enumerate(st.session_state.texts)]
    selected_option = st.selectbox("", options, key="trained_question_select")
    if st.button(t['ask_button'], use_container_width=True):
        idx = int(selected_option.split(":")[0]) - 1
        question = st.session_state.texts[idx]
        answer, is_fallback = generate_answer(question, st.session_state.ui_language)
        st.session_state.conversation_history.append({"role": "user", "content": question})
        st.session_state.conversation_history.append({"role": "assistant", "content": answer, "is_fallback": is_fallback})
        st.rerun()

# Chat input
user_input = st.text_input(t["chat_input_placeholder"], key="chat_input")
if st.button(t["send_button"], use_container_width=True):
    if user_input.strip():
        answer, is_fallback = generate_answer(user_input, st.session_state.ui_language)
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        st.session_state.conversation_history.append({"role": "assistant", "content": answer, "is_fallback": is_fallback})
        st.rerun()

# ---------- Training section with API key protection ----------
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔐 Training Access")
if not st.session_state.training_access:
    api_key_input = st.sidebar.text_input(t["api_key_label"], type="password")
    if st.sidebar.button(t["unlock_button"]):
        if api_key_input == REQUIRED_API_KEY:
            st.session_state.training_access = True
            st.rerun()
        else:
            st.sidebar.error(t["invalid_key"])
else:
    st.sidebar.success(t["training_active"])
    if st.sidebar.button(t["lock_button"]):
        st.session_state.training_access = False
        st.rerun()

if st.session_state.training_access:
    with st.expander(t["training_title"], expanded=True):
        # Character picker
        st.markdown(f"**{t['char_picker_label']}**")
        character_picker("train_text_area")
        # Text area for new fact
        new_fact = st.text_area(t["fact_label"], key="train_text_area", height=150)
        voice_file = st.file_uploader(t["voice_upload_label"], type=["wav","mp3"], key="train_voice")
        if st.button(t["learn_button"], use_container_width=True):
            if new_fact.strip():
                if voice_file:
                    save_voice_for_text(new_fact.strip(), voice_file.read())
                add_to_training(new_fact.strip())
                st.session_state.train_text_area = ""  # clear after training
                st.rerun()
            else:
                st.warning("Please enter some text.")
else:
    st.info(t["training_locked"])

# Sidebar utilities
with st.sidebar:
    st.markdown("---")
    st.markdown("## 🌍 GlobalInternet.py")
    st.markdown("**Gesner Deslandes – Coder in Chief**")
    st.markdown("📞 (509)-47385663  |  ✉️ deslandes78@gmail.com")
    st.markdown("---")
    if st.button(t["clear_chat"], use_container_width=True):
        st.session_state.conversation_history = []
        st.rerun()
    if st.button(t["reset_all"], use_container_width=True):
        st.session_state.training_data = []
        st.session_state.texts = []
        st.session_state.conversation_history = []
        if os.path.exists("training_data.json"):
            os.remove("training_data.json")
        import shutil
        if os.path.exists(VOICE_CACHE_DIR):
            shutil.rmtree(VOICE_CACHE_DIR)
            os.makedirs(VOICE_CACHE_DIR)
        build_tfidf()
        st.success("All knowledge erased. Start fresh!")
        time.sleep(1)
        st.rerun()

st.markdown(f'<div class="footer">{t["footer"]}</div>', unsafe_allow_html=True)
