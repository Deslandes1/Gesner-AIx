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

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
    .stMarkdown, .stTextInput label, .stButton button, h1, h2, p { color: #ffffff !important; }
    .stButton button { background-color: #e94560 !important; border-radius: 30px !important; }
    .stTextInput input, .stTextArea textarea { background-color: #0f3460 !important; border-radius: 12px; border: 1px solid #e94560; color: white; }
    .chat-message { padding: 1rem; border-radius: 20px; margin-bottom: 1rem; }
    .user-message { background: linear-gradient(135deg, #e94560, #ff6b6b); color: white; }
    .assistant-message { background: linear-gradient(135deg, #0f3460, #1a4a7a); color: white; }
    .footer { text-align: center; margin-top: 2rem; padding: 1rem; border-top: 1px solid #e94560; color: white; }
    </style>
    """, unsafe_allow_html=True
)

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

# ---------- API key for training (same as you defined) ----------
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

# ---------- built‑in answers ----------
def direct_keyword_answer(query):
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

def reason_about_question(query):
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

def generate_answer(query):
    direct = direct_keyword_answer(query)
    if direct:
        return direct
    facts = retrieve_relevant_facts(query, k=3)
    if facts:
        return facts[0]
    logic = reason_about_question(query)
    if logic:
        return logic
    return "Mwen poko gen repons sa. Tanpri anseye m nan seksyon 'Anseye m' anba a."

def play_voice_button(text, key_suffix=""):
    import base64
    voice = get_voice_for_text(text)
    if voice:
        b64 = base64.b64encode(voice).decode()
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
    else:
        safe_text = json.dumps(text)
        return f"""
        <button id="tts{key_suffix}" style="background:#ffaa33; border:none; border-radius:30px; padding:5px 12px;">🔊</button>
        <script>
            document.getElementById('tts{key_suffix}').onclick = () => {{
                var u = new SpeechSynthesisUtterance({safe_text});
                u.lang = 'fr-FR';
                speechSynthesis.speak(u);
            }};
        </script>
        """

# ---------- main UI ----------
load_previous_training()

st.markdown("<h1 style='text-align:center;'>💬 Gesner AI – Fast Chat</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>I learn from what you teach me. Ask me anything!</p>", unsafe_allow_html=True)

# Chat history display
for idx, msg in enumerate(st.session_state.conversation_history):
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-message user-message">🧑‍💻 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        col1, col2 = st.columns([10,1])
        with col1:
            st.markdown(f'<div class="chat-message assistant-message">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
        with col2:
            btn = play_voice_button(msg["content"], f"chat{idx}")
            if btn:
                st.components.v1.html(btn, height=50)

user_input = st.text_input("Your question:", key="chat_input")
if st.button("Send", use_container_width=True):
    if user_input.strip():
        ans = generate_answer(user_input)
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        st.session_state.conversation_history.append({"role": "assistant", "content": ans})
        st.rerun()

# ---------- Training section with API key protection ----------
st.sidebar.markdown("## 🔐 Training Access")
with st.sidebar:
    if not st.session_state.training_access:
        api_key_input = st.text_input("Enter API Key to teach me", type="password")
        if st.button("Unlock Training"):
            if api_key_input == REQUIRED_API_KEY:
                st.session_state.training_access = True
                st.success("Access granted!")
                st.rerun()
            else:
                st.error("Invalid API Key")
    else:
        st.success("Training mode active")
        if st.button("Lock Training"):
            st.session_state.training_access = False
            st.rerun()

if st.session_state.training_access:
    with st.expander("📚 Teach me something new", expanded=True):
        new_fact = st.text_area("Enter a fact, sentence, or Q&A pair:")
        voice_file = st.file_uploader("Optional: upload your voice for this text", type=["wav","mp3"])
        if st.button("Learn this", use_container_width=True):
            if new_fact.strip():
                if voice_file:
                    save_voice_for_text(new_fact.strip(), voice_file.read())
                add_to_training(new_fact.strip())
                st.rerun()
            else:
                st.warning("Please enter some text.")
else:
    st.info("🔒 Training is locked. Enter the API key in the sidebar to teach me new facts.")

# ---------- Sidebar utilities ----------
with st.sidebar:
    st.markdown("---")
    st.markdown("## 🌍 GlobalInternet.py")
    st.markdown("**Gesner Deslandes – Coder in Chief**")
    st.markdown("📞 (509)-47385663  |  ✉️ deslandes78@gmail.com")
    st.markdown("---")
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.conversation_history = []
        st.rerun()
    if st.button("🔥 Reset all knowledge", use_container_width=True):
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

st.markdown('<div class="footer">© GlobalInternet.py – Gesner AI | Fast, lightweight, always learning</div>', unsafe_allow_html=True)
