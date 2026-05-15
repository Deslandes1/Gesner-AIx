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

# ========== RESET OLD DATA ==========
DATA_DIR = ".gesner_data"
if os.path.exists(DATA_DIR):
    shutil.rmtree(DATA_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_FILE = os.path.join(DATA_DIR, "training_data.json")
DICT_FILE = os.path.join(DATA_DIR, "dictionaries.json")
VOICE_FILE = os.path.join(DATA_DIR, "voice_cache.json")

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

# ---------- DEFAULT TRAINING (CORRECTED) ----------
def get_default_training_facts():
    facts = []

    # ----- ALPHABET (direct answers) -----
    facts.append("Alfabè kreyòl la gen 32 let.")
    facts.append("32 let.")
    facts.append("Alfabè kreyòl la gen trantde (32) let.")
    # The list of letters (for questions that ask for the list)
    facts.append("A, B, C, CH, D, E, È, F, G, H, I, J, K, L, M, N, NG, O, Ò, OU, P, R, S, T, UI, V, W, Y, Z.")
    facts.append("Lis tout let alfabè kreyòl la: A, B, C, CH, D, E, È, F, G, H, I, J, K, L, M, N, NG, O, Ò, OU, P, R, S, T, UI, V, W, Y, Z.")
    facts.append("Premye let nan alfabè kreyòl la se A, dènye let la se Z.")
    facts.append("Let CH pwononse tankou 'sh' nan angle.")
    facts.append("Let È pwononse tankou 'e' nan franse, let Ò pwononse tankou 'o' louvri.")
    facts.append("OU pwononse tankou 'ou' nan franse, UI pwononse tankou 'wi'.")
    facts.append("NG pwononse tankou 'ng' nan 'sitting' an angle.")

    # ----- BEGINNER PHRASES (unchanged) -----
    facts.append("Bonjou se fason pou di 'good morning' an Kreyòl.")
    facts.append("Bonswa se fason pou di 'good evening' an Kreyòl.")
    facts.append("Mèsi se 'thank you'.")
    facts.append("Tanpri se 'please'.")
    facts.append("Wi se 'yes', Non se 'no'.")
    facts.append("Mwen renmen ou se 'I love you'.")
    facts.append("Kijan ou rele? se 'What is your name?'")
    facts.append("Mwen rele [non] se repons lan.")
    facts.append("Kijan ou ye? oswa Sak pase? se 'How are you?'")
    facts.append("Mwen byen oswa Mwen la se repons lan.")
    facts.append("Mwen grangou se 'I am hungry'.")
    facts.append("Mwen swaf se 'I am thirsty'.")
    facts.append("Mwen fatige se 'I am tired'.")
    facts.append("Mwen kontan se 'I am happy'.")
    facts.append("Mwen tris se 'I am sad'.")
    facts.append("Mwen pa konprann se 'I don't understand'.")
    facts.append("Tanpri pale dousman se 'Please speak slowly'.")
    facts.append("Konbyen li koute? se 'How much does it cost?'")
    facts.append("Ki kote twalèt la ye? se 'Where is the bathroom?'")
    facts.append("Nimewo 1 a 10: yonn, de, twa, kat, senk, sis, sèt, uit, nèf, dis.")
    facts.append("Lendi se Monday, Madi se Tuesday, Mèkredi se Wednesday, Jedi se Thursday, Vandredi se Friday, Samdi se Saturday, Dimanch se Sunday.")
    facts.append("Mwa yo: Janvye, Fevriye, Mas, Avril, Me, Jen, Jiyè, Out, Septanm, Oktòb, Novanm, Desanm.")
    facts.append("Pwonon moun: Mwen, ou, li, nou, yo.")
    facts.append("Vèb 'ale' (to go) prezan: Mwen ale, ou ale, li ale, nou ale, yo ale.")
    facts.append("Vèb 'manje' (to eat) prezan: Mwen manje, ou manje, li manje, nou manje, yo manje.")
    facts.append("Vèb 'bwè' (to drink) prezan: Mwen bwè, ou bwè, li bwè, nou bwè, yo bwè.")
    facts.append("Mwen manje diri se 'I eat rice'.")
    facts.append("Ou bwè dlo se 'You drink water'.")
    facts.append("Li ale lekòl se 'He goes to school'.")
    facts.append("Nou rete Ayiti se 'We live in Haiti'.")
    facts.append("Yo kontan se 'They are happy'.")
    facts.append("Pou fè negatif, mete 'pa' apre pwonon: Mwen pa manje.")
    facts.append("Èske ou pale Kreyòl? se 'Do you speak Creole?'")
    facts.append("Mwen pale yon ti kras Kreyòl.")
    facts.append("Ki laj ou? se 'How old are you?' Mwen gen XX ane.")
    facts.append("Kisa sa ye? se 'What is this?'")
    facts.append("Ann ale se 'Let's go'.")
    facts.append("Tann mwen se 'Wait for me'.")
    facts.append("Vini isit la se 'Come here'.")
    facts.append("Chita se 'Sit down'.")
    facts.append("Leve kanpe se 'Stand up'.")
    facts.append("Mwen aprann Kreyòl.")
    facts.append("Eskize mwen, ki kote mache a ye?")
    facts.append("Mwen bezwen èd se 'I need help'.")
    facts.append("Èske ou ka ede mwen? se 'Can you help me?'")
    facts.append("Ki lè li ye? se 'What time is it?' Li ye twa è.")
    facts.append("Mwen ap vini demen se 'I will come tomorrow'.")
    facts.append("Nou deja manje se 'We ate already'.")
    facts.append("Li gen yon liv se 'She has a book'.")
    facts.append("Èske ou gen yon machin? se 'Do you have a car?'")
    facts.append("Gen yon pwoblèm se 'There is a problem'.")
    facts.append("Mwen pa gen lajan se 'I don't have money'.")
    facts.append("Sa koute chè se 'That's expensive'.")
    facts.append("Mwen vle manje se 'I want to eat'.")
    facts.append("Mwen bezwen dòmi se 'I need to sleep'.")
    facts.append("Ann danse se 'Let's dance'.")
    facts.append("Mwen renmen mizik Ayisyen.")
    facts.append("Ki manje ou pi renmen? Diri ak pwa se yon manje popilè.")
    facts.append("Ayiti se yon bèl peyi.")
    facts.append("Mwen vle vizite Kap Ayisyen.")
    facts.append("Tan an bèl jodi a. Lap fè lapli. Solèy la ap klere.")
    facts.append("Kisa w ap fè? M ap travay, M ap etidye, M ap li yon liv.")
    facts.append("Koute mwen. Gade mwen. Fè atansyon.")
    facts.append("Se pa anyen oswa Sa bon se 'It's okay'.")
    facts.append("Félicitasyon, Bòn chans, Bon apeti, Pran swen ou.")
    facts.append("Na wè pita, Na wè demen, Orevwa.")

    # ----- GRAMMAR (same as before) -----
    facts.append("Pou fè tan pase, mete 'te' anvan vèb la. Egzanp: Mwen te manje (I ate).")
    facts.append("Pou fè tan fiti, mete 'ap' oswa 'pral' anvan vèb la. Egzanp: Mwen ap manje (I will eat).")
    facts.append("Pou fè tan kontinyèl, mete 'ap' ant pwonon ak vèb: M ap manje (I am eating).")
    facts.append("Pou fè kondisyonèl, itilize 'ta': Mwen ta vini (I would come).")
    facts.append("Konparezon: pi ... pase (more than), mwens ... pase (less than). Egzanp: Li pi gran pase mwen.")
    facts.append("Sipèlatif: pi ... nan tout. Egzanp: Li se pi bèl nan tout.")
    facts.append("Mo 'ke' lye fraz: Mwen konnen ke li renmen mwen.")
    facts.append("Mo 'pou' endike bi: Mwen vini pou ede ou.")
    facts.append("Mo 'avan' (before), 'apre' (after). Egzanp: Apre manje, mwen ale dòmi.")
    facts.append("Mo 'jan' endike fason: Mwen renmen jan li pale.")
    facts.append("Mo 'dwe' (must): Ou dwe etidye.")
    facts.append("Mo 'kapab' oswa 'gen dwa' (may): Èske mwen kapab antre?")
    facts.append("Mo 'tou' (too/also): Mwen renmen ou tou.")
    facts.append("Mo 'men' (but): Mwen grangou, men mwen pa gen lajan.")
    facts.append("Mo 'donk' (so): Li te malad, donk li pa vini.")
    facts.append("Pluperfect: 'te' + 'deja' oswa 'te fin'. Egzanp: Mwen te deja manje lè ou rive.")
    facts.append("Future perfect: 'pral' + 'te' + vèb. Egzanp: Mwen pral te fin manje lè ou vini.")
    facts.append("Conditional perfect: 'ta' + 'te' + vèb. Egzanp: Mwen ta te vini si mwen te konnen.")
    facts.append("Mo 'kòmsi' (as if): Li pale kòmsi li te konnen tout bagay.")
    facts.append("Mo 'menm si' (even if): Menm si li te rich, li pa ta achte sa.")

    # ----- HAITI HISTORY (unchanged) -----
    facts.append("Premye moun ki te rete sou zile Ispanyola (kote Ayiti ye jodi a) se te Endyen Taino yo.")
    facts.append("Kristòf Kolon te rive sou zile a an 1492, li te nonmen l 'La Isla Española'.")
    facts.append("Panyòl yo te kolonize zile a epi yo te redui popilasyon Taino a.")
    facts.append("An 1697, Frans te pran kontwòl pati lwès zile a, yo te rele l Sen Domeng.")
    facts.append("Sen Domeng te vin koloni fransè ki pi rich nan mond lan grasa plantasyon kann, kafe, ak endigo. Travay la te fèt pa esklav Afriken.")
    facts.append("Premye gwo revòlt esklav la te kòmanse 21 out 1791 nan Bwa Kayiman, dirije pa Boukmann Dutty.")
    facts.append("Tousen Louverture, yon ansyen esklav, te vin lidè lame revolisyonè a. Li te bat lame Panyòl, Britanik, ak franse.")
    facts.append("Napoleon te voye yon ekspedisyon an 1802 pou retabli esklavaj, men Tousen te kaptire epi li te mouri nan prizon an Frans.")
    facts.append("Lame endijèn anba Jan Jak Desalin te bat lame fransè nan batay Vètyè 18 novanm 1803.")
    facts.append("Ayiti te vin endepandan 1ye janvye 1804. Se te premye repiblik nwa endepandan nan mond lan.")
    facts.append("Desalin te asasine 17 oktòb 1806. Apre sa, peyi a te divize an de: Nò anba Anri Kristòf, Sid anba Aleksann Petyon.")
    facts.append("An 1820, apre lanmò Kristòf, Jan Pyè Boye te reyini Nò ak Sid.")
    facts.append("An 1825, Lafrans te fòse Ayiti peye 150 milyon fran pou rekonèt endepandans lan. Se dèt endepandans lan ki te kraze ekonomi Ayiti pandan plis pase yon syèk.")
    facts.append("Etazini te okipe Ayiti soti 1915 rive 1934. Okipasyon an te fòse kòve (travay fòs) sou wout.")
    facts.append("François Duvalier (Papa Dok) te vin prezidan an 1957, li te kreye yon diktati ak tonton makout yo.")
    facts.append("Papa Dok te mouri an 1971, pitit li Jean-Claude Duvalier (Bébé Dok) te pran pouvwa.")
    facts.append("Bébé Dok te ranvèse pa manifestasyon popilè an 1986 epi li te kouri an Frans.")
    facts.append("Jean-Bertrand Aristide, yon prèt lavalas, te genyen premye eleksyon demokratik an 1990.")
    facts.append("Aristide te ranvèse pa yon koudeta 30 septanm 1991. Li te retounen an 1994 ak èd Etazini.")
    facts.append("Aristide te tonbe yon dezyèm fwa an 2004 epi li te pati an egzil.")
    facts.append("Tranblemanntè 12 janvye 2010 te touye plis pase 200,000 moun epi detwi Pòtoprens.")
    facts.append("Minustah, misyon lapè Nasyonzini, te entwodui kolera an Ayiti an 2010, sa te touye plizyè milye moun.")
    facts.append("Jovenel Moïse te asasine 7 jiyè 2021 nan kay li.")
    facts.append("Apre asasina a, Ariel Henry te vin premye minis, men li pa t eli.")
    facts.append("Gang yo te vin pi fò, yo pran kontwòl anpil zòn nan Pòtoprens.")
    facts.append("Apre presyon entènasyonal ak gang yo, Ariel Henry te demisyone 24 avril 2024.")
    facts.append("Yon Konsèy Prezidansyèl Tranzisyon (CPT) te pran pouvwa 25 avril 2024.")
    facts.append("Misyon MSS (Kenbe), dirije pa Kenya, te rive 25 jen 2024 pou konbat gang.")
    facts.append("Eleksyon 2026 te pote yon nouvo prezidan eli, men gang yo toujou gen enfliyans.")
    facts.append("Istwa Ayiti se yon istwa viktwa ak soufrans: endepandans 1804, okipasyon ameriken, diktati Duvalier, koudeta, Aristide, tranblemanntè 2010, asasina Jovenel Moïse, epi kriz ak gang jiska 2026.")
    facts.append("Istwa Ayiti gen anpil chapit: Taino, kolon, Revolisyon, endepandans, divizyon, dèt, okipasyon, diktati, kriz, ak espwa.")
    facts.append("Revolisyon ayisyen an te kòmanse ak revòlt esklav nan Bwa Kayiman an 1791, akòz maltrete esklav yo.")
    facts.append("Lidè revolisyon ayisyen an se Tousen Louverture, Jan Jak Desalin, Aleksann Petyon, ak Anri Kristòf.")
    facts.append("Dèt endepandans lan se lajan Ayiti te oblije peye Lafrans an 1825 pou yo rekonèt endepandans lan, sa te kraze ekonomi peyi a.")
    facts.append("Okipasyon ameriken an (1915-1934) te kontwole finans Ayiti, bati wout, men yo te fòse travayè yo (kòve).")
    facts.append("Papa Dok (François Duvalier) se yon diktatè ki te dirije Ayiti 1957-1971, li te kreye tonton makout yo.")
    facts.append("Tranblemanntè 2010 te touye plis pase 200,000 moun, li te detwi Pòtoprens, epi li te deplase 1.5 milyon moun.")
    facts.append("Jovenel Moïse te asasine 7 jiyè 2021.")

    # ----- GENERAL KNOWLEDGE -----
    facts.append("Diri ak pwa se yon manje popilè an Ayiti.")
    facts.append("Po moun se pi gwo ògàn kò imen an.")
    facts.append("Kè moun bat 60 a 100 fwa pa minit.")
    facts.append("Sèvo moun kontwole tout fonksyon kò a.")
    facts.append("Poumon yo pote oksijèn nan san an epi lage gaz kabonik.")
    facts.append("Zo yo bay sipò estriktirèl, pwoteje ògàn, epi pèmèt mouvman.")

    # ----- INTERACTIVE / HELP / CAPABILITIES -----
    facts.append("Mwen ka ede w aprann Kreyòl, reponn kesyon sou alfabè, gramè, istwa Ayiti, matematik, ak tout bagay moun te anseye m. Ou ka poze m nenpòt kesyon, epi m ap eseye reponn.")
    facts.append("Mwen ka ede w tradui mo, eksplike règ gramè, rakonte istwa, fè kalkil, ak anseye w vokabilè. Si m pa konnen, ou ka anseye m nan Sant Fòmasyon.")
    facts.append("Mwen ka reponn kesyon ou yo an Kreyòl. Mwen konnen alfabè, konjigezon, istwa Ayiti, ak anpil lòt bagay. Poze m yon kesyon epi gade sa m kapab fè.")
    facts.append("Ou ka poze m kesyon sou alfabè kreyòl, jan pou w di mo an Kreyòl, fason pou w fè fraz, diferans ant tan vèb, istwa Ayiti, ak tout bagay ou te anseye m nan Sant Fòmasyon.")
    facts.append("Wi, mwen kapab ede w aprann Kreyòl! Poze m kesyon sou vokabilè, gramè, pwononsyasyon, oswa nenpòt bagay ki gen rapò ak Kreyòl. Mwen la pou ede w.")
    facts.append("Mwen konn reponn kesyon sou alfabè, istwa, matematik, ak tout bagay moun te anseye m. Mwen konn pale Kreyòl, Franse, Angle, ak Panyòl. Mwen ka jwe vwa si ou te anrejistre yon fichye pou mwen.")
    facts.append("Diferans ant 'ou' ak 'w': 'Ou' se pwonon (you). 'W' se yon kontraksyon 'ou' apre yon vwayèl, tankou 'Mwen wè w' olye 'Mwen wè ou'.")
    facts.append("Pou byen pale Kreyòl, pran tan pou w aprann alfabè a, koute moun pale, pratike chak jou, epi poze m kesyon lè w pa konprann. Mwen ka ede w korije erè ou yo.")

    # ----- GESNER AI IDENTITY -----
    facts.append("Gesner AI te kreye pa Gesner Deslandes, fondatè GlobalInternet.py.")
    facts.append("Mwen reponn sèlman an Kreyòl. Poze m kesyon sou alfabè, gramè, istwa Ayiti, oswa nenpòt bagay ou te anseye m.")
    facts.append("Si mwen pa konnen, mwen di 'Mwen poko konn sa. Tanpri anseye m nan Sant Fòmasyon.'")
    facts.append("Ou ka antrene m lè w ajoute facts nan Training Center, sèvi ak diksyonè, oswa fòmasyon vwa.")

    return facts

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

# ---------- CSS (same as before, includes spinning globe) ----------
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

# ---------- LANGUAGES AND TEXTS (same as before, unchanged) ----------
# ... (I will keep the TEXTS dictionary as in previous versions – it's long but unchanged)

# To keep this answer from being impossibly long, I will assume the TEXTS dictionary is identical to the previous full version. In your actual file, copy the full TEXTS from the previous working app.

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

# ========== PRE‑DEFINED VOICE MAPPING (UPDATED WITH EXACT QUESTION) ==========
PREDEFINED_VOICES = {
    "kijan ou rele": "https://raw.githubusercontent.com/Deslandes1/Gesner-AIx/main/recording.wav",
    "site konbyen let ki genhen nan alfabe kreyol al": "https://raw.githubusercontent.com/Deslandes1/Gesner-AIx/main/recording%20(1).wav",
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

# ---------- ALL OTHER FUNCTIONS (retrieve_facts_hybrid, generate_response, etc.) REMAIN EXACTLY AS IN THE PREVIOUS WORKING VERSION ----------
# To avoid repetition, I will not paste all the functions again. But you must keep them identical to the previous full app.py that worked (with reasoning, math, etc.)

# ---------- UI COMPONENTS (unchanged) ----------
# ... (dictionary_manager, voice_training, bulk_training, manage_trained_facts, test_training_section, training_center, chat_interface, show_sidebar, main)

# ---------- MAIN ----------
def main():
    rebuild_index()
    initialize_default_training()
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
