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

# ---------- DEFAULT TRAINING (FULL, INCLUDING NEW INTERACTIVE FACTS) ----------
def get_default_training_facts():
    facts = []

    # ----- ALPHABET (count and list) -----
    facts.append("Alfabè kreyòl la gen 32 let: A, B, C, CH, D, E, È, F, G, H, I, J, K, L, M, N, NG, O, Ò, OU, P, R, S, T, UI, V, W, Y, Z.")
    facts.append("Lis let alfabè kreyòl la se: A, B, C, CH, D, E, È, F, G, H, I, J, K, L, M, N, NG, O, Ò, OU, P, R, S, T, UI, V, W, Y, Z.")
    facts.append("Konbyen let nan alfabè kreyòl? 32 let.")
    facts.append("Site let nan alfabe kreyol la? A, B, C, CH, D, E, È, F, G, H, I, J, K, L, M, N, NG, O, Ò, OU, P, R, S, T, UI, V, W, Y, Z.")
    facts.append("Premye let nan alfabè kreyòl la se A, dènye let la se Z.")
    facts.append("Let CH pwononse tankou 'sh' nan angle.")
    facts.append("Let È pwononse tankou 'e' nan franse, let Ò pwononse tankou 'o' louvri.")
    facts.append("OU pwononse tankou 'ou' nan franse, UI pwononse tankou 'wi'.")
    facts.append("NG pwononse tankou 'ng' nan 'sitting' an angle.")

    # ----- BEGINNER PHRASES (essential) -----
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

    # ----- INTERMEDIATE / ADVANCED GRAMMAR -----
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

    # ----- HAITI HISTORY (extensive) -----
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
    facts.append("Rakontem istwa Ayiti. Istwa Ayiti se yon istwa viktwa ak soufrans. Li te kòmanse ak Endyen Taino, answit koloni fransè Sen Domeng, revolisyon esklav la, endepandans an 1804, premye repiblik nwa nan mond lan, okipasyon ameriken, diktati Duvalier, koudeta, Aristide, tranblemanntè 2010, asasina Jovenel Moïse, epi kriz ak gang jiska 2026.")
    facts.append("Rakonte m sou istwa Ayiti. Ayiti te gen revolisyon esklav la ki te kòmanse nan Bwa Kayiman an 1791, apre sa endepandans an 1804, okipasyon ameriken, diktati Duvalier, koudeta kont Aristide, tranblemanntè 2010, asasina Jovenel Moïse, ak kriz ak gang.")
    facts.append("Esplike m istwa Ayiti an brèf. Istwa Ayiti se revolisyon, endepandans, dèt, okipasyon, diktati, kriz, ak espwa. Li se premye repiblik nwa endepandan.")
    facts.append("Kijan Ayiti te vin endepandan? Apre revolisyon esklav la ak batay Vètyè an 1803, Jan Jak Desalin te pwoklame endepandans 1ye janvye 1804.")
    facts.append("Kisa ki te kòz revolisyon ayisyen an? Esklavaj mechan ak maltrete esklav yo te lakòz revòlt la nan Bwa Kayiman an 1791.")
    facts.append("Kiyès ki te lidè revolisyon ayisyen an? Tousen Louverture, Jan Jak Desalin, Aleksann Petyon, ak Anri Kristòf se kèk nan lidè yo.")
    facts.append("Kisa dèt endepandans lan ye? Se lajan Ayiti te oblije peye Lafrans an 1825 pou yo rekonèt endepandans lan, sa te kraze ekonomi peyi a.")
    facts.append("Kisa okipasyon ameriken an te fè? Ameriken yo te okipe Ayiti 1915-1934, yo te kontwole finans yo, bati wout, men yo te fòse travayè yo (kòve).")
    facts.append("Kiyès ki te Papa Dok? François Duvalier, yon diktatè ki te dirije Ayiti 1957-1971, li te kreye tonton makout yo.")
    facts.append("Kisa tranblemanntè 2010 te fè? Li te touye plis pase 200,000 moun, li te detwi Pòtoprens, epi li te deplase 1.5 milyon moun.")
    facts.append("Ki dat Jovenel Moïse te asasine? 7 jiyè 2021.")

    # ----- GENERAL KNOWLEDGE -----
    facts.append("Diri ak pwa se yon manje popilè an Ayiti.")
    facts.append("Po moun se pi gwo ògàn kò imen an.")
    facts.append("Kè moun bat 60 a 100 fwa pa minit.")
    facts.append("Sèvo moun kontwole tout fonksyon kò a.")
    facts.append("Poumon yo pote oksijèn nan san an epi lage gaz kabonik.")
    facts.append("Zo yo bay sipò estriktirèl, pwoteje ògàn, epi pèmèt mouvman.")

    # ========== NEW: INTERACTIVE / HELP / CAPABILITIES ==========
    facts.append("Kisa ou ka ede m fè? Mwen ka ede w aprann Kreyòl, reponn kesyon sou alfabè, gramè, istwa Ayiti, matematik, ak tout bagay moun te anseye m. Ou ka poze m nenpòt kesyon, epi m ap eseye reponn.")
    facts.append("Kisa ou kapab fè pou mwen? Mwen ka ede w tradui mo, eksplike règ gramè, rakonte istwa, fè kalkil, ak anseye w vokabilè. Si m pa konnen, ou ka anseye m nan Sant Fòmasyon.")
    facts.append("Kijan ou ka ede m? Mwen ka reponn kesyon ou yo an Kreyòl. Mwen konnen alfabè, konjigezon, istwa Ayiti, ak anpil lòt bagay. Poze m yon kesyon epi gade sa m kapab fè.")
    facts.append("Ki kalite kesyon mwen ka poze w? Ou ka poze m kesyon sou alfabè kreyòl, jan pou w di mo an Kreyòl, fason pou w fè fraz, diferans ant tan vèb, istwa Ayiti, ak tout bagay ou te anseye m nan Sant Fòmasyon.")
    facts.append("Èske ou ka ede m aprann Kreyòl? Wi, mwen kapab! Poze m kesyon sou vokabilè, gramè, pwononsyasyon, oswa nenpòt bagay ki gen rapò ak Kreyòl. Mwen la pou ede w.")
    facts.append("Kisa w konn fè? Mwen konn reponn kesyon sou alfabè, istwa, matematik, ak tout bagay moun te anseye m. Mwen konn pale Kreyòl, Franse, Angle, ak Panyòl. Mwen ka jwe vwa si ou te anrejistre yon fichye pou mwen.")
    facts.append("Kisa diferans ant 'ou' ak 'w'? 'Ou' se pwonon (you). 'W' se yon kontraksyon 'ou' apre yon vwayèl, tankou 'Mwen wè w' olye 'Mwen wè ou'.")
    facts.append("Kisa mwen dwe fè pou m byen pale Kreyòl? Pran tan pou w aprann alfabè a, koute moun pale, pratike chak jou, epi poze m kesyon lè w pa konprann. Mwen ka ede w korije erè ou yo.")

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
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- LANGUAGES AND TEXTS (UI only) ----------
LANGUAGES = {
    "English": "en",
    "Français": "fr",
    "Kreyòl Ayisyen": "ht",
    "Español": "es"
}

TEXTS = {
    "en": {
        "app_title": "💬 Gesner AI Chat",
        "chat_input": "Ask me anything in Haitian Creole...",
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
        "training_section": "🔧 Training Center",
        "test_training_section": "🧪 Test Training",
        "test_query_label": "Enter a query to test fact retrieval",
        "test_button": "Test Retrieval",
        "closest_fact": "Closest fact:",
        "no_fact": "No matching fact found.",
        "play_voice": "Play Voice",
        "voice_exists": "✅ Voice file attached",
        "voice_missing": "❌ No voice attached to this fact.",
        "attach_voice_here": "Attach voice to this fact now:",
        "upload_voice_label": "Upload voice (WAV/MP3)",
        "attach_voice_button": "Attach Voice to this Fact",
        "voice_attached_success": "Voice attached successfully!",
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
        "voice_success": "✅ Voice and text stored! Fact: '{fact}'",
        "test_this_fact": "🎧 Test this fact now",
        "record_btn": "🔴 Record",
        "stop_btn": "⏹️ Stop",
        "download_btn": "💾 Download",
        "bulk_training_title": "🚀 Bulk Training (Fast Import)",
        "bulk_csv_label": "Upload CSV file (columns: question, answer OR one column 'fact')",
        "bulk_json_label": "Upload JSON file (array of strings)",
        "bulk_text_label": "Paste text (one fact per line)",
        "bulk_import_button": "Import All Facts",
        "manage_facts": "📚 Manage Trained Facts",
        "edit_save": "✏️ Save",
        "delete": "🗑️ Delete",
        "test_voice_btn": "🔊 Test Voice",
        "footer": "© GlobalInternet.py – Gesner AI | Chat works only in Kreyòl."
    },
    "fr": {
        "app_title": "💬 Gesner IA Chat",
        "chat_input": "Posez votre question en créole haïtien...",
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
        "training_section": "🔧 Centre d'entraînement",
        "test_training_section": "🧪 Tester l'entraînement",
        "test_query_label": "Entrez une requête pour tester la recherche",
        "test_button": "Tester",
        "closest_fact": "Fait le plus proche :",
        "no_fact": "Aucun fait correspondant.",
        "play_voice": "Écouter la voix",
        "voice_exists": "✅ Voix attachée",
        "voice_missing": "❌ Aucune voix attachée à ce fait.",
        "attach_voice_here": "Attacher une voix à ce fait maintenant :",
        "upload_voice_label": "Télécharger la voix (WAV/MP3)",
        "attach_voice_button": "Attacher la voix à ce fait",
        "voice_attached_success": "Voix attachée avec succès !",
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
        "voice_success": "✅ Voix et texte enregistrés ! Fait : '{fact}'",
        "test_this_fact": "🎧 Tester ce fait maintenant",
        "record_btn": "🔴 Enregistrer",
        "stop_btn": "⏹️ Arrêter",
        "download_btn": "💾 Télécharger",
        "bulk_training_title": "🚀 Entraînement groupé (import rapide)",
        "bulk_csv_label": "Télécharger fichier CSV (colonnes: question, réponse OU une colonne 'fact')",
        "bulk_json_label": "Télécharger fichier JSON (tableau de chaînes)",
        "bulk_text_label": "Coller du texte (une ligne = un fait)",
        "bulk_import_button": "Importer tous les faits",
        "manage_facts": "📚 Gérer les faits appris",
        "edit_save": "✏️ Enregistrer",
        "delete": "🗑️ Supprimer",
        "test_voice_btn": "🔊 Tester la voix",
        "footer": "© GlobalInternet.py – Gesner IA | Chat fonctionne uniquement en créole haïtien."
    },
    "ht": {
        "app_title": "💬 Gesner AI Chat",
        "chat_input": "Poze m yon kesyon an Kreyòl...",
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
        "training_section": "🔧 Sant Fòmasyon",
        "test_training_section": "🧪 Tès Fòmasyon",
        "test_query_label": "Antre yon kesyon pou teste rekiperasyon",
        "test_button": "Tès",
        "closest_fact": "Fè ki pi pre:",
        "no_fact": "Pa gen fè ki matche.",
        "play_voice": "Jwe vwa",
        "voice_exists": "✅ Vwa atache",
        "voice_missing": "❌ Pa gen vwa atache ak fè sa.",
        "attach_voice_here": "Atache vwa ak fè sa kounye a:",
        "upload_voice_label": "Chaje vwa (WAV/MP3)",
        "attach_voice_button": "Atache vwa ak fè sa",
        "voice_attached_success": "Vwa atache avèk siksè!",
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
        "voice_success": "✅ Vwa ak tèks sove! Fè: '{fact}'",
        "test_this_fact": "🎧 Tès fè sa kounye a",
        "record_btn": "🔴 Anrejistre",
        "stop_btn": "⏹️ Sispann",
        "download_btn": "💾 Telechaje",
        "bulk_training_title": "🚀 Antreman an mas (enpòtasyon rapid)",
        "bulk_csv_label": "Chaje fichye CSV (kolòn: kesyon, repons OSWA yon sèl kolòn 'fact')",
        "bulk_json_label": "Chaje fichye JSON (tablo chèn karaktè)",
        "bulk_text_label": "Kole tèks (yon liy = yon reyalite)",
        "bulk_import_button": "Enpòte tout reyalite yo",
        "manage_facts": "📚 Jere Reyalite Aprann",
        "edit_save": "✏️ Sove",
        "delete": "🗑️ Efase",
        "test_voice_btn": "🔊 Tès Vwa",
        "footer": "© GlobalInternet.py – Gesner AI | Chat la sèlman an Kreyòl."
    },
    "es": {
        "app_title": "💬 Gesner AI Chat",
        "chat_input": "Haz tu pregunta en criollo haitiano...",
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
        "training_section": "🔧 Centro de Entrenamiento",
        "test_training_section": "🧪 Probar Entrenamiento",
        "test_query_label": "Ingrese una consulta para probar la recuperación",
        "test_button": "Probar",
        "closest_fact": "Hecho más cercano:",
        "no_fact": "No se encontró ningún hecho.",
        "play_voice": "Reproducir voz",
        "voice_exists": "✅ Voz adjunta",
        "voice_missing": "❌ No hay voz adjunta a este hecho.",
        "attach_voice_here": "Adjuntar voz a este hecho ahora:",
        "upload_voice_label": "Subir voz (WAV/MP3)",
        "attach_voice_button": "Adjuntar voz a este hecho",
        "voice_attached_success": "¡Voz adjuntada con éxito!",
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
        "voice_success": "✅ ¡Voz y texto guardados! Hecho: '{fact}'",
        "test_this_fact": "🎧 Probar este hecho ahora",
        "record_btn": "🔴 Grabar",
        "stop_btn": "⏹️ Detener",
        "download_btn": "💾 Descargar",
        "bulk_training_title": "🚀 Entrenamiento masivo (importación rápida)",
        "bulk_csv_label": "Subir archivo CSV (columnas: pregunta, respuesta O una columna 'fact')",
        "bulk_json_label": "Subir archivo JSON (arreglo de cadenas)",
        "bulk_text_label": "Pegar texto (una línea = un hecho)",
        "bulk_import_button": "Importar todos los hechos",
        "manage_facts": "📚 Gestionar hechos aprendidos",
        "edit_save": "✏️ Guardar",
        "delete": "🗑️ Eliminar",
        "test_voice_btn": "🔊 Probar voz",
        "footer": "© GlobalInternet.py – Gesner AI | El chat solo funciona en criollo haitiano."
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

# ========== REASONING FUNCTION ==========
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

def generate_response(user_input):
    with st.spinner("🧠 Gesner AI ap reflechi... (thinking...)"):
        time.sleep(0.8)
        direct = direct_keyword_answer(user_input)
        if direct:
            return direct, False
        facts = retrieve_facts_hybrid(user_input, k=5)
        if facts:
            reasoned = reason_answer(user_input, facts)
            return reasoned, False
        logic = reason_about_question(user_input)
        if logic:
            return logic, False
    return "Mwen poko konn sa. Tanpri anseye m nan Sant Fòmasyon.", True

def play_voice_button(text, button_label="🔊", key_suffix=""):
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

# ---------- UI COMPONENTS ----------
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
                fact = f"{w} means {m}"
                add_to_training(fact)
                save_dictionaries()
                st.success(t['trained_entry_success'].format(word=w, meaning=m))
                st.rerun()
        for word, meaning in list(dict_data.items()):
            col_a, col_b = st.columns([3,1])
            with col_a:
                st.text(f"{word}: {meaning}")
            with col_b:
                if st.button(t['dict_delete'], key=f"del_{lang_code}_{word}"):
                    del dict_data[word]
                    save_dictionaries()
                    st.rerun()
    with col1:
        display_dict("ht", t['dict_ht'], st.session_state.dictionaries["ht"])
    with col2:
        display_dict("fr", t['dict_fr'], st.session_state.dictionaries["fr"])
    with col3:
        display_dict("en", t['dict_en'], st.session_state.dictionaries["en"])

def voice_training(t):
    st.markdown(f"## {t['voice_training_title']}")
    st.info("🎙️ Record your voice, download it, then upload below.")
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
    uploaded_file = st.file_uploader(t['voice_upload'], type=["wav","mp3"], key="voice_upload")
    transcript = st.text_area(t['voice_transcribed_text'], key="voice_transcript")
    if uploaded_file and transcript.strip():
        if st.button(t['voice_train'], use_container_width=True):
            audio_bytes = uploaded_file.read()
            fact_text = transcript.strip()
            save_voice_for_text(fact_text, audio_bytes)
            add_to_training(fact_text)
            st.success(t['voice_success'].format(fact=fact_text))
            st.markdown("---")
            st.markdown(f"### {t['test_this_fact']}")
            st.write(f"**Fact:** {fact_text}")
            btn_html = play_voice_button(fact_text, t['play_voice'], "after_train")
            if btn_html:
                st.components.v1.html(btn_html, height=50)

def bulk_training(t):
    st.markdown(f"## {t['bulk_training_title']}")
    st.info("Import many facts at once.")
    def import_facts(facts):
        count = 0
        for fact in facts:
            if fact.strip():
                if add_to_training(fact.strip()):
                    count += 1
        st.success(f"Imported {count} facts.")
    csv_file = st.file_uploader(t['bulk_csv_label'], type=["csv"], key="bulk_csv")
    if csv_file:
        try:
            content = csv_file.read().decode('utf-8')
            reader = csv.DictReader(io.StringIO(content))
            facts = []
            for row in reader:
                if 'question' in row and 'answer' in row:
                    facts.append(f"{row['question']} → {row['answer']}")
                elif 'fact' in row:
                    facts.append(row['fact'])
                else:
                    first_key = list(row.keys())[0]
                    facts.append(row[first_key])
            if facts:
                st.info(f"Found {len(facts)} facts in CSV.")
                if st.button(t['bulk_import_button'], key="import_csv"):
                    import_facts(facts)
            else:
                st.warning("No valid facts found in CSV.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    json_file = st.file_uploader(t['bulk_json_label'], type=["json"], key="bulk_json")
    if json_file:
        try:
            data = json.load(json_file)
            if isinstance(data, list):
                facts = [str(item) for item in data]
                st.info(f"Found {len(facts)} facts in JSON.")
                if st.button(t['bulk_import_button'], key="import_json"):
                    import_facts(facts)
            else:
                st.warning("JSON must be an array of strings.")
        except Exception as e:
            st.error(f"Error reading JSON: {e}")
    text_facts = st.text_area(t['bulk_text_label'], height=150, key="bulk_text")
    if text_facts.strip():
        lines = [line.strip() for line in text_facts.split('\n') if line.strip()]
        st.info(f"Found {len(lines)} facts in text.")
        if st.button(t['bulk_import_button'], key="import_text"):
            import_facts(lines)

def manage_trained_facts(t):
    st.markdown(f"## {t['manage_facts']}")
    if not st.session_state.training_data:
        st.info("No facts trained yet. Use dictionaries, bulk import, or voice training to add facts.")
        return
    for idx, item in enumerate(st.session_state.training_data):
        original = item["text"]
        with st.expander(f"Fact #{idx+1}: {original[:60]}..."):
            character_picker(f"edit_{idx}", "Insert Kreyòl characters for this fact:")
            new_text = st.text_area("Edit text", value=original, key=f"edit_text_{idx}", height=100)
            col1, col2, col3 = st.columns([2,2,1])
            with col1:
                if st.button(t['edit_save'], key=f"save_{idx}"):
                    if new_text.strip() and new_text != original:
                        update_training_item(idx, new_text)
                        st.success("Updated")
                        st.rerun()
                    elif not new_text.strip():
                        st.warning("Text cannot be empty.")
                    else:
                        st.info("No changes made.")
            with col2:
                if st.button(t['delete'], key=f"delete_{idx}"):
                    delete_training_item(idx)
                    st.success("Deleted")
                    st.rerun()
            with col3:
                btn_html = play_voice_button(original, t['test_voice_btn'], f"test_{idx}")
                if btn_html:
                    st.components.v1.html(btn_html, height=50)
            st.markdown("---")
            st.markdown(f"### {t['upload_voice_label']}")
            uploaded_voice = st.file_uploader("Choose WAV or MP3 file", type=["wav","mp3"], key=f"attach_voice_{idx}")
            if st.button(t['attach_voice_button'], key=f"attach_btn_{idx}"):
                if uploaded_voice:
                    audio_bytes = uploaded_voice.read()
                    save_voice_for_text(original, audio_bytes)
                    st.success(t['voice_attached_success'])
                    st.rerun()
                else:
                    st.error("Please select a voice file first.")
            if get_voice_for_text(original) is not None:
                st.markdown("✅ **Voice is now attached!**")
                test_btn = play_voice_button(original, "🔊 Test attached voice", f"after_attach_{idx}")
                if test_btn:
                    st.components.v1.html(test_btn, height=50)

def test_training_section(t):
    st.markdown(f"## {t['test_training_section']}")
    st.info("Enter a query to see which fact the AI would retrieve, and play its voice.")
    test_query = st.text_input(t['test_query_label'], key="test_query_input")
    if st.button(t['test_button'], key="run_test"):
        if test_query.strip():
            facts = retrieve_facts_hybrid(test_query, k=1)
            if facts:
                best_fact = facts[0]
                st.success(f"{t['closest_fact']} \"{best_fact}\"")
                voice_bytes = get_voice_for_text(best_fact)
                if voice_bytes:
                    st.markdown(f"✅ {t['voice_exists']}")
                    btn_html = play_voice_button(best_fact, t['play_voice'], "test_retrieval")
                    if btn_html:
                        st.components.v1.html(btn_html, height=50)
                else:
                    st.warning(t['voice_missing'])
                    st.markdown(f"### {t['attach_voice_here']}")
                    attach_file = st.file_uploader(t['upload_voice_label'], type=["wav","mp3"], key="test_attach_voice")
                    if attach_file and st.button(t['attach_voice_button'], key="test_attach_btn"):
                        audio_bytes = attach_file.read()
                        save_voice_for_text(best_fact, audio_bytes)
                        st.success(t['voice_attached_success'])
                        st.rerun()
            else:
                st.warning(t['no_fact'])
        else:
            st.warning("Please enter a query.")

def training_center(t):
    st.markdown(f"<h1 style='text-align:center;'>🔧 {t['training_section']}</h1>", unsafe_allow_html=True)
    dictionary_manager(t)
    st.markdown("---")
    voice_training(t)
    st.markdown("---")
    bulk_training(t)
    st.markdown("---")
    test_training_section(t)
    st.markdown("---")
    manage_trained_facts(t)

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
                btn_html = play_voice_button(msg["content"], "🔊", f"chat_{idx}")
                if btn_html:
                    st.components.v1.html(btn_html, height=50)
                else:
                    if msg.get("is_fallback", False) and st.session_state.ui_language == "fr":
                        fallback_html = play_fallback_audio_french()
                        st.components.v1.html(fallback_html, height=50)
    user_input = st.text_input(t['chat_input'], key="chat_input")
    if st.button(t['send'], use_container_width=True, key="send_btn"):
        if user_input.strip():
            answer, is_fallback = generate_response(user_input)
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_history.append({"role": "assistant", "content": answer, "is_fallback": is_fallback})
            st.rerun()
    if st.button(t['clear'], use_container_width=True, key="clear_btn"):
        st.session_state.conversation_history = []
        st.rerun()

def show_sidebar():
    lang_names = list(LANGUAGES.keys())
    selected_lang_name = st.sidebar.selectbox("🌐 Language / Lang", lang_names, key="main_lang_selector")
    selected_lang_code = LANGUAGES[selected_lang_name]
    st.session_state.ui_language = selected_lang_code
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
