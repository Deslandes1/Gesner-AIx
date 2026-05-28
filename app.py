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
    "Ti Malice se yon lojisyèl edikatif ki anseye timoun yo Kreyòl Ayisyen atravè jwèt ak istwa.",
    "Ti Malice gen 12 chapit. Chapit 1: Alfabè, Chapit 2: Nonm, Chapit 3: Koulè, Chapit 4: Fanmi, Chapit 5: Manje, Chapit 6: Bèt, Chapit 7: Vèb, Chapit 8: Tan, Chapit 9: Fraz senp, Chapit 10: Konvèsasyon, Chapit 11: Pwovèb, Chapit 12: Istwa.",
]

# ========== DEFAULT TRAINING FACTS ==========
def get_default_training_facts():
    return HAITIAN_KNOWLEDGE_FACTS

def initialize_default_training():
    if not st.session_state.training_data:
        default_facts = get_default_training_facts()
        for fact in default_facts:
            if fact.strip():
                embedding = st.session_state.embedding_model.encode([fact])[0]
                st.session_state.training_data.append({"text": fact, "embedding": embedding.tolist()})
        rebuild_index()
        save_training_data()
    else:
        existing = {item["text"] for item in st.session_state.training_data}
        added = 0
        for fact in HAITIAN_KNOWLEDGE_FACTS:
            if fact not in existing:
                embedding = st.session_state.embedding_model.encode([fact])[0]
                st.session_state.training_data.append({"text": fact, "embedding": embedding.tolist()})
                added += 1
        if added > 0:
            rebuild_index()
            save_training_data()
            st.session_state._facts_added = added

# ---------- GROK API (Online Search & Image Recognition) ----------
def get_grok_api_key():
    try:
        return st.secrets["GROK_API_KEY"]
    except:
        return None

def call_grok_api(prompt, system_prompt="You are Gesner AI, a helpful assistant that answers in Haitian Creole. Provide accurate, concise responses.", image_base64=None):
    api_key = get_grok_api_key()
    if not api_key:
        return None
    endpoint = st.secrets.get("GROK_API_ENDPOINT", "https://api.x.ai/v1/chat/completions")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    if image_base64:
        # Grok may not support vision; we'll send as text and ignore image for now.
        # In production, use a vision model.
        pass
    payload = {
        "model": "grok-1",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500
    }
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error(f"Grok API error: {e}")
    return None

def identify_image_with_grok(image_bytes, user_question=""):
    """Try to identify an image using Grok (fallback to text description)."""
    if user_question:
        prompt = f"The user uploaded an image and asked: '{user_question}'. Since I cannot see the image, please provide a general answer based on the question. If the question asks to identify something, say you cannot see the image."
    else:
        prompt = "The user uploaded an image. Since I cannot see images, please ask the user to describe the image or tell me what they want to know about it."
    response = call_grok_api(prompt)
    if response:
        return response
    return "Mwen pa ka wè imaj la. Tanpri dekri li pou mwen, oswa di m sa w vle konnen."

# ---------- COGNITIVE TRAINING ----------
def add_cognitive_example(input_text, output_format, description=""):
    example = {
        "input": input_text.strip(),
        "output": output_format.strip(),
        "description": description,
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.cognitive_examples.append(example)
    save_cognitive_examples()
    marker = f"[COGNITIVE] Input: {input_text} → Output: {output_format}"
    if not any(item["text"] == marker for item in st.session_state.training_data):
        embedding = st.session_state.embedding_model.encode([marker])[0]
        st.session_state.training_data.append({"text": marker, "embedding": embedding.tolist()})
        rebuild_index()
        save_training_data()

def find_cognitive_match(query):
    if not st.session_state.cognitive_examples:
        return None
    query_emb = st.session_state.embedding_model.encode([query])[0]
    best_score = -1
    best_example = None
    for ex in st.session_state.cognitive_examples:
        ex_emb = st.session_state.embedding_model.encode([ex["input"]])[0]
        sim = np.dot(query_emb, ex_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(ex_emb) + 1e-8)
        if sim > best_score:
            best_score = sim
            best_example = ex
    if best_score > 0.7:
        return best_example
    return None

def apply_cognitive_format(query, matched_example):
    output = matched_example["output"]
    output = output.replace("{query}", query)
    return output

# ========== CENTRAL CORE ANSWERS DICTIONARY ==========
CORE_ANSWERS = {
    "site konbyen let ki genhen nan alfabe kreyol la": "A, AN, B, CH, D, E, È, EN, F, G, H, I, J, K, L, M, N, NG, O, Ò, ON, OU, OUN, P, R, S, T, UI, V, W, Y, Z",
    "konbyen let ki genhen nan alfabe kreyol la": "Nan alfabe kreyol la genhen 32 let.",
    "konbyen let ki gehen nan alfabe kreyol la": "Nan alfabe kreyol la gen 32 let.",
    "kijan ou rele": "Non pa mwen se Gesner L'AI kreyate mwen se Gesner Deslandes nan Globalinternet.py.",
    "ki moun ki dekouvri ayiti": "Kristòf Kolon te dekouvri zile Ispanyola (kote Ayiti ye jodi a) nan 5 desanm 1492.",
    "kiyès ki te dekouvri ayiti": "Kristòf Kolon te dekouvri Ayiti an 1492.",
    "kisa bwa kayiman ye": "Bwa Kayiman se kote seremoni sekrè esklav yo te fèt 21 out 1791 pou lanse revolisyon esklav la.",
    "ki dat ayiti te vin endepandan": "Ayiti te vin endepandan 1ye janvye 1804.",
    "ki moun ki te papa dok": "François Duvalier (Papa Dok) te vin prezidan an 1957 e li te kreye yon diktati.",
    "kisa tranblemanntè 2010 te fè": "Tranblemanntè 12 janvye 2010 te touye plis pase 200,000 moun, li te detwi Pòtoprens.",
    "ki moun ki te tousen louverture": "Tousen Louverture se yon lidè revolisyon esklav la, li te pran kontwòl tout Sen Domeng.",
    "ki moun ki te jan jak desalin": "Jan Jak Desalin se papa endepandans Ayiti.",
    "ki moun ki te anri kristòf": "Anri Kristòf se yon wa nan Nò Ayiti, li te bati Sitadèl Laferyè.",
    "kisa kapital ayiti ye": "Pòtoprens se kapital Ayiti.",
    "ki kote ayiti ye": "Ayiti sitiye nan Karayib la, sou zile Ispanyola.",
    "kisa diri ak pwa ye": "Diri ak pwa se manje nasyonal Ayiti.",
    "kisa soup joumou ye": "Soup joumou se soup joumou ke Ayisyen manje 1ye janvye pou fete endepandans.",
    "kisa kanaval ayiti ye": "Kanaval Ayiti se yon gwo fèt ak parad, mizik, ak danse anvan Karèm.",
    "kisa konpa ye": "Konpa (compas) se yon mizik ak dans ki popilè an Ayiti.",
    "ki moun ki kreye gesner ai": "Gesner AI te kreye pa Gesner Deslandes, fondatè GlobalInternet.py.",
    "kijan ou di mwen renmen ou an kreyòl": "Mwen renmen ou.",
}

def get_core_answer(question):
    q = question.strip().lower()
    q = re.sub(r'\s+', ' ', q)
    if q in CORE_ANSWERS:
        return CORE_ANSWERS[q]
    for key, answer in CORE_ANSWERS.items():
        if key in q:
            return answer
    return None

# ---------- RETRIEVAL & RESPONSE GENERATION ----------
def retrieve_facts_hybrid(query, k=5):
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
    if "ti malice" in q_lower:
        if "kiyès" in q_lower or "who" in q_lower or "kreyatè" in q_lower:
            return "Ti Malice se yon lojisyèl edikatif ki fèt pa Gesner Deslandes pou anseye Kreyòl Ayisyen atravè jwèt ak istwa."
        if "chapit" in q_lower or "chapter" in q_lower:
            return "Ti Malice gen 12 chapit. Chapit 1: Alfabè, Chapit 2: Nonm, Chapit 3: Koulè, Chapit 4: Fanmi, Chapit 5: Manje, Chapit 6: Bèt, Chapit 7: Vèb, Chapit 8: Tan, Chapit 9: Fraz senp, Chapit 10: Konvèsasyon, Chapit 11: Pwovèb, Chapit 12: Istwa."
        if "telechaje" in q_lower or "download" in q_lower:
            return "Ou ka telechaje Ti Malice sou sitwèb globalinternet.py."
        return "Ti Malice se yon lojisyèl k ap anseye Kreyòl Ayisyen. Li gen 12 chapit ak egzèsis."
    if any(w in q_lower for w in ["beginner", "debutan", "debutant", "aprann kreyòl deba"]):
        return "Kou Kreyòl pou debitan (Beginner): Alfabè 32 lèt, pwonon (mwen, ou, li, nou, yo), vèb 'se' ak 'gen', salitasyon (Bonjou, Bonswa), nonm 1-10, koulè debaz."
    if any(w in q_lower for w in ["intermediate", "entèmedyè", "mwayen", "intermédiaire"]):
        return "Kou Kreyòl entèmedyè: Tan pase ak 'te', tan kap vini ak 'ap' oswa 'pral', nègasyon ak 'pa', pwopozisyon (nan, sou, anba), fraz konplèks ak 'ki', 'kote', 'poukisa'."
    if any(w in q_lower for w in ["advanced", "avanse", "avancé"]):
        return "Kou Kreyòl avansé: Pawòl konpoze, pwovèb popilè, tan ki konpoze, vwa pasif, sijonktif, literati kreyòl."
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
    priority_keywords = ["ayiti", "haiti", "kolon", "tousen", "desalin", "kristòf", "bwa kayiman", "endepandan", "pòtoprens", "kapital", "kanaval", "diri", "soup joumou"]
    prioritized = []
    for f in retrieved_facts:
        if any(kw in f.lower() for kw in priority_keywords):
            prioritized.append(f)
    if prioritized:
        return ". ".join(prioritized[:2])
    return retrieved_facts[0]

def generate_response(user_input, uploaded_image_bytes=None):
    if uploaded_image_bytes:
        return identify_image_with_grok(uploaded_image_bytes, user_input), False, False
    
    core_answer = get_core_answer(user_input)
    if core_answer:
        return core_answer, False, False
    
    direct = direct_keyword_answer(user_input)
    if direct:
        return direct, False, False
    
    math_result = reason_about_question(user_input)
    if math_result:
        return math_result, False, False
    
    cog_match = find_cognitive_match(user_input)
    if cog_match:
        return apply_cognitive_format(user_input, cog_match), False, False
    
    facts = retrieve_facts_hybrid(user_input, k=7)
    if facts:
        return reason_answer(user_input, facts), False, False
    
    grok_answer = call_grok_api(user_input)
    if grok_answer:
        return grok_answer, False, False
    
    return "Mwen poko konn sa. Tanpri anseye m nan Sant Fòmasyon oswa ajoute yon egzanp kognitif.", True, False

# ---------- HELPER FUNCTIONS FOR INDEX, VOICE, ETC. ----------
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

def get_predefined_voice_url(user_question):
    norm_q = re.sub(r'\s+', ' ', user_question.strip().lower())
    predefined = {
        "site konbyen let ki genhen nan alfabe kreyol la": "https://raw.githubusercontent.com/Deslandes1/Gesner-AIx/main/recording%20(1).wav",
        "konbyen let ki genhen nan alfabe kreyol la": "https://raw.githubusercontent.com/Deslandes1/Gesner-AI/main/recording%20(3).wav",
        "kijan ou rele": "https://raw.githubusercontent.com/Deslandes1/Gesner-AIx/main/recording%20(4).wav"
    }
    for key, url in predefined.items():
        if key in norm_q:
            return url
    return None

def show_audio_button(text, user_question, key_suffix):
    url = get_predefined_voice_url(user_question) if user_question else None
    if url:
        if st.button("🔊", key=f"audio_btn_{key_suffix}", help="Play audio"):
            st.session_state.play_audio = ("url", url)
            st.rerun()
        return
    audio_bytes = get_voice_for_text(text)
    if audio_bytes:
        if st.button("🔊", key=f"audio_btn_{key_suffix}", help="Play audio"):
            st.session_state.play_audio = ("bytes", audio_bytes, "audio/wav")
            st.rerun()
        return

def render_audio_player():
    if st.session_state.play_audio:
        audio_type = st.session_state.play_audio[0]
        if audio_type == "url":
            url = st.session_state.play_audio[1]
            st.audio(url, format="audio/wav")
        elif audio_type == "bytes":
            _, data, mime = st.session_state.play_audio
            st.audio(data, format=mime)
        st.session_state.play_audio = None

# ---------- UI COMPONENTS (Training Center, Dictionary, Voice Training) ----------
def dictionary_manager(t):
    st.subheader(t['dictionary'])
    lang = st.selectbox("Select language", list(LANGUAGES.keys()), key="dict_lang")
    lang_code = LANGUAGES[lang]
    word = st.text_input("Word / Phrase", key="dict_word")
    meaning = st.text_area("Meaning / Translation", key="dict_meaning")
    if st.button("Add / Update", key="dict_add"):
        if word and meaning:
            st.session_state.dictionaries[lang_code][word] = meaning
            save_dictionaries()
            st.success("Saved!")
            st.rerun()
    st.markdown("---")
    st.write("**Existing entries**")
    for w, m in st.session_state.dictionaries[lang_code].items():
        col1, col2 = st.columns([3,1])
        with col1:
            st.write(f"**{w}**: {m}")
        with col2:
            if st.button("Delete", key=f"del_{lang_code}_{w}"):
                del st.session_state.dictionaries[lang_code][w]
                save_dictionaries()
                st.rerun()

def voice_training(t):
    st.subheader(t['voice_training'])
    fact_text = st.text_area(t['fact_text'], key="voice_fact")
    uploaded_audio = st.file_uploader(t['upload_audio'], type=["wav", "mp3"], key="voice_upload")
    if uploaded_audio:
        audio_bytes = uploaded_audio.read()
        st.audio(audio_bytes, format="audio/wav")
        if st.button(t['save_voice'], key="save_voice_btn"):
            save_voice_for_text(fact_text, audio_bytes)
            st.success("Voice saved!")
    st.markdown("---")
    st.write("**Existing voice mappings**")
    for idx, item in enumerate(st.session_state.training_data):
        text = item["text"]
        if get_voice_for_text(text):
            col1, col2 = st.columns([3,1])
            with col1:
                st.write(text[:60] + "..." if len(text) > 60 else text)
            with col2:
                if st.button(t['play'], key=f"play_voice_{idx}"):
                    audio_bytes = get_voice_for_text(text)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/wav")

def bulk_training(t):
    st.subheader(t['bulk_training'])
    uploaded_file = st.file_uploader(t['upload_csv'], type=["csv"], key="bulk_csv")
    if uploaded_file:
        content = uploaded_file.getvalue().decode("utf-8")
        reader = csv.DictReader(io.StringIO(content))
        facts = [row.get("fact") or row.get("text") for row in reader]
        if facts:
            if st.button("Import facts", key="bulk_import"):
                count = 0
                for fact in facts:
                    if fact and fact.strip():
                        if add_to_training(fact.strip()):
                            count += 1
                st.success(f"Imported {count} facts.")
                st.rerun()

def manage_trained_facts(t):
    st.subheader(t['manage_facts'])
    for idx, item in enumerate(st.session_state.training_data):
        col1, col2, col3 = st.columns([4,1,1])
        with col1:
            if f"edit_{idx}" in st.session_state and st.session_state[f"edit_{idx}"]:
                new_text = st.text_area("Edit", value=item["text"], key=f"edit_text_{idx}")
                if st.button("Save", key=f"save_edit_{idx}"):
                    update_training_item(idx, new_text)
                    st.session_state[f"edit_{idx}"] = False
                    st.rerun()
            else:
                st.write(item["text"])
        with col2:
            if st.button(t['edit'], key=f"edit_btn_{idx}"):
                st.session_state[f"edit_{idx}"] = True
                st.rerun()
        with col3:
            if st.button(t['delete'], key=f"del_btn_{idx}"):
                delete_training_item(idx)
                st.rerun()

def test_training_section(t):
    st.subheader(t['test_training'])
    query = st.text_input("Test query", key="test_query")
    if st.button("Test", key="test_btn"):
        if query:
            facts = retrieve_facts_hybrid(query, k=3)
            if facts:
                st.write("**Retrieved facts:**")
                for f in facts:
                    st.write(f"- {f}")
            else:
                st.write("No relevant facts found.")

def cognitive_training_ui(t):
    st.subheader("🧠 Cognitive Training (Example‑based Learning)")
    st.info("Teach Gesner AI how to analyze sentences and output specific formats.")
    col1, col2 = st.columns(2)
    with col1:
        example_input = st.text_area("Example Input (Kreyòl)", height=100, key="cognitive_input")
        example_output = st.text_area("Desired Output", height=150, key="cognitive_output")
        description = st.text_input("Description (optional)", key="cognitive_desc")
        if st.button("💾 Save Cognitive Example", key="save_cognitive"):
            if example_input and example_output:
                add_cognitive_example(example_input, example_output, description)
                st.success("Example saved!")
                st.rerun()
    with col2:
        st.markdown("### Existing Cognitive Examples")
        if st.session_state.cognitive_examples:
            for idx, ex in enumerate(st.session_state.cognitive_examples):
                with st.expander(f"Example {idx+1}: {ex['input'][:50]}..."):
                    st.write(f"**Input:** {ex['input']}")
                    st.write(f"**Output:** {ex['output']}")
                    if ex['description']:
                        st.write(f"**Description:** {ex['description']}")
                    if st.button("Delete", key=f"del_cog_{idx}"):
                        st.session_state.cognitive_examples.pop(idx)
                        save_cognitive_examples()
                        st.rerun()
        else:
            st.write("No cognitive examples yet.")

def training_center(t):
    st.markdown(f"## {t['training_center']}")
    if hasattr(st.session_state, '_facts_added') and st.session_state._facts_added:
        st.success(f"✅ {st.session_state._facts_added} nouvo reyalite Ayiti ajoute.")
        del st.session_state._facts_added
    tabs = st.tabs(["📝 Train New Fact", "🧠 Cognitive Training", "📚 Manage Facts", "🎙️ Voice Training", "📖 Dictionaries"])
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### {t['train_new']}")
            new_fact = st.text_area(t['fact_text'], key="new_fact", height=150)
            if st.button(t['add_fact'], key="add_fact_btn"):
                if new_fact.strip():
                    add_to_training(new_fact.strip())
                    st.success("Fact added!")
                    st.rerun()
        with col2:
            bulk_training(t)
    with tabs[1]:
        cognitive_training_ui(t)
    with tabs[2]:
        manage_trained_facts(t)
        test_training_section(t)
    with tabs[3]:
        voice_training(t)
    with tabs[4]:
        dictionary_manager(t)

# ---------- CHAT INTERFACE (SINGLE TEXT AREA WITH BLACK BACKGROUND) ----------
def chat_interface(t):
    st.markdown(f"<h1 style='text-align:center; color:#ffd966;'>{t['app_title']}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Mwen reponn sèlman an Kreyòl. Poze m kesyon oswa telechaje yon imaj.</p>", unsafe_allow_html=True)
    
    # Build the conversation text
    conversation_lines = []
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            conversation_lines.append(f"🧑‍💻 {msg['content']}")
            if msg.get("image"):
                conversation_lines.append("[Image uploaded]")
        else:
            conversation_lines.append(f"🤖 {msg['content']}")
    conversation_text = "\n\n".join(conversation_lines)
    
    # Single read‑only text area with black background and white text
    st.text_area("", value=conversation_text, height=400, key="chat_display", disabled=True, label_visibility="collapsed")
    
    # Input row
    col_input, col_upload, col_send = st.columns([6, 1, 1])
    with col_input:
        user_input = st.text_input("", key="chat_input", placeholder=t['chat_input'], label_visibility="collapsed")
    with col_upload:
        uploaded_file = st.file_uploader("📷", type=["jpg", "jpeg", "png", "gif"], key="image_upload", label_visibility="collapsed")
    with col_send:
        send_clicked = st.button(t['send'], key="send_btn", use_container_width=True)
    
    if send_clicked and user_input.strip():
        user_msg = {"role": "user", "content": user_input}
        if uploaded_file:
            img_bytes = uploaded_file.read()
            user_msg["image"] = img_bytes
            st.session_state.conversation_history.append(user_msg)
            answer, is_fallback, skip_audio = generate_response(user_input, img_bytes)
        else:
            st.session_state.conversation_history.append(user_msg)
            answer, is_fallback, skip_audio = generate_response(user_input, None)
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
    
    render_audio_player()

# ---------- STREAMLIT PAGE CONFIG & CSS (with dark textarea) ----------
st.set_page_config(page_title="Gesner AI", page_icon="🧠", layout="wide")
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
        border: none;
    }
    .stTextInput input {
        background-color: #0f3460 !important;
        color: white !important;
        border-radius: 12px;
        border: 1px solid #e94560;
    }
    /* Force all text areas to have black background and white text */
    .stTextArea textarea {
        background-color: #000000 !important;
        color: #ffffff !important;
        font-family: monospace;
        font-size: 1rem;
        border: 1px solid #e94560;
        border-radius: 12px;
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
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #e94560;
    }
    @keyframes spin-globe {
        0% { transform: rotate(0deg); filter: drop-shadow(0 0 2px gold); }
        50% { filter: drop-shadow(0 0 15px #ffaa33) drop-shadow(0 0 5px orange); }
        100% { transform: rotate(360deg); filter: drop-shadow(0 0 2px gold); }
    }
    .spinning-brain {
        animation: spin-globe 3s linear infinite;
        display: inline-block;
        font-size: 3rem;
        text-align: center;
        width: 100%;
    }
    .sidebar-info {
        text-align: center;
        margin-top: 1rem;
        padding: 0.5rem;
        border-top: 1px solid #e94560;
        font-size: 0.9rem;
    }
    .sidebar-info a {
        color: #ffaa33 !important;
        text-decoration: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- LANGUAGES AND TEXTS ----------
LANGUAGES = {
    "English": "en",
    "Français": "fr",
    "Kreyòl Ayisyen": "ht",
    "Español": "es"
}

TEXTS = {
    "en": {
        "app_title": "Gesner AI - Kreyòl Assistant",
        "chat_input": "Ask me anything in Kreyòl...",
        "send": "Send",
        "clear": "Clear Chat",
        "dictionary": "Dictionary",
        "voice_training": "Voice Training",
        "bulk_training": "Bulk Training",
        "manage_facts": "Manage Facts",
        "test_training": "Test Training",
        "training_center": "Training Center",
        "train_new": "Train New Fact",
        "fact_text": "Fact text",
        "add_fact": "Add Fact",
        "upload_csv": "Upload CSV",
        "upload_audio": "Upload Audio",
        "record_voice": "Record Voice",
        "save_voice": "Save Voice",
        "play": "Play",
        "delete": "Delete",
        "edit": "Edit",
        "update": "Update",
        "chat_interface_label": "Chat"
    },
    "fr": {
        "app_title": "Gesner IA - Assistant Kreyòl",
        "chat_input": "Posez-moi une question en Kreyòl...",
        "send": "Envoyer",
        "clear": "Effacer",
        "dictionary": "Dictionnaire",
        "voice_training": "Entraînement vocal",
        "bulk_training": "Formation en masse",
        "manage_facts": "Gérer les faits",
        "test_training": "Tester l'entraînement",
        "training_center": "Centre de formation",
        "train_new": "Ajouter un fait",
        "fact_text": "Texte du fait",
        "add_fact": "Ajouter",
        "upload_csv": "Importer CSV",
        "upload_audio": "Importer audio",
        "record_voice": "Enregistrer",
        "save_voice": "Sauvegarder",
        "play": "Écouter",
        "delete": "Supprimer",
        "edit": "Modifier",
        "update": "Mettre à jour",
        "chat_interface_label": "Discussion"
    },
    "ht": {
        "app_title": "Gesner AI - Asistan Kreyòl",
        "chat_input": "Pose m yon kesyon an Kreyòl...",
        "send": "Voye",
        "clear": "Efase",
        "dictionary": "Diksyonè",
        "voice_training": "Fòmasyon Vwa",
        "bulk_training": "Fòmasyon an mas",
        "manage_facts": "Jere reyalite yo",
        "test_training": "Tès fòmasyon",
        "training_center": "Sant Fòmasyon",
        "train_new": "Anseye yon nouvo reyalite",
        "fact_text": "Tèks reyalite a",
        "add_fact": "Ajoute",
        "upload_csv": "Chaje CSV",
        "upload_audio": "Chaje odyo",
        "record_voice": "Anrejistre",
        "save_voice": "Sove",
        "play": "Jwe",
        "delete": "Efase",
        "edit": "Modifye",
        "update": "Mete ajou",
        "chat_interface_label": "Chat"
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
if "cognitive_examples" not in st.session_state:
    st.session_state.cognitive_examples = load_cognitive_examples()
if "training_access" not in st.session_state:
    st.session_state.training_access = False
if "ui_language" not in st.session_state:
    st.session_state.ui_language = "en"
if "tfidf_vectorizer" not in st.session_state:
    st.session_state.tfidf_vectorizer = None
if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None
if "play_audio" not in st.session_state:
    st.session_state.play_audio = None

VOICE_CACHE = load_voice_cache()

def show_sidebar():
    with st.sidebar:
        st.markdown('<div class="spinning-brain">🧠</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sidebar-info">
                <strong>Gesner AI</strong><br>
                Created by <strong>Gesner Deslandes</strong><br>
                Founder of <strong>GlobalInternet.py</strong><br>
                ✉️ <a href="mailto:deslandes78@gmail.com">deslandes78@gmail.com</a><br>
                📞 +509 4738-5663<br>
                🌐 <a href="https://globalinternetsitepy-abh7v6tnmskxxnuplrdcgk.streamlit.app/" target="_blank">globalinternetsitepy-abh7v6tnmskxxnuplrdcgk.streamlit.app/</a>
            </div>
            """,
            unsafe_allow_html=True
        )
        lang_choice = st.selectbox("🌐 Interface Language", list(LANGUAGES.keys()), key="lang_select")
        st.session_state.ui_language = LANGUAGES[lang_choice]
        t = TEXTS.get(st.session_state.ui_language, TEXTS["en"])
        menu = st.radio("Menu", [t['chat_interface_label'], t['dictionary'], t['voice_training'], t['training_center']])
        return menu, t

def main():
    rebuild_index()
    initialize_default_training()
    menu, t = show_sidebar()
    if menu == t.get('chat_interface_label', "Chat"):
        chat_interface(t)
    elif menu == t['dictionary']:
        dictionary_manager(t)
    elif menu == t['voice_training']:
        voice_training(t)
    elif menu == t['training_center']:
        training_center(t)

if __name__ == "__main__":
    main()
