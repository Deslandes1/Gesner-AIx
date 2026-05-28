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

# ========== DATA DIRECTORY ==========
DATA_DIR = ".gesner_data"
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_FILE = os.path.join(DATA_DIR, "training_data.json")
DICT_FILE = os.path.join(DATA_DIR, "dictionaries.json")
VOICE_FILE = os.path.join(DATA_DIR, "voice_cache.json")
COGNITIVE_FILE = os.path.join(DATA_DIR, "cognitive_examples.json")

# ========== PERSISTENCE FUNCTIONS ==========
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

# ========== WORKRISE POLICY FACTS (unchanged from previous) ==========
WORKRISE_POLICY_FACTS = [
    # ... (keep the full list from your previous version)
    # For brevity, I'm not repeating the entire list here. In your final code, copy the full WORKRISE_POLICY_FACTS list from your working app.
    "Workrise pap peye konpansasyon aksidan travay pou blesi ki rive pandan anplwaye patisipe volontè nan aktivite lwazi, sosyal, oswa espòtif lè li pa nan travay, si aktivite sa a pa fè pati devwa travay li.",
    # ... (add all the Workrise facts you had)
]

# ========== DEFAULT TRAINING FACTS (INCLUDING WORKRISE) ==========
def get_default_training_facts():
    # This function should return a list of all default facts (Ti Malice, Kreyòl grammar, Workrise, etc.)
    # For space, I'll show a minimal version. In your final code, merge your existing get_default_training_facts() from your working app.
    facts = [
        "Ti Malice se yon lojisyèl edikatif ki anseye timoun yo Kreyòl Ayisyen atravè jwèt ak istwa.",
        # ... all your existing facts (Ti Malice chapters, grammar, Workrise policies)
    ]
    # Add Workrise facts
    facts.extend(WORKRISE_POLICY_FACTS)
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

# ========== GROK API INTEGRATION ==========
def get_grok_api_key():
    try:
        return st.secrets["GROK_API_KEY"]
    except:
        return None

def call_grok_api(prompt, system_prompt="You are Gesner AI, a helpful assistant that answers in Haitian Creole. Provide accurate, concise responses."):
    api_key = get_grok_api_key()
    if not api_key:
        return None
    endpoint = st.secrets.get("GROK_API_ENDPOINT", "https://api.x.ai/v1/chat/completions")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "grok-1",  # or the appropriate model name
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
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

# ========== COGNITIVE TRAINING (Example-based learning) ==========
def add_cognitive_example(input_text, output_format, description=""):
    """Store an example of how to transform a query into a structured output."""
    example = {
        "input": input_text.strip(),
        "output": output_format.strip(),
        "description": description,
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.cognitive_examples.append(example)
    save_cognitive_examples()
    # Also add to training data as a special fact with a marker for easier retrieval
    marker = f"[COGNITIVE] Input: {input_text} → Output: {output_format}"
    if not any(item["text"] == marker for item in st.session_state.training_data):
        add_to_training(marker)

def find_cognitive_match(query):
    """Find the most similar cognitive example by embedding similarity."""
    if not st.session_state.cognitive_examples:
        return None
    query_emb = st.session_state.embedding_model.encode([query])[0]
    best_score = -1
    best_example = None
    for ex in st.session_state.cognitive_examples:
        ex_emb = st.session_state.embedding_model.encode([ex["input"]])[0]
        sim = np.dot(query_emb, ex_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(ex_emb))
        if sim > best_score:
            best_score = sim
            best_example = ex
    if best_score > 0.7:
        return best_example
    return None

def apply_cognitive_format(query, matched_example):
    """Apply the output format from the matched example, replacing placeholders."""
    # Simple placeholder replacement: if the example output contains variables like {query}, we replace.
    output = matched_example["output"]
    # You can extend this to extract entities from the query and replace.
    return output

# ========== ENHANCED RESPONSE GENERATION ==========
def generate_response(user_input):
    normalized = user_input.strip().lower()
    
    # 1. Hardcoded special cases (alphabet list, etc.)
    if "site konbyen let ki genhen nan alfabe kreyol la" in normalized:
        answer = "A, AN, B, CH, D, E, È, EN, F, G, H, I, J, K, L, M, N, NG, O, Ò, ON, OU, OUN, P, R, S, T, UI, V, W, Y, Z"
        return answer, False, False
    if re.search(r"konbyen let ki (genhen|gehen) nan alfabe kreyol la", normalized):
        answer = "Nan alfabe kreyol la genhen 32 let."
        return answer, False, False
    if "kijan ou rele" in normalized or "ki jan ou rele" in normalized:
        answer = "Non pa mwen se Gesner L'AI kreyate mwen se Gesner Deslandes nan Globalinternet.py."
        return answer, False, False
    
    # 2. Direct keyword answers (Ti Malice, levels, etc.)
    direct = direct_keyword_answer(user_input)
    if direct:
        return direct, False, False
    
    # 3. Check cognitive examples (pattern matching)
    cognitive_match = find_cognitive_match(user_input)
    if cognitive_match:
        formatted = apply_cognitive_format(user_input, cognitive_match)
        return formatted, False, False
    
    # 4. Local retrieval from trained facts
    facts = retrieve_facts_hybrid(user_input, k=5)
    if facts:
        reasoned = reason_answer(user_input, facts)
        return reasoned, False, False
    
    # 5. Math or simple logic
    math_result = reason_about_question(user_input)
    if math_result:
        return math_result, False, False
    
    # 6. Fallback to Grok API (if available)
    grok_answer = call_grok_api(user_input)
    if grok_answer:
        return grok_answer, False, False
    
    return "Mwen poko konn sa. Tanpri anseye m nan Sant Fòmasyon oswa ajoute yon egzanp kognitif.", True, False

# ---------- COGNITIVE TRAINING UI ----------
def cognitive_training_ui(t):
    st.subheader("🧠 Cognitive Training (Example‑based Learning)")
    st.info("""
    Teach Gesner AI how to analyze sentences and output specific formats.
    Provide an example input and the desired output. The AI will learn to apply this pattern to similar queries.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        example_input = st.text_area("Example Input (Kreyòl)", height=100, key="cognitive_input",
                                     placeholder="e.g., 'Fè yon rezime legal sou seksyon 6.03 Workrise politik PTO'")
        example_output = st.text_area("Desired Output", height=150, key="cognitive_output",
                                      placeholder="e.g., 'Rezime: Kantite PTO depann sou ansyènte. Ou ka roule jiska 80 èdtan. Pou mande PTO, itilize Paylocity.'")
        description = st.text_input("Description (optional)", key="cognitive_desc")
        if st.button("💾 Save Cognitive Example", key="save_cognitive"):
            if example_input and example_output:
                add_cognitive_example(example_input, example_output, description)
                st.success("Example saved! Gesner AI will now use this pattern.")
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
            st.write("No cognitive examples yet. Add one above.")

# ========== MODIFIED TRAINING CENTER TO INCLUDE COGNITIVE TAB ==========
def training_center(t):
    st.markdown(f"## {t['training_center']}")
    # Show Workrise facts added message if any
    if hasattr(st.session_state, '_workrise_added') and st.session_state._workrise_added:
        st.success(f"✅ {st.session_state._workrise_added} nouvo reyalite Workrise ajoute nan baz konesans la.")
        del st.session_state._workrise_added
    
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

# ========== REMAINING FUNCTIONS (unchanged from your working app) ==========
# I will keep all the existing functions (retrieve_facts_hybrid, direct_keyword_answer,
# reason_about_question, reason_answer, etc.) exactly as they were in your previous version.
# For brevity, I'm not repeating them here. In your final code, you must copy them from your working app.
# Below is a placeholder – ensure you include the full implementations.

def direct_keyword_answer(query):
    # ... (copy your existing implementation)
    pass

def reason_about_question(query):
    # ... (copy your existing implementation)
    pass

def retrieve_facts_hybrid(query, k=5):
    # ... (copy your existing implementation)
    pass

def reason_answer(query, retrieved_facts):
    # ... (copy your existing implementation)
    pass

def bulk_training(t):
    # ... (copy your existing implementation)
    pass

def manage_trained_facts(t):
    # ... (copy your existing implementation)
    pass

def test_training_section(t):
    # ... (copy your existing implementation)
    pass

def voice_training(t):
    # ... (copy your existing implementation)
    pass

def dictionary_manager(t):
    # ... (copy your existing implementation)
    pass

def chat_interface(t):
    # ... (copy your existing implementation but use the new generate_response)
    pass

def show_sidebar():
    # ... (copy your existing implementation)
    pass

def main():
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
    
    global VOICE_CACHE
    VOICE_CACHE = load_voice_cache()
    
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
