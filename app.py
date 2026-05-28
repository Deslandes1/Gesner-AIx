import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
page_title="Gesner AI",
page_icon="🧠",
layout="wide"
)

st.markdown("""

<style>

.stApp{
    background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
}

h1,h2,h3,p,div,span,label{
    color:white !important;
}

.stTextInput input{
    background:#0f3460 !important;
    color:white !important;
    border-radius:12px;
    border:1px solid #e94560;
}

.stTextArea textarea{
    background:black !important;
    color:white !important;
    border-radius:12px;
    border:1px solid #e94560;
    font-weight:bold;
}

.stButton button{
    background:#e94560 !important;
    color:white !important;
    border:none !important;
    border-radius:30px !important;
    font-weight:bold !important;
}

</style>

""", unsafe_allow_html=True)

DATA_DIR = ".gesner_data"

os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_FILE = os.path.join(DATA_DIR, "training_data.json")

if "conversation_history" not in st.session_state:
st.session_state.conversation_history = []

if "embedding_model" not in st.session_state:

```
with st.spinner("Loading AI..."):

    st.session_state.embedding_model = SentenceTransformer(
        'all-MiniLM-L6-v2'
    )
```

if "training_data" not in st.session_state:

```
if os.path.exists(TRAINING_FILE):

    with open(TRAINING_FILE, "r", encoding="utf-8") as f:

        st.session_state.training_data = json.load(f)

else:

    st.session_state.training_data = []
```

if "texts" not in st.session_state:
st.session_state.texts = []

if "index" not in st.session_state:
st.session_state.index = None

if "tfidf_vectorizer" not in st.session_state:
st.session_state.tfidf_vectorizer = None

if "tfidf_matrix" not in st.session_state:
st.session_state.tfidf_matrix = None

DEFAULT_FACTS = [

```
"Ayiti te vin endepandan 1ye janvye 1804.",

"Pòtoprens se kapital Ayiti.",

"Tousen Louverture se yon lidè revolisyon ayisyen.",

"Jan Jak Desalin se papa endepandans Ayiti.",

"Ayiti sitiye nan Karayib la.",

"Diri ak pwa se manje nasyonal Ayiti.",

"Soup joumou se manje endepandans Ayiti.",

"Kompa se yon mizik ayisyen.",

"Gesner AI te kreye pa Gesner Deslandes.",

"Kreyòl ayisyen gen 32 lèt.",

"Bonjou vle di hello.",

"Bonswa vle di good evening."
```

]

def save_training_data():

```
with open(TRAINING_FILE, "w", encoding="utf-8") as f:

    json.dump(
        st.session_state.training_data,
        f,
        ensure_ascii=False,
        indent=2
    )
```

def rebuild_index():

```
if not st.session_state.training_data:
    return

st.session_state.texts = [
    item["text"]
    for item in st.session_state.training_data
]

embeddings = [

    np.array(
        item["embedding"],
        dtype=np.float32
    )

    for item in st.session_state.training_data
]

dim = len(embeddings[0])

index = faiss.IndexFlatL2(dim)

index.add(np.array(embeddings))

st.session_state.index = index

vectorizer = TfidfVectorizer()

matrix = vectorizer.fit_transform(
    st.session_state.texts
)

st.session_state.tfidf_vectorizer = vectorizer
st.session_state.tfidf_matrix = matrix
```

def initialize_default_training():

```
if not st.session_state.training_data:

    for fact in DEFAULT_FACTS:

        emb = (
            st.session_state.embedding_model
            .encode([fact])[0]
        )

        st.session_state.training_data.append({

            "text": fact,
            "embedding": emb.tolist()

        })

    save_training_data()

rebuild_index()
```

initialize_default_training()

def retrieve_facts_hybrid(query, k=8):

```
if (
    st.session_state.index is None
    or st.session_state.index.ntotal == 0
):
    return []

query_embedding = (
    st.session_state.embedding_model
    .encode([query])[0]
    .astype(np.float32)
    .reshape(1, -1)
)

distances, indices = st.session_state.index.search(
    query_embedding,
    k
)

results = []

for i, idx in enumerate(indices[0]):

    if idx != -1:

        distance = distances[0][i]

        if distance < 2.5:

            text = st.session_state.texts[idx]

            if text not in results:
                results.append(text)

if (
    st.session_state.tfidf_vectorizer is not None
    and st.session_state.tfidf_matrix is not None
):

    q_vec = st.session_state.tfidf_vectorizer.transform([query])

    scores = cosine_similarity(
        q_vec,
        st.session_state.tfidf_matrix
    ).flatten()

    top_indices = scores.argsort()[-k:][::-1]

    for idx in top_indices:

        score = scores[idx]

        if score > 0.05:

            text = st.session_state.texts[idx]

            if text not in results:
                results.append(text)

return results[:k]
```

def small_talk_response(query):

```
q = query.lower()

replies = {

    "bonjou": "Bonjou 👋 Kijan ou ye?",

    "bonswa": "Bonswa 😊",

    "alo": "Alo 👋",

    "kijan ou ye": "Mwen anfòm 😊",

    "sak pase": "Anyen 🙂 Mwen la.",

    "mesi": "Pa gen pwoblèm 😊",

    "mèsi": "Pa gen pwoblèm 😊",

    "bye": "Orevwa 👋",

    "kiyes ou ye": "Mwen se Gesner AI 🤖",

    "ki moun ou ye": "Mwen se Gesner AI 🤖",
}

for key, value in replies.items():

    if key in q:
        return value

return None
```

def reason_about_question(query):

```
q = query.lower()

math_match = re.search(
    r"(\\d+)\\s*([\\+\\-\\*\\/])\\s*(\\d+)",
    q
)

if math_match:

    a = int(math_match.group(1))
    op = math_match.group(2)
    b = int(math_match.group(3))

    try:

        if op == "+":
            result = a + b

        elif op == "-":
            result = a - b

        elif op == "*":
            result = a * b

        elif op == "/":
            result = a / b

        else:
            result = None

        return f"Repons lan se {result}"

    except:
        pass

return None
```

def generate_response(user_input):

```
try:

    q = user_input.strip()

    if not q:

        return "Tanpri ekri yon kestyon."

    small = small_talk_response(q)

    if small:
        return small

    math_result = reason_about_question(q)

    if math_result:
        return math_result

    facts = retrieve_facts_hybrid(q)

    if facts:

        return ". ".join(facts[:3])

except Exception as e:

    print(e)

fallback = [

    "Mwen poko konprann kestyon an 😊",

    "Tanpri bay plis detay 🙂",

    "Mwen toujou ap aprann 😊",

    "Eseye poze kestyon an yon lòt jan."
]

return random.choice(fallback)
```

st.markdown(
"<h1 style='text-align:center;color:#ffd966;'>Gesner AI</h1>",
unsafe_allow_html=True
)

conversation_lines = []

for msg in st.session_state.conversation_history:

```
if msg["role"] == "user":

    conversation_lines.append(
        f"🧑 {msg['content']}"
    )

else:

    conversation_lines.append(
        f"🤖 {msg['content']}"
    )
```

conversation_text = "\n\n".join(conversation_lines)

st.text_area(
"Chat",
value=conversation_text,
height=400,
disabled=True
)

col1, col2 = st.columns([6,1])

with col1:

```
user_input = st.text_input(
    "Message",
    placeholder="Poze kestyon an..."
)
```

with col2:

```
send = st.button("📤")
```

if send and user_input:

```
st.session_state.conversation_history.append({

    "role":"user",
    "content":user_input

})

answer = generate_response(user_input)

st.session_state.conversation_history.append({

    "role":"assistant",
    "content":answer

})

st.rerun()
```

st.markdown("---")

st.subheader("Train Gesner AI")

new_fact = st.text_area(
"New Fact"
)

if st.button("Add Fact"):

```
if new_fact.strip():

    emb = (
        st.session_state.embedding_model
        .encode([new_fact])[0]
    )

    st.session_state.training_data.append({

        "text": new_fact,
        "embedding": emb.tolist()

    })

    save_training_data()

    rebuild_index()

    st.success("Fact Added ✅")
```
