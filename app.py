import streamlit as st
import requests

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Gesner AI", page_icon="🧠", layout="wide")

# =========================
# TRAINING CENTER (READ ONLY)
# =========================
HAITIAN_KNOWLEDGE_FACTS = [
    "Kristòf Kolon te dekouvri Ayiti an 1492.",
    "Pòtoprens se kapital Ayiti.",
    "Ayiti pran endepandans 1 janvye 1804.",
    "Ayiti sitiye nan Karayib la sou zile Ispanyola.",
    "Tousen Louverture te yon lidè revolisyon Ayiti."
]

# =========================
# GROQ API KEY
# =========================
def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return None

# =========================
# GROQ CALL (ONLY BRAIN)
# =========================
def call_groq(prompt):
    key = get_groq_key()

    if not key:
        return "GROQ API key pa jwenn nan secrets."

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Gesner AI. "
                    "Answer directly in Haitian Creole. "
                    "Do NOT repeat the question. "
                    "Be short, correct, and clear."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]

        return f"Groq Error: {response.text}"

    except Exception:
        return "Pa ka konekte ak Groq kounye a."

# =========================
# LIGHT UI STYLE
# =========================
st.markdown("""
<style>
.stApp {
    background: #f6f8fc;
}

[data-testid="stSidebar"] {
    background: #e9eef7;
}

h1,h2,h3,p,div,label {
    color: #111 !important;
}

.stTextInput input {
    background: white !important;
    color: black !important;
    border-radius: 10px;
}

.stButton button {
    background: #4a6cff !important;
    color: white !important;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 🧠 Gesner AI")

    st.markdown("""
**Globalinternet.py/software**  
Built by Gesner Deslandes  

📞 (509)-47385663  
📧 deslandes78@gmail.com  
""")

    st.markdown("---")
    st.markdown("### 📚 Training Center")

    for fact in HAITIAN_KNOWLEDGE_FACTS:
        st.write("• " + fact)

# =========================
# CHAT MEMORY SAFE
# =========================
if "chat" not in st.session_state:
    st.session_state.chat = []

# =========================
# CHAT DISPLAY (SAFE FIX)
# =========================
st.title("🧠 Gesner AI (Groq Powered)")

for msg in st.session_state.chat:
    role = msg.get("role", "bot")
    text = msg.get("text", "")

    if role == "user":
        st.markdown("🧑 " + text)
    else:
        st.markdown("🤖 " + text)

# =========================
# INPUT
# =========================
user_input = st.text_input("Ask anything...")

if st.button("Send"):
    if user_input and user_input.strip():

        # save user message
        st.session_state.chat.append({
            "role": "user",
            "text": user_input.strip()
        })

        # GROQ ONLY RESPONSE
        answer = call_groq(user_input)

        # save assistant message
        st.session_state.chat.append({
            "role": "bot",
            "text": answer
        })

        st.rerun()
