import streamlit as st
import requests
import json
import os

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Gesner AI", page_icon="🧠", layout="wide")

# =========================
# DATA (TRAINING CENTER ONLY DISPLAY)
# =========================
HAITIAN_KNOWLEDGE_FACTS = [
    "Kristòf Kolon te dekouvri Ayiti an 1492.",
    "Pòtoprens se kapital Ayiti.",
    "Ayiti pran endepandans 1 janvye 1804.",
    "Ayiti sitiye nan Karayib la sou zile Ispanyola.",
    "Tousen Louverture te yon lidè revolisyon Ayiti."
]

# =========================
# GROK API
# =========================
def get_grok_key():
    try:
        return st.secrets["GROK_API_KEY"]
    except:
        return None

def call_grok(prompt):
    key = get_grok_key()
    if not key:
        return "Grok API key pa jwenn nan secrets."

    try:
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "grok-1",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are Gesner AI. "
                            "Answer ALL questions directly in Haitian Creole. "
                            "Do NOT repeat the question. "
                            "Be accurate and concise."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            },
            timeout=8
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]

        return "Erè nan Grok API."

    except Exception:
        return "Mwen pa ka konekte ak Grok kounye a."

# =========================
# UI STYLE (LIGHT MODE)
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
    st.markdown("### 📚 Training Center (Read Only)")
    for fact in HAITIAN_KNOWLEDGE_FACTS:
        st.write("• " + fact)

# =========================
# CHAT MEMORY
# =========================
if "chat" not in st.session_state:
    st.session_state.chat = []

# =========================
# CHAT DISPLAY
# =========================
st.title("🧠 Gesner AI (Grok Powered)")

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

        # GROK ONLY RESPONSE
        answer = call_grok(user_input)

        # save bot message
        st.session_state.chat.append({
            "role": "bot",
            "text": answer
        })

        st.rerun()
