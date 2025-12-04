import streamlit as st
import datetime
import base64
import html
from typing import Dict, List, Any
from translator import translate_text
from medical_model import generate_medical_answer
from utils import detect_language, is_medical_query

# --- Page Config ---
st.set_page_config(
    page_title="MediChat AI - Multilingual Medical Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
def load_css():
    st.markdown(""" 
    <style>
        /* Main container */
        .main {
            max-width: 900px;
            margin: 0 auto;
            padding: 1rem;
        }
        
        /* Header */
        .header {
            text-align: center;
            margin-bottom: 2rem;
            padding: 1.5rem;
            background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
            border-radius: 12px;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Chat container */
        .chat-container {
            background-color: #f8fafc;
            border-radius: 12px;
            padding: 1.0rem 1.25rem;
            margin-top: 1.25rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.1);
            border: 1px solid #e2e8f0;
        }
        
        /* Message bubbles */
        .user-message {
            background-color: #ffffff;
            padding: 12px 16px;
            border-radius: 18px 18px 0 18px;
            margin: 8px 0;
            max-width: 80%;
            margin-left: auto;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.15);
            color: #000000;
        }
        
        .assistant-message {
            background-color: #ffffff;
            padding: 12px 16px;
            border-radius: 18px 18px 18px 0;
            margin: 8px 0;
            max-width: 80%;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
            font-size: 0.95rem;
            line-height: 1.5;
            white-space: pre-wrap;
            color: #000000;
        }
        
        /* Input area */
        .stTextArea > div > div > textarea {
            min-height: 100px !important;
            border-radius: 12px !important;
            padding: 12px !important;
            font-size: 16px !important;
            border: 2px solid #e0e0e0 !important;
        }
        
        /* Buttons */
        .stButton > button {
            border-radius: 20px !important;
            padding: 0.5rem 2rem !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Language tag */
        .language-tag {
            font-size: 0.8rem;
            color: #666;
            margin-bottom: 4px;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            margin-top: 2rem;
            padding: 1rem;
            color: #666;
            font-size: 0.9rem;
        }
        /* Hide built-in screencast */
        li[title="Record a screencast"] {display:none !important;}
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "flan-t5-base"
if "show_english_answer" not in st.session_state:
    st.session_state.show_english_answer = True

# Load custom CSS
load_css()

# --- Sidebar Settings ---
with st.sidebar:
    st.subheader("Assistant Settings")
    st.session_state.selected_model = st.selectbox(
        "Medical model",
        ["flan-t5-base", "flan-t5", "biogpt"],
        index=["flan-t5-base", "flan-t5", "biogpt"].index(st.session_state.selected_model)
        if st.session_state.selected_model in ["flan-t5-base", "flan-t5", "biogpt"]
        else 0,
    )
    # Removed screenshot and screen recording functionality

    st.session_state.show_english_answer = st.checkbox(
        "Show original English answer",
        value=st.session_state.show_english_answer,
    )

    # Query log
    with st.expander("📜 View recent queries"):
        if st.session_state.chat_history:
            for item in reversed(st.session_state.chat_history[-20:]):
                if item["role"] == "user":
                    st.markdown(f"• {item['original']}")
        else:
            st.caption("No queries yet.")

# --- Header Section ---
st.markdown(
    """
    <div class="header">
        <h1 style="color: white; margin: 0;">💊 MediChat AI</h1>
        <p style="margin: 0.5rem 0 0; opacity: 0.9;">
            Your multilingual medical assistant for evidence-based health information
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Main Content ---
# --- Input area (prompt at the top) ---
with st.form("prompt_form"):
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_area(
            "",
            placeholder="Type your medical question here...",
            height=100,
            key="query_input"
        )
    with col2:
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("Send", use_container_width=True)

# --- Chat history below the prompt ---
with st.container():
    # Chat container (only shown once there is history)
    if st.session_state.chat_history:
        with st.container():
            st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
            
            for msg in st.session_state.chat_history:
                role = msg.get("role", "user")

                # Render user messages
                if role == "user":
                    user_text = html.escape(str(msg.get("original", ""))).replace("\n", "<br>")
                    st.markdown(
                        f"""
                        <div style="display: flex; flex-direction: column; align-items: flex-end; margin-bottom: 1.5rem;">
                            <div class="language-tag">You • {msg["user_lang"].upper()}</div>
                            <div class="user-message">
                                {user_text}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Render assistant messages
                elif role == "assistant":
                    user_lang = msg.get("user_lang", "en")
                    translated_raw = msg.get("final") or msg.get("eng_a") or ""
                    eng_raw = msg.get("eng_a", "")

                    translated_text = html.escape(str(translated_raw)).replace("\n", "<br>")
                    eng_text = html.escape(str(eng_raw)).replace("\n", "<br>")

                    # Main assistant bubble (translated or English depending on user language)
                    st.markdown(
                        f"""
                        <div style="display: flex; flex-direction: column; align-items: flex-start; margin-bottom: 0.75rem;">
                            <div class="language-tag">MediChat AI • {user_lang.upper()}</div>
                            <div class="assistant-message">
                                {translated_text}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Secondary bubble: original English answer for non-English queries
                    if (
                        st.session_state.show_english_answer
                        and user_lang != "en" and eng_raw
                    ):
                        st.markdown(
                            f"""
                            <div style="display: flex; flex-direction: column; align-items: flex-start; margin-bottom: 1.5rem;">
                                <div class="language-tag">Original answer • EN</div>
                                <div class="assistant-message" style="background-color:#f5f5f5;">
                                    {eng_text}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Welcome message for empty chat
        st.markdown(
            """
            <div style="text-align: center; padding: 1.5rem 1rem; margin: 1.5rem 0;">
                <h3 style="margin:0 0 1rem 0; color:#1e293b;">👋 Welcome to MediChat AI</h3>
                <p style="color:#475569; margin:0;">Ask me anything about medical conditions, symptoms, treatments, or medications.</p>
                <p style="color:#64748b; font-size:0.9rem; margin:0.5rem 0 0;">Available in multiple languages - I'll automatically detect your language!</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Clear conversation button
if st.session_state.chat_history:
    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Disclaimer
st.markdown(
    """
    <div class="footer" style="margin-top: 2rem; padding: 1rem 0;">
        <div style="background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%); padding: 1rem 1.5rem; border-radius: 12px; color: #ffffff;">
            <p style="margin:0 0 0.5rem 0; font-weight:500;">⚠️ Medical Disclaimer</p>
            <p style="margin:0; font-size:0.9em; opacity:0.9;">This tool provides general health information and is not a substitute for professional medical advice, diagnosis, or treatment. For medical emergencies, contact your local emergency services immediately.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Process form submission
if submit_btn and query.strip():
    with st.spinner("Processing your question..."):
        # Detect language
        detected = detect_language(query)
        
        # Add user message to chat history immediately
        user_msg = {
            "role": "user",
            "user_lang": detected,
            "original": query,
            "time": datetime.datetime.utcnow().isoformat() + "Z"
        }
        st.session_state.chat_history.append(user_msg)
        
        # Process the query
        try:
            # Show typing indicator
            placeholder_spinner = st.empty()
            placeholder_spinner.markdown(
                """
                <div style="display: flex; flex-direction: column; align-items: flex-start; margin-bottom: 1.5rem;">
                    <div class="language-tag">MediChat AI</div>
                    <div class="assistant-message">
                        <div class="typing-indicator">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Determine if this is a medical query (check original + English translation if needed)
            english_for_check = None
            medical_question = is_medical_query(query)
            if not medical_question and detected != "en":
                english_for_check = translate_text(query, detected, "en")
                medical_question = is_medical_query(english_for_check)

            if not medical_question:
                polite_decline = (
                    "I'm sorry, I can only answer medical or health-related questions. "
                    "Please ask about symptoms, treatments, medicines, or other healthcare topics."
                )
                final = polite_decline if detected == "en" else translate_text(polite_decline, "en", detected)
                assistant_msg = {
                    "role": "assistant",
                    "user_lang": detected,
                    "original": query,
                    "eng_q": english_for_check or (query if detected == "en" else ""),
                    "eng_a": polite_decline,
                    "final": final,
                    "time": datetime.datetime.utcnow().isoformat() + "Z"
                }
                st.session_state.chat_history[-1] = user_msg
                st.session_state.chat_history.append(assistant_msg)
                placeholder_spinner.empty()
                st.rerun()

            # Process the query (translation and medical answer generation)
            english_q = (
                english_for_check
                if english_for_check is not None
                else (query if detected == "en" else translate_text(query, detected, "en"))
            )
            english_a = generate_medical_answer(english_q, model_name=st.session_state.selected_model or "flan-t5-base")
            final = english_a if detected == "en" else translate_text(english_a, "en", detected)
            
            # Add assistant response to chat history
            assistant_msg = {
                "role": "assistant",
                "user_lang": detected,
                "original": query,
                "eng_q": english_q,
                "eng_a": english_a,
                "final": final,
                "time": datetime.datetime.utcnow().isoformat() + "Z"
            }
            
            # Update the last message in chat history
            st.session_state.chat_history[-1] = user_msg
            st.session_state.chat_history.append(assistant_msg)
            
        except Exception as e:
            st.error("An error occurred while processing your request. Please try again.")
            st.session_state.chat_history.pop()  # Remove the user message if there was an error
        
        # Rerun to update the UI
        st.rerun()

# Add JavaScript to auto-scroll to bottom of chat
st.markdown(
    """
    <script>
        // Auto-scroll to bottom of chat
        function scrollToBottom() {
            const container = document.getElementById('chat-container');
            container.scrollTop = container.scrollHeight;
        }
        
        // Run on page load and after each Streamlit event
        document.addEventListener('DOMContentLoaded', scrollToBottom);
        window.addEventListener('load', scrollToBottom);
    </script>
    """,
    unsafe_allow_html=True
)

