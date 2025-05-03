# app.py
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Custom CSS for styling and text sizing
def inject_custom_css():
    st.markdown("""
    <style>
        /* Increase base font size */
        html, body, [class*="View"] {
            font-size: 16px !important;
        }
        
        /* Chat message sizing */
        .stChatMessage {
            font-size: 1.1rem !important;
            line-height: 1.6 !important;
        }
        
        /* Message bubble styling */
        [data-testid="stChatMessage"] > div:has(div[role="img"]) {
            background-color: #f8f9fa;
            border-radius: 15px;
            padding: 1.2rem;
            margin: 10px 0;
            max-width: 80%;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* User message specific styling */
        [data-testid="stChatMessage"] > div:has(div[role="img"]:nth-child(1)) {
            margin-left: 20%;
            background-color: #e3f2fd;
        }
        
        /* Doctor message specific styling */
        [data-testid="stChatMessage"] > div:has(div[role="img"]:nth-child(2)) {
            margin-right: 20%;
        }
        
        /* Text wrapping and overflow */
        .stMarkdown {
            word-wrap: break-word;
            /*white-space: pre-wrap;*/
            overflow-wrap: anywhere;
        }
        
        /* Input text sizing */
        .stTextInput input {
            font-size: 1.1rem !important;
            padding: 1rem !important;
        }
        
        /* Chat input container */
        .stChatFloatingInputContainer {
            padding: 0 4% !important;
        }
        
        /* Typing animation */
        @keyframes typing {
            from { width: 0 }
            to { width: 100% }
        }
        
        .typing-indicator {
            display: inline-block;
            overflow: hidden;
            white-space: pre-wrap;
            animation: typing 1s steps(40, end);
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize page config
st.set_page_config(
    page_title="MedChat - AI Medical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()

# Load model with caching
@st.cache_resource
def load_medical_model():
    model_path = "../models/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer

# Header section
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3309/3309803.png", width=80)
with col2:
    st.markdown('<div class="header"><h1 class="title">MedChat AI Assistant 🩺</h1></div>', unsafe_allow_html=True)

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.history.append({
        "role": "assistant",
        "content": "Hello! I'am your AI medical assistant. How can I help you today? 😊"
    })

# Sidebar with information
with st.sidebar:
    st.header("Medical Chat History")
    if st.button("🧹 Clear Conversation"):
        st.session_state.history = []
        st.rerun()
    st.divider()
    st.markdown("### Recent Conversations")
    # Add your history navigation logic here

# Display chat messages
for message in st.session_state.history:
    avatar = "👤" if message["role"] == "user" else "🩺"
    with st.chat_message(message["role"], avatar=avatar):
        content = f"**{message['content']}**" if message["role"] == "assistant" else message["content"]
        st.markdown(f"""{content}""")
        # <div style='font-size: 1.1rem; line-height: 1.6;'>
        #     {content}
        # </div>
        # """, unsafe_allow_html=True)

# Generate medical response
def generate_medical_response(prompt):
    model, tokenizer = load_medical_model()
    # system_prompt = "You are a helpful medical assistant. Provide accurate and compassionate responses to patient queries. Explain medical terms in simple language."
    system_prompt = "You are a helpful medical assistant and I am your patient, I will explain to you my case: "
    
    full_prompt = f"Patient: {prompt}"
    
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=1024,
        max_new_tokens=1024,
        temperature=0.55,
        top_k=30,
        top_p=0.9,
        repetition_penalty=1.1,
        no_repeat_ngram_size=2,
        early_stopping=True,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
    # return response.split("Doctor:")[-1].strip()

# Chat input
if prompt := st.chat_input("Describe your symptoms or ask a medical question..."):
    # Add user message to history
    st.session_state.history.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"**{prompt}**")
    
    # Generate and display assistant response
    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Dr. AI is analyzing your query..."):
            try:
                response = generate_medical_response(prompt)
                response = response.split("Patient:")[-1]
                formatted_response = f"""
                <div style='font-size: 1.1rem;'>
                    <strong>Dr. AI:</strong> {response}
                </div>
                """
                st.markdown(formatted_response, unsafe_allow_html=True)
            except Exception as e:
                error_response = """
                <div style='color: #dc3545; font-size: 1.1rem;'>
                    I'm experiencing technical difficulties. Please try again later.
                </div>
                """
                st.markdown(error_response, unsafe_allow_html=True)
                response = "System Error"
        
    # Add assistant response to history
    st.session_state.history.append({"role": "assistant", "content": response})