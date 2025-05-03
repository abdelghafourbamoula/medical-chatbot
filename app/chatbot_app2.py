import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
# import torch
import time

# Load the model and tokenizer
model_name = "../models/flan-t5-small"  # Replace with your fine-tuned model path
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# Streamlit app layout
st.set_page_config(page_title="Clinical Chatbot", layout="centered")
st.title("💬 Clinical Chatbot Assistant")

st.markdown(
    """
    Welcome to the Clinical Chatbot powered by T5!  
    Enter your medical query, and receive AI-generated doctor responses.
    """
)

# Conversation history state
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []

# Function to generate chatbot responses
def generate_response(patient_query):
    # input_text = "hi doctor, " + patient_query.strip().lower()
    # input_text = "you are a doctor, this is you patient case : "# + patient_query.strip().lower()
    input_text = "I am your patient, I will explain to you my case: " + patient_query
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids

    # Generating with adjusted parameters
    output_ids = model.generate(
        input_ids,
        max_length=512,        # Limit response length
        top_k=30,             # Control randomness
        top_p=0.95,           # Control diversity
        # num_beams=3,          # Beam search for better output
        temperature=0.55,      # Reduce randomness
        no_repeat_ngram_size=2,  # Avoid repetitive text
        do_sample=True
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# User input
with st.form("query_form"):
    patient_query = st.text_input("Enter your medical query:", "")
    submitted = st.form_submit_button("Submit")

if submitted and patient_query.strip():
    # Show loading spinner while generating the response
    with st.spinner("Generating doctor's response..."):
        time.sleep(2)  # Simulate processing delay
        response = generate_response(patient_query)

        # Update conversation history
        st.session_state["conversation"].append(f"**Patient:** {patient_query}")
        st.session_state["conversation"].append(f"**Doctor:** {response}")

# Display conversation history
if st.session_state["conversation"]:
    st.markdown("### Conversation History")
    for chat in st.session_state["conversation"]:
        st.write(chat)

# Clear conversation button
if st.button("Clear Conversation"):
    st.session_state["conversation"] = []
    st.experimental_rerun()
