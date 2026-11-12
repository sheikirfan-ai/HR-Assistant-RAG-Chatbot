import os
import streamlit as st
from dataprocessor import run as build_index
from QueryProcessor import process_user_query

st.set_page_config(page_title="HR Policy Assistant", page_icon="💼")
st.title("💼 HR Policy Assistant")
st.caption("Ask questions about company HR policies — leave, WFH, onboarding, and more.")

# --- One-time index build ---
# If the local ChromaDB folder doesn't exist yet, build the index from the PDF first.
if not os.path.exists("./chroma_db"):
    with st.spinner("Indexing HR policy document for the first time... this may take a minute."):
        build_index()
    st.success("Index built! You can start asking questions.")

# --- Chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about leave policy, WFH, onboarding...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = process_user_query(user_input)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
