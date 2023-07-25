import streamlit as st
from streamlit_chat import message


st.title('Admissions Guide')
st.set_page_config(page_title="AU-Bot")
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
