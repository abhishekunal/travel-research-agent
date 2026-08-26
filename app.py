"""
Travel Research Agent — Streamlit UI
=====================================
A chat-style web interface for the LangChain agent in agent.py.

Run with:
    streamlit run app.py

Streamlit will open a browser tab at http://localhost:8501
"""

import streamlit as st
from agent import run_agent


# ---------------------------------------------------------------
# 1. PAGE CONFIG
# ---------------------------------------------------------------
# st.set_page_config MUST be the first Streamlit call in the script.
# It sets browser tab title, favicon, and layout.
st.set_page_config(
    page_title="Travel Research Agent",
    page_icon="✈️",
    layout="centered",
)


# ---------------------------------------------------------------
# 2. HEADER
# ---------------------------------------------------------------
st.title("✈️ Travel Research Agent")
st.caption("Ask me about weather in any destination, or general travel questions.")


# ---------------------------------------------------------------
# 3. SESSION STATE (conversation memory)
# ---------------------------------------------------------------
# IMPORTANT: Streamlit re-runs this entire script on every user action
# (typing, clicking, etc.). Regular Python variables get wiped each run.
# st.session_state is a dict that persists across re-runs for a given
# browser session — this is where we store the chat history.
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------
# 4. RENDER CHAT HISTORY
# ---------------------------------------------------------------
# On every re-run, we redraw the full conversation from session_state.
# st.chat_message() creates a styled chat bubble (user or assistant).
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------------------------------------------------------------
# 5. CHAT INPUT
# ---------------------------------------------------------------
# st.chat_input() renders a fixed input box at the bottom of the page.
# It returns the user's text when they hit Enter, otherwise None.
# The walrus operator (:=) both assigns and checks in one line.
if user_input := st.chat_input("e.g., What's the weather in Barcelona?"):

    # Store and display the user's message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call the agent and display its response
    # st.spinner shows a loading indicator while the agent thinks
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = run_agent(user_input)
            except Exception as e:
                # Catch-all so the UI never crashes on API/network errors
                response = f"Something went wrong: {e}"

        st.markdown(response)

    # Store the assistant response so it survives the next re-run
    st.session_state.messages.append({"role": "assistant", "content": response})


# ---------------------------------------------------------------
# 6. SIDEBAR (utility controls)
# ---------------------------------------------------------------
with st.sidebar:
    st.subheader("About")
    st.write(
        "This is a walking-skeleton demo of a travel research agent, "
        "built with LangChain, Claude, and OpenWeatherMap."
    )

    # A button to reset the conversation
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()  # force an immediate re-run so the UI updates