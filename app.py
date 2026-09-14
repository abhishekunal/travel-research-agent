"""
Travel Research Agent — Streamlit Chat UI
==========================================
Conversational trip planner: user types freeform requests, agent asks
clarifying questions when needed, and produces a structured trip brief
with reasoning.

Run with:
    streamlit run app.py
"""

import uuid
from datetime import timedelta

import streamlit as st

from agent import run_agent


# ---------------------------------------------------------------
# 1. PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Travel Research Agent",
    page_icon="✈️",
    layout="centered",
)


# ---------------------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# ---------------------------------------------------------------
# Streamlit re-runs this whole script on every user interaction. Any
# state that must survive a re-run (thread_id, chat history, cached
# briefs) lives in st.session_state. Initialize once per browser
# session, then read on every re-run.
#
# The "if key not in state" pattern is Streamlit's idiomatic way to
# initialize state only on the very first script run.
if "thread_id" not in st.session_state:
    # One UUID per browser session. This is what LangGraph's checkpointer
    # uses to keep this user's conversation state isolated from any other
    # user's. Same thread_id across turns = same conversation, memory
    # persists. New thread_id = fresh conversation.
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    # The UI-side copy of the conversation, used purely for rendering
    # the chat scroll. Each entry is a dict:
    #   {"role": "user" | "assistant", "content": str, "brief": TripBrief | None}
    # The agent's own conversation history lives separately in the
    # LangGraph checkpointer, keyed by thread_id — this list is what
    # the chat bubbles read from.
    st.session_state.messages = []


# ---------------------------------------------------------------
# 3. HEADER
# ---------------------------------------------------------------
st.title("✈️ Travel Research Agent")
st.caption(
    "Ask me to plan a trip. I'll pull weather, restaurants, and attractions, "
    "and I'll ask questions if I need more info."
)


# ---------------------------------------------------------------
# 4. RENDER THE CHAT HISTORY
# ---------------------------------------------------------------
# On every re-run, redraw every message we've accumulated in session_state.
# This is what makes the chat scroll feel persistent even though the
# script is re-executing top-to-bottom on every interaction.
def render_brief_card(brief):
    """Render a TripBrief as a stack of bordered cards inside a chat bubble."""

    # --- Header card: destination + dates ---
    with st.container(border=True):
        st.subheader(f"📍 {brief.destination}")
        st.write(
            f"**{brief.start_date.strftime('%b %d, %Y')} → "
            f"{brief.end_date.strftime('%b %d, %Y')}**"
        )

    # --- Notes banner (skip-and-note + any caveats) ---
    if brief.notes:
        for note in brief.notes:
            st.warning(note)

    # --- Weather card ---
    if brief.weather_summary:
        with st.container(border=True):
            st.markdown("### 🌤️ Weather")
            st.write(brief.weather_summary)

    # --- Restaurants card ---
    if brief.restaurants:
        with st.container(border=True):
            st.markdown("### 🍽️ Restaurants")
            for r in brief.restaurants:
                line = f"**{r.name}**  ·  _{r.type}_"
                if r.address and r.address != "No address on file":
                    line += f"  \n📍 {r.address}"
                st.write(line)

    # --- Attractions card ---
    if brief.attractions:
        with st.container(border=True):
            st.markdown("### 🎡 Attractions")
            for a in brief.attractions:
                line = f"**{a.name}**  ·  _{a.type}_"
                if a.address and a.address != "No address on file":
                    line += f"  \n📍 {a.address}"
                st.write(line)


# Draw every message that's already in the history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Every message has some text content — a user's question, an
        # agent's clarifying question, or an agent's reasoning paragraph
        # that accompanies the brief.
        if msg["content"]:
            st.markdown(msg["content"])

        # Assistant turns that include a trip brief render the cards
        # below the reasoning text, all inside the same chat bubble.
        if msg.get("brief"):
            render_brief_card(msg["brief"])


# ---------------------------------------------------------------
# 5. HANDLE THE USER'S NEW INPUT
# ---------------------------------------------------------------
# st.chat_input renders the text box pinned to the bottom of the page.
# It returns whatever the user typed on this re-run, or None if they
# haven't typed anything since the last re-run.
user_input = st.chat_input("Ask me about a trip...")

if user_input:
    # Step A: Add the user's message to history and render it immediately
    # so they see it appear before we start the (slow) agent call.
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "brief": None,
    })
    with st.chat_message("user"):
        st.markdown(user_input)

    # Step B: Run the agent. This is the slow part — real API calls to
    # Claude + OpenWeatherMap + OSM. Show a spinner so the user knows
    # something is happening.
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = run_agent(
                    user_input,
                    thread_id=st.session_state.thread_id,
                )
            except Exception as e:
                # Fail visibly but don't crash the whole app. The user
                # can retry with a different message.
                st.error(f"Something went wrong: {e}")
                st.stop()

        # Step C: The agent's response can be one of two shapes:
        #   (a) A clarifying question — scope needed more info
        #   (b) A completed brief — scope had enough, synthesize ran
        # Handle both and add to history for future re-renders.

        if result["clarifying_question"]:
            # Path (a): render the question, save to history.
            question = result["clarifying_question"]
            st.markdown(question)
            st.session_state.messages.append({
                "role": "assistant",
                "content": question,
                "brief": None,
            })

        elif result["trip_brief"]:
            # Path (b): render the reasoning paragraph and the brief cards.
            reasoning = result["reasoning"] or ""
            if reasoning:
                st.markdown(reasoning)
            render_brief_card(result["trip_brief"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": reasoning,
                "brief": result["trip_brief"],
            })

        else:
            # Defensive: shouldn't hit this path given the graph's routing.
            # If we do, surface it as an error rather than silently failing.
            st.error(
                "The agent didn't produce a clarifying question or a trip "
                "brief. This is likely a bug — try resetting the conversation."
            )


# ---------------------------------------------------------------
# 6. SIDEBAR
# ---------------------------------------------------------------
with st.sidebar:
    st.subheader("About")
    st.write(
        "Travel Research Agent — Weekend 3 build.\n\n"
        "A conversational agent built with LangGraph, Claude Sonnet 4.5, "
        "OpenStreetMap, and OpenWeatherMap. Ask about a destination, "
        "answer any follow-up questions, and get a structured trip brief."
    )

    st.markdown("---")

    if st.button("Reset conversation"):
        # New thread_id = fresh conversation. The old thread's state
        # still lives in the checkpointer, we just stop pointing to it.
        # UI history clears alongside.
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()  # force an immediate re-render with the empty state