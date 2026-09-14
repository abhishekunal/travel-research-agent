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

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# The soft ceiling for per-session queries. Not a hard rate limit —
# just a signal for accidental over-use. Browser refresh bypasses it.
# Real cost backstop is the $20/mo Anthropic spending cap set in
# the platform dashboard.
MAX_QUERIES_PER_SESSION = 10


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
# 4b. EMPTY-STATE PROMPT GALLERY
# ---------------------------------------------------------------
# When the conversation is empty, show clickable examples so users
# don't face a cold blank screen. Once they've interacted at all,
# this section renders nothing — the gallery is first-impression only.
#
# Clicking an example writes it to session_state.pending_input; the
# input-handling block below reads either pending_input OR the live
# chat input, whichever is present this run.
EXAMPLE_PROMPTS = [
    "Plan a trip to Austin, TX from Dec 15 to Dec 17, 2026. I'm into live music and BBQ.",
    "I want to spend a long weekend in Lisbon in early October — love food and architecture.",
    "Plan a 3-day trip to Kyoto next spring, focused on temples and quiet gardens.",
    "Help me plan a weekend in Chicago — I like museums and deep-dish pizza.",
]

if not st.session_state.messages:
    st.markdown("**Try one of these to get started:**")
    # Two columns × two rows layout; keeps buttons readable, doesn't
    # sprawl on wider screens.
    col1, col2 = st.columns(2)
    for i, example in enumerate(EXAMPLE_PROMPTS):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            # use_container_width makes buttons the same size in each column
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.pending_input = example
                st.rerun()


# ---------------------------------------------------------------
# 5. HANDLE THE USER'S NEW INPUT
# ---------------------------------------------------------------
# st.chat_input renders the text box pinned to the bottom of the page.
# It returns whatever the user typed on this re-run, or None if they
# haven't typed anything since the last re-run.
user_input = st.chat_input("Ask me about a trip...")

# The user's input this run comes from EITHER the chat input box OR
# a click on an empty-state example. Whichever is present, use it.
if not user_input and st.session_state.get("pending_input"):
    user_input = st.session_state.pending_input
    st.session_state.pending_input = None  # consume so it doesn't re-fire

if user_input:
    # Step 0: Cost guardrail. If this session has already hit the soft
    # limit, decline gracefully and tell the user how to continue.
    # We still render their attempted message so they see it wasn't
    # dropped silently.
    if st.session_state.query_count >= MAX_QUERIES_PER_SESSION:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.info(
                f"**You've hit the {MAX_QUERIES_PER_SESSION}-query limit for this session.**  \n\n"
                "This app is a portfolio project with API costs I'm covering personally, "
                "so I cap queries per browser session to keep things sustainable. "
                "**Refresh the page** to start a new session and keep exploring."
            )
        st.stop()

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
                # Fail visibly but don't crash the whole app. Wrap the
                # error in a friendlier explanation with next steps —
                # a raw exception makes the app feel broken; a scoped
                # error card makes it feel intentional and recoverable
                
                st.error(
                    "**Something went wrong on this turn.**  \n\n"
                    "This is usually a temporary issue with one of the "
                    "external services (weather, places data, or the "
                    "language model). You can:\n\n"
                    "- Try rephrasing your question\n"
                    "- Try again in a moment\n"
                    "- Click **Reset conversation** in the sidebar and start over"
                )
                # Keep the technical detail visible but out of the way,
                # so we can debug without cluttering the main UI.
                with st.expander("Technical details"):
                    st.code(f"{type(e).__name__}: {e}")
                st.stop()

        # Successful agent invocation — count it against the session budget.
        # This happens after the try/except, so failed calls don't count.
        st.session_state.query_count += 1
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

    st.markdown(
        "[View on GitHub](https://github.com/abhishekunal/travel-research-agent)"
    )
    st.caption(
        f"Queries this session: {st.session_state.query_count} / {MAX_QUERIES_PER_SESSION}"
    )

    st.markdown("---")

    if st.button("Reset conversation"):
        # New thread_id = fresh conversation. The old thread's state
        # still lives in the checkpointer, we just stop pointing to it.
        # UI history clears alongside.
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()  # force an immediate re-render with the empty state