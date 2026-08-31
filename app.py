"""
Travel Research Agent — Streamlit UI (Form-based)
==================================================
Structured trip planner: user fills in destination, dates, and duration;
guardrails validate; agent runs; TripBrief renders as cards.

Run with:
    streamlit run app.py
"""

from datetime import date, timedelta

import streamlit as st

from agent import run_agent, to_trip_brief
from guardrails import MAX_TRIP_DAYS, MIN_TRIP_DAYS, validate_trip_inputs


# ---------------------------------------------------------------
# 1. PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Travel Research Agent",
    page_icon="✈️",
    layout="centered",
)


# ---------------------------------------------------------------
# 2. HEADER
# ---------------------------------------------------------------
st.title("✈️ Travel Research Agent")
st.caption("Plan a trip: pick a destination, dates, and length. I'll pull weather, restaurants, and attractions.")


# ---------------------------------------------------------------
# 3. INPUT FORM
# ---------------------------------------------------------------
# st.form batches inputs — nothing runs until the user hits the submit
# button. Without this, Streamlit would re-run and re-validate on every
# keystroke, which would spam Nominatim with geocode calls.
with st.form("trip_form"):
    city = st.text_input(
        "Destination city",
        placeholder="e.g., Tokyo, Barcelona, Lisbon",
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start date",
            value=date.today() + timedelta(days=14),
            min_value=date.today(),
        )
    with col2:
        days = st.number_input(
            "Trip length (days)",
            min_value=MIN_TRIP_DAYS,
            max_value=MAX_TRIP_DAYS,
            value=5,
            step=1,
        )

    submitted = st.form_submit_button("Plan trip", type="primary")


# ---------------------------------------------------------------
# 4. RESULT CACHE
# ---------------------------------------------------------------
# Streamlit re-runs the whole script on any interaction. Without caching
# the result in session_state, expanding a card or resizing the window
# would re-run the agent. We stash the last brief and re-render it on
# subsequent runs until the user submits a new trip.
if "last_brief" not in st.session_state:
    st.session_state.last_brief = None
    st.session_state.last_raw = None
    st.session_state.last_inputs = None


# ---------------------------------------------------------------
# 5. HANDLE SUBMISSION
# ---------------------------------------------------------------
if submitted:
    # Step A: Guardrails. Cheap validation before we spend an LLM call.
    validation = validate_trip_inputs(city, start_date, days)

    if not validation.ok:
        st.error(validation.error)
        st.stop()  # halt this run; the form stays populated for retry

    inputs = validation.value

    # Step B: Build the agent prompt from validated inputs.
    end_date = inputs["start_date"] + timedelta(days=inputs["days"] - 1)
    prompt = (
        f"Plan a {inputs['days']}-day trip to {inputs['city_input']} "
        f"from {inputs['start_date'].isoformat()} to {end_date.isoformat()}. "
        f"Include weather, restaurant recommendations, and attractions."
    )

    # Step C: Run the agent, then convert to structured TripBrief.
    with st.spinner("Researching your trip..."):
        try:
            raw_response = run_agent(prompt)
            brief = to_trip_brief(prompt, raw_response)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    # Cache for re-renders
    st.session_state.last_brief = brief
    st.session_state.last_raw = raw_response
    st.session_state.last_inputs = inputs


# ---------------------------------------------------------------
# 6. RENDER TRIP BRIEF AS CARDS
# ---------------------------------------------------------------
def render_brief(brief, inputs):
    """Render a TripBrief as a stack of bordered cards."""

    end_date = inputs["start_date"] + timedelta(days=inputs["days"] - 1)

    # --- Header card: destination + dates ---
    with st.container(border=True):
        st.subheader(f"📍 {brief.destination}")
        st.write(
            f"**{brief.start_date.strftime('%b %d, %Y')} → "
            f"{brief.end_date.strftime('%b %d, %Y')}**  ·  {inputs['days']} days"
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

    # --- Debug: raw agent output (collapsed by default) ---
    with st.expander("🔍 Raw agent output (debug)"):
        st.text(st.session_state.last_raw or "")

# ---------------------------------------------------------------
# 6b. INVOKE THE RENDERER
# ---------------------------------------------------------------
# render_brief() is just a function definition until we call it.
# We call it on every re-run (not just after form submit) so the cards
# stay on screen when the user interacts with the sidebar or expander.
if st.session_state.last_brief is not None:
    render_brief(st.session_state.last_brief, st.session_state.last_inputs)

# ---------------------------------------------------------------
# 7. SIDEBAR
# ---------------------------------------------------------------
with st.sidebar:
    st.subheader("About")
    st.write(
        "Travel Research Agent — Weekend 2 build. "
        "Built with LangChain + LangGraph, Claude Sonnet 4.5, "
        "OpenStreetMap, and OpenWeatherMap."
    )

    if st.button("Reset"):
        st.session_state.last_brief = None
        st.session_state.last_raw = None
        st.session_state.last_inputs = None
        st.rerun()