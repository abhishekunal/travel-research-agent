# ✈️ Travel Research Agent

A conversational AI agent for travel research, built as a portfolio project to demonstrate applied agentic AI skills.

**🔗 Try it live:** [travel-research-agent-abhishekunal.streamlit.app](https://travel-research-agent-abhishekunal.streamlit.app/)

**Current status:** Weekend 3 complete — conversational agent with prompt chaining, memory, and hosted deployment. Weekend 4 (RAG + public launch) in progress.

---

## Roadmap

- [x] **Weekend 1** — Walking skeleton: LangChain agent, one tool (weather), Streamlit UI
- [x] **Weekend 2** — Multi-tool orchestration + structured output (attractions, restaurants via OpenStreetMap, Pydantic schemas, input validation)
- [x] **Weekend 3** — Prompt chaining architecture, conversation memory, chat UI, quiet-launch deployment
- [ ] **Weekend 4** — Intent router, places API upgrade, RAG pipeline, public launch

Detailed plans and post-hoc build notes for each weekend live in `plans/WEEKEND_N_PLAN.md`.

---

## What it does

Have a conversation with the agent about a trip. It asks when it needs more information, remembers what you've said across turns, and produces a structured trip brief with reasoning about why it made the picks it did.

**A real example flow from the live app:**

> **You:** *"Help me plan a weekend in Chicago — I like museums and deep-dish pizza."*
>
> **Agent:** *"What dates are you planning to visit Chicago?"*
>
> **You:** *"October 30 to November 2."*
>
> **Agent:** *(produces a trip brief with weather, restaurants, attractions, plus a reasoning paragraph explaining why these picks fit your interests and honestly flagging any data gaps.)*

What the agent can currently do:
- Interpret natural-language trip requests and extract structured intent (destination, dates, interests, constraints)
- Ask targeted clarifying questions when information is missing rather than guess
- Remember context across turns within a conversation (e.g. change dates without re-specifying the city)
- Call live APIs for weather (OpenWeatherMap) and places (OpenStreetMap)
- Compose a structured `TripBrief` with reasoning about why picks fit stated interests
- Handle tool failures gracefully — when a data source fails, disclose the gap rather than fabricate

---

## Architecture (Weekend 3)

The agent is a four-node LangGraph `StateGraph` with cross-turn memory:

User query
│
▼
┌────────────────────┐
│ SCOPE │ Extract structured intent, detect missing info
│ (LLM call) │
└─────────┬──────────┘
│
▼
Missing info?
│ │
YES NO
│ │
▼ ▼
┌─────────┐ ┌────────────────────┐
│ ASK │ │ RESEARCH │ Call weather + places tools
│ USER │ │ (ReAct sub-agent) │
└─────────┘ └─────────┬──────────┘
│
▼
┌────────────────────┐
│ SYNTHESIZE │ Compose TripBrief + reasoning
│ (LLM call) │
└─────────┬──────────┘
│
▼
Trip brief
rendered in
chat UI


State flows through nodes as a single `TripState` object. A `MemorySaver` checkpointer persists that state between turns, scoped per browser session, so multi-turn refinement ("actually change the dates to Dec 20–23") works naturally.

---

## Tech stack

- **LLM:** Claude Sonnet 4.5 (Anthropic API)
- **Agent framework:** LangGraph `StateGraph` + LangChain
- **External APIs:** OpenWeatherMap (weather), OpenStreetMap Nominatim + Overpass (places)
- **UI:** Streamlit (chat interface with `st.chat_input` + `st.chat_message`)
- **State validation:** Pydantic (TripBrief, Intent, ScopeResult, SynthesisResult)
- **Persistence:** LangGraph `MemorySaver` (in-memory, per-session)
- **Deployment:** Streamlit Community Cloud
- **Language / runtime:** Python 3.12, virtual environment
- **Secrets management:** `.env` locally, Streamlit secrets in deployment

---

## Known limitations

Honest documentation of what's not yet working, so you have accurate expectations:

- **Follow-up questions re-run the full pipeline.** Ask *"Plan a trip to Chicago"* → get a brief → ask *"what about deep-dish pizza specifically?"* and the agent will re-run scope, research, and synthesize instead of answering from the existing brief. This is a known intent-routing gap; being addressed in Weekend 4.
- **Places data is patchy.** OpenStreetMap coverage for restaurants and attractions varies significantly by city. Chicago returns thin results (Pizano's as the only pizza spot, Money Museum as the only museum); other cities can be worse. The agent honestly flags these gaps in its reasoning, but the underlying data source is the limitation. Weekend 4 will evaluate migrating to a paid places API.
- **Weather is current conditions only.** The OpenWeatherMap free tier doesn't include forecasts, so trips scheduled more than a few days out get "current conditions" caveats. The agent discloses this explicitly.
- **Memory is per-session and in-memory.** State resets on browser refresh and when Streamlit Cloud sleeps the container after inactivity. Documented tradeoff for a portfolio-scope deployment.

None of these are secrets — the app is transparent about them in its own output. Naming them here so anyone browsing the repo has the same picture the app itself gives its users.

---

## Setup (run locally)

### Prerequisites
- Python 3.12+
- An [Anthropic API key](https://console.anthropic.com/)
- An [OpenWeatherMap API key](https://openweathermap.org/api) (free tier works)

### Installation

```bash
git clone https://github.com/abhishekunal/travel-research-agent.git
cd travel-research-agent

python3 -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

Create a `.env` file in the project root:


Run the app:

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

### Testing the agent directly (without UI)

```bash
python agent.py
```

Runs two hardcoded test queries — one that exercises the full graph, one that exercises the clarifying-question path.

---

## Project structure

travel-research-agent/
├── agent.py # StateGraph wiring, MemorySaver, run_agent entry point
├── nodes.py # The four node functions (scope, ask, research, synthesize)
├── tools.py # @tool functions for weather, restaurants, attractions
├── schemas.py # Pydantic models — TripBrief, Intent, TripState, helpers
├── osm_client.py # OpenStreetMap plumbing (Nominatim + Overpass, with mirror failover)
├── guardrails.py # Input validators (currently unused post-chat-UI migration)
├── app.py # Streamlit chat UI + session state + cost guardrail
├── plans/ # Per-weekend plan and build-log files
├── requirements.txt
├── .env # API keys (not committed)
├── .gitignore
└── README.md


---

## Why this project

I'm a technical product manager with 8+ years in fintech, transitioning deeper into applied AI. This project is my hands-on way of learning agentic AI patterns — tool use, orchestration, prompt chaining, memory, deployment — by building rather than just reading. Each weekend's scope is deliberately small so the project stays shippable and the concepts stay learnable.

Per-weekend plans and post-hoc build notes live in `plans/`. If you're curious about the reasoning behind architectural choices — including the ones I rejected — those files are where I documented them.

---