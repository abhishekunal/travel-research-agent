# ✈️ Travel Research Agent

A conversational AI agent for travel research, built as a portfolio project to demonstrate applied agentic AI skills.

**🔗 Try it live:** [travel-research-agent-abhishekunal.streamlit.app](https://travel-research-agent-abhishekunal.streamlit.app/)

**Current status:** Weekend 4 complete — intent routing, Google Places upgrade, RAG pipeline over a curated 8-city knowledge base, UI polish, and public launch all shipped.

---

## Roadmap

- [x] **Weekend 1** — Walking skeleton: LangChain agent, one tool (weather), Streamlit UI
- [x] **Weekend 2** — Multi-tool orchestration + structured output (attractions, restaurants via OpenStreetMap, Pydantic schemas, input validation)
- [x] **Weekend 3** — Prompt chaining architecture, conversation memory, chat UI, quiet-launch deployment
- [x] **Weekend 4** — Intent router, Google Places migration, RAG pipeline, public launch

Detailed per-weekend plans and build logs live in `plans/WEEKEND_N_PLAN.md`.

---

## What it does

Have a conversation with the agent about a trip. It asks when it needs more information, remembers what you've said across turns, and produces a structured trip brief with reasoning about why it made the picks it did. For a curated set of cities, it also pulls cultural context, logistics, and etiquette advice from a built-in knowledge base.

**A real example flow from the live app:**

> **You:** *"Help me plan a weekend in Chicago — I like museums and deep-dish pizza."*
>
> **Agent:** *"What dates are you planning to visit Chicago?"*
>
> **You:** *"October 30 to November 2."*
>
> **Agent:** *(produces a trip brief with weather, restaurants, attractions, cultural context from the knowledge base, plus a reasoning paragraph explaining why these picks fit your interests and honestly flagging any data gaps.)*
>
> **You:** *"What about dining etiquette in Japan?"*
>
> **Agent:** *(answers from the existing brief + knowledge base without re-running the full pipeline.)*

What the agent can currently do:

- Interpret natural-language trip requests and extract structured intent (destination, dates, interests, constraints)
- Ask targeted clarifying questions when information is missing rather than guess
- Remember context across turns within a conversation (e.g. change dates without re-specifying the city)
- Classify every turn as new-trip, refinement, follow-up, or other — and route accordingly instead of re-running the full pipeline
- Call live APIs for weather (OpenWeatherMap) and places (Google Places New Text Search)
- Retrieve cultural, logistics, and etiquette context from a curated 8-city knowledge base (Tokyo, NYC, Chicago, Barcelona, Paris, Bangkok, Mumbai, Los Angeles)
- Compose a structured `TripBrief` with a reasoning paragraph explaining why picks fit stated interests
- Handle tool failures gracefully — when a data source fails, disclose the gap rather than fabricate

---

## Architecture

The agent is a LangGraph `StateGraph` with cross-turn memory. Weekend 4 added an upstream intent router and a knowledge-base retrieval tool.

```
User query
    │
    ▼
┌──────────────────────┐
│   INTENT ROUTER       │  Classify: new_trip / refinement
│   (LLM call)          │           / follow_up / other
└──────────┬───────────┘
           │
    ┌──────┼──────┬──────────┐
    ▼      ▼      ▼          ▼
new_trip  follow_up         other
refinement│                   │
    │    ▼                   ▼
    │  ┌──────────┐    ┌──────────┐
    │  │ BRIEF QA │    │  OTHER   │
    │  │(LLM call)│    │(deflects)│
    │  └──────────┘    └──────────┘
    ▼
┌──────────────────────┐
│       SCOPE           │  Extract structured intent, detect missing info
│   (LLM call)          │
└──────────┬───────────┘
           │
     Missing info?
           │
    ┌──────┴──────┐
    │             │
   YES            NO
    │             │
    ▼             ▼
┌─────────┐  ┌──────────────────────┐
│  ASK     │  │       RESEARCH        │  Weather + Places + RAG tools
│  USER    │  │   (ReAct sub-agent)   │
└─────────┘  └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │     SYNTHESIZE        │  Compose TripBrief + reasoning
             │   (LLM call)          │
             └──────────┬───────────┘
                        │
                        ▼
                  Trip brief
                  rendered in chat UI
```

State flows through nodes as a single `TripState` object (a TypedDict with LangGraph's `add_messages` reducer). A `MemorySaver` checkpointer persists that state between turns, scoped per browser session, so multi-turn refinement ("actually change the dates to Dec 20–23") works naturally.

---

## Tech stack

- **LLM:** Claude Sonnet 4.5 (Anthropic API)
- **Agent framework:** LangGraph `StateGraph` + LangChain
- **External APIs:** OpenWeatherMap (weather), Google Places (New) Text Search (restaurants, attractions)
- **Knowledge base:** 8 city markdown files chunked and embedded with OpenAI `text-embedding-3-small`, stored in ChromaDB (local, embedded)
- **UI:** Streamlit (chat interface with `st.chat_input` + `st.chat_message`)
- **State validation:** Pydantic (`TripBrief`, `Intent`, `ScopeResult`, `SynthesisResult`, `RouterDecision`, `BriefQAResult`)
- **Persistence:** LangGraph `MemorySaver` (in-memory, per-session)
- **Deployment:** Streamlit Community Cloud
- **Language / runtime:** Python 3.12, virtual environment
- **Secrets management:** `.env` locally, Streamlit secrets in deployment

---

## The RAG pipeline

Added in Weekend 4. The knowledge base covers eight cities with hand-curated markdown — one file per city, 13 sections each (Understand, Best Time to Visit, Neighborhoods, Get in, Get around, See, Do, Day Trips, Eat & Drink, Budget and Costs, Practical Travel Information, Stay safe, Talk).

**Indexing** (one-time, re-run only when corpus changes): `scripts/index_corpus.py` splits each file on `##` headers using LangChain's `MarkdownHeaderTextSplitter`, prefixes each chunk with its city + section, embeds with OpenAI `text-embedding-3-small`, and persists to ChromaDB at `vector_db/`. 104 chunks total, ~25,500 tokens, ~$0.0005 per full re-index.

**Retrieval** (every query): the `search_travel_knowledge` tool is called by the research agent when the destination is in the corpus. The agent passes a natural-language query ("dining etiquette", "best time for cherry blossoms") plus an optional city filter. ChromaDB returns the top-3 nearest-neighbor chunks by cosine similarity. These get synthesized into the final trip brief alongside weather and places data.

**Routing** (when RAG fires): the research agent decides which tools to call based on each tool's docstring plus the research-stage system prompt. For cities not in the corpus (Rome, Istanbul, Milan, Dallas, etc.) the agent skips RAG and transparently notes "City not in the knowledge base" in the output.

A standalone retrieval test (`scripts/test_retrieval.py`) queries the vector DB with sample prompts to sanity-check that retrieval is finding the right chunks before any changes get deployed.

---

## Known limitations

Honest documentation of what's not yet working, so you have accurate expectations:

- **Knowledge base covers 8 cities only.** For destinations outside the corpus, the agent falls back to Claude's general training knowledge and transparently flags "City not in the knowledge base" in the output. Expansion to more cities is deliberately deferred — the architectural pattern is what matters, not corpus size.
- **Weather is current conditions only.** The OpenWeatherMap free tier doesn't include forecasts, so trips scheduled more than a few days out get "current conditions" caveats. The agent discloses this explicitly in the notes.
- **Memory is per-session and in-memory.** State resets on browser refresh and when Streamlit Cloud sleeps the container after inactivity. Documented tradeoff for a portfolio-scope deployment.
- **No API layer.** Streamlit calls the agent in-process. This is a deliberate tradeoff — see "Lessons learned" below.
- **ChromaDB telemetry noise in logs.** ChromaDB 0.6.3 ships broken telemetry events that fail noisily in server logs. Cosmetic; doesn't affect retrieval. `anonymized_telemetry=False` partially suppresses it.

None of these are secrets — the app is transparent about them in its own output. Naming them here so anyone browsing the repo has the same picture the app itself gives its users.

---

## Lessons learned

Candid notes to myself on what shaped this project from a bare-bones ReAct demo to something that actually works end-to-end. Organized by topic, not by weekend.

### Foundations — LLM + tool use

- **Tool docstrings are the routing contract.** The `@tool` docstring is what the LLM reads to pick a tool. "Use when / do NOT use when" phrasing worked better than I expected.
- **Skip-and-note beats hard-fail.** On tool failure, catch and add a note instead of raising. Changes the feel from "broken" to "honest about what it couldn't do."
- **ReAct agents don't play nice with Pydantic structured output.** Had to split it: agent produces prose, second LLM call converts prose to typed data.
- **`default_factory=list` on Pydantic list fields saved me on partial failures.** A failed tool returns empty cleanly instead of missing from the response.

### LangGraph architecture

- **Went with StateGraph over writing my own chaining.** MemorySaver integration and conditional edges came free.
- **TypedDict for graph state, Pydantic for LLM output.** Different jobs, kept them separate.
- **Each node reads state, returns a dict of updates. Nothing else.** Lets me unit test nodes with hand-crafted state.
- **The W3 graph re-ran everything on every turn.** W4 added an intent router upstream — new-trip / refinement / follow-up / other. Architecture evolved once the gap was obvious.

### Conversation & memory

- **When info is missing, pause and ask. Don't guess.** Users are fine waiting; less fine with hallucinated dates.
- **UI render list and checkpointer log have to be separate.** UI needs TripBrief objects for cards; checkpointer holds conversation state. Mixing them bit me.
- **MemorySaver + a documented tradeoff beats SqliteSaver on Streamlit Cloud.** The filesystem is ephemeral; "memory lost on container sleep" is more honest than fake persistence.
- **The intent router made follow-ups actually cheap.** "What about pizza?" doesn't trigger four tool calls anymore.

### RAG pipeline

- **Chunked on `##` markdown headers instead of fixed-size.** Each chunk covers one topic (etiquette, budget, etc.). Fixed-size chunks split ideas and hurt retrieval.
- **Prefixed each chunk with city + section before indexing.** Without it, "Metro single ride: ¥170–320" could be any city.
- **When RAG gets called vs weather vs places is a prompt problem, not a code problem.** The code for all four tools looks identical; the docstring decides routing.

### Deployment & resilience

- **LLMs drop required fields on complex structured output.** Claude kept dropping `reasoning` from `SynthesisResult`. Fix: reorder fields in schema (reasoning first) + move the reasoning rule to #1 in the prompt + `default=""` as a backup.
- **Local secrets are not cloud secrets.** `.env` for local, Streamlit secrets for cloud. Forgot to add OpenAI key to cloud — the failure looked exactly like a code bug for a while.

---

## Setup (run locally)

### Prerequisites

- Python 3.12+
- An [Anthropic API key](https://console.anthropic.com/)
- An [OpenWeatherMap API key](https://openweathermap.org/api) (free tier works)
- An [OpenAI API key](https://platform.openai.com/api-keys) (for RAG embeddings; ~$0.001 to re-index corpus, pennies per month at demo traffic)
- A [Google Cloud API key](https://console.cloud.google.com/apis/credentials) with the Places API (New) enabled

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

```
ANTHROPIC_API_KEY=sk-ant-...
OPENWEATHER_API_KEY=...
OPENAI_API_KEY=sk-proj-...
GOOGLE_PLACES_API_KEY=...
```

Run the app:

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The vector database for RAG is committed to the repo at `vector_db/`, so no indexing step is needed on first run.

### Re-indexing the knowledge base (optional)

If you modify any file in `corpus/` or add new cities, re-generate the vector DB:

```bash
python scripts/index_corpus.py
```

Takes ~15 seconds, costs well under a cent.

### Sanity-checking retrieval (optional)

```bash
python scripts/test_retrieval.py
```

Runs seven sample queries against the vector DB and prints the top-3 nearest chunks for each. Useful after re-indexing to confirm retrieval is finding the right content.

---

## Project structure

```
travel-research-agent/
├── agent.py              # StateGraph wiring, MemorySaver, run_agent entry point
├── nodes.py              # Node functions: intent_router, scope, ask_clarification,
│                         #                 research, synthesize, brief_qa, other_response
├── tools.py              # @tool functions: weather, restaurants, attractions,
│                         #                  search_travel_knowledge (RAG)
├── schemas.py            # Pydantic models: TripBrief, Intent, TripState, RouterDecision,
│                         #                  ScopeResult, SynthesisResult, BriefQAResult
├── places_client.py      # Google Places (New) Text Search API plumbing
├── osm_client.py         # Legacy OSM client (kept for guardrails.py's geocoder)
├── guardrails.py         # Input validators (city, dates, geographic sanity)
├── app.py                # Streamlit chat UI + session state + render cards
├── corpus/               # 8 hand-curated city markdown files (RAG source)
├── scripts/
│   ├── index_corpus.py   # One-time indexer: corpus/ → vector_db/
│   └── test_retrieval.py # Read-only retrieval sanity check
├── vector_db/            # ChromaDB persistent collection (committed)
├── plans/                # Per-weekend plans and build logs
├── requirements.txt
├── .env                  # API keys (not committed)
├── .gitignore
└── README.md
```

---

## Why this project

I'm a technical product manager with 8+ years in fintech, transitioning deeper into applied AI. This project is my hands-on way of learning agentic AI patterns — tool use, orchestration, prompt chaining, memory, RAG, deployment — by building rather than just reading. Each weekend's scope is deliberately small so the project stays shippable and the concepts stay learnable.

Per-weekend plans and post-hoc build notes live in `plans/`. If you're curious about the reasoning behind architectural choices — including the ones I rejected — those files are where I documented them.

---
