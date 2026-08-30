# ✈️ Travel Research Agent

A conversational AI agent for travel research, built as a portfolio project to demonstrate applied agentic AI skills.

**Current status:** Weekend 1 complete — walking skeleton shipped.

---

## Roadmap

- [x] **Weekend 1** — Walking skeleton: LangChain agent, one tool (weather), Streamlit UI
- [ ] **Weekend 2** — Multi-tool orchestration + structured output (attractions, restaurants, Pydantic schemas, input validation)
- [ ] **Weekend 3** — TBD (likely: real conversation memory + polish/deployment)
- [ ] **Weekend 4** — TBD (likely: evaluation harness + advanced patterns like RAG or MCP)

Detailed plans for each weekend live in `WEEKEND_N_PLAN.md` files as they're scoped.
---

## What it does

Ask the agent travel-related questions in natural language and it responds conversationally. It can:

- Fetch live weather for any city (via OpenWeatherMap)
- Answer general travel questions from Claude's built-in knowledge (attractions, culture, logistics, etc.)
- Decide autonomously when to use a tool vs. answer directly

Example queries:
- *"What's the weather in Barcelona right now?"* → calls the weather tool
- *"What are the top 3 things to do in Kyoto?"* → answers from general knowledge
- *"How does Paris weather compare to London?"* → calls the weather tool twice

---

## Tech stack

- **LLM:** Claude (Anthropic API) — decision-making + response generation
- **Agent framework:** LangChain + LangGraph — ReAct-pattern orchestration
- **External API:** OpenWeatherMap — live weather data
- **UI:** Streamlit — browser-based chat interface
- **Language / runtime:** Python 3.12, virtual environment
- **Secrets management:** `.env` file loaded via `python-dotenv`

---

## Architecture

```
User query
    │
    ▼
Streamlit UI (app.py)
    │
    ▼
LangChain agent (agent.py)
    │
    ├──► Claude (Anthropic API) ── decides whether a tool is needed
    │
    └──► get_weather() tool ──► OpenWeatherMap API
              │
              ▼
    Response returned to Claude for final answer
              │
              ▼
    Rendered in Streamlit chat UI
```

Under the hood, this is the **ReAct pattern** (Reasoning + Acting): the LLM alternates between thinking about the query and taking actions (tool calls) until it has enough information to respond.

---

## Setup

### Prerequisites
- Python 3.12+
- macOS, Linux, or Windows with WSL
- An [Anthropic API key](https://console.anthropic.com/)
- An [OpenWeatherMap API key](https://openweathermap.org/api) (free tier works)

### Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/abhishekunal/travel-research-agent.git
   cd travel-research-agent
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS/Linux
   # venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file** in the project root with your API keys:
   ```
   ANTHROPIC_API_KEY=your_anthropic_key_here
   OPENWEATHER_API_KEY=your_openweather_key_here
   ```

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```

   The app will open in your browser at `http://localhost:8501`.

### Testing the agent directly (without UI)

```bash
python agent.py
```

This runs a hardcoded test query and prints the response to the terminal — useful for debugging the agent without touching the UI.

---

## Project structure

```
travel-research-agent/
├── agent.py            # LangChain agent + weather tool + Claude wiring
├── app.py              # Streamlit chat UI
├── requirements.txt    # Pinned Python dependencies
├── .env                # API keys (not committed)
├── .gitignore
└── README.md
```

---

---

## Why this project

I'm a technical product manager with 8+ years in fintech, transitioning deeper into applied AI. This project is my hands-on way of learning agentic AI patterns — tool use, orchestration, structured outputs, evaluation — by building rather than just reading. Each weekend's scope is deliberately small so the project stays shippable and the concepts stay learnable.

Build logs and design decisions are captured per milestone in the `WEEKEND_N_PLAN.md` files.

---
