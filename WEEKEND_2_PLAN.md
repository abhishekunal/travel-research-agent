# Weekend 2 Plan — Multi-tool Orchestration + Structured Output

**Status:** Planned
**Estimated time:** 10–15 hours (may extend to 15–20 if debugging is heavy)
**Prerequisite:** Weekend 1 walking skeleton complete ✅

---

## Goal

Evolve the walking skeleton from a single-tool agent into a real multi-tool orchestrator that returns structured, validated output — the foundation of a production-quality travel research agent.

---

## Scope

### 1. Add two new tools
- **Attractions lookup** — returns top attractions for a given city
- **Restaurants lookup** — returns notable restaurants for a given city
- **API choice:** Google Places (single API covers both) OR Yelp Fusion for restaurants + OpenTripMap for attractions
  - *Leaning Google Places for the consolidated-vendor story and richer data, pending credit card / billing setup friction*

### 2. Multi-tool orchestration
- Refactor the agent so it dynamically decides which of the 3 tools (weather + attractions + restaurants) to call based on user intent
- No hardcoded sequences — the LLM should route autonomously
- A single query like *"Plan a weekend in Austin"* should trigger all three tool calls; a query like *"What's the weather in Tokyo?"* should trigger only one

### 3. Structured JSON output via Pydantic
- Define a `TripBrief` schema with fields: `destination`, `dates`, `weather_summary`, `attractions[]`, `restaurants[]`, `notes`
- Force the agent to return this schema instead of free-form text
- **Design first, code second:** write the Pydantic model before touching any tool code, so tools have a clear target

### 4. Input validation guardrails
- Cities must resolve to real locations
- Dates must be in the future
- Trip length must be 1–14 days
- Invalid inputs return friendly, actionable error messages — not stack traces

### 5. Streamlit UI upgrade
- Render structured output as sections/cards, not a wall of text
- Layout: destination header → weather block → attractions list → restaurants list → notes
- Retain conversation history from Weekend 1

---

## Agentic AI concepts I'll learn

- **Multi-tool orchestration** — how LLMs route between multiple available tools
- **Tool description design** — how docstring quality affects tool selection accuracy
- **Structured output** — forcing LLMs to conform to a schema (Pydantic + LangChain integration)
- **Input guardrails** — validating user inputs before they reach the LLM
- **Prompt engineering for tool selection** — system prompts that shape routing behavior

---

## Success criteria

**Happy path:**
> Input: `"Austin, TX"` + `"Dec 15–17"`
> Behavior: agent makes 3 tool calls (weather, attractions, restaurants)
> Output: structured `TripBrief` with 5 attractions, 5 restaurants, weather summary
> UI: rendered as clean sections/cards in Streamlit

**Guardrail path:**
> Input: `"Trip to Atlantis"`
> Output: graceful error message ("Sorry, I couldn't find that destination. Try a real city name.")

**Selective routing:**
> Input: `"What's the weather in Denver?"`
> Behavior: agent calls only the weather tool, not the others

---

## Out of scope for Weekend 2

Deliberately deferring the following to later weekends:
- Flight and hotel APIs (Weekend 4 or later)
- RAG / vector databases (Weekend 4)
- MCP integrations (Weekend 4)
- Automated evaluations (Weekend 4)
- Deployment to Streamlit Cloud (Weekend 3)
- Conversation memory across turns (Weekend 3)

---

## Sequencing (rough time budget)

| Step | Estimated time |
|---|---|
| API onboarding (Google Places or Yelp) + smoke test in Python REPL | 2–3 hrs |
| Design Pydantic schemas (`TripBrief`, `Attraction`, `Restaurant`, `DateRange`) | 1 hr |
| Build 2 new tool functions with `@tool` decorators | 1–2 hrs |
| Wire structured output through the agent (LangChain's structured output pattern) | 3–4 hrs |
| Input validation guardrails | 1–2 hrs |
| Streamlit UI refactor for card-based rendering | 2–3 hrs |
| Debugging + iteration | 2–3 hrs |
| **Total** | **12–18 hrs** |

**Scope-cut plan if I run over:** drop input validation guardrails first (defer to Weekend 3). They're the most polish-adjacent item and don't block the core multi-tool + structured output learning goal.

---

## Open questions to resolve before starting

- [ ] Google Places vs. Yelp Fusion — which API am I actually using?
- [ ] Does LangChain's `create_react_agent` support structured output natively, or do I need a different agent constructor?
- [ ] How do I handle the case where one tool fails mid-orchestration? Retry, skip, or fail the whole request?

---

## Definition of done

- [ ] All 3 tools registered with the agent and callable
- [ ] Agent routes autonomously (verified with 5+ test queries)
- [ ] Output conforms to `TripBrief` Pydantic schema
- [ ] Invalid inputs return friendly errors
- [ ] Streamlit UI renders structured output as sections
- [ ] `README.md` roadmap updated with Weekend 2 checkbox ticked
- [ ] All work committed and pushed to GitHub
- [ ] Weekend 2 build log added to notes (debugging lessons, decisions, vocabulary)