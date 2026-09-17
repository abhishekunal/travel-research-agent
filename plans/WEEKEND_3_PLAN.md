# Weekend 3 Plan — Prompt Chaining + Memory + Quiet Deployment

**Status:** In progress — Steps 1–5 complete, Steps 6–8 remaining
**Estimated time:** 16–22 hours
**Prerequisite:** Weekend 2 complete (multi-tool orchestration + structured output shipped) ✅

---

## Goal

Evolve the agent architecture from a single-loop ReAct pattern to a multi-stage workflow with prompt chaining, add cross-turn conversation memory, and quietly deploy the app to Streamlit Community Cloud so it's live and battle-tested before the Weekend 4 public launch.

**Framing:** Weekend 3 is the "invisible foundations" weekend. The user-visible changes are modest (memory, better multi-turn behavior), but the internals get significantly more sophisticated. This weekend builds the platform on which Weekend 4's RAG + launch will sit.

---

## Scope

### 1. Refactor to prompt chaining (scope → research → synthesize)

Replace the single-loop ReAct agent with an explicit 3-stage workflow:

- **Stage 1 — Scope:** LLM interprets the user's request and extracts structured intent (destination, dates, interests, constraints). If information is missing or ambiguous, agent asks clarifying questions instead of guessing.
- **Stage 2 — Research:** Agent orchestrates tool calls (weather, attractions, restaurants) based on scoped intent. This is where the current ReAct logic lives, but now with cleaner input.
- **Stage 3 — Synthesize:** A dedicated final LLM call composes the structured `TripBrief` output with reasoning about *why* certain recommendations were chosen.

Each stage has its own prompt, its own responsibilities, and can be tested in isolation.

**Why this matters:** Most portfolio AI agents are single-loop ReAct. Building an explicit multi-stage workflow demonstrates you understand agent architecture beyond the tutorial default. It also produces better output — the synthesis stage can reason across all tool results holistically instead of accumulating them incrementally.

### 2. Cross-turn conversation memory

- Use LangGraph's built-in `MemorySaver` checkpointer for state persistence
- Add a session/thread ID concept — each Streamlit user gets their own conversation state
- Verify multi-turn refinement flows work:
  - *"Plan a trip to Barcelona Dec 15–17"* → *"What about vegetarian restaurants?"* (no city repeat needed)
  - *"Change the dates to Dec 20–22"* → agent should update existing brief, not start over
  - *"What was the weather again?"* → agent recalls prior tool result without re-calling
- Handle edge cases: memory reset button, memory overflow (what happens after 20 turns?)

**Why this matters:** Stateless agents are toys. Real agents remember. This is one of the highest-leverage user experience improvements you can make.

### 3. UI polish for demo-quality experience

- Empty-state prompt gallery: *"Try: Plan a weekend in Austin"* clickable examples
- Tool-call transparency: show *which tool* is running ("Looking up weather... Searching attractions...")
- Better error rendering — card-style errors instead of raw stack traces
- Sidebar additions: "About this project" section, link to GitHub, memory reset button
- Optional: "Export as JSON" button on the trip brief (portfolio flex — makes structured output visible)

**Why this matters:** The difference between "cool project" and "someone would actually use this" is polish. Every UI improvement compounds when someone lands on your live URL cold.

### 4. Deploy to Streamlit Community Cloud (quiet launch)

- Sign up at streamlit.io/cloud (free tier, GitHub-connected)
- Configure Streamlit secrets management (API keys move from `.env` to Streamlit's secrets UI)
- Handle environment differences (file paths, package versions, memory persistence in deployed vs local)
- Get a live URL like `travel-research-agent-abhishekunal.streamlit.app`
- **Do NOT announce on LinkedIn yet** — this is a private launch to shake out deployment bugs before Weekend 4's public post

**Why deploy this weekend instead of Weekend 4:** Deployment always breaks things in unexpected ways (paths, secrets, dependencies, memory persistence). Discovering those bugs a week before your public launch — not the night before — is the whole point.

### 5. Cost protection guardrails

- Set spending caps on Anthropic and OpenWeatherMap accounts (both platforms support this)
- Add lightweight per-session query counter in `session_state` (soft limit of e.g. 10 queries per session with a friendly "please refresh to continue" message)
- Not real rate limiting (out of scope) — just enough to prevent accidental cost blowouts from a link being shared

**Why this matters:** Once your app is public, anyone can spam it. Cost protection before public launch is non-negotiable.

### 6. README updates (light touch, save the big overhaul for Weekend 4)

- Add "Live URL" line at top (even though not publicly announced yet)
- Update architecture diagram to include memory + prompt chaining stages
- Roadmap checkbox: Weekend 3 ✅

---

## Agentic AI concepts learned this weekend

- **Prompt chaining / multi-stage workflows** — explicit decomposition of agent tasks into stages
- **Stateful agent design** — persisting conversation state across turns
- **LangGraph checkpointers** — LangGraph's memory persistence primitives
- **Thread-scoped memory** — session isolation for multi-user apps
- **Production deployment** — secrets management, environment separation, cloud deployment workflows
- **Cost engineering** — spending caps, session-level rate limiting

---

## Success criteria

**Prompt chaining works:**
> Input: *"Plan a trip to Tokyo"* (ambiguous — no dates)
> Behavior: Stage 1 asks *"When are you planning to go and for how long?"* instead of guessing
> After user responds: Stages 2 and 3 run cleanly with full context

**Memory works:**
> Input 1: *"Plan a weekend in Austin Dec 15–17"*
> Input 2: *"What about vegetarian restaurants?"*
> Behavior: Agent uses Austin + those dates from prior turn without re-asking

**Deployment works:**
> Anyone with the URL can load the app and use it
> API keys are not exposed anywhere in the deployed code or logs
> Memory persists within a session but doesn't leak across users

**Cost protection works:**
> Session query counter hits limit and shows friendly message
> Spending caps configured on both API dashboards

---

## Out of scope for Weekend 3

Deliberately deferring to Weekend 4:
- RAG pipeline
- Evaluation suite (DeepEval)
- Public LinkedIn launch
- Big README overhaul with screenshots + demo GIF

Deferring beyond Weekend 4:
- MCP server integration (Weekend 5+)
- Cost estimation feature (Weekend 5+ product feature, not agentic AI learning)
- User authentication / persistent accounts
- Multi-modal (images of destinations)

---

## Sequencing (rough time budget)

| Step | Estimated time |
|---|---|
| Design prompt chaining architecture (sketch stages, prompts, data flow) | 1–2 hrs |
| Refactor agent.py into 3 stages with clean interfaces | 4–5 hrs |
| Add LangGraph MemorySaver + thread management | 3–4 hrs |
| UI polish (empty state, tool transparency, error rendering, sidebar) | 3–4 hrs |
| Deploy to Streamlit Cloud + debug environment issues | 2–3 hrs |
| Cost guardrails (spending caps + session counter) | 1 hr |
| Light README updates | 1 hr |
| Debugging + iteration | 2–3 hrs |
| **Total** | **17–23 hrs** |

## Scope-cut plan — none triggered

The plan protected prompt chaining as the highest-value item, and it shipped clean. Memory shipped. UI polish shipped (with tool-call transparency and JSON export deliberately skipped as low-value/high-effort). No cuts were needed.

---

## Open questions — resolved during Step 1

- **LangGraph graph builder vs. manual chaining?** → `StateGraph`, chosen because MemorySaver integrates natively and conditional edges are first-class primitives for scope's clarifying-question routing.
- **`MemorySaver` vs. `SqliteSaver`?** → `MemorySaver`. Streamlit Cloud has an ephemeral filesystem, so `SqliteSaver` doesn't buy real persistence there. Documented tradeoff: memory lost when the container sleeps.
- **Testing deployment without public URL?** → Deploy as public but share with 2–3 trusted people. Streamlit Cloud URLs have no discovery mechanism; a URL not shared isn't findable.
- **App sleep memory behavior?** → Memory is lost on sleep. Accepted; document in README. Weekend 5+ backlog: detect sleep on wake and show "Previous context cleared due to inactivity" banner.

---

## Definition of done

- [x] Agent refactored into scope → research → synthesize stages
- [x] Multi-turn conversations work with verifiable memory recall
- [ ] Live URL exists on Streamlit Cloud (not yet publicized)
- [x] Spending caps configured on Anthropic + OpenWeatherMap dashboards
- [x] Session-level query counter in place
- [x] UI has empty state, tool transparency, and friendly error rendering (tool-call transparency deferred to Weekend 4+)
- [ ] README updated with live URL and new architecture
- [ ] All work committed and pushed to GitHub - Steps 1–5 pushed; Steps 6–7 remaining
- [x] Weekend 3 build log added to notes (debugging lessons, decisions, vocabulary)
- [ ] Personal validation: I can send the URL to a friend and they can plan a trip without confusion

---

## Real-world learnings surfaced during the build

Things that only became visible from actually building and testing:

- **Pre-existing OWM bug found during refactor testing.** OpenWeatherMap's `q` parameter accepts `City` or `City,CountryCode` but 404s on `City,StateCode`. Skip-and-note masked this since Weekend 1. Fixed by stripping everything after the first comma. A reminder that graceful degradation can hide real bugs — diagnostic runs matter.

- **Deprecated `create_react_agent` kept intentionally.** The recommended replacement (`langchain.agents.create_agent`) has a broken transitive dependency in the current package set. Documented the choice in code with a NOTE comment. Revisit when the ecosystem stabilizes.

- **Intent router gap identified from real use.** After Step 5, testing surfaced that follow-up questions about an existing brief ("what about Beverly Hills?") re-run the full pipeline instead of narrowing scope. This is "Fork B" from Weekend 2 notes, now with a concrete symptom. Weekend 5+ backlog.

- **OSM data quality varies by city.** Chicago tests returned rich results (real museums, real restaurants with addresses). Dallas and Tokyo tests returned patchy results. This is an OSM data completeness issue, not an agent issue. Weekend 5+ backlog: migrate to Google Places or Foursquare (paid).

- **Weather is current-only.** OWM's free `/weather` endpoint returns current conditions, not forecasts. Synthesize node discloses this honestly ("current, not December forecast"). Weekend 5+ backlog: `/onecall` (paid) or alternative API.

- **msgpack serialization warning logged.** Future LangGraph will require explicit type registration for custom Pydantic types in checkpointer state. Working today with a deprecation warning; Weekend 5+ backlog.

---


## Note on the "quiet launch" strategy

The rationale for deploying but not announcing this weekend:

1. **Deployment surfaces bugs local development hides.** File paths differ. Secrets management differs. Persistence differs. You want a full week to find and fix these bugs.
2. **Weekend 4's LinkedIn post is stronger with RAG included.** A live URL with just memory reads as "solid tutorial project." A live URL with memory + prompt chaining + RAG reads as "sophisticated agent architecture."
3. **Risk mitigation.** If Weekend 4 hits problems with RAG or evals, you still have a working live app to fall back on. You can publicly launch the memory-only version and defer RAG to Weekend 5. Your launch isn't blocked on any single component.

This is how real product launches work — deploy quietly, test in production, announce when it's solid.
