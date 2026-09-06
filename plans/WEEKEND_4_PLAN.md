# Weekend 4 Plan — RAG Pipeline + Eval Suite + Public Launch

**Status:** Planned
**Estimated time:** 18–24 hours
**Prerequisite:** Weekend 3 complete (prompt chaining + memory + quiet deployment shipped) ✅

---

## Goal

Add retrieval-augmented generation (RAG) grounded in a curated travel knowledge base, build a lightweight evaluation suite to prove the agent works reliably, and launch the project publicly on LinkedIn with a working URL.

**Framing:** Weekend 4 is the "launch" weekend. Every task ladders up to a public post that positions this project as a serious portfolio piece — not just working, but *provably* working via evals, and *distinctively* designed via RAG + prompt chaining.

---

## Scope

### 1. RAG pipeline over curated travel knowledge base

- **Corpus curation** (10–15 destinations)
  - Source: Wikivoyage articles (Creative Commons-licensed, travel-focused)
  - Mix: popular destinations (Tokyo, Paris, Barcelona, Rome, NYC) + a few interesting ones (Marrakech, Lisbon, Chiang Mai)
  - Store as markdown files in a `corpus/` directory in the repo
  - Clean the content: remove nav junk, focus on cultural context, neighborhood guides, safety, best time to visit

- **Vector database + embedding pipeline**
  - Use **Chroma** (local, embedded, no cloud setup — ideal for portfolio project that deploys)
  - Use **OpenAI's `text-embedding-3-small`** for embeddings (cheap, high quality, well-documented)
  - Alternative: Voyage AI embeddings if I want to stay closer to the Anthropic ecosystem
  - Build a one-time indexing script: chunk corpus → embed → persist to Chroma
  - Persist the vector DB in the repo so it deploys with the app (no external DB dependency)

- **New retrieval tool**
  - Add `search_travel_knowledge` tool alongside existing tools
  - Update system prompt so the agent knows *when* to use RAG (cultural questions, neighborhood advice, "what's it like") vs. structured APIs (weather, current restaurants)
  - Handle edge cases: destination not in corpus, retrieval returns nothing relevant, low similarity scores

**Why this matters:** RAG is the most common enterprise AI pattern right now. Demonstrating you've built one end-to-end — with a real corpus, real chunking decisions, real retrieval tuning — is portfolio-critical for anyone hiring for applied AI roles.

### 2. Evaluation suite with DeepEval

- Install DeepEval and integrate with the agent
- Build a test dataset of 15–20 representative queries covering:
  - **Tool routing accuracy** — does the right tool get called for the right query?
  - **Structured output validity** — does the output conform to the `TripBrief` schema?
  - **Guardrail behavior** — do invalid inputs get rejected gracefully?
  - **RAG grounding** — does the agent use retrieval when it should, and cite the corpus?
  - **Multi-turn memory** — does the agent recall context from prior turns?
- Use **LLM-as-judge** patterns for open-ended quality checks (does the trip brief actually make sense?)
- Produce a simple pass/fail report + latency + cost per query metrics
- Save results to a `evals/results/` directory as JSON or markdown for the README

**Why this matters:** Evals are one of the highest-signal portfolio items in applied AI. Most hobby projects skip them entirely. Having even a basic eval harness signals you understand what production AI actually requires — not just that it runs, but that it works reliably and you can measure regressions.

### 3. Full README overhaul for public launch

- **"Try it live" section at top** with prominent URL + one-line hook
- **Demo GIF or screen recording** — record a 30–60s walkthrough showing a full trip planning flow
- **Screenshots** — 2–3 static shots of the app in action
- **Updated architecture diagram** — reflect prompt chaining stages, memory, RAG, all tools
- **Eval results section** — show the pass rate, key metrics, honest discussion of weaknesses
- **Lessons learned section** — capture the PM-to-engineer journey, biggest surprises, what you'd do differently
- **What's next section** — foreshadow MCP integration as Weekend 5+ direction

**Why this matters:** The README is the first thing a recruiter, hiring manager, or curious LinkedIn scroller will see when they click through from your post. It has to convince someone in 30 seconds that this project is worth their time. Screenshots and a demo GIF do 80% of that work.

### 4. Public LinkedIn launch post

- Draft a post that tells the full 4-weekend story: what you built, why, what you learned, what's next
- Include the live URL, GitHub link, screenshots/demo GIF
- Frame it as: *"I'm a PM in fintech who wanted to learn agentic AI hands-on. Over 4 weekends I built a travel research agent with multi-tool orchestration, structured output, memory, and RAG — and shipped it live. Here's what I learned and where I want to go next."*
- End with an invite to try the app + a hook about what's next (MCP, evals expansion, etc.)
- Post on a weekday morning for maximum engagement

**Why this matters:** This is the whole point of the project as a portfolio piece. If you don't post, the work exists in a vacuum. If you post well, it opens conversations, doors, and future opportunities.

### 5. Final polish + edge case fixes

- Test the live app extensively — send the URL to 2–3 friends before public post
- Fix anything embarrassing that shows up
- Verify cost caps are still holding
- Make sure the eval results are honest, not cherry-picked

---

## Agentic AI concepts learned this weekend

- **Retrieval-augmented generation (RAG)** — grounding LLM responses in external knowledge
- **Embeddings and vector similarity search** — how semantic retrieval actually works
- **Chunking strategies** — how document splitting affects retrieval quality
- **Vector databases (Chroma)** — local vs. hosted, persistence, query patterns
- **Hybrid tool + retrieval routing** — when to use structured APIs vs. semantic search
- **LLM evaluation** — DeepEval, LLM-as-judge patterns, test dataset design
- **Regression testing for AI systems** — how to know when a change breaks something
- **Cost/latency metrics** — measuring what matters in production AI

---

## Success criteria

**RAG works and adds value:**
> Input: *"What's it like to visit Kyoto in autumn?"*
> Behavior: Agent calls `search_travel_knowledge`, retrieves relevant Kyoto corpus chunks, incorporates them into response
> Verifiable: response contains information NOT available from weather/attractions/restaurants APIs

**RAG knows its limits:**
> Input: *"What's the weather in Kyoto?"*
> Behavior: Agent uses weather tool, not RAG (retrieval isn't for real-time data)

**Evals produce credible results:**
> Full test suite runs in under 5 minutes
> Pass rate > 80% on tool routing and structured output tests
> LLM-as-judge scores are documented, not just claimed
> Failures are logged and analyzed, not hidden

**Launch is real:**
> LinkedIn post is published
> Live URL works when clicked from the post
> README is embarrassment-proof
> At least 2 friends have tested the app and given feedback

---

## Out of scope for Weekend 4

Deliberately deferring to Weekend 5+ (if continuing):
- MCP server integration
- Cost estimation feature
- User authentication / saved trips
- Multi-modal (destination images)
- Advanced eval features (adversarial testing, continuous eval pipeline)
- Additional destinations beyond the initial 10–15 in the corpus
- Multi-language support

---

## Sequencing (rough time budget)

| Step | Estimated time |
|---|---|
| Corpus curation (research + clean 10–15 destinations) | 2–3 hrs |
| Chunking + embedding + Chroma indexing script | 2–3 hrs |
| RAG retrieval tool + integration with agent | 3–4 hrs |
| Test RAG end-to-end + tune retrieval quality | 2 hrs |
| DeepEval setup + test dataset design | 2–3 hrs |
| Run evals + iterate on failures | 2–3 hrs |
| Redeploy to Streamlit Cloud + verify RAG works in production | 1–2 hrs |
| README overhaul + demo GIF + screenshots | 3–4 hrs |
| LinkedIn post drafting + editing | 1–2 hrs |
| Friend testing + final polish | 1–2 hrs |
| **Total** | **19–28 hrs** |

**Scope-cut plan if I run over:**
1. First cut: eval suite depth (ship with 8–10 tests instead of 15–20; note in README that this is v1 eval coverage)
2. Second cut: corpus size (ship with 5–8 destinations instead of 10–15; add more post-launch)
3. Third cut (last resort): demo GIF (screenshots only for launch, add GIF later)

**Protect at all costs:** RAG working end-to-end, live URL functional, LinkedIn post shipped. Everything else is polish.

---

## Open questions to resolve before starting

- [ ] Wikivoyage API vs. manual download vs. hand-curated markdown — which is fastest for getting a clean corpus?
- [ ] OpenAI embeddings vs. Voyage AI — cost/quality tradeoff for this use case?
- [ ] Chunking strategy: fixed size vs. semantic chunking vs. structure-aware (by markdown headers)?
- [ ] Does DeepEval work well with LangGraph-based agents, or do I need to wrap the agent for testing?
- [ ] How do I handle the eval scoring for open-ended outputs (LLM-as-judge) — which model do I use as judge?
- [ ] For the LinkedIn post: single post vs. carousel/thread format?

---

## Definition of done

- [ ] RAG pipeline works end-to-end with a curated corpus of 10+ destinations
- [ ] Vector database persists correctly in deployed environment
- [ ] Agent routes intelligently between structured APIs and RAG
- [ ] Eval suite runs and produces documented results
- [ ] README fully overhauled with live URL, screenshots, demo GIF, eval results
- [ ] Live app tested by at least 2 people other than me
- [ ] LinkedIn post published with live URL, GitHub link, and story
- [ ] All work committed and pushed to GitHub
- [ ] Weekend 4 build log added to notes (debugging lessons, decisions, vocabulary)
- [ ] Personal validation: I can point to specific queries where RAG demonstrably improves the answer

---

## Post-launch considerations

Things to think about AFTER the LinkedIn post goes live:

- Monitor API costs for the first week — if the app gets more traffic than expected, tighten caps or add real rate limiting
- Respond to LinkedIn comments and questions (this is where the actual networking value happens)
- Track any bugs users report — the first week post-launch is the best time to fix embarrassing issues
- Consider a follow-up post 2–4 weeks later about lessons learned from real user feedback

---

## The bigger arc — what this project demonstrates by end of Weekend 4

Looking back across all 4 weekends, the project demonstrates:

- ✅ **Weekend 1:** Walking skeleton — LangChain, Claude, tool use, ReAct pattern, Streamlit
- ✅ **Weekend 2:** Multi-tool orchestration, structured output with Pydantic, input validation
- ✅ **Weekend 3:** Prompt chaining architecture, conversation memory, production deployment
- ✅ **Weekend 4:** RAG pipeline, evaluation suite, public launch

For a PM transitioning to applied AI, this hits every major pattern hiring managers look for: agents, tool use, structured outputs, memory, RAG, evals, deployment, and shipping publicly. **It's a legitimately strong portfolio project** — not because any single piece is world-class, but because the arc shows you can scope, plan, build, ship, and communicate.

The MCP work in Weekend 5+ becomes the "and I kept going" chapter — always a good signal to future employers.
