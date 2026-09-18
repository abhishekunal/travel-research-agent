# Weekend 4 Plan — RAG Pipeline + Eval Suite + Public Launch

**Status:** Planned
**Estimated time:** 18–24 hours
**Prerequisite:** Weekend 3 complete (prompt chaining + memory + quiet deployment shipped) ✅

---

## Goal

"Fix the two visible product gaps surfaced during Weekend 3's quiet-launch testing (intent routing, places data quality), add a scoped RAG pipeline grounded in a curated travel knowledge base, and launch the project publicly on LinkedIn with a working URL."

"Framing: Weekend 4 is the 'launch-ready' weekend. Weekend 3's hosted smoke test revealed that follow-up questions re-run the full pipeline (intent router gap) and that even best-case cities like Chicago return thin OSM results (data quality gap). Both are demo-killers for a public launch. This weekend closes those two gaps, adds RAG as the differentiating architectural piece, and ships publicly. Evals are deliberately deferred to Weekend 5+ — a launch that works and looks credible is worth more than a launch with evals but visible bugs."

---

## Scope

### 1. Intent router node (protected — first to build, last to cut)

Add classifier node at top of graph: new-trip / refinement / brief-question / other
Add brief_qa_node that answers follow-up questions from existing state without re-invoking research tools
Update route_after_scope logic (or add a new router before scope)
Test all four intent paths through the hosted app
Why this is protected: Weekend 3 testing confirmed this is what users notice first. A working app is worth more than a sophisticated broken one.

### 2. Places API migration (depends on cost research — see open questions)

Verify current pricing on Google Places and Foursquare before starting
Set hard billing cap ($30 ceiling, verify automated shutdown works)
New places_client.py replacing OSM plumbing
Update search_restaurants and search_attractions — keep return shape identical so downstream stays untouched
If cost research shows neither provider fits the $30 budget: stay on OSM, document the limitation more prominently in the README
Why this matters: OSM's Chicago pizza gap is real and visible on the public URL right now.

### 3. RAG pipeline (scope-reduced from original plan)

Original scope was too ambitious for Weekend 4's expanded workload. Reduce to 5–8 destinations (was 10–15), targeting cities where you can vouch for the content quality personally.
Wikivoyage source (Creative Commons), stored as markdown in corpus/
Chroma (local, embedded) + OpenAI text-embedding-3-small
One-time indexing script; persist the vector DB in the repo
New search_travel_knowledge tool alongside existing tools
Update scope prompt so the agent knows when to reach for RAG vs. structured APIs
What "scope-reduced" means: get the architecture into your portfolio (embeddings, chunking, retrieval), not a comprehensive corpus. The pattern is what matters for the AI PM story.

### 4. Full README overhaul + LinkedIn launch

Live URL prominence, demo GIF or short screen recording, 2–3 static screenshots
Updated architecture diagram reflecting intent router + RAG + memory
Honest "known limitations" section — Weekend 5+ backlog items become visible portfolio strength
Lessons learned section covering the PM-to-engineer journey
LinkedIn post drafted, edited, posted on a weekday morning
Post-launch monitoring plan documented — API cost check, comment responsiveness

### 5. Final polish + friend testing

Send URL to 2–3 friends before public launch
Fix anything embarrassing that surfaces
Verify all cost caps are still holding after places API migration


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

Deferred from Weekend 4: Evaluation suite (DeepEval). Originally planned for this weekend. Deferred to Weekend 5+ so this weekend can fit the intent router and places API work required for a credible public launch. Rationale: a portfolio project that ships with intent routing and thin data quality (OSM) reads worse than one that ships with intent routing, better data, and no evals yet. Evals are highest-value when there's something worth measuring; they belong to the "how do we know it works reliably?" question, which is a Weekend 5+ discipline layer. Naming this cut explicitly rather than silently pretending it wasn't planned.

---

## Sequencing (rough time budget)

Step	Estimated time
Intent router node (scope, code, test)	3–4 hrs
Places API cost research (verify Google/Foursquare pricing, set billing cap)	1 hr
Places API migration (assuming go/no-go decision made)	2–3 hrs
RAG corpus curation (5–8 destinations)	1–2 hrs
RAG indexing + retrieval tool + agent integration	3–4 hrs
RAG end-to-end test + tuning	1–2 hrs
Redeploy to Streamlit Cloud + verify hosted	1–2 hrs
README overhaul + demo GIF + screenshots	3–4 hrs
LinkedIn post drafting + editing	1–2 hrs
Friend testing + final polish	1–2 hrs
Total	17–26 hrs

**Scope-cut plan if I run over:**
First cut: RAG corpus size (ship with 3–5 destinations instead of 5–8)
Second cut: Places API migration deferred to post-launch if cost research is inconclusive; stay on OSM with better README framing
Third cut: Demo GIF (screenshots only for launch, add GIF later)

**Protect at all costs:** RAG working end-to-end, live URL functional, LinkedIn post shipped. Everything else is polish.

---

## Open questions to resolve before starting

Google Places vs. Foursquare pricing — which fits the $30/month ceiling with real cost caps? Verify at source before Weekend 4 begins.
 Wikivoyage API vs. manual download vs. hand-curated markdown — fastest path to a clean corpus?
 OpenAI embeddings vs. Voyage AI — cost/quality tradeoff for this use case?
 Chunking strategy: fixed size vs. semantic vs. structure-aware (by markdown headers)?
 For the LinkedIn post: single post vs. carousel/thread format?

---

## Definition of done

Intent router node classifies turns and routes to appropriate downstream node
 Follow-up questions about existing brief return scoped prose answers, not full TripBrief re-runs
 Places API decision made (migrated OR OSM retained with documented limitation)
 RAG pipeline works end-to-end with 5+ destinations
 Vector database persists correctly in deployed environment
 Agent routes intelligently between structured APIs and RAG
 README fully overhauled with live URL, screenshots, demo GIF, honest limitations
 Live app tested by at least 2 people other than me
 All work committed and pushed to GitHub
 Weekend 4 build log added to notes

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
