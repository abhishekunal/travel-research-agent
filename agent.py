"""
Travel Research Agent — LangGraph StateGraph
=============================================
This module owns:
  - The compiled LangGraph StateGraph that orchestrates the workflow:
      intent_router → (scope → (research → synthesize | ask_clarification))
                    → brief_qa
                    → other_response
  - Router functions for the conditional edges
  - run_agent(): the entry point the app calls to invoke the graph

Nodes live in nodes.py. Tools live in tools.py. External data plumbing
lives in osm_client.py.

Weekend 4 note: intent_router became the graph entry point. scope is no
longer entered directly from START — the router decides whether the turn
warrants a full pipeline run (new_trip, refinement), a scoped answer from
existing state (follow_up), or a canned deflection (other).
"""

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from schemas import TripState
from nodes import (
    intent_router_node,
    scope_node,
    ask_clarification_node,
    research_node,
    synthesize_node,
    brief_qa_node,
    other_response_node,
)


load_dotenv()


# ──────────────────────────────────────────────────────────────────────
# 1. ROUTERS for the conditional edges
# ──────────────────────────────────────────────────────────────────────
# Two conditional edges now:
#   route_after_intent_router — picks scope / brief_qa / other_response
#   route_after_scope         — picks ask_clarification / research
#
# Both are pure functions of state. They contain the ONLY runtime logic
# in the graph — every other edge is static wiring.
def route_after_intent_router(state: TripState) -> str:
    """Pick the next node based on the router's classification.

    Special case: when the previous turn ended with a clarifying question,
    the router bypasses classification (see intent_router_node). In that
    case we route to scope regardless of the placeholder 'route' value.
    """
    # The bypass path: intent_router_node cleared pending_clarification
    # this turn, but if the ROUTE field is our sentinel from that bypass
    # (or if state signals it another way), we still want scope.
    # Simplest way: the bypass sets route to "new_trip" as a placeholder,
    # but we can detect the bypass by checking whether the router actually
    # ran a classification. In our implementation the bypass short-circuits
    # BEFORE the LLM call, and always returns route="new_trip". Because
    # "new_trip" also legitimately routes to scope, we don't need a
    # special case here — both paths land at scope.
    route = state.get("route")

    if route == "follow_up":
        return "brief_qa"
    if route == "other":
        return "other_response"
    # new_trip and refinement both go to scope (and so does the bypass)
    return "scope"


def route_after_scope(state: TripState) -> str:
    """Route to research if scope has enough info; otherwise ask user."""
    if state["missing_fields"]:
        return "ask_clarification"
    return "research"


# ──────────────────────────────────────────────────────────────────────
# 2. BUILD AND COMPILE THE GRAPH
# ──────────────────────────────────────────────────────────────────────
_graph_builder = StateGraph(TripState)

# Register every node under a string name. Router is new for Weekend 4;
# brief_qa and other_response are also new.
_graph_builder.add_node("intent_router", intent_router_node)
_graph_builder.add_node("scope", scope_node)
_graph_builder.add_node("ask_clarification", ask_clarification_node)
_graph_builder.add_node("research", research_node)
_graph_builder.add_node("synthesize", synthesize_node)
_graph_builder.add_node("brief_qa", brief_qa_node)
_graph_builder.add_node("other_response", other_response_node)

# Entry point is now the intent router, not scope.
_graph_builder.add_edge(START, "intent_router")

# Conditional edge: router → scope / brief_qa / other_response
_graph_builder.add_conditional_edges(
    "intent_router",
    route_after_intent_router,
    {
        "scope": "scope",
        "brief_qa": "brief_qa",
        "other_response": "other_response",
    },
)

# The scope → research | ask_clarification branch is untouched from W3.
_graph_builder.add_conditional_edges(
    "scope",
    route_after_scope,
    {
        "ask_clarification": "ask_clarification",
        "research": "research",
    },
)

# Terminal edges. Three nodes now end a graph invocation:
#   ask_clarification — waiting for user's answer
#   synthesize        — brief delivered
#   brief_qa          — follow-up answered
#   other_response    — canned deflection delivered
_graph_builder.add_edge("ask_clarification", END)
_graph_builder.add_edge("research", "synthesize")
_graph_builder.add_edge("synthesize", END)
_graph_builder.add_edge("brief_qa", END)
_graph_builder.add_edge("other_response", END)

# Compile with a MemorySaver checkpointer so state persists between
# invocations of the same thread_id. See Weekend 3 plan for the
# ephemeral-filesystem tradeoff on Streamlit Cloud.
_checkpointer = MemorySaver()
graph = _graph_builder.compile(checkpointer=_checkpointer)


# ──────────────────────────────────────────────────────────────────────
# 3. RUN FUNCTION (used by Streamlit and for direct testing)
# ──────────────────────────────────────────────────────────────────────
def run_agent(user_query: str, thread_id: str) -> dict:
    """Invoke the graph with a user query on a given conversation thread.

    Args:
        user_query: The latest user message.
        thread_id: An opaque identifier that groups turns into a conversation.

    Returns:
        dict with keys:
          - trip_brief: TripBrief | None — set if synthesize ran this turn
          - reasoning: str | None — set if synthesize ran this turn
          - clarifying_question: str | None — set if scope needed more info
          - assistant_message: str | None — set if brief_qa or other_response
                               ran (a prose reply to render as a plain chat
                               bubble, no card stack)
          - route: str | None — router's classification (for logging/debug)

    The UI branches on which of these fields is populated. exactly one of
    {trip_brief, clarifying_question, assistant_message} should be set on
    any given turn.
    """
    config = {"configurable": {"thread_id": thread_id}}
    turn_input = {"messages": [("human", user_query)]}

    final_state = graph.invoke(turn_input, config=config)

    # For brief_qa / other_response: the node wrote an AIMessage to
    # messages. Pull the last one as the turn's assistant reply. We only
    # surface it if this turn's route was follow_up or other — otherwise
    # the last AIMessage might be a clarifying question we've already
    # surfaced via clarifying_question, and returning it twice would
    # double-render.
    assistant_message = None
    route = final_state.get("route")
    if route in ("follow_up", "other"):
        messages = final_state.get("messages", [])
        if messages:
            last = messages[-1]
            # last will be an AIMessage from brief_qa or other_response
            assistant_message = getattr(last, "content", None)

    return {
        "trip_brief": final_state.get("trip_brief"),
        "reasoning": final_state.get("reasoning"),
        "clarifying_question": final_state.get("clarifying_question"),
        "assistant_message": assistant_message,
        "route": route,
    }


# ──────────────────────────────────────────────────────────────────────
# 4. QUICK TEST (runs when this file is executed directly)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uuid

    def _header(title: str) -> None:
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)

    def _summarize(result: dict) -> None:
        print(f"  route: {result['route']}")
        if result["clarifying_question"]:
            print(f"  clarifying_question: {result['clarifying_question']}")
        if result["assistant_message"]:
            preview = result["assistant_message"][:200]
            print(f"  assistant_message: {preview}"
                  f"{'...' if len(result['assistant_message']) > 200 else ''}")
        if result["trip_brief"]:
            b = result["trip_brief"]
            print(f"  trip_brief: {b.destination}, {b.start_date} → {b.end_date}")
            print(f"    restaurants: {len(b.restaurants)}, attractions: {len(b.attractions)}")
        if result["reasoning"]:
            print(f"  reasoning: {result['reasoning'][:150]}...")

    # ── Test 1: single-turn complete query ──
    _header("Test 1: Complete new-trip query, fresh thread")
    thread_a = str(uuid.uuid4())
    r = run_agent(
        "Plan a trip to Austin, TX from Dec 15 to Dec 17, 2026. "
        "I'm into live music and BBQ.",
        thread_id=thread_a,
    )
    _summarize(r)

    # ── Test 2: ambiguous, should ask for info ──
    _header("Test 2: Ambiguous new-trip, fresh thread")
    thread_b = str(uuid.uuid4())
    r = run_agent("I want to travel somewhere fun.", thread_id=thread_b)
    _summarize(r)

    # ── Test 3: clarification loop, one thread ──
    _header("Test 3: Multi-turn clarification (bypass path)")
    thread_c = str(uuid.uuid4())
    print("\n--- Turn 1: ambiguous ---")
    r1 = run_agent("Plan a trip to Barcelona.", thread_id=thread_c)
    _summarize(r1)
    print("\n--- Turn 2: answer the question ---")
    r2 = run_agent(
        "Dec 20 to Dec 23, 2026. Love architecture.",
        thread_id=thread_c,
    )
    _summarize(r2)

    # ── Test 4: follow-up question flow ──
    _header("Test 4: New trip + follow-up on brief")
    thread_d = str(uuid.uuid4())
    print("\n--- Turn 1: plan a Chicago trip ---")
    r_plan = run_agent(
        "Plan a weekend in Chicago Oct 3-5, 2026. I like museums.",
        thread_id=thread_d,
    )
    _summarize(r_plan)
    print("\n--- Turn 2: follow-up about the existing brief ---")
    r_qa = run_agent(
        "Which of those restaurants is closest to downtown?",
        thread_id=thread_d,
    )
    _summarize(r_qa)

    # ── Test 5: refinement flow ──
    _header("Test 5: New trip + refinement (destination swap)")
    thread_e = str(uuid.uuid4())
    print("\n--- Turn 1: plan an Austin trip ---")
    r_a = run_agent(
        "Plan Austin Dec 15-17, 2026. Live music and BBQ.",
        thread_id=thread_e,
    )
    _summarize(r_a)
    print("\n--- Turn 2: swap to Denver ---")
    r_d = run_agent(
        "Actually let's do Denver instead.",
        thread_id=thread_e,
    )
    _summarize(r_d)
    # Weekend 4 Step 5 check: refinement should produce an acknowledgment
    # prefix at the start of the reasoning field.
    reasoning = r_d.get("reasoning") or ""
    prefix_ok = "updated" in reasoning.lower()[:100] or "revised" in reasoning.lower()[:100]
    marker = "✅" if prefix_ok else "❌"
    print(f"  {marker} refinement acknowledgment prefix present: {prefix_ok}")
    print(f"     reasoning starts with: {reasoning[:120]}...")

    # ── Test 6: other-response flow ──
    _header("Test 6: Off-topic / other")
    thread_f = str(uuid.uuid4())
    r_o = run_agent("What's the stock market doing today?", thread_id=thread_f)
    _summarize(r_o)

    # ── Test 7: thread isolation still works ──
    _header("Test 7: Thread isolation (fresh thread, no context)")
    thread_g = str(uuid.uuid4())
    r_iso = run_agent(
        "What's the weather like there?", thread_id=thread_g
    )
    _summarize(r_iso)
    print("(Router should route to scope; scope should ask WHERE.)")