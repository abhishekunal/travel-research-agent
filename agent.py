"""
Travel Research Agent — LangGraph StateGraph
=============================================
This module owns:
  - The compiled LangGraph StateGraph that orchestrates the four-stage
    workflow: scope → (ask_clarification | research → synthesize)
  - The router function for the conditional edge after scope
  - run_agent(): the entry point the app calls to invoke the graph

Nodes live in nodes.py. Tools live in tools.py. External data plumbing
lives in osm_client.py.

Weekend 3 note: the graph runs stateless in this sub-step. Memory
(MemorySaver + thread-scoped state) is added in Step 3 of the plan.
"""

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from schemas import TripState
from nodes import (
    scope_node,
    ask_clarification_node,
    research_node,
    synthesize_node,
)


load_dotenv()


# ──────────────────────────────────────────────────────────────────────
# 1. ROUTER for the conditional edge after scope
# ──────────────────────────────────────────────────────────────────────
# LangGraph's add_conditional_edges() takes a function that inspects state
# and returns the name of the next node to run. This is where the graph's
# ONE piece of runtime logic lives — everything else is static wiring.
def route_after_scope(state: TripState) -> str:
    """Route to research if scope has enough info; otherwise ask user."""
    if state["missing_fields"]:
        return "ask_clarification"
    return "research"


# ──────────────────────────────────────────────────────────────────────
# 2. BUILD AND COMPILE THE GRAPH
# ──────────────────────────────────────────────────────────────────────
# Same construction pattern every LangGraph app uses:
#   1) Instantiate StateGraph with the state shape (TripState)
#   2) Add each node function under a string name
#   3) Add edges (static) and conditional edges (via router functions)
#   4) Compile — this validates the graph and returns an executable
_graph_builder = StateGraph(TripState)

_graph_builder.add_node("scope", scope_node)
_graph_builder.add_node("ask_clarification", ask_clarification_node)
_graph_builder.add_node("research", research_node)
_graph_builder.add_node("synthesize", synthesize_node)

_graph_builder.add_edge(START, "scope")
_graph_builder.add_conditional_edges(
    "scope",
    route_after_scope,
    {
        "ask_clarification": "ask_clarification",
        "research": "research",
    },
)
_graph_builder.add_edge("ask_clarification", END)
_graph_builder.add_edge("research", "synthesize")
_graph_builder.add_edge("synthesize", END)

# Compile once at module load. `graph` is the executable object the app
# invokes. In Step 3 of the plan, .compile() will get a checkpointer=MemorySaver()
# argument to enable persistent state across turns.
# Compile with a MemorySaver checkpointer so state persists between
# invocations of the same thread_id. This is what enables multi-turn
# conversations: each turn is a separate graph.invoke() call, but they
# share state because they share thread_id.
#
# In-memory storage means state is lost when the Python process ends
# (Streamlit Cloud sleeping the container, local restart, etc.). That's
# an accepted tradeoff for this deployment — see WEEKEND_3_PLAN.md.
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
            Callers (Streamlit, tests, etc.) issue this. Same thread_id
            across calls = same conversation, state persists. Different
            thread_id = fresh conversation.

    Returns:
        dict with keys:
          - trip_brief: TripBrief | None (set if synthesize ran this turn)
          - reasoning: str | None (set if synthesize ran this turn)
          - clarifying_question: str | None (set if scope needed more info)

    Note: on turns after the first, most of the state fields we pass here
    are effectively overrides — but the checkpointer will merge them with
    whatever state was saved for this thread_id at the end of the last turn.
    We only pass the fields we want to *update* this turn (the new message);
    everything else is reloaded from the checkpoint.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # On the first turn, thread has no saved state — the checkpointer
    # treats the input as the starting state. On subsequent turns, the
    # checkpointer merges this input with the prior saved state (the
    # add_messages reducer appends our new human message to the existing
    # conversation history).
    turn_input = {"messages": [("human", user_query)]}

    final_state = graph.invoke(turn_input, config=config)

    return {
        "trip_brief": final_state.get("trip_brief"),
        "reasoning": final_state.get("reasoning"),
        "clarifying_question": final_state.get("clarifying_question"),
    }


# ──────────────────────────────────────────────────────────────────────
# 4. QUICK TEST (runs when this file is executed directly)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uuid

    # ── Test 1: single-turn queries with fresh threads ──
    print("=" * 60)
    print("Test 1: Complete query, fresh thread")
    print("=" * 60)
    thread_a = str(uuid.uuid4())
    result = run_agent(
        "Plan a trip to Austin, TX from Dec 15 to Dec 17, 2026. "
        "I'm into live music and BBQ.",
        thread_id=thread_a,
    )
    print(f"clarifying_question: {result['clarifying_question']}")
    if result["trip_brief"]:
        print(f"destination: {result['trip_brief'].destination}")
        print(f"reasoning (first 200 chars): {result['reasoning'][:200]}...")

    print("\n" + "=" * 60)
    print("Test 2: Ambiguous query, fresh thread — should ask for info")
    print("=" * 60)
    thread_b = str(uuid.uuid4())
    result = run_agent("I want to travel somewhere fun.", thread_id=thread_b)
    print(f"clarifying_question: {result['clarifying_question']}")
    print(f"trip_brief: {result['trip_brief']}")

    # ── Test 3: multi-turn conversation on ONE thread ──
    # This is the crux — turn 2 must see turn 1's context via checkpointer.
    print("\n" + "=" * 60)
    print("Test 3: Multi-turn on one thread (memory recall)")
    print("=" * 60)
    thread_c = str(uuid.uuid4())

    print("\n--- Turn 1: ambiguous, should ask a question ---")
    r1 = run_agent("Plan a trip to Barcelona.", thread_id=thread_c)
    print(f"clarifying_question: {r1['clarifying_question']}")
    print(f"trip_brief: {r1['trip_brief']}")

    print("\n--- Turn 2: answer the question, same thread ---")
    r2 = run_agent(
        "I'm going Dec 20 to Dec 23, 2026. Love architecture.",
        thread_id=thread_c,
    )
    print(f"clarifying_question: {r2['clarifying_question']}")
    if r2["trip_brief"]:
        print(f"destination: {r2['trip_brief'].destination}")
        print(f"start_date: {r2['trip_brief'].start_date}")
        print(f"end_date: {r2['trip_brief'].end_date}")
        print(f"reasoning (first 200 chars): {r2['reasoning'][:200]}...")

    # ── Test 4: thread isolation ──
    # Two threads running the same query should be independent — thread D
    # should NOT see thread_c's Barcelona context.
    print("\n" + "=" * 60)
    print("Test 4: Thread isolation")
    print("=" * 60)
    thread_d = str(uuid.uuid4())
    print(f"\n--- New thread, asking a follow-up-shaped question ---")
    r_iso = run_agent(
        "What's the weather like there?", thread_id=thread_d
    )
    print(f"clarifying_question: {r_iso['clarifying_question']}")
    print("(Should ask WHERE — should NOT assume Barcelona from thread_c)")