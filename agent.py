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
graph = _graph_builder.compile()


# ──────────────────────────────────────────────────────────────────────
# 3. RUN FUNCTION (used by Streamlit and for direct testing)
# ──────────────────────────────────────────────────────────────────────
def run_agent(user_query: str) -> dict:
    """Invoke the graph with a single user query; return final state fields.

    Returns:
        dict with keys:
          - trip_brief: TripBrief | None (set if scope had enough info)
          - reasoning: str | None (set if synthesize ran)
          - clarifying_question: str | None (set if scope needed more info)

    In the current stateless implementation, each call gets a fresh state
    with no prior conversation history. Memory (Step 3) will change this.
    """
    initial_state: TripState = {
        "messages": [("human", user_query)],
        "intent": None,
        "missing_fields": [],
        "clarifying_question": None,
        "tool_results": {},
        "tool_errors": [],
        "trip_brief": None,
        "reasoning": None,
    }

    final_state = graph.invoke(initial_state)

    return {
        "trip_brief": final_state.get("trip_brief"),
        "reasoning": final_state.get("reasoning"),
        "clarifying_question": final_state.get("clarifying_question"),
    }


# ──────────────────────────────────────────────────────────────────────
# 4. QUICK TEST (runs when this file is executed directly)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Fully specified query (should route scope → research → synthesize)")
    print("=" * 60)
    result = run_agent(
        "Plan a trip to Austin, TX from Dec 15 to Dec 17, 2026. "
        "I'm into live music and BBQ."
    )
    print(f"\nclarifying_question: {result['clarifying_question']}")
    if result["trip_brief"]:
        print(f"\ntrip_brief (JSON):")
        print(result["trip_brief"].model_dump_json(indent=2))
        print(f"\nreasoning:\n{result['reasoning']}")

    print("\n" + "=" * 60)
    print("Test 2: Ambiguous query (should route scope → ask_clarification)")
    print("=" * 60)
    result = run_agent("I want to travel somewhere fun.")
    print(f"\nclarifying_question: {result['clarifying_question']}")
    print(f"trip_brief: {result['trip_brief']}")
    print(f"reasoning: {result['reasoning']}")