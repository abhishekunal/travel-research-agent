"""
nodes.py
========
Node functions for the Travel Research Agent's LangGraph StateGraph.

Each node is a plain Python function with the signature:
    (state: TripState) -> dict

It reads whatever it needs from state, does its work (LLM call, tool call,
logic), and returns a dict of state updates. LangGraph merges the returned
dict into the shared state and hands it to the next node.

The four nodes:
    scope_node               — extract intent from conversation, detect gaps
    ask_clarification_node   — write clarifying question into messages, pause
    research_node            — invoke ReAct agent to gather tool data
    synthesize_node          — compose final TripBrief with reasoning

Graph wiring (which node runs after which) lives in agent.py, not here.
"""

# NOTE: LangGraph 1.0 deprecated create_react_agent and recommends
# `from langchain.agents import create_agent`. On this project's current
# package versions those imports have a broken transitive dependency
# (langchain-core removed the memory submodule the older langchain
# package still expects). Keeping the stable pre-deprecation import
# until the ecosystem versions align. Revisit when upgrading LangChain.
from langgraph.prebuilt import create_react_agent
from tools import get_weather, search_restaurants, search_attractions
from datetime import date
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage

from schemas import Intent, ScopeResult, TripState


load_dotenv()

# One shared LLM instance for the whole module. temperature=0 gives us
# deterministic routing — same input, same extraction, same question.
llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    temperature=0,
)


# ──────────────────────────────────────────────────────────────────────
# Node 1: SCOPE
# ──────────────────────────────────────────────────────────────────────
SCOPE_PROMPT = """You are the SCOPE stage of a travel research agent. Your \
job is to read the conversation so far and produce a structured trip intent.

Extract these fields:
- city: destination city (include country/state only if the user did)
- start_date: trip start date, as YYYY-MM-DD
- end_date: trip end date, as YYYY-MM-DD
- interests: themes the user mentioned (architecture, food, nightlife, etc.)
- constraints: requirements or restrictions (vegetarian, budget, mobility, etc.)

Required fields are: city, start_date, end_date.
If any required field is missing or ambiguous, DO NOT guess. Instead:
1. Leave that field as null in the intent
2. Add its name to missing_fields
3. Write ONE natural-sounding clarifying_question that asks about the missing \
information as concisely as possible. If multiple fields are missing, ask \
about them together in one question.

If all required fields are extracted with confidence, set missing_fields to \
an empty list and clarifying_question to null.

Today's date is {today}. If the user gave a relative date like "next weekend" \
or an ambiguous date like "Dec 15-17" without a year, resolve it against today.
"""


def scope_node(state: TripState) -> dict:
    """Extract structured intent from the conversation; detect missing info."""
    prompt = SCOPE_PROMPT.format(today=date.today().isoformat())

    structured_llm = llm.with_structured_output(ScopeResult)
    result: ScopeResult = structured_llm.invoke(
        [("system", prompt), *_to_lc_messages(state["messages"])]
    )

    return {
        "intent": result.intent,
        "missing_fields": result.missing_fields,
        "clarifying_question": result.clarifying_question,
    }

# ──────────────────────────────────────────────────────────────────────
# Node 2: ASK CLARIFICATION
# ──────────────────────────────────────────────────────────────────────
def ask_clarification_node(state: TripState) -> dict:
    """Append the clarifying question to messages as an assistant turn.

    Terminal node for this graph invocation — the graph run ends here and
    waits for the user's next message. When the user responds, a new graph
    invocation begins, re-entering at scope_node with full history.
    """
    question = state["clarifying_question"]

    # Defensive: if this node is ever reached without a question set, don't
    # crash — surface it as a bug in a way the user can see. Should never
    # happen once the graph routing is wired correctly.
    if not question:
        question = "Could you tell me more about your trip?"

    return {"messages": [AIMessage(content=question)]}
# ──────────────────────────────────────────────────────────────────────
# Node 3: RESEARCH
# ──────────────────────────────────────────────────────────────────────
RESEARCH_PROMPT = """You are the RESEARCH stage of a travel research agent. \
You have access to three tools:
  - get_weather: current weather in a city
  - search_restaurants: dining options in a city
  - search_attractions: sightseeing, museums, and landmarks in a city

Given a structured trip intent, call ALL three tools for the destination \
city. Do not skip any tool. If a tool fails or returns no results, note it \
and move on — the downstream synthesis stage will handle partial data \
gracefully.

Report your findings as a clear, structured summary. Do not editorialize, \
plan the trip, or make recommendations — that is the synthesis stage's job. \
Your job is only to gather.
"""

# Build the ReAct agent once at module load — reused across all invocations.
_research_agent = create_react_agent(
    model=llm,
    tools=[get_weather, search_restaurants, search_attractions],
    prompt=RESEARCH_PROMPT,
)

def research_node(state: TripState) -> dict:
    """Invoke the ReAct agent to gather tool data for the scoped destination."""
    intent = state["intent"]

    # Build a compact instruction to the agent that includes just the fields
    # it needs. We're translating structured intent → prose because the ReAct
    # agent takes conversational input, not typed inputs.
    agent_instruction = (
        f"Gather travel information for {intent.city} "
        f"from {intent.start_date} to {intent.end_date}."
    )
    if intent.interests:
        agent_instruction += f" The traveler is interested in: {', '.join(intent.interests)}."
    if intent.constraints:
        agent_instruction += f" Constraints to respect: {', '.join(intent.constraints)}."

    result = _research_agent.invoke(
        {"messages": [("human", agent_instruction)]}
    )

    # The agent's final message contains the full research summary.
    # We stash it in tool_results under a single key; synthesize will
    # parse what it needs.
    final_message = result["messages"][-1].content

    return {
        "tool_results": {"research_summary": final_message},
        "tool_errors": [],  # skip-and-note errors are inside the summary text
    }
# ──────────────────────────────────────────────────────────────────────
# Node 4: SYNTHESIZE
# ──────────────────────────────────────────────────────────────────────
SYNTHESIZE_PROMPT = """You are the SYNTHESIZE stage of a travel research \
agent. You have a structured trip intent and raw research findings. Your \
job is to compose a final trip brief that a real traveler could use.

Rules:

1. Fill every field in the TripBrief schema from the research summary. \
Extract restaurants and attractions as Place objects. Preserve names and \
addresses exactly as given.

2. For weather_summary: report what the research shows, but frame it \
correctly. If the trip is more than a few days out, the weather data is \
CURRENT conditions in the destination — not a forecast. Say so plainly \
(e.g. "Current conditions in Austin are..."). Do NOT invent forecasts.

3. In the notes field, capture: any tool that failed, any constraint that \
couldn't be verified against the available data, and any relevant caveats \
about the picks (seasonal issues, missing addresses, etc.).

4. In the reasoning field, write 2-4 sentences explaining WHY these \
specific picks fit the traveler's stated interests and constraints. If \
the research didn't turn up enough to confidently match a constraint, \
say so honestly. This field is what makes your brief useful vs. a bare \
list — it should read like advice from a thoughtful friend, not a \
summary of the data.

Today's date is {today}."""


def synthesize_node(state: TripState) -> dict:
    """Compose the final TripBrief and reasoning from intent + research."""
    from schemas import SynthesisResult  # local import to avoid circular deps

    intent = state["intent"]
    research_summary = state["tool_results"].get("research_summary", "")
    tool_errors = state["tool_errors"]

    prompt = SYNTHESIZE_PROMPT.format(today=date.today().isoformat())

    user_content = (
        f"TRIP INTENT:\n"
        f"  City: {intent.city}\n"
        f"  Dates: {intent.start_date} to {intent.end_date}\n"
        f"  Interests: {', '.join(intent.interests) if intent.interests else '(none stated)'}\n"
        f"  Constraints: {', '.join(intent.constraints) if intent.constraints else '(none stated)'}\n"
        f"\n"
        f"RESEARCH SUMMARY:\n{research_summary}\n"
        f"\n"
        f"TOOL ERRORS (if any):\n{tool_errors if tool_errors else '(none)'}"
    )

    structured_llm = llm.with_structured_output(SynthesisResult)
    result: SynthesisResult = structured_llm.invoke(
        [("system", prompt), ("human", user_content)]
    )

    return {
        "trip_brief": result.trip_brief,
        "reasoning": result.reasoning,
    }
# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _to_lc_messages(messages: list) -> list:
    """Convert LangGraph message objects into the tuple format ChatAnthropic
    accepts. Handles both raw LangChain message objects (from add_messages
    reducer) and pre-tupled inputs (from tests). Idempotent."""
    converted = []
    for m in messages:
        if isinstance(m, tuple):
            converted.append(m)
        else:
            # LangChain message object — HumanMessage, AIMessage, etc.
            role = "human" if m.type == "human" else "ai"
            converted.append((role, m.content))
    return converted