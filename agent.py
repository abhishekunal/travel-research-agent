"""
Travel Research Agent — LLM + ReAct agent + structured output
==============================================================
This module owns:
  - The LLM (Claude via ChatAnthropic)
  - The ReAct agent that wires the LLM to tools
  - The two-stage structured output post-processing (to_trip_brief)

Tools live in tools.py. External data plumbing lives in osm_client.py.

(Weekend 3 note: this module is about to grow into a LangGraph StateGraph
with scope/research/synthesize nodes. The tool extraction was a preparatory
cleanup so the graph refactor happens on a tidier starting file.)
"""

from datetime import date

from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from schemas import TripBrief
from tools import get_weather, search_restaurants, search_attractions


# ---------------------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------------
# 2. INITIALIZE THE LLM
# ---------------------------------------------------------------
# temperature=0 gives deterministic outputs — same input → same output.
# Good for tool-use agents where we want reliable routing decisions.
llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    temperature=0,
)


# ---------------------------------------------------------------
# 3. CREATE THE AGENT
# ---------------------------------------------------------------
# ReAct = Reasoning + Acting. The agent loops (think → tool → observe)
# until it has enough to give a final answer.
SYSTEM_PROMPT = (
    "You are a helpful travel research assistant with access to three tools:\n"
    "  - get_weather: for current weather in any city\n"
    "  - search_restaurants: for dining recommendations in any city\n"
    "  - search_attractions: for sightseeing, museums, and landmarks in any city\n"
    "\n"
    "When a user asks about a destination, use these tools to gather live data "
    "rather than relying on your training knowledge. For trip planning questions "
    "that touch multiple topics (weather AND food AND attractions), call all "
    "relevant tools before answering. Always call get_weather when a destination "
    "is mentioned, even for future trips — current conditions give useful context. "
    "Keep final responses concise and practical."
)

agent = create_react_agent(
    model=llm,
    tools=[get_weather, search_restaurants, search_attractions],
    prompt=SYSTEM_PROMPT,
)


# ---------------------------------------------------------------
# 4. RUN FUNCTION (used by Streamlit and for direct testing)
# ---------------------------------------------------------------
def run_agent(user_query: str) -> str:
    """Send a query to the agent and return the final text response."""
    result = agent.invoke({"messages": [("human", user_query)]})
    return result["messages"][-1].content


# ---------------------------------------------------------------
# 5. STRUCTURED OUTPUT VIA POST-PROCESSING
# ---------------------------------------------------------------
# Stage 2 of the two-stage pattern: take the agent's free-form prose
# and convert it into a validated TripBrief object.
def to_trip_brief(user_query: str, agent_response: str) -> TripBrief:
    """Convert the agent's prose response into a structured TripBrief.

    Args:
        user_query: The original user question (has dates, destination)
        agent_response: The agent's prose answer (has weather + place data)

    Returns:
        A validated TripBrief object.

    Raises:
        pydantic.ValidationError: if the LLM's JSON doesn't match the schema.
    """
    structured_llm = llm.with_structured_output(TripBrief)

    formatting_prompt = f"""Convert the following travel research response into a structured trip brief.

Original user question:
{user_query}

Agent's research findings:
{agent_response}

Instructions:
- Extract destination, start_date, and end_date from the user's question.
- Use today's date as context if the user gave a year-less date like "Dec 15-17".
- Summarize the weather findings into weather_summary.
- Extract each restaurant and attraction as a Place with name, type, and address.
- If the agent noted any data was unavailable, add that to notes.
- If no restaurants or attractions were found, leave those lists empty.

Today's date is {date.today().isoformat()}."""

    return structured_llm.invoke(formatting_prompt)


# ---------------------------------------------------------------
# 6. QUICK TEST (only runs if you execute this file directly)
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("Testing agent with structured output...\n")

    test_query = "Plan a trip to Austin, TX from Dec 15 to Dec 17, 2026. What's the weather, where should I eat, and what should I see?"
    print(f"Query: {test_query}\n")

    prose_response = run_agent(test_query)
    print("── Prose response ──")
    print(prose_response)

    print("\n── Structured TripBrief ──")
    trip_brief = to_trip_brief(test_query, prose_response)
    print(trip_brief.model_dump_json(indent=2))