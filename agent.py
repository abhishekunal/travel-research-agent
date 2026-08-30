"""
Travel Research Agent — Weekend 1 Walking Skeleton
===================================================
A minimal LangChain agent that uses Claude as the LLM
and can look up current weather via OpenWeatherMap.

Architecture (what happens when a user asks a question):
  1. User query comes in (e.g., "What's the weather in Tokyo?")
  2. LangChain sends the query + tool descriptions to Claude
  3. Claude decides: do I need a tool, or can I answer directly?
  4. If Claude picks the weather tool → LangChain calls OpenWeatherMap
  5. The API response goes back to Claude for a natural-language summary
  6. Claude's final answer is returned to the caller
"""

import os
import requests
from dotenv import load_dotenv
import osm_client
from datetime import date
from schemas import TripBrief

# --- LangChain imports ---
# ChatAnthropic: the bridge between LangChain and Claude's API
from langchain_anthropic import ChatAnthropic

# @tool: a decorator that turns any Python function into a "tool"
# that LangChain can offer to the LLM
from langchain_core.tools import tool

# create_react_agent: builds a ReAct agent — the pattern where the LLM
# loops through Reasoning → Acting → Observing until it has an answer
from langgraph.prebuilt import create_react_agent


# ---------------------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------------------
# load_dotenv() reads your .env file and puts ANTHROPIC_API_KEY,
# OPENWEATHER_API_KEY, etc. into os.environ so the code can use them
# without hardcoding secrets.
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


# ---------------------------------------------------------------
# 2. DEFINE THE WEATHER TOOL
# ---------------------------------------------------------------
# The @tool decorator does two things:
#   a) Registers this function so LangChain can offer it to Claude
#   b) Uses the docstring as the tool description — Claude reads this
#      to decide WHEN to call the tool and WHAT arguments to pass
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Use this when the user asks
    about weather, temperature, or conditions in a specific location.
    Pass the city name as a string, e.g. 'Tokyo' or 'New York'."""

    # Call the OpenWeatherMap API
    # - 'q' is the city name
    # - 'appid' is your API key
    # - 'units=metric' gives Celsius (the API defaults to Kelvin)
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # raises an error for 4xx/5xx status codes
        data = response.json()

        # Pull out the fields we care about from the API response
        city_name = data["name"]
        country = data["sys"]["country"]
        temp_c = data["main"]["temp"]
        temp_f = round(temp_c * 9 / 5 + 32, 1)  # convert for US users
        feels_like_c = data["main"]["feels_like"]
        feels_like_f = round(feels_like_c * 9 / 5 + 32, 1)
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]

        # Return a formatted string — Claude will use this raw text
        # to compose a natural-sounding answer for the user
        return (
            f"Weather in {city_name}, {country}:\n"
            f"  Condition: {description}\n"
            f"  Temperature: {temp_c}°C ({temp_f}°F)\n"
            f"  Feels like: {feels_like_c}°C ({feels_like_f}°F)\n"
            f"  Humidity: {humidity}%"
        )

    except requests.exceptions.HTTPError as e:
        return f"Could not find weather for '{city}'. Check the city name. Error: {e}"
    except requests.exceptions.RequestException as e:
        return f"Weather API request failed: {e}"

@tool
def search_restaurants(city: str) -> str:
    """Find restaurants in a specific city for travel planning.
    
    Use this tool when the user asks about:
    - Places to eat or dining options in a destination
    - Restaurant recommendations for a trip
    - Food or cuisine available in a city they're visiting
    
    Do NOT use this tool for:
    - Weather questions (use get_weather instead)
    - Sightseeing or things to do (use search_attractions instead)
    
    Args:
        city: The city to search in. Include state or country for clarity
              (e.g., "Austin, TX", "Paris, France", "Tokyo").
    
    Returns:
        A formatted list of up to 5 restaurants with name, cuisine, and address.
        Returns an error message if the city can't be found.
    """
    try:
        restaurants = osm_client.search_restaurants(city)
        if not restaurants:
            return f"No restaurants found in {city}."
        
        lines = [f"Restaurants in {city}:"]
        for r in restaurants:
            lines.append(f"- {r['name']} ({r['type']}) at {r['address']}")
        return "\n".join(lines)
    
    except ValueError:
        # Nominatim couldn't geocode the city — it's not a real place
        return f"Could not find city '{city}'. Please provide a valid city name."
    except Exception as e:
        # Any other failure (network, timeout, etc.) — skip and note
        return f"Restaurant search unavailable right now: {str(e)}"


@tool
def search_attractions(city: str) -> str:
    """Find tourist attractions, museums, and landmarks in a specific city.
    
    Use this tool when the user asks about:
    - Things to do or see in a destination
    - Sightseeing, tourist spots, or points of interest
    - Museums, galleries, landmarks, or scenic viewpoints
    
    Do NOT use this tool for:
    - Weather questions (use get_weather instead)
    - Restaurants or food (use search_restaurants instead)
    
    Args:
        city: The city to search in. Include state or country for clarity
              (e.g., "Austin, TX", "Paris, France", "Tokyo").
    
    Returns:
        A formatted list of up to 5 attractions with name, type, and address.
        Returns an error message if the city can't be found.
    """
    try:
        attractions = osm_client.search_attractions(city)
        if not attractions:
            return f"No attractions found in {city}."
        
        lines = [f"Attractions in {city}:"]
        for a in attractions:
            lines.append(f"- {a['name']} ({a['type']}) at {a['address']}")
        return "\n".join(lines)
    
    except ValueError:
        return f"Could not find city '{city}'. Please provide a valid city name."
    except Exception as e:
        return f"Attractions search unavailable right now: {str(e)}"


# ---------------------------------------------------------------
# 3. INITIALIZE THE LLM
# ---------------------------------------------------------------
# ChatAnthropic automatically reads ANTHROPIC_API_KEY from env.
# model: which Claude model to use
# temperature: 0 = deterministic (same input → same output),
#   higher = more creative/varied. 0 is good for tool-use agents.
llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
    temperature=0,
)


# ---------------------------------------------------------------
# 4. CREATE THE AGENT
# ---------------------------------------------------------------
# create_react_agent wires everything together:
#   - The LLM (Claude) decides what to do
#   - The tools list tells it what actions are available
#   - The system prompt sets the agent's personality/role
#
# "ReAct" = Reasoning + Acting. The agent can loop multiple times
# (think → use tool → read result → think again) before giving
# a final answer.
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
# 5. RUN FUNCTION (used by Streamlit and for direct testing)
# ---------------------------------------------------------------
def run_agent(user_query: str) -> str:
    """Send a query to the agent and return the final text response."""

    # invoke() runs the full ReAct loop and returns a dict
    # with a "messages" key containing the conversation history
    result = agent.invoke({"messages": [("human", user_query)]})

    # The last message in the list is Claude's final answer
    return result["messages"][-1].content

# ---------------------------------------------------------------
# 6. STRUCTURED OUTPUT VIA POST-PROCESSING
# ---------------------------------------------------------------
# Stage 2 of the two-stage pattern: take the agent's free-form prose
# and convert it into a TripBrief object.
#
# We call Claude a second time with a strict instruction to produce
# only JSON. Pydantic then parses and validates that JSON.
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
    # We use the same LLM but bind it to the TripBrief schema.
    # with_structured_output() tells Claude "return output that matches
    # this Pydantic model" — LangChain handles the JSON schema conversion
    # under the hood.
    structured_llm = llm.with_structured_output(TripBrief)

    # The prompt has one job: extract fields from the two inputs.
    # We're not asking Claude to think or plan here — just to format.
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
# 7. QUICK TEST (only runs if you execute this file directly)
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("Testing agent with structured output...\n")

    test_query = "Plan a trip to Austin, TX from Dec 15 to Dec 17, 2026. What's the weather, where should I eat, and what should I see?"
    print(f"Query: {test_query}\n")

    # Stage 1: get the prose response
    prose_response = run_agent(test_query)
    print("── Prose response ──")
    print(prose_response)

    # Stage 2: convert to structured TripBrief
    print("\n── Structured TripBrief ──")
    trip_brief = to_trip_brief(test_query, prose_response)
    print(trip_brief.model_dump_json(indent=2))