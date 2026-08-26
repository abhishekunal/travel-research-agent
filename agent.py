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
    "You are a helpful travel research assistant. "
    "When users ask about weather in a destination, use the get_weather tool. "
    "For other travel questions, answer from your general knowledge. "
    "Keep responses concise and practical for travelers."
)

agent = create_react_agent(
    model=llm,
    tools=[get_weather],
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
# 6. QUICK TEST (only runs if you execute this file directly)
# ---------------------------------------------------------------
if __name__ == "__main__":
    # This block runs when you type: python agent.py
    # It does NOT run when Streamlit imports this file
    print("Testing agent...\n")

    test_query = "What's the weather like in Tokyo right now?"
    print(f"Query: {test_query}")
    print(f"Response:\n{run_agent(test_query)}")