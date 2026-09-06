"""
tools.py
========
Tool definitions for the Travel Research Agent.

A "tool" is a Python function the LLM can choose to call when it needs
information it doesn't have (live weather, real places). The @tool decorator
registers each function with LangChain and turns its docstring into the
description the LLM reads to decide when to call it.

This module owns:
  - Tool functions the agent can invoke
  - The external API glue for the OpenWeatherMap weather endpoint
  - Delegation to osm_client.py for OpenStreetMap-backed tools

This module deliberately does NOT know about:
  - The LLM, the agent, or the graph — that's agent.py's job
  - How results are rendered — that's app.py's job

Adding a new tool = add a new @tool function here, then include it in
the tools list where the agent is assembled in agent.py.
"""

import os
import requests
from dotenv import load_dotenv

from langchain_core.tools import tool

import osm_client


# ── Environment ───────────────────────────────────────────────────────
# load_dotenv() is idempotent — safe to call in every module that needs
# env vars, so this module can be imported standalone without depending
# on agent.py having run first.
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


# ── Weather tool ──────────────────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Use this when the user asks
    about weather, temperature, or conditions in a specific location.
    Pass the city name as a string, e.g. 'Tokyo' or 'New York'."""

    # OWM's `q` parameter accepts "City" or "City,CountryCode" (ISO 3166),
    # but NOT "City,StateCode" like "Austin, TX". Strip anything after the
    # first comma so US-style "City, State" inputs resolve correctly.
    city = city.split(",")[0].strip()

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }


    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        city_name = data["name"]
        country = data["sys"]["country"]
        temp_c = data["main"]["temp"]
        temp_f = round(temp_c * 9 / 5 + 32, 1)
        feels_like_c = data["main"]["feels_like"]
        feels_like_f = round(feels_like_c * 9 / 5 + 32, 1)
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]

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


# ── Restaurant tool ───────────────────────────────────────────────────
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
        return f"Could not find city '{city}'. Please provide a valid city name."
    except Exception as e:
        return f"Restaurant search unavailable right now: {str(e)}"


# ── Attractions tool ──────────────────────────────────────────────────
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