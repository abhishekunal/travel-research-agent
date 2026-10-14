"""
tools.py
========
Tool definitions for the Travel Research Agent.

A "tool" is a Python function the LLM can choose to call when it needs
information it doesn't have (live weather, real places, curated travel
knowledge). The @tool decorator registers each function with LangChain
and turns its docstring into the description the LLM reads to decide
when to call it.

This module owns:
  - Tool functions the agent can invoke
  - The external API glue for the OpenWeatherMap weather endpoint
  - Google Places API calls for restaurants and attractions
  - Chromadb RAG retrieval over the curated travel knowledge base

This module deliberately does NOT know about:
  - The LLM, the agent, or the graph — that's agent.py's job
  - How results are rendered — that's app.py's job

Adding a new tool = add a new @tool function here, then include it in
the tools list where the agent is assembled in nodes.py.
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from langchain_core.tools import tool

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

import places_client


# ── Environment ───────────────────────────────────────────────────────
# load_dotenv() is idempotent — safe to call in every module that needs
# env vars, so this module can be imported standalone without depending
# on agent.py having run first.
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# RAG configuration. Paths are resolved from this file so tools.py can
# be imported from any working directory (streamlit, pytest, repl).
_TOOLS_DIR = Path(__file__).resolve().parent
VECTOR_DB_DIR = _TOOLS_DIR / "vector_db"
COLLECTION_NAME = "travel_knowledge"
EMBEDDING_MODEL = "text-embedding-3-small"

# Exact city names stored as metadata in the vector DB. The LLM uses
# these when passing the `city` filter to search_travel_knowledge — any
# other spelling returns no results (chromadb metadata filters are exact
# match). Keep this in sync with the YAML frontmatter in corpus/*.md.
AVAILABLE_CITIES = [
    "Tokyo", "New York City", "Chicago", "Barcelona",
    "Paris", "Bangkok", "Mumbai", "Los Angeles",
]


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
def search_restaurants(city: str, focus: str | None = None) -> str:
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
        focus: Optional. A cuisine, dish, or dietary need taken from the
               traveler's interests or constraints, in a few words
               (e.g., "deep dish pizza", "vegetarian", "ramen"). Do not
               include the word "restaurants". Omit it if the traveler
               has no food-related interest.

    Returns:
        A formatted list of up to 5 restaurants with name, cuisine, and address.
        Returns an error message if the city can't be found.
    """
    try:
        restaurants = places_client.search_restaurants(city, focus=focus)
        label = f"{focus} restaurants" if focus else "restaurants"
        if not restaurants:
            return f"No {label} found in {city}."

        lines = [f"Search: {label} in {city}"]
        for r in restaurants:
            lines.append(f"- {r['name']} ({r['type']}) at {r['address']}")
        return "\n".join(lines)

    except ValueError:
        return f"Could not find city '{city}'. Please provide a valid city name."
    except Exception as e:
        return f"Restaurant search unavailable right now: {str(e)}"


# ── Attractions tool ──────────────────────────────────────────────────
@tool
def search_attractions(city: str, focus: str | None = None) -> str:
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
        focus: Optional. A sightseeing or activity interest from the
               traveler's interests, in a few words (e.g., "architecture",
               "jazz clubs", "art museums"). Omit it if the traveler has
               no sightseeing-related interest.

    Returns:
        A formatted list of up to 5 attractions with name, type, and address.
        Returns an error message if the city can't be found.
    """
    try:
        attractions = places_client.search_attractions(city, focus=focus)
        label = focus if focus else "attractions"
        if not attractions:
            return f"No {label} found in {city}."

        lines = [f"Search: {label} in {city}"]
        for a in attractions:
            lines.append(f"- {a['name']} ({a['type']}) at {a['address']}")
        return "\n".join(lines)

    except ValueError:
        return f"Could not find city '{city}'. Please provide a valid city name."
    except Exception as e:
        return f"Attractions search unavailable right now: {str(e)}"


# ── Travel knowledge tool (RAG) ───────────────────────────────────────
# Lazy module-level cache for the Chroma collection. Loading it costs a
# second or two (sqlite open + HNSW index mmap + embedding function init)
# so we only pay that once, on the first tool call. Subsequent calls
# reuse the same collection handle.
_collection = None


def _get_collection():
    """Open the Chroma collection on first use, then reuse it.

    Raises:
        FileNotFoundError: if vector_db/ hasn't been built yet.
        ValueError: if OPENAI_API_KEY is missing.
    """
    global _collection
    if _collection is not None:
        return _collection

    if not VECTOR_DB_DIR.exists():
        raise FileNotFoundError(
            f"Vector DB not found at {VECTOR_DB_DIR}. "
            "Run `python scripts/index_corpus.py` first."
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment.")

    # anonymized_telemetry=False silences chromadb 0.6.3's noisy (and
    # broken) telemetry events. Cosmetic — doesn't affect retrieval.
    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL,
    )
    _collection = client.get_collection(
        COLLECTION_NAME, embedding_function=embedding_fn
    )
    return _collection


@tool
def search_travel_knowledge(query: str, city: str | None = None) -> str:
    """Search a curated travel knowledge base for cultural context,
    local customs, logistics advice, and when-to-go guidance about a city.

    Use this tool when the user asks about:
    - Cultural context, local customs, or etiquette (dining, dress, tipping)
    - When the best time of year to visit a city is, or weather patterns
      across seasons (as opposed to CURRENT weather — use get_weather for that)
    - Budget and typical daily spend
    - How to get around a city (transit systems, driving advice)
    - Day trips or weekend getaways from a city
    - Neighborhoods, districts, or arrondissements and their character
    - Safety advice and common scams
    - Language and communication tips
    - General orientation ("what's Tokyo like?", "what should I know about Mumbai?")

    Do NOT use this tool for:
    - CURRENT weather (use get_weather)
    - Specific restaurant venues or where to eat tonight (use search_restaurants)
    - Specific attractions or landmarks to visit (use search_attractions)
    - Cities NOT in the available list below — this tool returns nothing useful
      for those, so skip it and rely on general knowledge instead.

    AVAILABLE CITIES (use these exact spellings when passing `city`):
      Tokyo, New York City, Chicago, Barcelona,
      Paris, Bangkok, Mumbai, Los Angeles

    Args:
        query: A natural-language description of what the user wants to know.
               Can be a question or a phrase. Examples:
               - "dining etiquette and tipping customs"
               - "best time to visit for cherry blossoms"
               - "how to get around the city"
               - "pickpocketing warnings"
               - "weekend day trips"
        city: Optional. One of the AVAILABLE CITIES above (exact spelling).
              Pass this when the user asked about a specific city — it
              narrows the search to that city's content. Omit it for
              general questions that could apply across cities or when
              the user hasn't named a city.

    Returns:
        Up to 3 relevant passages from the knowledge base, each labeled
        with the city and section it came from. If no good match is found,
        returns a short message saying so.
    """
    try:
        collection = _get_collection()

        # Chromadb's `where` clause is exact-match. Only pass it when we
        # actually have a city — otherwise None means "search all cities."
        where = {"city": city} if city else None

        results = collection.query(
            query_texts=[query],
            n_results=3,
            where=where,
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        if not docs:
            if city:
                return (
                    f"No travel knowledge found for '{query}' in {city}. "
                    f"The city may not be in the knowledge base — "
                    f"available cities: {', '.join(AVAILABLE_CITIES)}."
                )
            return f"No travel knowledge found for '{query}'."

        # Format for the LLM: labeled sections, body text, blank line
        # between. The chunks already include a "# City\n## Section\n\n"
        # prefix from indexing, but we strip that and add our own cleaner
        # header so the LLM sees city/section at a glance.
        lines = [f"Travel knowledge for query: '{query}'", ""]
        for doc, meta in zip(docs, metas):
            body = doc.split("\n\n", 1)[1] if "\n\n" in doc else doc
            lines.append(f"=== {meta['city']} / {meta['section']} ===")
            lines.append(body.strip())
            lines.append("")
        return "\n".join(lines).rstrip()

    except FileNotFoundError as e:
        return f"Travel knowledge base is not available: {e}"
    except Exception as e:
        return f"Travel knowledge search unavailable right now: {e}"
