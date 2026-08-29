"""
osm_client.py

Helper module for OpenStreetMap data.

Responsibility: talk to OSM's public APIs (Nominatim + Overpass) and return
clean, structured Python lists. Callers (the agent tools) never see raw
JSON responses, HTTP details, or Overpass QL syntax.

If we ever swap OSM for another provider (Google Places, Foursquare, etc.),
only this file changes — the agent tools import the same function names
and get back the same shapes.

Public functions:
    geocode_city(city)          → (lat, lon)
    search_restaurants(city)    → list of restaurant dicts
    search_attractions(city)    → list of attraction dicts

Each place dict has a stable shape:
    {"name": str, "type": str, "address": str}
"""

import requests

# ── Endpoints ─────────────────────────────────────────────────────────
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Overpass has multiple public mirrors. We try them in order until one
# responds successfully. This is a standard "failover" resilience pattern:
# no single server is a guaranteed point of failure. The main endpoint is
# most authoritative but often busiest; the Kumi mirror is community-run
# and frequently faster during peak hours.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# ── Etiquette: identify yourself ──────────────────────────────────────
# Nominatim requires a User-Agent identifying your app. This is how OSM's
# public services enforce fair-use policy — no keys, but they track callers.
HEADERS = {
    "User-Agent": "TravelResearchAgent/1.0 (portfolio project)"
}

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_RADIUS_M = 5000    # 5 km — reasonable for city-level search
DEFAULT_LIMIT = 5          # Enough for the agent to work with, not overwhelming
NOMINATIM_TIMEOUT = 10     # Geocoding should be fast
OVERPASS_TIMEOUT = 30      # Overpass can be slow; be patient


# ── Internal helpers ──────────────────────────────────────────────────

def _format_place(element: dict) -> dict:
    """
    Convert a raw Overpass 'element' into our clean dict shape.
    
    OSM data is user-contributed, so many fields are optional. We provide
    sensible fallbacks so downstream code never crashes on missing keys.
    """
    tags = element.get("tags", {})
    
    name = tags.get("name", "(unnamed)")
    
    # 'type' is a generic label — cuisine for restaurants, tourism/historic
    # for attractions. We check keys in priority order.
    place_type = (
        tags.get("cuisine")
        or tags.get("tourism")
        or tags.get("historic")
        or tags.get("amenity")
        or "unknown"
    )
    
    # Address is spread across multiple tags; stitch what's available
    street = tags.get("addr:street", "")
    housenumber = tags.get("addr:housenumber", "")
    address = f"{housenumber} {street}".strip() or "No address on file"
    
    return {
        "name": name,
        "type": place_type,
        "address": address
    }


def _query_overpass(overpass_query: str) -> list[dict]:
    """
    Run an Overpass QL query and return the list of raw elements.
    
    Tries each endpoint in OVERPASS_URLS in order. If one returns a
    timeout or server error, falls through to the next. Only raises
    if every endpoint fails — that way, transient issues on one server
    don't break the agent.
    """
    last_error = None
    
    for url in OVERPASS_URLS:
        try:
            response = requests.post(
                url,
                data={"data": overpass_query},
                headers=HEADERS,
                timeout=OVERPASS_TIMEOUT
            )
            response.raise_for_status()
            return response.json().get("elements", [])
        except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            # Server was slow or overloaded — try the next endpoint
            last_error = e
            continue
    
    # If we got here, every endpoint failed
    raise RuntimeError(f"All Overpass endpoints failed. Last error: {last_error}")


# ── Public functions ──────────────────────────────────────────────────

def geocode_city(city: str) -> tuple[float, float]:
    """
    Convert a city name like "Austin, TX" into (latitude, longitude).
    
    Raises ValueError if the city can't be found — this is useful for
    input validation later (we'll use it as our "is this a real place?" check).
    """
    params = {"q": city, "format": "json", "limit": 1}
    response = requests.get(
        NOMINATIM_URL,
        params=params,
        headers=HEADERS,
        timeout=NOMINATIM_TIMEOUT
    )
    response.raise_for_status()
    
    results = response.json()
    if not results:
        raise ValueError(f"Could not find location: '{city}'")
    
    return float(results[0]["lat"]), float(results[0]["lon"])


def search_restaurants(city: str, limit: int = DEFAULT_LIMIT) -> list[dict]:
    """
    Find restaurants in the given city.
    Returns a list of {"name", "type", "address"} dicts.
    """
    lat, lon = geocode_city(city)
    
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="restaurant"](around:{DEFAULT_RADIUS_M},{lat},{lon});
    );
    out center {limit};
    """
    
    elements = _query_overpass(query)
    return [_format_place(el) for el in elements]


def search_attractions(city: str, limit: int = DEFAULT_LIMIT) -> list[dict]:
    """
    Find tourist attractions in the given city.
    Returns a list of {"name", "type", "address"} dicts.
    
    Searches multiple tag types — attractions, museums, galleries, viewpoints —
    because "attraction" is a fuzzier concept in OSM than "restaurant."
    """
    lat, lon = geocode_city(city)
    
    # Overpass optimizes a union of simple tag matches better than a single
    # regex match. Same result, noticeably faster server-side response.
    query = f"""
    [out:json][timeout:25];
    (
      node["tourism"="attraction"](around:{DEFAULT_RADIUS_M},{lat},{lon});
      node["tourism"="museum"](around:{DEFAULT_RADIUS_M},{lat},{lon});
      node["tourism"="gallery"](around:{DEFAULT_RADIUS_M},{lat},{lon});
      node["tourism"="viewpoint"](around:{DEFAULT_RADIUS_M},{lat},{lon});
    );
    out center {limit};
    """
    
    elements = _query_overpass(query)
    return [_format_place(el) for el in elements]


# ── Optional: run this file directly to sanity-check ──────────────────
if __name__ == "__main__":
    # Quick self-test — same idea as the smoke test but shorter
    print("Testing osm_client with 'Austin, TX'...\n")
    print("Restaurants:")
    for r in search_restaurants("Austin, TX", limit=3):
        print(f"  • {r['name']} ({r['type']}) — {r['address']}")
    print("\nAttractions:")
    for a in search_attractions("Austin, TX", limit=3):
        print(f"  • {a['name']} ({a['type']}) — {a['address']}")