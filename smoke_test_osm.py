"""
Smoke test: Verify OpenStreetMap works for both restaurants and attractions.
No API key needed — OSM services are free and public.

Two APIs used together:
1. Nominatim  — geocodes a city name into latitude/longitude
2. Overpass   — searches OSM's tagged point-of-interest database

Design pattern: geocode first, then query.
This is more reliable than trying to match city names directly in Overpass,
because it decouples "where is the city" from "what's in it."

Usage: python smoke_test_osm.py
"""

import requests

# ── Endpoints ─────────────────────────────────────────────────────────
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL  = "https://overpass-api.de/api/interpreter"

# ── Etiquette: identify yourself ──────────────────────────────────────
# Nominatim requires a User-Agent header identifying your app.
# This is how public OSM services enforce their fair-use policy —
# no keys, but they want to know who's calling. Skipping this can
# get you rate-limited or blocked.
HEADERS = {
    "User-Agent": "TravelResearchAgent/1.0 (portfolio project)"
}

TEST_CITY = "Austin, TX"


def geocode_city(city: str) -> tuple[float, float]:
    """
    Convert a city name to (latitude, longitude) using Nominatim.
    Returns the top match.
    """
    params = {
        "q": city,
        "format": "json",
        "limit": 1
    }
    response = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=10)
    response.raise_for_status()
    
    results = response.json()
    if not results:
        raise ValueError(f"Nominatim couldn't find '{city}'")
    
    lat = float(results[0]["lat"])
    lon = float(results[0]["lon"])
    return lat, lon


def query_overpass(overpass_query: str) -> dict:
    """
    Run an Overpass QL query and return the JSON response.
    
    Overpass QL is OSM's query language. It's unusual-looking but powerful.
    We wrap it here so the rest of the code doesn't need to know the syntax.
    """
    # Overpass expects the query in the request body as form data
    response = requests.post(
        OVERPASS_URL,
        data={"data": overpass_query},
        headers=HEADERS,
        timeout=30       # Overpass can be slow (5-15s is normal)
    )
    response.raise_for_status()
    return response.json()


def search_restaurants(lat: float, lon: float, radius_m: int = 5000, limit: int = 5) -> list[dict]:
    """
    Find restaurants within `radius_m` meters of the given point.
    
    Overpass QL breakdown:
    - [out:json][timeout:25]  → response format and server-side timeout
    - node[...] (around:...)  → find map "nodes" (points) with the tag,
                                 within a radius around lat/lon
    - out center N            → return up to N results with coordinates
    """
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="restaurant"](around:{radius_m},{lat},{lon});
    );
    out center {limit};
    """
    data = query_overpass(query)
    return data.get("elements", [])


def search_attractions(lat: float, lon: float, radius_m: int = 5000, limit: int = 5) -> list[dict]:
    """
    Find tourist attractions within `radius_m` meters of the given point.
    
    We search for multiple tag types with a regex — attractions, museums,
    galleries, and viewpoints. This is broader than restaurants because
    "attraction" is a fuzzier concept in OSM tagging.
    """
    query = f"""
    [out:json][timeout:25];
    (
      node["tourism"~"attraction|museum|gallery|viewpoint"](around:{radius_m},{lat},{lon});
    );
    out center {limit};
    """
    data = query_overpass(query)
    return data.get("elements", [])


def print_results(label: str, places: list[dict]):
    """Pretty-print results so you can verify the data shape."""
    print(f"\n{'='*60}")
    print(f"  {label} near {TEST_CITY} — {len(places)} result(s)")
    print(f"{'='*60}")
    
    if not places:
        print("  ⚠️  Query worked but returned nothing. Try a bigger radius.")
        return
    
    for i, place in enumerate(places, 1):
        # OSM data is user-contributed, so many fields may be missing.
        # Everything comes back inside a "tags" dict.
        tags = place.get("tags", {})
        name = tags.get("name", "(unnamed)")
        cuisine = tags.get("cuisine", tags.get("tourism", tags.get("amenity", "—")))
        
        # Address is spread across multiple keys; stitch them together
        street = tags.get("addr:street", "")
        housenumber = tags.get("addr:housenumber", "")
        address = f"{housenumber} {street}".strip() or "No address on file"
        
        print(f"\n  {i}. {name}")
        print(f"     Type:    {cuisine}")
        print(f"     Address: {address}")
    print()


# ── Run the smoke test ────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n🔍 Smoke-testing OpenStreetMap with city: {TEST_CITY}")
    
    # Step 1: Geocode
    try:
        print("\n⏳ Geocoding city via Nominatim...")
        lat, lon = geocode_city(TEST_CITY)
        print(f"   → Found: lat={lat}, lon={lon}")
    except Exception as e:
        print(f"❌ Geocoding failed: {e}")
        exit(1)
    
    # Step 2: Restaurants
    try:
        print("\n⏳ Searching restaurants via Overpass...")
        restaurants = search_restaurants(lat, lon)
        print_results("🍽️  RESTAURANTS", restaurants)
    except Exception as e:
        print(f"❌ Restaurant search failed: {e}")
    
    # Step 3: Attractions
    try:
        print("⏳ Searching attractions via Overpass...")
        attractions = search_attractions(lat, lon)
        print_results("🏛️  ATTRACTIONS", attractions)
    except Exception as e:
        print(f"❌ Attraction search failed: {e}")
    
    print("✅ Smoke test complete. If you saw results above, you're ready to build the tools.")