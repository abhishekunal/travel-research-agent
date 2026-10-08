"""
Smoke test: Verify Google Places API (New) Text Search works before we
build places_client.py around it.

What this proves:
  1. The API key in .env is valid and restricted correctly
  2. The field mask keeps us on the Text Search PRO tier (cheapest
     tier that includes name + address + type)
  3. Google returns better data than OSM for the city that exposed
     OSM's gap (Chicago — no deep-dish places came back from OSM)

Cost: 3 Text Search calls. Free tier is 5,000/month, so this is free.

Usage: python smoke_test_google_places.py
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()  # reads GOOGLE_PLACES_API_KEY from .env into os.environ

# ── Endpoint ──────────────────────────────────────────────────────────
# Places API (New) uses a single POST endpoint for text search. This is
# different from OSM: no separate geocoding step — "restaurants in
# Chicago, IL" is understood directly.
TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"

# ── Field mask: THE cost lever ────────────────────────────────────────
# Google bills each request at the most expensive field you ask for.
# These three are all "Pro" tier fields. Adding places.rating,
# places.priceLevel, or opening hours would silently bump every call
# to the Enterprise tier (1,000 free/month instead of 5,000).
# DO NOT add fields here without checking the SKU table first.
FIELD_MASK = ",".join([
    "places.displayName",            # → our Place.name
    "places.primaryTypeDisplayName", # → our Place.type (human-readable, e.g. "Pizza Restaurant")
    "places.formattedAddress",       # → our Place.address
])

TEST_CITY = "Chicago, IL"


def text_search(query: str, page_size: int = 5) -> list[dict]:
    """Run one Text Search call and return the raw list of places."""
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_PLACES_API_KEY is not set in .env")

    headers = {
        # Places API (New) takes the key and field mask as headers,
        # not URL parameters. Keeps the key out of server logs/URLs.
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": FIELD_MASK,
        "Content-Type": "application/json",
    }
    body = {
        "textQuery": query,
        "pageSize": page_size,  # max results for this call (1-20)
    }

    response = requests.post(TEXT_SEARCH_URL, json=body, headers=headers, timeout=10)

    # 429 = quota exhausted (our 160/day cap). In places_client.py this
    # becomes an exception the tool catches → skip-and-note in the brief.
    if response.status_code == 429:
        raise RuntimeError("Quota exhausted (HTTP 429) — daily cap reached")

    # Any other non-2xx: print Google's error body, it's usually specific
    # (e.g. "API key not valid", "API not enabled for this project").
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

    # An empty result set comes back as {} with no "places" key at all.
    return response.json().get("places", [])


def print_results(label: str, places: list[dict]):
    """Print results in the same {name, type, address} shape our Place schema uses."""
    print(f"\n{'=' * 60}")
    print(f"  {label} — {len(places)} result(s)")
    print(f"{'=' * 60}")

    if not places:
        print("  ⚠️  Call worked but returned nothing.")
        return

    for i, p in enumerate(places, 1):
        # displayName and primaryTypeDisplayName are objects like
        # {"text": "Lou Malnati's", "languageCode": "en"} — we want .text
        name = p.get("displayName", {}).get("text", "(unnamed)")
        ptype = p.get("primaryTypeDisplayName", {}).get("text", "—")
        address = p.get("formattedAddress", "No address on file")
        print(f"\n  {i}. {name}")
        print(f"     Type:    {ptype}")
        print(f"     Address: {address}")
    print()


if __name__ == "__main__":
    print(f"\n🔍 Smoke-testing Google Places API (New) with city: {TEST_CITY}")

    tests = [
        ("🍽️  RESTAURANTS (plain — what the swap gives us)", f"restaurants in {TEST_CITY}"),
        ("🍕 DEEP DISH (interest-aware — the future upgrade)", f"deep dish pizza restaurants in {TEST_CITY}"),
        ("🏛️  ATTRACTIONS", f"tourist attractions in {TEST_CITY}"),
    ]

    for label, query in tests:
        try:
            print(f"⏳ Query: {query!r}")
            print_results(label, text_search(query))
        except Exception as e:
            print(f"❌ {label} failed: {e}")

    print("✅ Smoke test complete. Compare against what OSM returned for Chicago.")