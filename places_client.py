"""
places_client.py

Helper module for Google Places API (New) — Text Search.

Responsibility: talk to Google Places and return clean, structured Python
lists. Drop-in replacement for the two search functions in osm_client.py:
same function names, same arguments, same return shape. tools.py swaps
which module it calls; nothing downstream (nodes, schemas, UI) changes.

osm_client.py is NOT retired — guardrails.py still uses its Nominatim
geocoder to validate that a city is real. Only the places searches moved.

Public functions:
    search_restaurants(city)    → list of restaurant dicts
    search_attractions(city)    → list of attraction dicts

Each place dict has a stable shape (matches schemas.Place):
    {"name": str, "type": str, "address": str}

Cost controls (see Weekend 4 decision log):
    1. FIELD_MASK keeps every call on the Text Search PRO tier
       (5,000 free calls/month). Adding fields can silently move
       every call to a pricier tier.
    2. An in-process daily call counter refuses calls past
       PLACES_DAILY_CAP (default 160 → 4,960/month max, under the free
       tier). This is a backstop for the Google Cloud per-day quota,
       which can't be set during the free trial. Caveat: the counter
       lives in memory, so it resets if Streamlit Cloud restarts the app.
"""

import os
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Endpoint ──────────────────────────────────────────────────────────
TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"

# ── Field mask: THE cost lever ────────────────────────────────────────
# Google bills each request at the most expensive field requested.
# All three below are Pro-tier. Do NOT add rating / priceLevel /
# opening hours / editorialSummary without re-checking the SKU table.
FIELD_MASK = ",".join([
    "places.displayName",             # → name
    "places.primaryTypeDisplayName",  # → type (e.g. "Pizza Restaurant")
    "places.formattedAddress",        # → address
])

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_LIMIT = 5        # Same as osm_client — enough for the agent, not overwhelming
REQUEST_TIMEOUT = 10     # Google is fast; 10s is generous

# Daily cap, overridable via env var. Setting PLACES_DAILY_CAP=0 is a
# free way to test the "quota exhausted" path without calling Google.
DAILY_CAP = int(os.getenv("PLACES_DAILY_CAP", "160"))

# Google's quotas reset at midnight Pacific time, so we count days the
# same way rather than in UTC or the server's local time.
_QUOTA_TZ = ZoneInfo("America/Los_Angeles")


# ── Daily call counter (the backstop) ─────────────────────────────────
# Module-level variables live once per Python process. Streamlit runs
# every browser session inside the same process, so this counter is
# shared across ALL users of the app — which is what we want for a
# spending cap.
#
# Streamlit serves sessions on separate threads, so two users could hit
# the counter at the same instant. The Lock makes "check, then
# increment" one indivisible step, so we can't overshoot the cap.
_counter_lock = threading.Lock()
_counter = {"day": None, "count": 0}


def _reserve_call() -> None:
    """Count one API call against today's cap, or raise if the cap is hit."""
    today = datetime.now(_QUOTA_TZ).date()
    with _counter_lock:
        if _counter["day"] != today:
            # New day — reset the count
            _counter["day"] = today
            _counter["count"] = 0
        if _counter["count"] >= DAILY_CAP:
            raise RuntimeError(
                "daily places lookup limit reached for this demo app; "
                "try again tomorrow"
            )
        _counter["count"] += 1


# ── Internal helpers ──────────────────────────────────────────────────

def _format_place(place: dict) -> dict:
    """Convert a raw Google place into our clean {name, type, address} dict."""
    # displayName / primaryTypeDisplayName are objects like
    # {"text": "Lou Malnati's", "languageCode": "en"} — we want .text
    return {
        "name": place.get("displayName", {}).get("text", "(unnamed)"),
        "type": place.get("primaryTypeDisplayName", {}).get("text", "unknown"),
        "address": place.get("formattedAddress", "No address on file"),
    }


def _text_search(query: str, limit: int) -> list[dict]:
    """
    Run one Text Search call and return formatted places.

    Raises RuntimeError on any failure (missing key, cap reached, quota
    exhausted, HTTP error). tools.py catches it and returns an
    "unavailable right now" string, which synthesize turns into a
    skip-and-note — the user still gets a brief, minus this section.
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_PLACES_API_KEY is not configured")

    # Check the cap BEFORE calling Google, so a refused call costs nothing.
    _reserve_call()

    response = requests.post(
        TEXT_SEARCH_URL,
        json={"textQuery": query, "pageSize": limit},
        headers={
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": FIELD_MASK,
            "Content-Type": "application/json",
        },
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code == 429:
        # Google-side quota exhausted (per-minute limit, or the per-day
        # quota once it's set after the free trial).
        raise RuntimeError("places quota exhausted (HTTP 429)")
    if not response.ok:
        raise RuntimeError(f"places API error HTTP {response.status_code}: {response.text[:200]}")

    # No matches comes back as {} with no "places" key at all.
    return [_format_place(p) for p in response.json().get("places", [])]


# ── Public functions (same signatures as osm_client) ──────────────────

def search_restaurants(city: str, limit: int = DEFAULT_LIMIT) -> list[dict]:
    """
    Find restaurants in the given city.
    Returns a list of {"name", "type", "address"} dicts.
    """
    return _text_search(f"restaurants in {city}", limit)


def search_attractions(city: str, limit: int = DEFAULT_LIMIT) -> list[dict]:
    """
    Find tourist attractions in the given city.
    Returns a list of {"name", "type", "address"} dicts.

    Google's "tourist attractions" query already spans museums, landmarks,
    viewpoints and parks — no need for OSM's union of tag types.
    """
    return _text_search(f"tourist attractions in {city}", limit)


# ── Optional: run this file directly to sanity-check ──────────────────
if __name__ == "__main__":
    # 2 calls against the 5,000/month free tier.
    print(f"Testing places_client with 'Austin, TX' (daily cap = {DAILY_CAP})...\n")
    print("Restaurants:")
    for r in search_restaurants("Austin, TX", limit=3):
        print(f"  • {r['name']} ({r['type']}) — {r['address']}")
    print("\nAttractions:")
    for a in search_attractions("Austin, TX", limit=3):
        print(f"  • {a['name']} ({a['type']}) — {a['address']}")
    print(f"\nCalls counted today: {_counter['count']}")