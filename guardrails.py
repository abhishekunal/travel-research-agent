"""
guardrails.py — Input validation for the Travel Research Agent.

Runs BEFORE the agent is invoked. Catches bad inputs cheaply (and with
friendly messages) instead of letting them waste an LLM call or crash
mid-tool.

Validates:
- City is real and geocodable (reuses osm_client.geocode_city)
- Start date is today or later
- Trip length is 1–14 days

Design:
- Each validator returns a ValidationResult(ok, value, error).
- validate_trip_inputs() checks all three at once and collects EVERY
  error, so users see all their problems in one pass instead of
  fix-one-retry-fix-next.
"""

import requests  # for distinguishing network errors from not-found
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional, Union

from osm_client import geocode_city


# Trip-length bounds. Change here, not sprinkled in messages.
MIN_TRIP_DAYS = 1
MAX_TRIP_DAYS = 14


@dataclass
class ValidationResult:
    ok: bool
    value: Any = None            # cleaned/normalized value when ok=True
    error: Optional[str] = None  # user-friendly message when ok=False


def validate_city(city: str) -> ValidationResult:
    """Confirm the city exists by geocoding it via OSM Nominatim."""
    if not city or not city.strip():
        return ValidationResult(ok=False, error="Please enter a city name.")

    try:
        result = geocode_city(city.strip())
    except (requests.RequestException, TimeoutError, ConnectionError):
        # Real infrastructure problem — Nominatim unreachable, timeout, etc.
        return ValidationResult(
            ok=False,
            error="Couldn't reach the geocoding service. Please try again in a moment.",
        )
    except Exception:
        # geocode_city raises for "not found" too. Anything that isn't a
        # network error, we treat as an unknown place.
        return ValidationResult(
            ok=False,
            error=f"Couldn't find '{city}'. Check the spelling, or try a nearby larger city.",
        )

    if not result:
        return ValidationResult(
            ok=False,
            error=f"Couldn't find '{city}'. Check the spelling, or try a nearby larger city.",
        )

    return ValidationResult(ok=True, value=result)

def validate_start_date(start_date: Union[date, str]) -> ValidationResult:
    """Ensure the start date is today or in the future."""
    # Accept a date object (what Streamlit's date_input returns) OR
    # a YYYY-MM-DD string (handy for tests and CLI use).
    if isinstance(start_date, str):
        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        except ValueError:
            return ValidationResult(
                ok=False,
                error="Date must be in YYYY-MM-DD format (e.g., 2026-10-15).",
            )

    if not isinstance(start_date, date):
        return ValidationResult(ok=False, error="Invalid start date.")

    if start_date < date.today():
        return ValidationResult(
            ok=False,
            error=f"Start date must be today or later. You entered {start_date.isoformat()}.",
        )

    return ValidationResult(ok=True, value=start_date)


def validate_duration(days: Any) -> ValidationResult:
    """Ensure trip length is a whole number in [MIN_TRIP_DAYS, MAX_TRIP_DAYS]."""
    # bool is a subclass of int in Python — reject before int() accepts True as 1.
    if isinstance(days, bool):
        return ValidationResult(ok=False, error="Trip length must be a whole number of days.")

    # Reject non-whole floats before int() silently truncates 3.5 → 3.
    if isinstance(days, float) and not days.is_integer():
        return ValidationResult(ok=False, error="Trip length must be a whole number of days.")

    try:
        days_int = int(days)
    except (TypeError, ValueError):
        return ValidationResult(ok=False, error="Trip length must be a whole number of days.")

    if days_int < MIN_TRIP_DAYS or days_int > MAX_TRIP_DAYS:
        return ValidationResult(
            ok=False,
            error=f"Trip length must be between {MIN_TRIP_DAYS} and {MAX_TRIP_DAYS} days.",
        )

    return ValidationResult(ok=True, value=days_int)


def validate_trip_inputs(
    city: str,
    start_date: Union[date, str],
    days: Any,
) -> ValidationResult:
    """
    Validate all three inputs at once. Collects ALL errors, not just the first,
    so the user sees every problem in a single pass.

    On success: .value is a dict with cleaned inputs and the cached geocode.
    On failure: .error is a newline-joined list of friendly messages.
    """
    city_r = validate_city(city)
    date_r = validate_start_date(start_date)
    dur_r = validate_duration(days)

    errors = [r.error for r in (city_r, date_r, dur_r) if not r.ok]
    if errors:
        return ValidationResult(ok=False, error="\n".join(errors))

    return ValidationResult(
        ok=True,
        value={
            "city_input": city.strip(),
            "geocode": city_r.value,   # reuse downstream, don't re-geocode
            "start_date": date_r.value,
            "days": dur_r.value,
        },
    )


# ---------------------------------------------------------------------------
# Self-test: `python guardrails.py`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import timedelta

    print("=== guardrails self-test ===\n")

    print("Cities:")
    for c in ["Tokyo", "Paris", "asdfghjkl", "", "  "]:
        r = validate_city(c)
        print(f"  [{'OK  ' if r.ok else 'FAIL'}] '{c}' -> {r.error or 'geocoded'}")

    print("\nDates:")
    for d in [
        date.today(),
        date.today() + timedelta(days=30),
        date.today() - timedelta(days=1),
        "2026-12-01",
        "not-a-date",
    ]:
        r = validate_start_date(d)
        print(f"  [{'OK  ' if r.ok else 'FAIL'}] {d} -> {r.error or r.value}")

    print("\nDurations:")
    for n in [1, 7, 14, 0, 15, "5", "abc", 3.5]:
        r = validate_duration(n)
        print(f"  [{'OK  ' if r.ok else 'FAIL'}] {n!r} -> {r.error or r.value}")

    print("\nCombined — valid case:")
    r = validate_trip_inputs("Tokyo", date.today() + timedelta(days=30), 7)
    print(f"  ok={r.ok}, value={r.value if r.ok else r.error}")

    print("\nCombined — all invalid:")
    r = validate_trip_inputs("asdfgh", date.today() - timedelta(days=5), 20)
    print(f"  ok={r.ok}")
    print(f"  errors:\n{r.error}")