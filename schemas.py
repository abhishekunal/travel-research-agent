"""
schemas.py

Pydantic models defining the structured output of the Travel Research Agent.

The agent produces free-form prose. This module defines the shape we want
that prose to take once it's parsed into structured data — one canonical
schema that the agent produces, the UI consumes, and validation enforces.

Public models:
    Place       — a restaurant or attraction (name, type, address)
    TripBrief   — the full agent response (destination, dates, weather,
                  restaurants[], attractions[], notes)
"""

from datetime import date
from pydantic import BaseModel, Field, field_validator


# ── Sub-model: a single restaurant or attraction ─────────────────────
# We use one shared model for both because they have the same shape.
# This matches what osm_client returns, which keeps the code simple.
class Place(BaseModel):
    """A single restaurant or attraction."""
    
    name: str = Field(description="Name of the place, e.g. 'Galaxy Cafe'")
    type: str = Field(description="Category or cuisine, e.g. 'american' or 'museum'")
    address: str = Field(description="Street address, or 'No address on file' if unknown")


# ── Main model: the full trip brief ──────────────────────────────────
class TripBrief(BaseModel):
    """
    Structured output for a trip research request.
    
    This is the contract between the agent (which produces one) and the
    UI (which renders one). Every trip brief has the same shape, whether
    the agent had full data or had to skip a failed tool.
    """
    
    destination: str = Field(
        description="City name including state or country, e.g. 'Austin, TX'"
    )
    
    start_date: date = Field(
        description="Trip start date"
    )
    
    end_date: date = Field(
        description="Trip end date"
    )
    
    weather_summary: str = Field(
        description="One or two sentences describing the weather during the trip"
    )
    
    # Default empty list means: if the tool failed and produced no data,
    # the field still exists (as []) rather than being missing entirely.
    # This is what makes "skip and note" work — the UI can render a
    # partial brief without special-casing missing fields.
    restaurants: list[Place] = Field(
        default_factory=list,
        description="Recommended restaurants; empty if data unavailable"
    )
    
    attractions: list[Place] = Field(
        default_factory=list,
        description="Recommended attractions; empty if data unavailable"
    )
    
    notes: list[str] = Field(
        default_factory=list,
        description="Any caveats, warnings, or explanations of missing data"
    )
    
    # ── Cross-field validation ────────────────────────────────────────
    # @field_validator runs after individual field types are validated.
    # We use it here to enforce a business rule that spans two fields:
    # end_date must not be before start_date.
    @field_validator("end_date")
    @classmethod
    def end_date_after_start_date(cls, end_date: date, info) -> date:
        start_date = info.data.get("start_date")
        if start_date and end_date < start_date:
            raise ValueError(
                f"end_date ({end_date}) must be on or after start_date ({start_date})"
            )
        return end_date


# ── Sanity check ──────────────────────────────────────────────────────
# Run `python schemas.py` to verify the schema parses valid input and
# rejects invalid input. This is a poor-man's unit test.
if __name__ == "__main__":
    print("Testing TripBrief schema...\n")
    
    # Test 1: valid input parses correctly
    valid_brief = TripBrief(
        destination="Austin, TX",
        start_date=date(2026, 12, 15),
        end_date=date(2026, 12, 17),
        weather_summary="Hot and sunny, around 100°F.",
        restaurants=[
            Place(name="Galaxy Cafe", type="american", address="1000 West Lynn St"),
        ],
        attractions=[
            Place(name="Elisabet Ney Museum", type="museum", address="No address on file"),
        ],
        notes=["Extreme heat warning in effect."],
    )
    print("✅ Valid brief parsed successfully:")
    print(valid_brief.model_dump_json(indent=2))
    
    # Test 2: invalid input (end before start) should raise
    print("\n" + "─" * 50)
    print("Testing invalid input (end_date before start_date)...")
    try:
        invalid_brief = TripBrief(
            destination="Austin, TX",
            start_date=date(2026, 12, 17),
            end_date=date(2026, 12, 15),  # Before start — should fail
            weather_summary="Test",
        )
        print("❌ Should have raised an error but didn't!")
    except Exception as e:
        print(f"✅ Correctly rejected: {e}")