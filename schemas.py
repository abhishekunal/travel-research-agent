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

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
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
# ── Intent: what the SCOPE node extracts from user input ─────────────
# This is the structured version of "what does the user actually want?"
# The scope node reads the conversation, populates an Intent, and either
# routes to research (if complete) or asks a clarifying question (if not).
#
# Why a Pydantic model rather than a dict: the scope node calls the LLM
# with .with_structured_output(Intent), which uses this schema to force
# Claude to return valid JSON matching these fields. Pydantic then
# validates it — dates are real dates, lists are real lists, etc.
class Intent(BaseModel):
    """Structured trip intent extracted from user conversation.

    Populated by the SCOPE node. Consumed by the RESEARCH and SYNTHESIZE
    nodes. If required fields (city, start_date, end_date) can't be
    confidently extracted, they stay None and missing_fields is populated
    on TripState instead.
    """

    city: str | None = Field(
        default=None,
        description="Destination city; include country if ambiguous, e.g. 'Paris, France'"
    )

    start_date: date | None = Field(
        default=None,
        description="Trip start date"
    )

    end_date: date | None = Field(
        default=None,
        description="Trip end date"
    )

    interests: list[str] = Field(
        default_factory=list,
        description="Themes the user mentioned, e.g. ['architecture', 'food', 'nightlife']"
    )

    constraints: list[str] = Field(
        default_factory=list,
        description="Requirements or restrictions, e.g. ['vegetarian only', 'no walking >20min']"
    )

    # Same cross-field rule as TripBrief: if both dates exist, end >= start.
    # We only enforce when both are non-None because scope may extract
    # one date and leave the other for a clarifying question.
    @field_validator("end_date")
    @classmethod
    def end_date_after_start_date(cls, end_date: date | None, info) -> date | None:
        start_date = info.data.get("start_date")
        if start_date and end_date and end_date < start_date:
            raise ValueError(
                f"end_date ({end_date}) must be on or after start_date ({start_date})"
            )
        return end_date


# ── TripState: the shared "clipboard" flowing through the graph ──────
# Every LangGraph node reads from and writes to this single object.
# TypedDict (not Pydantic) because LangGraph's state machinery has
# native support for TypedDict — including the add_messages reducer
# used below, which APPENDS to the messages list instead of REPLACING
# it (the default merge behavior for other fields).
#
# Field lifecycle:
#   messages           — grows across the whole conversation
#   intent             — set by scope, read by research + synthesize
#   missing_fields     — set by scope; if non-empty, we route to clarify
#   clarifying_question — set by scope, read by ask_clarification
#   tool_results       — set by research, read by synthesize
#   tool_errors        — set by research (skip-and-note), read by synthesize
#   trip_brief         — set by synthesize, read by the UI
#   reasoning          — set by synthesize, read by the UI
class TripState(TypedDict):
    """Shared state passed between nodes in the travel research graph."""

    # `Annotated[..., add_messages]` tells LangGraph: when a node returns
    # an update to `messages`, APPEND those messages to the existing list
    # instead of replacing the list. This is how conversation history
    # accumulates across turns and across nodes.
    messages: Annotated[list, add_messages]

    # Populated by scope
    intent: Intent | None
    missing_fields: list[str]
    clarifying_question: str | None

    # Populated by research
    tool_results: dict
    tool_errors: list[str]

    # Populated by synthesize
    trip_brief: TripBrief | None
    reasoning: str | None

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
    # Test 3: Intent accepts partial data (both dates missing is fine)
    print("\n" + "─" * 50)
    print("Testing Intent with partial data...")
    partial_intent = Intent(city="Barcelona", interests=["architecture", "food"])
    print(f"✅ Partial Intent parsed: {partial_intent.model_dump()}")

    # Test 4: Intent rejects invalid date ordering (same rule as TripBrief)
    print("\n" + "─" * 50)
    print("Testing Intent with end_date before start_date...")
    try:
        Intent(
            city="Barcelona",
            start_date=date(2026, 12, 17),
            end_date=date(2026, 12, 15),
        )
        print("❌ Should have raised an error but didn't!")
    except Exception as e:
        print(f"✅ Correctly rejected: {e}")

    # Test 5: TripState is a valid TypedDict shape (this is a type check,
    # not a runtime check — TypedDict doesn't enforce at construction,
    # it just gives static type checkers a shape to verify. Creating one
    # here proves the imports work and the type is well-formed.)
    print("\n" + "─" * 50)
    print("Testing TripState construction...")
    initial_state: TripState = {
        "messages": [],
        "intent": None,
        "missing_fields": [],
        "clarifying_question": None,
        "tool_results": {},
        "tool_errors": [],
        "trip_brief": None,
        "reasoning": None,
    }
    print(f"✅ TripState constructed with {len(initial_state)} fields") 