"""
Adversa — Pydantic v2 Models for Multi-Agent Courtroom Environment

Defines the core data contracts for the OpenEnv-compliant environment:
- AdversaAction: What agents can DO (present_evidence, object, cross_examine, etc.)
- AdversaObservation: What agents can SEE (role-specific, information-asymmetric)
- AdversaState: Full internal state (for grading/debugging, not sent to agents)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════

class Role(str, Enum):
    PROSECUTOR = "prosecutor"
    DEFENSE = "defense"
    JUDGE = "judge"


class Phase(str, Enum):
    OPENING = "opening"
    PROSECUTION_CASE = "prosecution_case"
    DEFENSE_CASE = "defense_case"
    CLOSING = "closing"
    VERDICT = "verdict"


class ActionType(str, Enum):
    PRESENT_EVIDENCE = "present_evidence"
    OBJECT = "object"
    CROSS_EXAMINE = "cross_examine"
    CLOSING_ARGUMENT = "closing_argument"
    OPENING_STATEMENT = "opening_statement"
    SUSTAIN = "sustain"
    OVERRULE = "overrule"
    INSTRUCT_JURY = "instruct_jury"
    PASS = "pass"


class Framing(str, Enum):
    FACTUAL = "factual"
    EMOTIONAL = "emotional"
    AUTHORITY = "authority"


class ObjectionType(str, Enum):
    HEARSAY = "hearsay"
    RELEVANCE = "relevance"
    COERCED = "coerced"
    LEADING = "leading"


class Verdict(str, Enum):
    GUILTY = "guilty"
    NOT_GUILTY = "not_guilty"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase → Valid Actions Mapping
# ═══════════════════════════════════════════════════════════════════════════════

PHASE_VALID_ACTIONS: dict[Phase, dict[Role, list[str]]] = {
    Phase.OPENING: {
        Role.PROSECUTOR: ["opening_statement", "pass"],
        Role.DEFENSE: ["opening_statement", "pass"],
        Role.JUDGE: ["pass"],
    },
    Phase.PROSECUTION_CASE: {
        Role.PROSECUTOR: ["present_evidence", "pass"],
        Role.DEFENSE: ["object", "cross_examine", "pass"],
        Role.JUDGE: ["sustain", "overrule", "pass"],
    },
    Phase.DEFENSE_CASE: {
        Role.PROSECUTOR: ["object", "cross_examine", "pass"],
        Role.DEFENSE: ["present_evidence", "pass"],
        Role.JUDGE: ["sustain", "overrule", "pass"],
    },
    Phase.CLOSING: {
        Role.PROSECUTOR: ["closing_argument", "pass"],
        Role.DEFENSE: ["closing_argument", "pass"],
        Role.JUDGE: ["pass"],
    },
    Phase.VERDICT: {
        Role.PROSECUTOR: ["pass"],
        Role.DEFENSE: ["pass"],
        Role.JUDGE: ["instruct_jury", "pass"],
    },
}


def get_valid_actions(phase: Phase, role: Role) -> list[str]:
    """Return list of valid action type strings for a given phase and role."""
    return PHASE_VALID_ACTIONS.get(phase, {}).get(role, ["pass"])


# ═══════════════════════════════════════════════════════════════════════════════
# Action Model
# ═══════════════════════════════════════════════════════════════════════════════

class AdversaAction(BaseModel):
    """An action taken by an LLM agent in the courtroom."""

    action_type: str = Field(
        ...,
        description="Type of action: present_evidence, object, cross_examine, "
                    "closing_argument, opening_statement, sustain, overrule, "
                    "instruct_jury, pass",
    )
    evidence_id: Optional[str] = Field(
        None, description="ID of evidence to present (e.g. 'E1')"
    )
    framing: Optional[str] = Field(
        None, description="How to frame the argument: factual, emotional, authority"
    )
    objection_type: Optional[str] = Field(
        None, description="Type of objection: hearsay, relevance, coerced, leading"
    )
    target: Optional[str] = Field(
        None, description="Target of objection or cross-examination (evidence_id or witness)"
    )
    argument_text: Optional[str] = Field(
        None, description="Free-text argument for opening/closing statements"
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> "AdversaAction":
        """Ensure required fields are present based on action_type."""
        if self.action_type == "present_evidence" and not self.evidence_id:
            raise ValueError("evidence_id required for present_evidence action")
        if self.action_type == "object" and not self.objection_type:
            raise ValueError("objection_type required for object action")
        if self.action_type == "object" and not self.target:
            raise ValueError("target required for object action")
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# Observation Model (Role-Specific View)
# ═══════════════════════════════════════════════════════════════════════════════

class EvidenceView(BaseModel):
    """How a piece of evidence appears to an agent."""
    id: str
    description: str
    strength: float
    emotional_impact: float
    admissible: bool
    presented: bool = False


class PublicRecordEntry(BaseModel):
    """An entry in the public trial record visible to all agents."""
    step: int
    role: str
    action_type: str
    evidence_id: Optional[str] = None
    framing: Optional[str] = None
    argument_text: Optional[str] = None
    objection_type: Optional[str] = None
    objection_ruling: Optional[str] = None  # "sustained" or "overruled"
    target: Optional[str] = None


class JurySentiment(BaseModel):
    """Current jury sentiment scores (visible to agents as feedback signal)."""
    analytical: float = 0.5
    empathetic: float = 0.5
    skeptical: float = 0.5


class AdversaObservation(BaseModel):
    """What an agent can observe — role-specific, information-asymmetric."""

    role: str = Field(..., description="Agent's role: prosecutor, defense, judge")
    phase: str = Field(..., description="Current trial phase")
    step: int = Field(..., description="Current step number (1-indexed)")
    max_steps: int = Field(30, description="Maximum steps in episode")

    # Role-specific evidence (only YOUR side's evidence)
    my_evidence: list[EvidenceView] = Field(
        default_factory=list,
        description="Evidence available to this role (hidden from opponent)",
    )

    # Public record (visible to all)
    public_record: list[PublicRecordEntry] = Field(
        default_factory=list,
        description="All publicly presented evidence and arguments",
    )

    # Jury state (visible as strategic feedback)
    jury_sentiment: JurySentiment = Field(
        default_factory=JurySentiment,
        description="Current juror sentiment scores",
    )

    # Opponent context
    last_opponent_action: Optional[PublicRecordEntry] = Field(
        None, description="Last action taken by opposing counsel"
    )

    # Phase-aware action space
    available_actions: list[str] = Field(
        default_factory=list,
        description="Valid action types for current phase and role",
    )

    # Judge-specific: pending objection to rule on
    objection_pending: Optional[dict[str, Any]] = Field(
        None, description="For judge: objection awaiting ruling"
    )

    # Case metadata
    case_name: str = Field("", description="Name of the current case")
    charges: str = Field("", description="What the defendant is charged with")


# ═══════════════════════════════════════════════════════════════════════════════
# Full State Model (Internal — for grading, not sent to agents)
# ═══════════════════════════════════════════════════════════════════════════════

class RewardBreakdown(BaseModel):
    """Detailed reward components for transparency."""
    verdict_reward: float = 0.0
    jury_persuasion_reward: float = 0.0
    evidence_timing_reward: float = 0.0
    procedural_reward: float = 0.0
    objection_reward: float = 0.0
    total: float = 0.0


class AdversaState(BaseModel):
    """Full internal state — used for grading and debugging."""

    # Episode metadata
    case_id: str = ""
    case_name: str = ""
    seed: int = 0
    ground_truth: str = Field("", description="Correct verdict: guilty or not_guilty")

    # Trial state
    phase: str = "opening"
    step: int = 0
    max_steps: int = 30
    done: bool = False
    verdict: Optional[str] = None

    # Current turn
    current_role: str = "prosecutor"

    # Evidence tracking
    prosecution_evidence_presented: list[str] = Field(default_factory=list)
    defense_evidence_presented: list[str] = Field(default_factory=list)

    # Jury state
    jury_sentiment: JurySentiment = Field(default_factory=JurySentiment)
    jury_sentiment_history: list[dict[str, float]] = Field(default_factory=list)

    # Public record
    public_record: list[PublicRecordEntry] = Field(default_factory=list)

    # Scores
    procedural_violations: int = 0
    successful_objections: int = 0
    total_objections: int = 0
    correct_rulings: int = 0
    total_rulings: int = 0

    # Reward tracking
    cumulative_reward: dict[str, float] = Field(
        default_factory=lambda: {"prosecutor": 0.0, "defense": 0.0, "judge": 0.0}
    )
    last_reward_breakdown: Optional[RewardBreakdown] = None

    # Evidence timing scores
    prosecution_timing_score: float = 0.0
    defense_timing_score: float = 0.0
