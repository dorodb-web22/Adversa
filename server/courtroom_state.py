"""
Adversa — CourtroomState

World state machine managing:
- Phase transitions (Opening → Prosecution → Defense → Closing → Verdict)
- Evidence registry (presented / hidden)
- Jury sentiment (deterministic formula, NOT an LLM)
- Public record
- Objection handling
- Reward signal components
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from server.case_templates import CaseTemplate, Evidence, JurorProfile, get_case


# ══════════════════════════════════════════════════════════════════════════════
# Phase ordering
# ══════════════════════════════════════════════════════════════════════════════

PHASES = ["opening", "prosecution_case", "defense_case", "closing", "verdict"]

PHASE_STEP_RANGES = {
    "opening":          (1, 4),
    "prosecution_case": (5, 14),
    "defense_case":     (15, 24),
    "closing":          (25, 28),
    "verdict":          (29, 30),
}

# Turn order within each phase: list of (role, sub_action_budget)
PHASE_TURN_ORDER = {
    "opening":          ["prosecutor", "defense"],
    "prosecution_case": ["prosecutor", "defense", "prosecutor", "defense",
                         "prosecutor", "defense", "prosecutor", "defense",
                         "prosecutor", "defense"],
    "defense_case":     ["defense", "prosecutor", "defense", "prosecutor",
                         "defense", "prosecutor", "defense", "prosecutor",
                         "defense", "prosecutor"],
    "closing":          ["prosecutor", "defense", "prosecutor", "defense"],
    "verdict":          ["judge", "judge"],
}


# ══════════════════════════════════════════════════════════════════════════════
# Juror state
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class JurorState:
    profile: JurorProfile
    sentiment: float          # 0 = certain innocent, 1 = certain guilty
    prev_sentiment: float = 0.5
    argument_history: list[str] = field(default_factory=list)

    def shift(self, amount: float) -> None:
        self.prev_sentiment = self.sentiment
        self.sentiment = max(0.0, min(1.0, self.sentiment + amount))

    def vote(self) -> str:
        return "guilty" if self.sentiment > 0.5 else "not_guilty"


def compute_juror_shift(
    evidence: Evidence,
    framing: str,
    juror: JurorState,
    is_consistent_with_prior: bool,
) -> float:
    """
    Deterministic jury persuasion formula.
    Returns delta sentiment (positive = toward guilty).
    Framing multiplies the relevant weight component.
    """
    framing_multiplier = {"factual": 1.0, "emotional": 1.0, "authority": 1.0}

    if framing == "factual":
        ev_score = evidence.strength * juror.profile.evidence_weight * 1.2
        em_score = evidence.emotional_impact * juror.profile.emotion_weight * 0.7
        au_score = evidence.authority_appeal * juror.profile.authority_weight * 0.9
    elif framing == "emotional":
        ev_score = evidence.strength * juror.profile.evidence_weight * 0.7
        em_score = evidence.emotional_impact * juror.profile.emotion_weight * 1.3
        au_score = evidence.authority_appeal * juror.profile.authority_weight * 0.8
    elif framing == "authority":
        ev_score = evidence.strength * juror.profile.evidence_weight * 0.9
        em_score = evidence.emotional_impact * juror.profile.emotion_weight * 0.7
        au_score = evidence.authority_appeal * juror.profile.authority_weight * 1.4
    else:
        ev_score = evidence.strength * juror.profile.evidence_weight
        em_score = evidence.emotional_impact * juror.profile.emotion_weight
        au_score = evidence.authority_appeal * juror.profile.authority_weight

    consistency = juror.profile.consistency_bonus * 0.05 if is_consistent_with_prior else 0.0

    raw = (ev_score + em_score + au_score + consistency) * 0.08  # scale to ~0–0.15
    return raw


# ══════════════════════════════════════════════════════════════════════════════
# Public record entry
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RecordEntry:
    step: int
    role: str
    action_type: str
    evidence_id: Optional[str] = None
    framing: Optional[str] = None
    argument_text: Optional[str] = None
    objection_type: Optional[str] = None
    objection_ruling: Optional[str] = None
    target: Optional[str] = None
    jury_shift: Optional[dict] = None  # {juror_type: delta}


@dataclass
class PendingObjection:
    step: int
    raised_by: str
    objection_type: str
    target_evidence_id: str
    target_role: str


# ══════════════════════════════════════════════════════════════════════════════
# CourtroomState
# ══════════════════════════════════════════════════════════════════════════════

class CourtroomState:
    """
    Core world state for one trial episode.
    Manages phases, evidence, jury, public record, and objection handling.
    """

    def __init__(self, case: CaseTemplate, seed: int = 0):
        self.case = case
        self.seed = seed

        # ── Phase / Step ──────────────────────────────────────────────────
        self.phase: str = "opening"
        self.step: int = 0
        self.done: bool = False
        self.verdict: Optional[str] = None

        # Turn tracking within phase
        self._phase_idx: int = 0          # index in PHASE_TURN_ORDER[phase]
        self.current_role: str = "prosecutor"

        # ── Evidence registry ─────────────────────────────────────────────
        self.prosecution_evidence: dict[str, Evidence] = {
            e.id: e for e in case.prosecution_evidence
        }
        self.defense_evidence: dict[str, Evidence] = {
            e.id: e for e in case.defense_evidence
        }
        self.presented_evidence: set[str] = set()
        self.suppressed_evidence: set[str] = set()  # ruled inadmissible

        # ── Jury ──────────────────────────────────────────────────────────
        self.jurors: list[JurorState] = [
            JurorState(profile=j, sentiment=j.initial_sentiment,
                       prev_sentiment=j.initial_sentiment)
            for j in case.jurors
        ]

        # ── Public record ─────────────────────────────────────────────────
        self.public_record: list[RecordEntry] = []

        # ── Objection tracking ────────────────────────────────────────────
        self.pending_objection: Optional[PendingObjection] = None
        self.total_objections: int = 0
        self.sustained_objections: int = 0
        self.overruled_objections: int = 0
        self.total_rulings: int = 0
        self.correct_rulings: int = 0
        self.procedural_violations: int = 0

        # ── Reward components ─────────────────────────────────────────────
        self.prev_jury_sentiment: dict[str, float] = self._jury_dict()
        self.prosecution_timing_score: float = 0.0
        self.defense_timing_score: float = 0.0
        self.new_violations: int = 0
        self.new_successful_objections: int = 0

        # ── Cumulative rewards ────────────────────────────────────────────
        self.cumulative: dict[str, float] = {
            "prosecutor": 0.0, "defense": 0.0, "judge": 0.0
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Accessors
    # ─────────────────────────────────────────────────────────────────────────

    def _jury_dict(self) -> dict[str, float]:
        return {j.profile.type: j.sentiment for j in self.jurors}

    def jury_sentiment(self) -> dict[str, float]:
        return self._jury_dict()

    def get_evidence_for_role(self, role: str) -> dict[str, Evidence]:
        if role == "prosecutor":
            return self.prosecution_evidence
        elif role == "defense":
            return self.defense_evidence
        return {}

    def get_available_evidence(self, role: str) -> list[Evidence]:
        """Return unpresented (and not suppressed) evidence for a role."""
        pool = self.get_evidence_for_role(role)
        return [e for e in pool.values()
                if e.id not in self.presented_evidence
                and e.id not in self.suppressed_evidence]

    # ─────────────────────────────────────────────────────────────────────────
    # Phase transitions
    # ─────────────────────────────────────────────────────────────────────────

    def _advance_phase(self) -> None:
        idx = PHASES.index(self.phase)
        if idx < len(PHASES) - 1:
            self.phase = PHASES[idx + 1]
            self._phase_idx = 0
            self.current_role = PHASE_TURN_ORDER[self.phase][0]
        else:
            self.done = True

    def _advance_turn(self) -> None:
        """Move to next turn within phase, or advance phase."""
        order = PHASE_TURN_ORDER[self.phase]
        self._phase_idx += 1
        if self._phase_idx >= len(order):
            self._advance_phase()
        else:
            self.current_role = order[self._phase_idx]

    def is_correct_role(self, role: str) -> bool:
        return role == self.current_role

    # ─────────────────────────────────────────────────────────────────────────
    # Action processing
    # ─────────────────────────────────────────────────────────────────────────

    def process_action(self, role: str, action: dict) -> dict:
        """
        Process a single action from an agent.
        Returns a result dict with: reward, info, error (if any).
        """
        self.new_violations = 0
        self.new_successful_objections = 0
        reward = 0.0
        info: dict = {}

        action_type = action.get("action_type", "pass")

        # ── Role mismatch: penalize but DO NOT advance turn or count step ─
        # This preserves the correct agent's turn slot.
        if not self.is_correct_role(role):
            self.procedural_violations += 1
            self.new_violations += 1
            # Do NOT call _advance_turn() — the correct role must still act
            return {"reward": -0.5, "info": {"error": "not_your_turn",
                                              "expected_role": self.current_role}}

        # Step increments only when the correct role acts
        self.step += 1

        # ── Pending objection: judge must rule; other roles pass silently ─
        if self.pending_objection and role != "judge":
            # Consume the turn (correct role is acting) but do nothing
            entry = RecordEntry(step=self.step, role=role, action_type="pass")
            self.public_record.append(entry)
            self._advance_turn()
            return {"reward": 0.0, "info": {"waiting_for_judge_ruling": True}}

        entry = RecordEntry(
            step=self.step,
            role=role,
            action_type=action_type,
        )

        # ── Dispatch ──────────────────────────────────────────────────────
        if action_type == "present_evidence":
            reward, info = self._handle_present_evidence(role, action, entry)

        elif action_type == "object":
            reward, info = self._handle_objection(role, action, entry)

        elif action_type == "cross_examine":
            reward, info = self._handle_cross_examine(role, action, entry)

        elif action_type in ("opening_statement", "closing_argument"):
            reward, info = self._handle_argument(role, action, entry)

        elif action_type == "sustain":
            reward, info = self._handle_ruling("sustain", role, entry)

        elif action_type == "overrule":
            reward, info = self._handle_ruling("overrule", role, entry)

        elif action_type == "instruct_jury":
            reward, info = self._handle_instruct_jury(role, entry)

        elif action_type == "pass":
            reward = 0.0
            info = {"passed": True}

        else:
            # Unknown action type = procedural violation
            self.procedural_violations += 1
            self.new_violations += 1
            reward = -0.5
            info = {"error": f"unknown_action_type: {action_type}"}

        entry.jury_shift = {j.profile.type: j.sentiment - j.prev_sentiment
                            for j in self.jurors}
        self.public_record.append(entry)
        self.prev_jury_sentiment = self._jury_dict()

        # ── Check end condition ───────────────────────────────────────────
        if not self.done:
            self._advance_turn()

        if self.step >= 30 and not self.done:
            self._force_verdict()

        if self.done and self.verdict is None:
            self._force_verdict()

        self.cumulative[role] = self.cumulative.get(role, 0.0) + reward
        return {"reward": reward, "info": info}

    # ─────────────────────────────────────────────────────────────────────────
    # Action handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_present_evidence(self, role: str, action: dict,
                                  entry: RecordEntry) -> tuple[float, dict]:
        evidence_id = action.get("evidence_id")
        framing = action.get("framing", "factual")
        entry.evidence_id = evidence_id
        entry.framing = framing

        pool = self.get_evidence_for_role(role)
        if evidence_id not in pool:
            self.procedural_violations += 1
            self.new_violations += 1
            return -0.5, {"error": "evidence_not_yours"}

        if evidence_id in self.presented_evidence:
            self.procedural_violations += 1
            self.new_violations += 1
            return -0.3, {"error": "already_presented"}

        if evidence_id in self.suppressed_evidence:
            self.procedural_violations += 1
            self.new_violations += 1
            return -0.5, {"error": "evidence_suppressed"}

        evidence = pool[evidence_id]

        if not evidence.admissible:
            # Presenting inadmissible evidence is a procedural risk
            # Opponent can object and get it suppressed
            pass  # allowed for now; opponent must object

        self.presented_evidence.add(evidence_id)

        # Compute jury persuasion
        all_presented = list(self.presented_evidence)
        is_consistent = len(all_presented) >= 3  # simplistic consistency check
        jury_reward = 0.0
        for juror in self.jurors:
            shift = compute_juror_shift(evidence, framing, juror, is_consistent)
            # Prosecution wants sentiment high; defense wants it low
            if role == "prosecutor":
                juror.shift(+shift)
                jury_reward += 0.3 * shift
            else:
                juror.shift(-shift)
                jury_reward += 0.3 * shift

        # Timing reward: presenting strong evidence late (after building context) is better
        n_presented_so_far = len(self.presented_evidence)
        timing_bonus = 0.0
        if evidence.strength > 0.7:
            # Reward presenting strong evidence after at least 2 prior pieces
            if n_presented_so_far >= 3:
                timing_bonus = 0.8
                if role == "prosecutor":
                    self.prosecution_timing_score += timing_bonus
                else:
                    self.defense_timing_score += timing_bonus
            else:
                timing_bonus = 0.0  # no penalty but missed optimal timing bonus

        return jury_reward + timing_bonus, {
            "evidence_id": evidence_id,
            "framing": framing,
            "jury_shift": {j.profile.type: j.sentiment - j.prev_sentiment
                           for j in self.jurors},
        }

    def _handle_objection(self, role: str, action: dict,
                           entry: RecordEntry) -> tuple[float, dict]:
        objection_type = action.get("objection_type")
        target = action.get("target")
        entry.objection_type = objection_type
        entry.target = target

        self.total_objections += 1

        # Check if target evidence exists in opponent's presented set
        opponent_pool = (self.defense_evidence if role == "prosecutor"
                         else self.prosecution_evidence)

        if target not in opponent_pool or target not in self.presented_evidence:
            # Objection to non-existent/not-yet-presented evidence
            self.procedural_violations += 1
            self.new_violations += 1
            return -0.5, {"error": "invalid_objection_target"}

        target_evidence = opponent_pool[target]

        # Determine if objection is legally valid
        valid_objection = False
        if objection_type == "coerced" and not target_evidence.admissible:
            valid_objection = True
        elif objection_type == "hearsay" and target_evidence.strength < 0.5:
            valid_objection = True
        elif objection_type == "relevance" and target_evidence.strength < 0.4:
            valid_objection = True
        elif objection_type == "leading":
            valid_objection = True  # always plausible

        self.pending_objection = PendingObjection(
            step=self.step,
            raised_by=role,
            objection_type=objection_type,
            target_evidence_id=target,
            target_role="prosecution" if role == "defense" else "defense",
        )
        self.pending_objection._valid = valid_objection  # type: ignore

        # Small immediate reward for raising a valid objection
        if valid_objection:
            return 0.2, {"objection_raised": True, "likely_valid": True}
        else:
            return 0.0, {"objection_raised": True, "likely_valid": False}

    def _handle_ruling(self, ruling: str, role: str,
                        entry: RecordEntry) -> tuple[float, dict]:
        if role != "judge":
            self.procedural_violations += 1
            self.new_violations += 1
            return -0.5, {"error": "only_judge_can_rule"}

        if not self.pending_objection:
            self.procedural_violations += 1
            self.new_violations += 1
            return -0.3, {"error": "no_pending_objection"}

        self.total_rulings += 1
        obj = self.pending_objection
        correct = getattr(obj, "_valid", False)

        if ruling == "sustain":
            entry.objection_ruling = "sustained"
            self.sustained_objections += 1
            self.new_successful_objections += 1
            # Suppress the evidence
            self.suppressed_evidence.add(obj.target_evidence_id)
            self.presented_evidence.discard(obj.target_evidence_id)
            # Reverse jury impact of suppressed evidence
            for juror in self.jurors:
                juror.shift(-0.05)  # mild reversal
            judge_reward = 2.0 if correct else -1.0
            self.correct_rulings += (1 if correct else 0)
        else:  # overrule
            entry.objection_ruling = "overruled"
            self.overruled_objections += 1
            judge_reward = 2.0 if not correct else -1.0
            self.correct_rulings += (1 if not correct else 0)

        self.pending_objection = None
        return judge_reward, {"ruling": ruling, "correct": correct}

    def _handle_cross_examine(self, role: str, action: dict,
                               entry: RecordEntry) -> tuple[float, dict]:
        target = action.get("target")
        entry.target = target

        # Cross-examination: undermines witness credibility
        # Find the witness in opponent's case
        opponent_witnesses = self.case.witnesses
        witness = next((w for w in opponent_witnesses
                        if w.id == target and w.side != role), None)

        if not witness:
            return 0.0, {"info": "witness_not_found"}

        # Credibility attack reduces jury persuasion slightly
        for juror in self.jurors:
            juror.shift(-0.02)

        return 0.1, {"cross_examined": target, "witness_credibility": witness.credibility}

    def _handle_argument(self, role: str, action: dict,
                          entry: RecordEntry) -> tuple[float, dict]:
        text = action.get("argument_text", "")
        framing = action.get("framing", "factual")
        entry.argument_text = text
        entry.framing = framing

        # Closing arguments: weighted by consistency with prior evidence
        n_presented = len([e for e in self.presented_evidence
                           if e in self.get_evidence_for_role(role)])
        consistency_bonus = min(0.5, n_presented * 0.1)

        # Jury shift: smaller than evidence, but still meaningful
        reward = 0.0
        for juror in self.jurors:
            base_shift = 0.03
            if framing == "emotional":
                shift = base_shift * juror.profile.emotion_weight
            elif framing == "authority":
                shift = base_shift * juror.profile.authority_weight
            else:
                shift = base_shift * juror.profile.evidence_weight

            shift += consistency_bonus * 0.01
            if role == "prosecutor":
                juror.shift(+shift)
                reward += 0.1 * shift
            else:
                juror.shift(-shift)
                reward += 0.1 * shift

        return reward + consistency_bonus * 0.1, {"argument": True}

    def _handle_instruct_jury(self, role: str,
                               entry: RecordEntry) -> tuple[float, dict]:
        if role != "judge":
            return -0.5, {"error": "only_judge_can_instruct"}

        # Trigger verdict calculation
        self._force_verdict()
        return 1.0, {"verdict_triggered": True}

    # ─────────────────────────────────────────────────────────────────────────
    # Verdict
    # ─────────────────────────────────────────────────────────────────────────

    def _force_verdict(self) -> None:
        """Compute verdict via majority jury vote."""
        votes = [j.vote() for j in self.jurors]
        guilty_count = votes.count("guilty")
        self.verdict = "guilty" if guilty_count >= 2 else "not_guilty"
        self.done = True
        self.phase = "verdict"

    def is_verdict_correct(self) -> bool:
        return self.verdict == self.case.ground_truth

    # ─────────────────────────────────────────────────────────────────────────
    # Serialization helpers
    # ─────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "case_id": self.case.case_id,
            "case_name": self.case.name,
            "phase": self.phase,
            "step": self.step,
            "done": self.done,
            "verdict": self.verdict,
            "ground_truth": self.case.ground_truth,
            "current_role": self.current_role,
            "jury_sentiment": self._jury_dict(),
            "presented_evidence": list(self.presented_evidence),
            "suppressed_evidence": list(self.suppressed_evidence),
            "procedural_violations": self.procedural_violations,
            "sustained_objections": self.sustained_objections,
            "total_objections": self.total_objections,
            "correct_rulings": self.correct_rulings,
            "total_rulings": self.total_rulings,
            "prosecution_timing_score": self.prosecution_timing_score,
            "defense_timing_score": self.defense_timing_score,
            "verdict_correct": self.is_verdict_correct() if self.done else None,
        }
