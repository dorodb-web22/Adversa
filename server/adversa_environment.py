"""
Adversa — AdversaEnvironment

OpenEnv-compliant environment wrapping CourtroomState.
Implements reset(), step(), state() per OpenEnv spec.
"""
from __future__ import annotations

import random
from typing import Any, Optional

from server.case_templates import ALL_CASES, get_case
from server.courtroom_state import CourtroomState
from models import (
    AdversaAction,
    AdversaObservation,
    AdversaState,
    EvidenceView,
    JurySentiment,
    PublicRecordEntry,
    RewardBreakdown,
    get_valid_actions,
    Phase,
    Role,
)


class AdversaEnvironment:
    """
    Multi-agent adversarial courtroom environment.

    OpenEnv API:
        reset(seed, options) → observation
        step(action)         → observation, reward, done, info
        state()              → full internal state dict
    """

    def __init__(self) -> None:
        self._courtroom: Optional[CourtroomState] = None
        self._current_role: str = "prosecutor"
        self._seed: int = 0
        self._case_id: str = "C1"

    # ══════════════════════════════════════════════════════════════════════════
    # reset
    # ══════════════════════════════════════════════════════════════════════════

    def reset(
        self,
        seed: int = 0,
        options: Optional[dict[str, Any]] = None,
    ) -> dict:
        """
        Reset environment to a new episode.

        options:
            case_id (str):  Which case template to use (default: random)
            role    (str):  Which role the caller is playing (default: "prosecutor")
        """
        opts = options or {}
        self._seed = seed
        self._case_id = opts.get("case_id", random.choice(list(ALL_CASES.keys())))
        self._current_role = opts.get("role", "prosecutor")

        case = get_case(self._case_id, seed)
        self._courtroom = CourtroomState(case=case, seed=seed)

        obs = self._build_observation(self._current_role)
        return obs.model_dump()

    # ══════════════════════════════════════════════════════════════════════════
    # step
    # ══════════════════════════════════════════════════════════════════════════

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """
        Execute one action in the environment.

        Returns:
            observation (dict)
            reward      (float)
            done        (bool)
            info        (dict)
        """
        if self._courtroom is None:
            raise RuntimeError("Environment not initialized — call reset() first.")

        ct = self._courtroom
        role = action.get("role", self._current_role)

        # Validate action via Pydantic (soft validation — fix and continue)
        try:
            validated = AdversaAction(**action)
        except Exception as exc:
            # Invalid action = procedural violation
            ct.procedural_violations += 1
            ct.new_violations += 1
            ct.step += 1
            obs = self._build_observation(role)
            return obs.model_dump(), -0.5, ct.done, {"error": str(exc)}

        result = ct.process_action(role, validated.model_dump())
        raw_reward = result.get("reward", 0.0)
        info = result.get("info", {})

        # Auto-trigger verdict if episode is done but no verdict computed yet
        if (ct.done or ct.step >= 30) and ct.verdict is None:
            ct._force_verdict()

        # Compute full reward with verdict component if done
        reward = self._compute_reward(role, raw_reward)

        obs = self._build_observation(role)
        return obs.model_dump(), reward, ct.done, info

    # ══════════════════════════════════════════════════════════════════════════
    # state
    # ══════════════════════════════════════════════════════════════════════════

    def state(self) -> dict:
        """Return full internal state (for grading/debugging, not sent to agents)."""
        if self._courtroom is None:
            return {"error": "not_initialized"}
        return self._courtroom.to_dict()

    # ══════════════════════════════════════════════════════════════════════════
    # Reward computation
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_reward(self, role: str, step_reward: float) -> float:
        """
        Full reward signal including sparse verdict reward at episode end.
        """
        ct = self._courtroom
        if ct is None:
            return 0.0

        if role in ("prosecutor", "defense"):
            # Dense components already in step_reward (jury shifts, timing, procedure)
            dense = step_reward

            # Sparse: verdict reward only at end
            verdict_r = 0.0
            if ct.done and ct.verdict is not None:
                my_win = (
                    (role == "prosecutor" and ct.verdict == "guilty") or
                    (role == "defense" and ct.verdict == "not_guilty")
                )
                verdict_correct = ct.is_verdict_correct()
                if my_win and verdict_correct:
                    verdict_r = +5.0   # won AND correct verdict
                elif my_win and not verdict_correct:
                    verdict_r = +2.0   # won but verdict wrong (partial credit)
                elif not my_win and verdict_correct:
                    verdict_r = -5.0   # lost AND correct verdict (fully outplayed)
                else:
                    verdict_r = -2.0   # lost but verdict wrong

            return dense + verdict_r

        elif role == "judge":
            # Judge reward: correct rulings + fairness + efficiency
            fairness = (
                2.0 * ct.correct_rulings / max(ct.total_rulings, 1)
                if ct.total_rulings > 0 else 0.0
            )
            efficiency = -0.01 * ct.step
            procedure = 1.0 if ct.new_violations == 0 else 0.0
            return step_reward + procedure + efficiency

        return step_reward

    # ══════════════════════════════════════════════════════════════════════════
    # Observation builder
    # ══════════════════════════════════════════════════════════════════════════

    def _build_observation(self, role: str) -> AdversaObservation:
        ct = self._courtroom
        assert ct is not None

        # Role-specific evidence (hidden from opponent)
        my_evidence_dict = ct.get_evidence_for_role(role)
        my_evidence = [
            EvidenceView(
                id=e.id,
                description=e.description,
                strength=e.strength,
                emotional_impact=e.emotional_impact,
                admissible=e.admissible,
                presented=e.id in ct.presented_evidence,
            )
            for e in my_evidence_dict.values()
        ]

        # Public record → serializable entries
        public_record = [
            PublicRecordEntry(
                step=r.step,
                role=r.role,
                action_type=r.action_type,
                evidence_id=r.evidence_id,
                framing=r.framing,
                argument_text=r.argument_text,
                objection_type=r.objection_type,
                objection_ruling=r.objection_ruling,
                target=r.target,
            )
            for r in ct.public_record
        ]

        # Last opponent action
        opponent = "defense" if role == "prosecutor" else "prosecutor"
        last_opponent = next(
            (PublicRecordEntry(
                step=r.step, role=r.role, action_type=r.action_type,
                evidence_id=r.evidence_id, framing=r.framing,
                argument_text=r.argument_text, objection_type=r.objection_type,
                objection_ruling=r.objection_ruling, target=r.target,
            )
             for r in reversed(ct.public_record) if r.role == opponent),
            None,
        )

        # Judge: pending objection
        objection_pending = None
        if role == "judge" and ct.pending_objection:
            obj = ct.pending_objection
            objection_pending = {
                "raised_by": obj.raised_by,
                "objection_type": obj.objection_type,
                "target_evidence_id": obj.target_evidence_id,
                "target_role": obj.target_role,
            }

        jury = ct.jury_sentiment()
        phase_enum = Phase(ct.phase)
        role_enum = Role(role)

        return AdversaObservation(
            role=role,
            phase=ct.phase,
            step=ct.step,
            max_steps=30,
            my_evidence=my_evidence,
            public_record=public_record,
            jury_sentiment=JurySentiment(
                analytical=jury.get("analytical", 0.5),
                empathetic=jury.get("empathetic", 0.5),
                skeptical=jury.get("skeptical", 0.5),
            ),
            last_opponent_action=last_opponent,
            available_actions=get_valid_actions(phase_enum, role_enum),
            objection_pending=objection_pending,
            case_name=ct.case.name,
            charges=ct.case.charges,
        )
