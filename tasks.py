"""
Adversa — Task Definitions + Graders (3 difficulty tiers)

Compliant with OpenEnv task spec:
    - Each task has an id, description, and grader function
    - Grader takes episode trajectory and returns a score [0, 1]
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class AdversaTask:
    id: str
    name: str
    description: str
    difficulty: str          # "easy" | "medium" | "hard"
    case_ids: list[str]      # Which case templates apply
    role: str                # Which role is being trained
    max_steps: int
    grader: Callable         # fn(trajectory: list[dict]) -> float


# ══════════════════════════════════════════════════════════════════════════════
# Grader functions
# ══════════════════════════════════════════════════════════════════════════════

def grade_easy(trajectory: list[dict]) -> float:
    """
    Easy: Did the agent get the correct verdict?
    Score = 1.0 if correct verdict, 0.0 otherwise.
    Also partial credit for jury persuasion > 0.6 on 2+ jurors.
    """
    if not trajectory:
        return 0.0

    final = trajectory[-1]
    state = final.get("state", {})

    verdict_correct = state.get("verdict_correct", False)
    if verdict_correct:
        return 1.0

    # Partial credit: jury persuasion
    jury = state.get("jury_sentiment", {})
    high_persuasion = sum(1 for v in jury.values() if v > 0.6)
    return 0.3 * high_persuasion / 3.0


def grade_medium(trajectory: list[dict]) -> float:
    """
    Medium: Correct verdict + evidence strategy + at least 1 valid objection.
    Score is composite:
        - 0.5 for correct verdict
        - 0.25 for presenting 3+ pieces of evidence in correct order
        - 0.25 for at least 1 sustained objection
    """
    if not trajectory:
        return 0.0

    final = trajectory[-1]
    state = final.get("state", {})

    score = 0.0

    # Verdict
    if state.get("verdict_correct", False):
        score += 0.5

    # Evidence count
    presented = state.get("presented_evidence", [])
    if len(presented) >= 3:
        score += 0.25

    # Objections
    sustained = state.get("sustained_objections", 0)
    if sustained >= 1:
        score += 0.25

    return score


def grade_hard(trajectory: list[dict]) -> float:
    """
    Hard: Full strategic mastery.
    Score components:
        - 0.4 for correct verdict
        - 0.2 for all 3 jurors persuaded (sentiment > 0.55)
        - 0.2 for evidence timing (strong evidence presented 3rd or later)
        - 0.1 for valid objections
        - 0.1 for no procedural violations
    """
    if not trajectory:
        return 0.0

    final = trajectory[-1]
    state = final.get("state", {})

    score = 0.0

    # Verdict
    if state.get("verdict_correct", False):
        score += 0.4

    # Jury persuasion
    jury = state.get("jury_sentiment", {})
    persuaded = sum(1 for v in jury.values() if v > 0.55)
    score += 0.2 * persuaded / 3.0

    # Evidence timing (prosecution_timing_score or defense_timing_score)
    timing = max(
        state.get("prosecution_timing_score", 0.0),
        state.get("defense_timing_score", 0.0)
    )
    score += min(0.2, timing * 0.05)

    # Objections
    total_obj = state.get("total_objections", 0)
    sustained = state.get("sustained_objections", 0)
    if total_obj > 0:
        score += 0.1 * (sustained / total_obj)

    # Procedural compliance
    violations = state.get("procedural_violations", 0)
    if violations == 0:
        score += 0.1

    return min(1.0, score)


# ══════════════════════════════════════════════════════════════════════════════
# Task Definitions
# ══════════════════════════════════════════════════════════════════════════════

TASKS: list[AdversaTask] = [
    AdversaTask(
        id="adversa_easy",
        name="First Chair (Easy)",
        description=(
            "You are the Defense Attorney in a clear-cut case. "
            "Win the trial by achieving the correct verdict. "
            "Cases: The Contract (C5), The Landlord (C7). "
            "Graded on verdict correctness and basic jury persuasion."
        ),
        difficulty="easy",
        case_ids=["C5", "C7"],
        role="defense",
        max_steps=30,
        grader=grade_easy,
    ),
    AdversaTask(
        id="adversa_medium",
        name="Strategic Counsel (Medium)",
        description=(
            "You are the Defense Attorney in a complex case. "
            "Win by presenting evidence strategically, objecting to inadmissible "
            "prosecution evidence, and persuading all 3 juror types. "
            "Cases: The Teacher (C1), The Accident (C3), The Intern (C9). "
            "Graded on verdict + evidence strategy + objection mastery."
        ),
        difficulty="medium",
        case_ids=["C1", "C3", "C9"],
        role="defense",
        max_steps=30,
        grader=grade_medium,
    ),
    AdversaTask(
        id="adversa_hard",
        name="Master Litigator (Hard)",
        description=(
            "You are counsel in the most complex cases. "
            "Must demonstrate full strategic mastery: optimal evidence sequencing, "
            "theory-of-mind jury targeting, procedural compliance, and correct verdict. "
            "Cases: The Startup (C2), The Whistleblower (C4), The Mayor (C10). "
            "Graded on all 5 strategic dimensions simultaneously."
        ),
        difficulty="hard",
        case_ids=["C2", "C4", "C10"],
        role="defense",
        max_steps=30,
        grader=grade_hard,
    ),
]

TASKS_BY_ID: dict[str, AdversaTask] = {t.id: t for t in TASKS}


def get_task(task_id: str) -> AdversaTask:
    if task_id not in TASKS_BY_ID:
        raise KeyError(f"Unknown task '{task_id}'. Valid: {list(TASKS_BY_ID)}")
    return TASKS_BY_ID[task_id]


def sample_task_episode(task_id: str, seed: int | None = None) -> dict:
    """Return a sampled episode config (case_id + seed) for a task."""
    task = get_task(task_id)
    rng = random.Random(seed)
    case_id = rng.choice(task.case_ids)
    episode_seed = rng.randint(0, 9999)
    return {
        "task_id": task_id,
        "case_id": case_id,
        "seed": episode_seed,
        "role": task.role,
        "max_steps": task.max_steps,
    }
