"""
Adversa — Environment Tests
Run with: pytest tests/ -v
"""
import sys
import pytest
sys.path.insert(0, ".")

from server.adversa_environment import AdversaEnvironment
from server.case_templates import ALL_CASES, get_case
from server.courtroom_state import compute_juror_shift, JurorState, JurorProfile
from server.case_templates import Evidence
from models import AdversaAction, AdversaObservation, get_valid_actions, Phase, Role
from tasks import TASKS, get_task, grade_easy, grade_medium, grade_hard


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    e = AdversaEnvironment()
    e.reset(seed=42, options={"case_id": "C1", "role": "defense"})
    return e


@pytest.fixture
def full_env():
    """Environment with both roles ready to play."""
    e = AdversaEnvironment()
    e.reset(seed=0, options={"case_id": "C5"})
    return e


# ── Import tests ──────────────────────────────────────────────────────────────

def test_models_import():
    action = AdversaAction(action_type="pass")
    assert action.action_type == "pass"


def test_action_validation_present_evidence_requires_evidence_id():
    with pytest.raises(Exception):
        AdversaAction(action_type="present_evidence")  # missing evidence_id


def test_action_validation_object_requires_objection_type():
    with pytest.raises(Exception):
        AdversaAction(action_type="object", target="E1")  # missing objection_type


# ── Case loading ──────────────────────────────────────────────────────────────

def test_all_10_cases_load():
    for cid in ["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10"]:
        case = get_case(cid, seed=42)
        assert len(case.prosecution_evidence) > 0
        assert len(case.defense_evidence) > 0
        assert case.ground_truth in ("guilty", "not_guilty")


def test_seed_reproducibility():
    case_a = get_case("C1", seed=999)
    case_b = get_case("C1", seed=999)
    assert case_a.prosecution_evidence[0].strength == case_b.prosecution_evidence[0].strength


def test_different_seeds_produce_different_strengths():
    case_a = get_case("C1", seed=1)
    case_b = get_case("C1", seed=2)
    # Very unlikely to be identical
    strengths_a = [e.strength for e in case_a.prosecution_evidence]
    strengths_b = [e.strength for e in case_b.prosecution_evidence]
    assert strengths_a != strengths_b


# ── OpenEnv API compliance ────────────────────────────────────────────────────

def test_reset_returns_correct_keys():
    e = AdversaEnvironment()
    obs = e.reset(seed=42, options={"case_id": "C1", "role": "defense"})
    required = ["role", "phase", "step", "max_steps", "my_evidence",
                "public_record", "jury_sentiment", "available_actions",
                "case_name", "charges"]
    for k in required:
        assert k in obs, f"Missing key: {k}"


def test_step_returns_four_tuple():
    e = AdversaEnvironment()
    e.reset(seed=0, options={"case_id": "C5"})
    obs, reward, done, info = e.step({
        "role": "prosecutor", "action_type": "opening_statement",
        "argument_text": "x", "framing": "factual"
    })
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_state_returns_required_keys():
    e = AdversaEnvironment()
    e.reset(seed=0)
    state = e.state()
    required = ["case_id", "phase", "step", "done", "verdict",
                "ground_truth", "jury_sentiment", "verdict_correct"]
    for k in required:
        assert k in state, f"Missing key: {k}"


# ── Role isolation ────────────────────────────────────────────────────────────

def test_prosecution_does_not_see_defense_evidence():
    e = AdversaEnvironment()
    obs = e.reset(seed=0, options={"case_id": "C1", "role": "prosecutor"})
    ids = {ev["id"] for ev in obs["my_evidence"]}
    defense_ids = {"E6", "E7", "E8", "E9", "E10"}
    assert not ids.intersection(defense_ids), f"Leaked defense evidence: {ids & defense_ids}"


def test_defense_does_not_see_prosecution_evidence():
    e = AdversaEnvironment()
    obs = e.reset(seed=0, options={"case_id": "C1", "role": "defense"})
    ids = {ev["id"] for ev in obs["my_evidence"]}
    prosecution_ids = {"E1", "E2", "E3", "E4", "E5"}
    assert not ids.intersection(prosecution_ids), f"Leaked prosecution evidence: {ids & prosecution_ids}"


# ── Phase-specific available actions ─────────────────────────────────────────

def test_opening_phase_actions():
    e = AdversaEnvironment()
    obs = e.reset(seed=0, options={"case_id": "C1", "role": "prosecutor"})
    assert obs["phase"] == "opening"
    assert "opening_statement" in obs["available_actions"]
    assert "present_evidence" not in obs["available_actions"]


def test_prosecution_case_defense_cannot_present_evidence():
    e = AdversaEnvironment()
    e.reset(seed=0, options={"case_id": "C1", "role": "defense"})
    ct = e._courtroom
    # Advance to prosecution_case
    e.step({"role": "prosecutor", "action_type": "opening_statement", "argument_text": "x", "framing": "factual"})
    e.step({"role": "defense", "action_type": "opening_statement", "argument_text": "x", "framing": "emotional"})
    assert ct.phase == "prosecution_case"
    obs_d = e._build_observation("defense").model_dump()
    assert "present_evidence" not in obs_d["available_actions"]
    assert "object" in obs_d["available_actions"]


# ── Wrong-turn handling (Bug #1 fix) ─────────────────────────────────────────

def test_wrong_turn_does_not_skip_correct_role():
    e = AdversaEnvironment()
    e.reset(seed=42, options={"case_id": "C1", "role": "defense"})
    ct = e._courtroom
    # Advance to prosecution_case
    e.step({"role": "prosecutor", "action_type": "opening_statement", "argument_text": "x", "framing": "factual"})
    e.step({"role": "defense", "action_type": "opening_statement", "argument_text": "x", "framing": "emotional"})
    assert ct.current_role == "prosecutor"

    # Defense acts out of turn
    _, r, _, info = e.step({"role": "defense", "action_type": "present_evidence",
                            "evidence_id": "E6", "framing": "factual"})
    assert r == -0.5
    assert "not_your_turn" in info.get("error", "")
    # Prosecutor must still be current role
    assert ct.current_role == "prosecutor"

    # Prosecutor can now act correctly
    _, r2, _, _ = e.step({"role": "prosecutor", "action_type": "present_evidence",
                          "evidence_id": "E1", "framing": "factual"})
    assert r2 > 0, f"Expected positive reward after correct action, got {r2}"


# ── Reward correctness ────────────────────────────────────────────────────────

def test_evidence_gives_positive_reward():
    e = AdversaEnvironment()
    e.reset(seed=0, options={"case_id": "C1"})
    # Skip opening
    e.step({"role": "prosecutor", "action_type": "opening_statement", "argument_text": "x", "framing": "factual"})
    e.step({"role": "defense", "action_type": "opening_statement", "argument_text": "x", "framing": "emotional"})
    # Prosecutor presents strong evidence
    _, r, _, _ = e.step({"role": "prosecutor", "action_type": "present_evidence",
                         "evidence_id": "E1", "framing": "factual"})
    assert r > 0, f"Evidence presentation should give positive reward, got {r}"


def test_timing_bonus_fires_for_strong_evidence_presented_third():
    e = AdversaEnvironment()
    e.reset(seed=0, options={"case_id": "C1"})
    e.step({"role": "prosecutor", "action_type": "opening_statement", "argument_text": "x", "framing": "factual"})
    e.step({"role": "defense", "action_type": "opening_statement", "argument_text": "x", "framing": "emotional"})
    # Present 2 weak items first
    e.step({"role": "prosecutor", "action_type": "present_evidence", "evidence_id": "E4", "framing": "factual"})
    e.step({"role": "defense", "action_type": "pass"})
    e.step({"role": "prosecutor", "action_type": "present_evidence", "evidence_id": "E5", "framing": "factual"})
    e.step({"role": "defense", "action_type": "pass"})
    # Now present strong evidence (E1, strength=~0.9) — should get timing bonus
    _, r, _, _ = e.step({"role": "prosecutor", "action_type": "present_evidence",
                         "evidence_id": "E1", "framing": "factual"})
    state = e.state()
    assert state["prosecution_timing_score"] > 0, "Timing bonus should have fired"


def test_verdict_reward_only_on_done():
    e = AdversaEnvironment()
    e.reset(seed=0, options={"case_id": "C5"})
    ct = e._courtroom
    rewards = []
    for _ in range(40):
        current = ct.current_role
        avail = e._build_observation(current).model_dump()["available_actions"]
        ev = [x for x in e._build_observation(current).model_dump().get("my_evidence", []) if not x.get("presented")]
        if "present_evidence" in avail and ev:
            action = {"role": current, "action_type": "present_evidence", "evidence_id": ev[0]["id"], "framing": "factual"}
        elif "opening_statement" in avail:
            action = {"role": current, "action_type": "opening_statement", "argument_text": "x", "framing": "factual"}
        elif "closing_argument" in avail:
            action = {"role": current, "action_type": "closing_argument", "argument_text": "x", "framing": "factual"}
        elif "instruct_jury" in avail:
            action = {"role": current, "action_type": "instruct_jury"}
        else:
            action = {"role": current, "action_type": "pass"}
        _, r, done, _ = e.step(action)
        rewards.append((r, done))
        if done:
            break
    final_r, final_done = rewards[-1]
    pre_rewards = [r for r, d in rewards[:-1]]
    assert final_done is True
    assert abs(final_r) >= 1.0, f"Final reward should include verdict signal: {final_r}"
    assert all(abs(r) < 3.0 for r in pre_rewards), f"Pre-done rewards too large: {max(abs(r) for r in pre_rewards)}"


# ── Jury system ───────────────────────────────────────────────────────────────

def test_factual_framing_moves_analytical_most():
    evidence = Evidence("E_T", "Test", "prosecution", 0.8, 0.6, True, 0.5)
    analytical = JurorState(JurorProfile("analytical", 1.5, 0.3, 0.5, 0.5, 0.5), 0.5)
    empathetic = JurorState(JurorProfile("empathetic", 0.5, 1.5, 0.7, 0.3, 0.5), 0.5)
    shift_a = compute_juror_shift(evidence, "factual", analytical, False)
    shift_e = compute_juror_shift(evidence, "factual", empathetic, False)
    assert shift_a > shift_e, f"Factual should move analytical more: {shift_a} vs {shift_e}"


def test_emotional_framing_moves_empathetic_most():
    evidence = Evidence("E_T", "Test", "prosecution", 0.8, 0.6, True, 0.5)
    analytical = JurorState(JurorProfile("analytical", 1.5, 0.3, 0.5, 0.5, 0.5), 0.5)
    empathetic = JurorState(JurorProfile("empathetic", 0.5, 1.5, 0.7, 0.3, 0.5), 0.5)
    shift_a = compute_juror_shift(evidence, "emotional", analytical, False)
    shift_e = compute_juror_shift(evidence, "emotional", empathetic, False)
    assert shift_e > shift_a, f"Emotional should move empathetic more: {shift_e} vs {shift_a}"


def test_consistency_bonus_moves_skeptical_more():
    evidence = Evidence("E_T", "Test", "prosecution", 0.8, 0.6, True, 0.5)
    skeptical = JurorState(JurorProfile("skeptical", 1.0, 0.2, 0.3, 2.0, 0.5), 0.5)
    shift_no = compute_juror_shift(evidence, "factual", skeptical, False)
    shift_yes = compute_juror_shift(evidence, "factual", skeptical, True)
    assert shift_yes > shift_no


def test_jury_majority_vote():
    j1 = JurorState(JurorProfile("analytical", 1.5, 0.3, 0.5, 0.5, 0.5), 0.7)  # guilty
    j2 = JurorState(JurorProfile("empathetic", 0.5, 1.5, 0.7, 0.3, 0.5), 0.3)  # not_guilty
    j3 = JurorState(JurorProfile("skeptical",  1.0, 0.2, 0.3, 2.0, 0.5), 0.6)  # guilty
    votes = [j1.vote(), j2.vote(), j3.vote()]
    assert votes.count("guilty") >= 2


# ── Full episode ──────────────────────────────────────────────────────────────

def test_full_episode_terminates():
    e = AdversaEnvironment()
    e.reset(seed=7, options={"case_id": "C5"})
    ct = e._courtroom
    for _ in range(60):
        current = ct.current_role
        avail = e._build_observation(current).model_dump()["available_actions"]
        ev = [x for x in e._build_observation(current).model_dump().get("my_evidence", []) if not x.get("presented")]
        if "present_evidence" in avail and ev:
            action = {"role": current, "action_type": "present_evidence", "evidence_id": ev[0]["id"], "framing": "factual"}
        elif "opening_statement" in avail:
            action = {"role": current, "action_type": "opening_statement", "argument_text": "x", "framing": "factual"}
        elif "closing_argument" in avail:
            action = {"role": current, "action_type": "closing_argument", "argument_text": "x", "framing": "factual"}
        elif "instruct_jury" in avail:
            action = {"role": current, "action_type": "instruct_jury"}
        else:
            action = {"role": current, "action_type": "pass"}
        _, _, done, _ = e.step(action)
        if done:
            break
    state = e.state()
    assert state["done"] is True
    assert state["verdict"] in ("guilty", "not_guilty")
    assert state["verdict_correct"] is not None


def test_verdict_never_none_when_done():
    for case_id in ["C1", "C5", "C9"]:
        e = AdversaEnvironment()
        e.reset(seed=0, options={"case_id": case_id})
        ct = e._courtroom
        for _ in range(60):
            current = ct.current_role
            e.step({"role": current, "action_type": "pass"})
            if ct.done:
                break
        state = e.state()
        assert state["verdict"] is not None, f"Verdict is None for {case_id}"


# ── Tasks ─────────────────────────────────────────────────────────────────────

def test_all_three_tasks_defined():
    assert "adversa_easy" in [t.id for t in TASKS]
    assert "adversa_medium" in [t.id for t in TASKS]
    assert "adversa_hard" in [t.id for t in TASKS]


def test_graders_return_float_0_to_1():
    dummy_correct = [{"state": {"verdict_correct": True, "jury_sentiment": {"a": 0.7, "b": 0.7, "c": 0.7},
                                "presented_evidence": ["E1","E2","E3","E4"],
                                "sustained_objections": 1, "total_objections": 1,
                                "prosecution_timing_score": 2.0, "defense_timing_score": 2.0,
                                "procedural_violations": 0}}]
    dummy_wrong = [{"state": {"verdict_correct": False, "jury_sentiment": {"a": 0.3, "b": 0.3, "c": 0.3},
                              "presented_evidence": [], "sustained_objections": 0, "total_objections": 0,
                              "prosecution_timing_score": 0.0, "defense_timing_score": 0.0,
                              "procedural_violations": 5}}]
    for grade_fn in [grade_easy, grade_medium, grade_hard]:
        r_correct = grade_fn(dummy_correct)
        r_wrong = grade_fn(dummy_wrong)
        assert 0.0 <= r_correct <= 1.0
        assert 0.0 <= r_wrong <= 1.0
        assert r_correct > r_wrong, f"{grade_fn.__name__}: correct should score higher"
