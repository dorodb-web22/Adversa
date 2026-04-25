"""
Adversa — Baseline LLM Agent (inference.py)

Role-specific system prompts + greedy action selection.
Used to:
1. Demonstrate the environment works end-to-end
2. Generate baseline trajectories (before GRPO training)
3. Show "before" behavior in the demo

Usage:
    python inference.py --role defense --case C1 --seed 42 --env-url http://localhost:7860
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Optional

import requests

# ══════════════════════════════════════════════════════════════════════════════
# Role-Specific System Prompts
# ══════════════════════════════════════════════════════════════════════════════

ROLE_PROMPTS = {
    "prosecutor": """You are the PROSECUTOR in an Indian courtroom trial.
OBJECTIVE: Prove the defendant is GUILTY beyond reasonable doubt.
INFORMATION: You can ONLY see prosecution evidence. Defense evidence is hidden from you.
STRATEGY: Present evidence strategically. Object to inadmissible defense evidence. 
Tailor arguments to persuade different juror types (analytical, empathetic, skeptical).

Available action types by phase:
- opening: {"action_type": "opening_statement", "argument_text": "...", "framing": "factual|emotional|authority"}
- prosecution_case: {"action_type": "present_evidence", "evidence_id": "E1", "framing": "factual|emotional|authority"}
- defense_case: {"action_type": "object", "objection_type": "hearsay|relevance|coerced|leading", "target": "E6"}
- closing: {"action_type": "closing_argument", "argument_text": "...", "framing": "factual|emotional|authority"}

Respond with ONLY a valid JSON action object. No explanation.""",

    "defense": """You are the DEFENSE ATTORNEY in an Indian courtroom trial.
OBJECTIVE: Prove the defendant is NOT GUILTY / create reasonable doubt.
INFORMATION: You can ONLY see defense evidence. Prosecution evidence is hidden from you.
STRATEGY: Build context before presenting key evidence. Object to inadmissible prosecution 
evidence. Target arguments to specific juror types. Reveal your strongest evidence AFTER 
establishing context (not first!).

Jury types you can see:
- analytical: responds to facts and logic (weight: high evidence, low emotion)
- empathetic: responds to human stories and emotion (weight: high emotion, low facts)  
- skeptical: responds to consistency over time (weight: consistency bonus x2)

Available action types by phase:
- opening: {"action_type": "opening_statement", "argument_text": "...", "framing": "factual|emotional|authority"}
- defense_case: {"action_type": "present_evidence", "evidence_id": "E6", "framing": "factual|emotional|authority"}
- prosecution_case: {"action_type": "object", "objection_type": "hearsay|relevance|coerced|leading", "target": "E3"}
- closing: {"action_type": "closing_argument", "argument_text": "...", "framing": "emotional"}

Respond with ONLY a valid JSON action object. No explanation.""",

    "judge": """You are the JUDGE in an Indian courtroom trial.
OBJECTIVE: Ensure procedural fairness. Rule on objections correctly.
INFORMATION: You see all public arguments but NOT hidden evidence.

Ruling guide:
- sustain objection: when evidence is coerced, hearsay without foundation, or clearly irrelevant
- overrule objection: when objection is frivolous or evidence is clearly admissible

Available actions:
- {"action_type": "sustain"} — uphold the objection, suppress the evidence
- {"action_type": "overrule"} — reject the objection, evidence stands
- {"action_type": "instruct_jury"} — give final jury instructions (verdict phase only)
- {"action_type": "pass"} — no action needed

Respond with ONLY a valid JSON action object. No explanation.""",
}


def build_prompt(role: str, observation: dict) -> str:
    system = ROLE_PROMPTS[role]
    obs_str = json.dumps(
        {k: v for k, v in observation.items() if k != "public_record"
         or len(observation.get("public_record", [])) <= 5},
        indent=2, ensure_ascii=False
    )
    return f"""{system}

--- CURRENT OBSERVATION ---
{obs_str}

Your action (JSON only):"""


# ══════════════════════════════════════════════════════════════════════════════
# Action parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_action(text: str, role: str, observation: dict) -> dict:
    """Extract JSON action from LLM output. Falls back to 'pass' on failure."""
    # Try to find JSON block
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            action = json.loads(match.group())
            if "action_type" in action:
                action["role"] = role
                return action
        except json.JSONDecodeError:
            pass

    # Fallback: select first available action
    available = observation.get("available_actions", ["pass"])
    action_type = available[0] if available else "pass"
    action: dict = {"role": role, "action_type": action_type}

    if action_type == "present_evidence":
        evidence = observation.get("my_evidence", [])
        unpresented = [e for e in evidence if not e.get("presented", False)]
        if unpresented:
            action["evidence_id"] = unpresented[0]["id"]
            action["framing"] = "factual"

    elif action_type == "object":
        public = observation.get("public_record", [])
        opponent = "prosecution" if role == "defense" else "defense"
        opp_evidence = [r for r in public if r.get("role") == opponent
                        and r.get("evidence_id")]
        if opp_evidence:
            action["objection_type"] = "relevance"
            action["target"] = opp_evidence[-1]["evidence_id"]

    elif action_type in ("opening_statement", "closing_argument"):
        action["argument_text"] = (
            "We will demonstrate the truth through evidence and testimony."
            if action_type == "opening_statement"
            else "The evidence clearly supports our position. Justice demands this verdict."
        )
        action["framing"] = "factual"

    return action


# ══════════════════════════════════════════════════════════════════════════════
# Heuristic baseline (no LLM — for testing without API key)
# ══════════════════════════════════════════════════════════════════════════════

def heuristic_action(role: str, observation: dict) -> dict:
    """
    Rule-based baseline agent. Demonstrates 'untrained' behavior:
    - Dumps strongest evidence first (suboptimal)
    - Never objects
    - Ignores jury types
    """
    phase = observation.get("phase", "opening")
    available = observation.get("available_actions", ["pass"])

    if "opening_statement" in available:
        return {
            "role": role,
            "action_type": "opening_statement",
            "argument_text": "The evidence will speak for itself.",
            "framing": "factual",
        }

    if "present_evidence" in available:
        evidence = observation.get("my_evidence", [])
        unpresented = [e for e in evidence if not e.get("presented", False)]
        if unpresented:
            # Untrained: always picks highest strength first (suboptimal timing)
            strongest = max(unpresented, key=lambda e: e.get("strength", 0))
            return {
                "role": role,
                "action_type": "present_evidence",
                "evidence_id": strongest["id"],
                "framing": "factual",  # ignores jury types
            }

    if "closing_argument" in available:
        return {
            "role": role,
            "action_type": "closing_argument",
            "argument_text": "The evidence proves our case beyond doubt.",
            "framing": "factual",
        }

    if "instruct_jury" in available:
        return {"role": role, "action_type": "instruct_jury"}

    return {"role": role, "action_type": "pass"}


# ══════════════════════════════════════════════════════════════════════════════
# Episode runner
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(
    env_url: str,
    role: str,
    case_id: str,
    seed: int,
    use_heuristic: bool = True,
    llm_fn=None,
) -> dict:
    """
    Run one complete trial episode.
    Returns trajectory with all steps and final metrics.
    """
    # Reset
    resp = requests.post(f"{env_url}/reset", json={
        "seed": seed,
        "options": {"case_id": case_id, "role": role}
    }, timeout=30)
    resp.raise_for_status()
    obs = resp.json()["observation"]

    trajectory = []
    total_reward = 0.0

    for step_num in range(30):
        # Select action
        if use_heuristic or llm_fn is None:
            action = heuristic_action(role, obs)
        else:
            prompt = build_prompt(role, obs)
            raw_output = llm_fn(prompt)
            action = parse_action(raw_output, role, obs)

        # Step
        step_resp = requests.post(f"{env_url}/step", json={"action": action}, timeout=30)
        step_resp.raise_for_status()
        step_data = step_resp.json()

        reward = step_data["reward"]
        done = step_data["done"]
        info = step_data.get("info", {})
        obs = step_data["observation"]

        # Get full state for grading
        state_resp = requests.get(f"{env_url}/state", timeout=10)
        state = state_resp.json()

        trajectory.append({
            "step": step_num + 1,
            "action": action,
            "reward": reward,
            "done": done,
            "info": info,
            "state": state,
        })

        total_reward += reward

        if done:
            break

    final_state = requests.get(f"{env_url}/state", timeout=10).json()
    return {
        "trajectory": trajectory,
        "total_reward": total_reward,
        "steps": len(trajectory),
        "verdict": final_state.get("verdict"),
        "verdict_correct": final_state.get("verdict_correct"),
        "jury_sentiment": final_state.get("jury_sentiment"),
        "case_id": case_id,
        "seed": seed,
        "role": role,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Adversa Baseline Agent")
    parser.add_argument("--role", default="defense",
                        choices=["prosecutor", "defense", "judge"])
    parser.add_argument("--case", default="C1",
                        help="Case ID (C1–C10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-url", default="http://localhost:7860",
                        help="URL of the Adversa environment server")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Adversa Baseline Agent")
    print(f"Role: {args.role} | Case: {args.case} | Seed: {args.seed}")
    print(f"{'='*60}\n")

    all_results = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        print(f"Episode {ep+1}/{args.episodes} (seed={seed})...", end=" ")
        result = run_episode(
            env_url=args.env_url,
            role=args.role,
            case_id=args.case,
            seed=seed,
            use_heuristic=True,
        )
        all_results.append(result)
        verdict_icon = "✅" if result["verdict_correct"] else "❌"
        print(f"{verdict_icon} Verdict: {result['verdict']} | "
              f"Reward: {result['total_reward']:.2f} | Steps: {result['steps']}")

        if args.verbose:
            for step in result["trajectory"]:
                print(f"  Step {step['step']:2d} | {step['action'].get('action_type'):20s} "
                      f"| reward={step['reward']:+.3f}")

    # Summary
    correct = sum(1 for r in all_results if r["verdict_correct"])
    avg_reward = sum(r["total_reward"] for r in all_results) / len(all_results)
    print(f"\n{'='*60}")
    print(f"Summary: {correct}/{args.episodes} correct verdicts | "
          f"Avg reward: {avg_reward:.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
